"""
Camera sensor: YOLO detection, stop-line zone counting, and road-plane
speed measurement.

What this gives you, and what it does not
-----------------------------------------
Object detection yields counts and positions in image space. Queue length,
occupancy, arrival gap and speed are *traffic* quantities, so each one needs
a homography from the camera to the road plane and a hand-drawn stop-line
polygon per approach. Those live in a calibration file; without one this
sensor refuses to start rather than returning pixel counts dressed up as
vehicle queues.

Speed is measured by tracking ground-contact points across frames in road
coordinates (``speed.py``), not inferred from occupancy. A single frame
produces no speed at all and reports zero samples, which is the honest
answer: one observation of a car cannot tell you how fast it is going.

Glare is treated as a sensor-integrity problem rather than a cosmetic one.
When the UV/light sensor reports low-angle glare on an approach and mean
detection confidence on that approach has collapsed, the approach is
reported ``degraded``. The controller then places it on constant recall
instead of reading the thin detections as an empty road.

Install the extras before selecting this backend::

    pip install "ultralytics>=8.1" "opencv-python-headless>=4.9"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from app.core.phases import APPROACHES, CROSSWALKS
from app.core.types import ApproachReading, SensorFrame, SignalCommand

from .base import TrafficSensor
from .environment import ModelledUVSensor, UVLightSensor
from .speed import ApproachSpeedTracker

logger = logging.getLogger(__name__)


class CameraSensor(TrafficSensor):
    name = "camera"

    #: COCO class ids for road vehicles.
    VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    def __init__(
        self,
        source: str = "0",
        weights: str = "yolov8n.pt",
        calibration: Optional[str] = None,
        confidence: float = 0.35,
        environment: Optional[UVLightSensor] = None,
        glare_confidence_floor: float = 0.45,
    ) -> None:
        try:
            import cv2  # noqa: F401
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "camera backend needs 'ultralytics' and 'opencv-python-headless'"
            ) from exc

        if not calibration or not Path(calibration).exists():
            raise RuntimeError(
                "camera backend needs a calibration file mapping each approach to a "
                "stop-line polygon and a road-plane homography; see "
                "docs/camera-calibration.md"
            )

        import cv2

        self.zones: Dict[str, dict] = json.loads(Path(calibration).read_text())
        missing = [a for a in APPROACHES if a not in self.zones]
        if missing:
            raise RuntimeError(f"calibration is missing approaches: {missing}")
        without_h = [a for a in APPROACHES if "homography" not in self.zones[a]]
        if without_h:
            raise RuntimeError(
                "calibration has no image-to-road homography for approaches "
                f"{without_h}; speed and queue length are road-plane quantities "
                "and cannot be derived from pixels without one"
            )

        self.model = YOLO(weights)
        self.confidence = confidence
        self.glare_confidence_floor = glare_confidence_floor
        self.environment = environment or ModelledUVSensor()
        self.capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video source {source!r}")
        self._prev_counts: Dict[str, int] = {a: 0 for a in APPROACHES}
        self._gap: Dict[str, float] = {a: 99.0 for a in APPROACHES}
        self._wait: Dict[str, float] = {a: 0.0 for a in APPROACHES}
        self._trackers: Dict[str, ApproachSpeedTracker] = {
            a: ApproachSpeedTracker(
                homography=self.zones[a]["homography"],
                speed_limit_kmh=float(self.zones[a].get("speed_limit_kmh", 50.0)),
            )
            for a in APPROACHES
        }

    def read(self, dt: float, command: Optional[SignalCommand] = None) -> SensorFrame:
        import cv2
        import numpy as np

        ok, image = self.capture.read()
        if not ok:
            raise RuntimeError("video source returned no frame")

        result = self.model.predict(
            image,
            conf=self.confidence,
            verbose=False,
            classes=sorted(self.VEHICLE_CLASSES),
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        env = self.environment.read()
        approaches = {}
        degraded = []

        for approach in APPROACHES:
            zone = np.array(self.zones[approach]["stop_line_polygon"], dtype=np.int32)
            lanes = int(self.zones[approach].get("lanes", 2))
            area_m2 = float(self.zones[approach].get("zone_area_m2", 60.0))
            limit = float(self.zones[approach].get("speed_limit_kmh", 50.0))

            inside = 0
            ground_points = []
            confidences = []
            for box, score in zip(boxes, scores):
                cx = float((box[0] + box[2]) / 2)
                cy = float(box[3])  # ground contact point, not box centre
                if cv2.pointPolygonTest(zone, (cx, cy), False) >= 0:
                    inside += 1
                    ground_points.append((cx, cy))
                    confidences.append(float(score))

            speed = self._trackers[approach].update(ground_points, dt)

            arrived = max(0, inside - self._prev_counts[approach])
            self._gap[approach] = 0.0 if arrived else self._gap[approach] + dt
            serving = bool(command and approach in command.greens())
            self._wait[approach] = (
                0.0 if serving or inside == 0 else self._wait[approach] + dt
            )
            self._prev_counts[approach] = inside

            # Glare on this leg plus weak detections means the counts cannot
            # be trusted. Saying so is better than passing a thin frame to
            # the controller as though the road were empty.
            mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
            is_degraded = (
                approach in env.approaches_affected
                and mean_conf < self.glare_confidence_floor
            )
            if is_degraded:
                degraded.append(approach)

            approaches[approach] = ApproachReading(
                queue=inside,
                flow=round(arrived / max(dt, 1e-6) * 60.0, 2),
                occupancy=round(min(1.0, inside * 7.0 * lanes / max(area_m2, 1.0)), 3),
                mean_speed=speed.mean,
                speed_85th=speed.p85,
                speed_limit=limit,
                speeding=speed.speeding,
                speed_samples=speed.samples,
                gap=round(self._gap[approach], 2),
                max_wait=round(self._wait[approach], 1),
                degraded=is_degraded,
            )

        return SensorFrame(
            approaches=approaches,
            pedestrian_calls={c: False for c in CROSSWALKS},
            environment=env,
            degraded_approaches=degraded,
        )

    def health(self) -> dict:
        return {
            "sensor": self.name,
            "status": "ok",
            "tracks": {a: t.active_tracks for a, t in self._trackers.items()},
            "rejected_associations": {
                a: t.rejected_associations for a, t in self._trackers.items()
            },
        }

    def close(self) -> None:
        try:
            self.capture.release()
        except Exception:  # pragma: no cover
            logger.warning("failed to release video capture", exc_info=True)
