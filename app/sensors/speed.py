"""
Speed monitoring.

A detection model gives boxes in image space. Pixels per frame is not a
speed, and the conversion is not a scale factor: a road receding from the
camera compresses metres into fewer and fewer pixels, so the same physical
speed measured near the stop line and at the far end of the zone differs by
a large factor. Every measurement here is therefore taken *after* projecting
the vehicle's ground-contact point through the approach homography into road
coordinates in metres.

Two further rules, both of which exist to stop this module reporting numbers
it has not earned:

* A speed requires the same vehicle in at least two frames. A single pass of
  the detector yields tracks with no history and therefore no speed, and the
  sample count says so rather than the mean quietly reading zero.
* Displacements implying an implausible speed are discarded as track swaps
  rather than averaged in. A detector that drops a frame and re-associates
  two different cars will otherwise produce 300 km/h and poison the mean.

The reported statistic is the 85th percentile as well as the mean, because
the 85th percentile is what speed studies and limit-setting actually use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


def project(homography: Sequence[Sequence[float]], x: float, y: float) -> Tuple[float, float]:
    """Map an image point to road-plane metres through a 3×3 homography."""
    h = homography
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(w) < 1e-12:
        raise ValueError("degenerate homography: point maps to the line at infinity")
    return (
        (h[0][0] * x + h[0][1] * y + h[0][2]) / w,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / w,
    )


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. No numpy dependency for 20 numbers."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass
class Track:
    """One vehicle being followed across frames, in road coordinates."""

    track_id: int
    x: float
    y: float
    age: float = 0.0
    missing: float = 0.0
    speeds: List[float] = field(default_factory=list)

    @property
    def speed(self) -> float:
        """Smoothed speed in km/h over this track's observed history."""
        if not self.speeds:
            return 0.0
        recent = self.speeds[-5:]
        return sum(recent) / len(recent)


@dataclass
class SpeedStats:
    mean: float = 0.0
    p85: float = 0.0
    samples: int = 0
    speeding: int = 0


class ApproachSpeedTracker:
    """Nearest-neighbour tracker over road-plane positions for one approach.

    Association is done in metres with a gate derived from the maximum
    plausible speed, which is the right space for it: an image-space gate is
    tight at the horizon and loose at the stop line for no physical reason.
    """

    def __init__(
        self,
        homography: Sequence[Sequence[float]],
        speed_limit_kmh: float = 50.0,
        max_plausible_kmh: float = 160.0,
        #: Association gate as a multiple of the maximum plausible
        #: displacement. Deliberately larger than 1: if the gate equalled
        #: the plausibility limit, a track swap would simply fail to
        #: associate and would be silently replaced by a fresh track. A
        #: generous gate followed by a strict speed check means the swap is
        #: caught, counted and excluded instead of disappearing.
        assoc_slack: float = 1.6,
        drop_after: float = 1.5,
        window: int = 40,
    ) -> None:
        self.homography = homography
        self.speed_limit = speed_limit_kmh
        self.max_plausible = max_plausible_kmh
        self.assoc_slack = assoc_slack
        self.drop_after = drop_after
        self.window = window
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._recent: List[float] = []
        self._rejected = 0

    # ───────────────────────── tracking ─────────────────────────

    def update(self, ground_points: Sequence[Tuple[float, float]], dt: float) -> SpeedStats:
        """Associate this frame's detections and return the speed statistics.

        ``ground_points`` are image-space ground-contact points — the bottom
        centre of each box, not its centroid, since the centroid of a tall
        vehicle projects well behind where it is standing.
        """
        if dt <= 0:
            return self.stats()

        observed = []
        for px, py in ground_points:
            try:
                observed.append(project(self.homography, px, py))
            except ValueError:
                continue

        gate = (self.max_plausible * self.assoc_slack / 3.6) * dt
        unmatched = set(range(len(observed)))
        for track in self._tracks.values():
            best: Optional[int] = None
            best_d = gate
            for i in unmatched:
                d = math.dist((track.x, track.y), observed[i])
                if d < best_d:
                    best, best_d = i, d
            if best is None:
                track.missing += dt
                continue
            unmatched.discard(best)
            kmh = (best_d / dt) * 3.6
            if kmh <= self.max_plausible:
                track.speeds.append(kmh)
                self._recent.append(kmh)
                del self._recent[: max(0, len(self._recent) - self.window)]
            else:
                self._rejected += 1
            track.x, track.y = observed[best]
            track.age += dt
            track.missing = 0.0

        for i in unmatched:
            self._tracks[self._next_id] = Track(
                track_id=self._next_id, x=observed[i][0], y=observed[i][1]
            )
            self._next_id += 1

        for tid in [t for t, tr in self._tracks.items() if tr.missing > self.drop_after]:
            del self._tracks[tid]

        return self.stats()

    def stats(self) -> SpeedStats:
        """Statistics over the sample window.

        ``samples`` is the number of speed measurements behind the figures.
        Zero samples means no vehicle has been seen twice yet, and the mean
        is reported as 0.0 — which downstream code must read as *unmeasured*,
        which is why the count travels with it.
        """
        moving = [t.speed for t in self._tracks.values() if t.speeds]
        if not moving:
            return SpeedStats(samples=0)
        return SpeedStats(
            mean=round(sum(moving) / len(moving), 1),
            p85=round(percentile(self._recent or moving, 0.85), 1),
            samples=len(self._recent),
            speeding=sum(1 for s in moving if s > self.speed_limit),
        )

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    @property
    def rejected_associations(self) -> int:
        """Displacements discarded as implausible. A rising count means the
        detector is dropping frames or the gate is mis-tuned."""
        return self._rejected
