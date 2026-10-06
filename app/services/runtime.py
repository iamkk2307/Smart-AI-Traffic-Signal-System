"""
The runtime: one object that owns one intersection.

Everything the API exposes reads from here, and the control loop is the only
writer. The loop is deliberately boring — read a frame, tick the controller,
verify the command, record — because each of those steps is independently
testable and the interesting logic lives in ``app.core``.

Two things are worth pointing out. First, the interlock runs on every
command and trips the controller to flashing red on any finding, so a
controller bug degrades to a known-safe failure instead of an unknown
unsafe one. Second, the forecaster is trained in a worker thread on a timer
and only *promoted* if it beats a persistence baseline, so a bad retrain
cannot stall the control loop or quietly make timing worse.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional

from app.config import Settings, get_settings
from app.core.controller import AdaptiveController
from app.core.phases import APPROACHES, CROSSWALKS, PHASES
from app.core.safety import SafetyInterlock
from app.core.types import SensorFrame, SignalCommand
from app.db import IncidentRecord, ModelMetric, PhaseLog, SafetyLog, SessionLocal, TrafficLog
from app.ml.features import minute_of_week
from app.ml.incidents import IncidentDetector
from app.ml.predictor import ArrivalForecaster
from app.sensors import create_sensor
from app.sensors.environment import ModelledUVSensor, PushedUVSensor
from app.sim.intersection import SimConfig

logger = logging.getLogger(__name__)


class IntersectionRuntime:
    def __init__(self, settings: Optional[Settings] = None,
                 intersection_id: int = 1) -> None:
        self.settings = settings or get_settings()
        self.intersection_id = intersection_id
        self.timing = self.settings.timing_plan()

        self.sensor = create_sensor(
            self.settings.sensor_mode,
            **self._sensor_kwargs(),
        )
        self.forecaster = ArrivalForecaster(
            horizon=self.settings.forecast_horizon,
            model_dir=self.settings.model_dir,
        )
        restored = self.forecaster.load()
        if restored:
            logger.info("restored forecasters for %s", ", ".join(restored))

        self.controller = AdaptiveController(
            timing=self.timing,
            mode=self.settings.control_mode,
            forecast=(
                self.forecaster.as_callable()
                if self.settings.forecast_enabled
                else None
            ),
        )
        self.interlock = SafetyInterlock()
        self.detector = IncidentDetector()

        self.command: Optional[SignalCommand] = None
        self.frame: Optional[SensorFrame] = None
        self.history: Deque[dict] = deque(maxlen=self.settings.live_history)
        self.decision_feed: Deque[dict] = deque(maxlen=60)
        self.incident_feed: Deque[dict] = deque(maxlen=60)
        self.safety_feed: Deque[dict] = deque(maxlen=40)

        self._decisions_seen = 0
        self._last_log = 0.0
        self._last_train = 0.0
        self._last_minute = -1
        self._training = False
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def _sensor_kwargs(self) -> dict:
        """Arguments for the selected backend only.

        Passing camera arguments to the MQTT backend (or the reverse) would
        be accepted silently by ``**kwargs`` and then ignored, so each mode
        gets exactly what it understands.
        """
        settings = self.settings
        mode = settings.sensor_mode
        if mode == "simulation":
            return {
                "sim_config": SimConfig(
                    latitude=settings.site_latitude,
                    longitude=settings.site_longitude,
                    utc_offset_hours=settings.site_utc_offset_hours,
                ),
                "timing": self.timing,
            }
        if mode == "camera":
            return {
                "source": settings.camera_source,
                "weights": settings.camera_weights,
                "calibration": settings.camera_calibration,
                "environment": ModelledUVSensor(**settings.uv_sensor_kwargs()),
                "glare_confidence_floor": settings.camera_glare_confidence_floor,
            }
        if mode == "iot":
            return {
                "broker_host": settings.mqtt_host,
                "broker_port": settings.mqtt_port,
                "prefix": settings.mqtt_prefix,
                "username": settings.mqtt_username,
                "password": settings.mqtt_password,
                "max_age": settings.iot_detector_max_age,
                "emergency_ttl": settings.iot_emergency_ttl,
                "environment": PushedUVSensor(
                    max_age=settings.uv_max_age, **settings.uv_sensor_kwargs()
                ),
            }
        return {}

    # ───────────────────────── loop ─────────────────────────

    def step(self, dt: Optional[float] = None) -> SignalCommand:
        """Advance one tick. Synchronous, so tests can drive it directly."""
        dt = dt or self.settings.tick_seconds

        frame = self.sensor.read(dt, self.command)
        command = self.controller.tick(frame, dt)

        findings = self.interlock.check(self.controller.t, command)
        if findings:
            detail = "; ".join(f.detail for f in findings)
            logger.error("interlock tripped: %s", detail)
            self.controller.trip_fault(detail)
            command = self.controller.tick(frame, 0.001)
            for finding in findings:
                self.safety_feed.appendleft(finding.to_dict())
            self._persist_safety(findings)

        # Clear the pedestrian pushbuttons the controller has now served.
        served = [
            c for c, w in command.pedestrian.items() if w.value != "DONT_WALK"
        ]
        if served and hasattr(self.sensor, "sim"):
            self.sensor.sim.clear_ped_calls(served)

        for incident in self.detector.update(frame, command, dt):
            payload = incident.to_dict()
            if payload not in self.incident_feed:
                self.incident_feed.appendleft(payload)
                self._persist_incident(incident)

        self.frame = frame
        self.command = command
        self._collect_decisions()
        self._aggregate_for_ml(frame)
        self._record_sample(frame, command)
        self._maybe_log(frame, command)
        self._maybe_train()

        return command

    async def run(self) -> None:
        self._running = True
        interval = self.settings.tick_seconds * self.settings.realtime_factor
        logger.info(
            "control loop started: mode=%s sensor=%s tick=%.2fs",
            self.settings.control_mode,
            self.settings.sensor_mode,
            interval,
        )
        try:
            while self._running:
                started = time.perf_counter()
                async with self._lock:
                    try:
                        self.step()
                    except Exception:
                        logger.exception("control tick failed; holding all-red")
                        self.controller.trip_fault("control tick raised")
                drift = time.perf_counter() - started
                await asyncio.sleep(max(0.0, interval - drift))
        except asyncio.CancelledError:
            logger.info("control loop cancelled")
            raise

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.sensor.close()

    # ───────────────────────── operator actions ─────────────────────────

    def place_ped_call(self, crosswalk: str) -> dict:
        self.controller.place_ped_call(crosswalk)
        if hasattr(self.sensor, "sim"):
            self.sensor.sim.inject_ped_call(crosswalk)
        return {"crosswalk": crosswalk, "queued": True}

    def inject_emergency(self, approach: str, kind: str = "ambulance",
                         duration: float = 25.0) -> dict:
        if approach not in APPROACHES:
            raise ValueError(f"unknown approach {approach!r}")
        if not hasattr(self.sensor, "sim"):
            raise RuntimeError(
                "emergency injection is a simulation facility; with a live sensor "
                "the preemption input comes from the vehicle's transponder"
            )
        self.sensor.sim.inject_emergency(approach, kind, duration)
        return {
            "approach": approach,
            "kind": kind,
            "duration": duration,
            "budget_remaining": round(
                max(0.0, self.timing.preempt_budget - self.controller.preempt_used), 1
            ),
        }

    def set_weather(self, weather: str) -> dict:
        if hasattr(self.sensor, "sim"):
            self.sensor.sim.set_weather(weather)
        glare = bool(self.frame and self.frame.environment.glare)
        yellow, all_red = self.timing.clearance_for(weather, glare)
        return {
            "weather": weather,
            "glare": glare,
            "yellow": round(yellow, 2),
            "all_red": round(all_red, 2),
            "discharge_rate": round(self.timing.discharge_rate(weather, glare), 3),
        }

    def set_clock(self, hour: float) -> dict:
        """Move the simulated wall clock, in local hours.

        Glare depends on where the sun is, so demonstrating it means being
        able to put the junction at 07:00 or 17:30. Only the simulation
        backend has a clock to move; the camera and MQTT backends read the
        real one.
        """
        if not hasattr(self.sensor, "sim"):
            raise ValueError("only the simulation backend has a settable clock")
        sim = self.sensor.sim
        hour = float(hour) % 24.0
        sim.start_time = sim.start_time.replace(
            hour=int(hour), minute=int((hour % 1) * 60), second=0, microsecond=0
        ) - timedelta(seconds=sim.t * sim.config.time_scale)
        env = sim.environment.read(sim.solar_time, sim.weather)
        return {
            "wall_time": sim.wall_time.isoformat(),
            "environment": asdict(env),
        }

    def set_mode(self, mode: str) -> dict:
        if mode not in ("adaptive", "fixed_time"):
            raise ValueError(f"unknown mode {mode!r}")
        self.controller.mode = mode
        return {"mode": mode}

    def clear_fault(self) -> dict:
        self.controller.clear_fault()
        return {"faulted": False}

    def train_now(self) -> dict:
        reports = self.forecaster.train()
        self._persist_model_metrics(reports)
        anomaly = self.detector.fit_anomaly()
        return {"forecasters": reports, "anomaly": anomaly}

    # ───────────────────────── views ─────────────────────────

    def state(self) -> dict:
        if self.command is None or self.frame is None:
            return {"status": "starting"}
        metrics = (
            self.sensor.sim.metrics() if hasattr(self.sensor, "sim") else {}
        )
        pressure, terms = self.controller._pressure(self.frame)
        return {
            "status": "faulted" if self.controller.faulted else "running",
            "intersection_id": self.intersection_id,
            "command": self.command.to_dict(),
            "frame": self.frame.to_dict(),
            "controller": self.controller.snapshot(),
            "pressure": {k: round(v, 3) for k, v in pressure.items()},
            "pressure_terms": terms,
            "phases": {
                k: {
                    "label": p.label,
                    "vehicles": sorted(p.vehicles),
                    "pedestrians": sorted(p.pedestrians),
                }
                for k, p in PHASES.items()
            },
            "metrics": metrics,
            "interlock": self.interlock.statistics(),
            "sensor": self.sensor.health(),
            "forecast": {
                a: round(self.forecaster.predict(a), 2) for a in APPROACHES
            },
            "incidents": list(self.incident_feed)[:12],
            "decisions": list(self.decision_feed)[:12],
            "safety_events": list(self.safety_feed)[:8],
        }

    def series(self, limit: int = 240) -> List[dict]:
        return list(self.history)[-limit:]

    # ───────────────────────── internals ─────────────────────────

    def _collect_decisions(self) -> None:
        decisions = self.controller.decisions
        while self._decisions_seen < len(decisions):
            decision = decisions[self._decisions_seen]
            self._decisions_seen += 1
            self.decision_feed.appendleft(decision.to_dict())
            self._persist_decision(decision)

    def _aggregate_for_ml(self, frame: SensorFrame) -> None:
        """Feed the forecaster one row per wall-clock minute, not per tick."""
        minute = minute_of_week(frame.timestamp)
        if minute != self._last_minute:
            self._last_minute = minute
            self.forecaster.observe_frame(frame, minute)

    def _record_sample(self, frame: SensorFrame, command: SignalCommand) -> None:
        self.history.append(
            {
                "t": round(self.controller.t, 1),
                "timestamp": frame.timestamp.isoformat(),
                "phase": command.phase,
                "interval": command.interval.value,
                "queue": {a: frame.approaches[a].queue for a in APPROACHES},
                "flow": {a: frame.approaches[a].flow for a in APPROACHES},
                "forecast": {
                    a: round(self.forecaster.predict(a), 2) for a in APPROACHES
                },
                "total_queue": sum(frame.approaches[a].queue for a in APPROACHES),
            }
        )

    def _maybe_log(self, frame: SensorFrame, command: SignalCommand) -> None:
        if self.controller.t - self._last_log < self.settings.log_interval:
            return
        self._last_log = self.controller.t
        metrics = self.sensor.sim.metrics() if hasattr(self.sensor, "sim") else {}
        try:
            with SessionLocal() as session:
                session.add(
                    TrafficLog(
                        intersection_id=self.intersection_id,
                        timestamp=frame.timestamp.replace(tzinfo=None),
                        phase=command.phase,
                        interval=command.interval.value,
                        cycle=command.cycle,
                        weather=frame.weather,
                        total_queue=sum(
                            frame.approaches[a].queue for a in APPROACHES
                        ),
                        mean_delay=float(metrics.get("mean_delay_s", 0.0)),
                        approaches=frame.to_dict()["approaches"],
                    )
                )
                session.commit()
        except Exception:
            logger.warning("traffic log write failed", exc_info=True)

    def _maybe_train(self) -> None:
        if not self.settings.forecast_enabled or self._training:
            return
        if self.controller.t - self._last_train < self.settings.train_interval_seconds:
            return
        self._last_train = self.controller.t
        self._training = True

        async def _worker():
            try:
                result = await asyncio.to_thread(self.train_now)
                promoted = [
                    a for a, r in result["forecasters"].items() if r.get("promoted")
                ]
                logger.info(
                    "retrain complete; promoted=%s", promoted or "none (baseline kept)"
                )
            except Exception:
                logger.warning("retrain failed", exc_info=True)
            finally:
                self._training = False

        try:
            asyncio.get_running_loop().create_task(_worker())
        except RuntimeError:
            # No loop (synchronous test driver) — train inline.
            try:
                self.train_now()
            finally:
                self._training = False

    def _persist_decision(self, decision) -> None:
        try:
            with SessionLocal() as session:
                session.add(
                    PhaseLog(
                        intersection_id=self.intersection_id,
                        timestamp=decision.timestamp.replace(tzinfo=None),
                        cycle=decision.cycle,
                        from_phase=decision.from_phase,
                        to_phase=decision.to_phase,
                        trigger=decision.trigger,
                        reason=decision.reason[:640],
                        served_green_time=decision.served_green_time,
                        pressure=decision.pressure,
                        terms=decision.terms,
                    )
                )
                session.commit()
        except Exception:
            logger.warning("phase log write failed", exc_info=True)

    def _persist_incident(self, incident) -> None:
        try:
            with SessionLocal() as session:
                session.add(
                    IncidentRecord(
                        intersection_id=self.intersection_id,
                        detected_at=incident.timestamp.replace(tzinfo=None),
                        approach=incident.approach,
                        kind=incident.kind,
                        severity=incident.severity,
                        layer=incident.layer,
                        confidence=incident.confidence,
                        detail=incident.detail[:640],
                    )
                )
                session.commit()
        except Exception:
            logger.warning("incident write failed", exc_info=True)

    def _persist_safety(self, findings) -> None:
        try:
            with SessionLocal() as session:
                for finding in findings:
                    session.add(
                        SafetyLog(
                            intersection_id=self.intersection_id,
                            timestamp=finding.timestamp.replace(tzinfo=None),
                            kind=finding.kind,
                            detail=finding.detail[:640],
                            command=finding.command,
                        )
                    )
                session.commit()
        except Exception:
            logger.warning("safety log write failed", exc_info=True)

    def _persist_model_metrics(self, reports: Dict[str, dict]) -> None:
        try:
            with SessionLocal() as session:
                for approach, report in reports.items():
                    if "model_mae" not in report:
                        continue
                    session.add(
                        ModelMetric(
                            approach=approach,
                            estimator=report.get("estimator", "unknown"),
                            rows=int(report.get("rows", 0)),
                            model_mae=float(report["model_mae"]),
                            baseline_mae=float(report["baseline_mae"]),
                            promoted=bool(report.get("promoted")),
                        )
                    )
                session.commit()
        except Exception:
            logger.warning("model metric write failed", exc_info=True)


_runtime: Optional[IntersectionRuntime] = None


def get_runtime() -> IntersectionRuntime:
    global _runtime
    if _runtime is None:
        _runtime = IntersectionRuntime()
    return _runtime


def reset_runtime() -> None:
    global _runtime
    _runtime = None
