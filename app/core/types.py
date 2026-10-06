"""Value types passed between sensors, controller, interlock and API."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .phases import APPROACHES, CROSSWALKS, Interval, Signal, Walk


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ApproachReading:
    """One detector station's view of one approach."""

    queue: int = 0  #: stopped vehicles, count
    flow: float = 0.0  #: vehicles per minute crossing the stop line
    occupancy: float = 0.0  #: loop occupancy 0..1
    mean_speed: float = 0.0  #: km/h over the detector, road plane
    gap: float = 99.0  #: seconds since last arrival at the stop line
    max_wait: float = 0.0  #: seconds the head-of-queue vehicle has waited
    #: Speed monitoring. ``speed_85th`` is the 85th percentile of tracked
    #: vehicles in the sample window — the figure traffic engineering
    #: actually uses — and ``speeding`` counts tracks over the posted limit.
    speed_85th: float = 0.0
    speed_limit: float = 50.0  #: km/h posted on this approach
    speeding: int = 0
    speed_samples: int = 0  #: tracks the speed figures are derived from
    #: True when this station's data is stale or unavailable. A degraded
    #: approach is placed on constant recall rather than read as empty.
    degraded: bool = False


@dataclass
class EnvironmentReading:
    """Ambient optical conditions from the roadside UV/light sensor.

    The UV index is a proxy for direct-beam irradiance. Combined with the
    sun's position it identifies low-angle glare, which degrades both
    driver sight distance and camera detection confidence. It is recorded
    as a measurement with a named mechanism, not as a decorative reading:
    ``glare`` lengthens change intervals and derates discharge, and
    ``approaches_affected`` names the approaches the sun is actually
    shining into.
    """

    uv_index: float = 0.0
    solar_elevation: float = 90.0  #: degrees above the horizon
    solar_azimuth: float = 180.0  #: degrees clockwise from north
    illuminance_lux: float = 0.0
    glare: bool = False
    approaches_affected: List[str] = field(default_factory=list)
    source: str = "model"  #: model | sensor
    stale: bool = False


@dataclass
class SensorFrame:
    """A complete, timestamped observation of the intersection."""

    timestamp: datetime = field(default_factory=_utcnow)
    approaches: Dict[str, ApproachReading] = field(
        default_factory=lambda: {a: ApproachReading() for a in APPROACHES}
    )
    pedestrian_calls: Dict[str, bool] = field(
        default_factory=lambda: {c: False for c in CROSSWALKS}
    )
    emergency_approach: Optional[str] = None
    emergency_kind: Optional[str] = None  #: ambulance | fire | police
    emergency_eta: Optional[float] = None  #: seconds to stop line
    weather: str = "clear"  #: clear | rain | fog | snow | thunderstorm
    visibility_km: float = 10.0
    surface_wet: bool = False
    environment: EnvironmentReading = field(default_factory=EnvironmentReading)
    #: Detector stations whose data could not be trusted on this frame.
    degraded_approaches: List[str] = field(default_factory=list)

    def queue(self, approach: str) -> int:
        return self.approaches[approach].queue

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "approaches": {k: asdict(v) for k, v in self.approaches.items()},
            "pedestrian_calls": dict(self.pedestrian_calls),
            "emergency_approach": self.emergency_approach,
            "emergency_kind": self.emergency_kind,
            "emergency_eta": self.emergency_eta,
            "weather": self.weather,
            "visibility_km": self.visibility_km,
            "surface_wet": self.surface_wet,
            "environment": asdict(self.environment),
            "degraded_approaches": list(self.degraded_approaches),
        }


@dataclass
class SignalCommand:
    """What the controller is telling the signal heads to display, right now."""

    timestamp: datetime
    phase: str
    interval: Interval
    vehicle: Dict[str, Signal]
    pedestrian: Dict[str, Walk]
    interval_elapsed: float
    interval_remaining: Optional[float]
    cycle: int
    #: True while an emergency preemption is holding this phase.
    preempted: bool = False

    def greens(self) -> List[str]:
        return [a for a, s in self.vehicle.items() if s is Signal.GREEN]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase,
            "interval": self.interval.value,
            "vehicle": {k: v.value for k, v in self.vehicle.items()},
            "pedestrian": {k: v.value for k, v in self.pedestrian.items()},
            "interval_elapsed": round(self.interval_elapsed, 1),
            "interval_remaining": (
                None if self.interval_remaining is None
                else round(self.interval_remaining, 1)
            ),
            "cycle": self.cycle,
            "preempted": self.preempted,
        }


@dataclass
class PhaseDecision:
    """Audit record for one phase-selection decision.

    ``reason`` is written for a human reading the operations log. ``pressure``
    and ``terms`` carry the numbers behind it so the reason is checkable
    rather than merely plausible.
    """

    timestamp: datetime
    cycle: int
    from_phase: str
    to_phase: str
    trigger: str  #: gap_out | max_out | preempt | ped_call | starvation | recall
    reason: str
    pressure: Dict[str, float]
    terms: Dict[str, Dict[str, float]]
    served_green_time: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cycle": self.cycle,
            "from_phase": self.from_phase,
            "to_phase": self.to_phase,
            "trigger": self.trigger,
            "reason": self.reason,
            "pressure": {k: round(v, 2) for k, v in self.pressure.items()},
            "terms": {
                p: {k: round(v, 3) for k, v in t.items()}
                for p, t in self.terms.items()
            },
            "served_green_time": round(self.served_green_time, 1),
        }


@dataclass
class SafetyEvent:
    """An interlock finding. Anything recorded here is a controller bug."""

    timestamp: datetime
    kind: str
    detail: str
    command: dict

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "kind": self.kind,
            "detail": self.detail,
            "command": self.command,
        }
