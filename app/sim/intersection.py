"""
Point-queue microsimulation of one four-leg intersection.

The simulation is closed-loop: arrivals are exogenous, but discharge depends
on what the controller is actually displaying. That matters for two reasons.
A simulator that generates queue numbers independently of the signal cannot
show whether the controller is any good, and a test suite built on such a
simulator cannot fail. Here, holding a red grows the queue and the delay
counter, which is what makes the adaptive-versus-fixed-time comparison in
``scripts/benchmark.py`` mean something.

The model is a vertical (point) queue per approach with start-up lost time
and a weather-derated saturation flow. It is not a car-following model and
makes no claim to be: it is the right fidelity for evaluating signal timing
policy, and nothing in the controller depends on anything finer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from app.core.controller import TimingPlan
from app.core.phases import APPROACHES, CROSSWALKS, Signal
from app.core.types import ApproachReading, SensorFrame, SignalCommand
from app.sensors.environment import ModelledUVSensor
from app.sensors.speed import percentile

#: Relative demand by hour of day, normalised so 1.0 is the daily mean.
#: Twin peaks, morning sharper than evening, overnight floor.
HOURLY_PROFILE = (
    0.22, 0.15, 0.12, 0.13, 0.25, 0.55, 1.05, 1.72, 1.95, 1.48, 1.12, 1.05,
    1.15, 1.10, 1.05, 1.18, 1.52, 1.84, 1.66, 1.20, 0.88, 0.66, 0.46, 0.31,
)

#: Start-up lost time: the first vehicles off the line are slower than
#: saturation flow.
STARTUP_LOST_TIME = 2.0


@dataclass
class ApproachDemand:
    """Baseline arrival rate, lane count and posted limit for one approach."""

    vehicles_per_hour: float
    lanes: int = 2
    speed_limit_kmh: float = 50.0
    #: Free-flow speed with an empty queue ahead, as a fraction of the
    #: posted limit. Above 1.0 the approach habitually speeds, which is how
    #: a real site behaves and what makes the speeding count non-trivial.
    free_flow_factor: float = 1.06


@dataclass
class SimConfig:
    demand: Dict[str, ApproachDemand] = field(
        default_factory=lambda: {
            "north": ApproachDemand(760, 2),
            "south": ApproachDemand(690, 2),
            "east": ApproachDemand(480, 2),
            "west": ApproachDemand(520, 2),
        }
    )
    ped_call_rate_per_hour: float = 26.0
    emergency_rate_per_hour: float = 2.4
    emergency_duration: float = 22.0
    incident_rate_per_hour: float = 0.7
    incident_duration: float = 180.0
    weather: str = "clear"
    #: Site location, used for solar geometry in the UV/light model.
    latitude: float = 17.385
    longitude: float = 78.4867
    #: The simulated wall clock is local time, because a demand profile is
    #: indexed by local hour. Solar position needs the UTC instant, so the
    #: offset is recorded rather than assumed to be zero — without it a
    #: 17:30 peak would be placed at 17:30 UTC and the sun would be in
    #: entirely the wrong part of the sky.
    utc_offset_hours: float = 5.5
    seed: Optional[int] = 7
    start_time: Optional[datetime] = None
    #: Compress the clock so a short run still covers a demand peak.
    time_scale: float = 1.0


@dataclass
class ApproachState:
    queue: float = 0.0
    waiting: float = 0.0  #: seconds the head of queue has been stopped
    gap: float = 99.0  #: seconds since the last arrival
    green_elapsed: float = 0.0
    arrivals: float = 0.0
    departures: float = 0.0
    delay: float = 0.0  #: vehicle-seconds
    blocked_until: float = 0.0


class IntersectionSim:
    """Advance queues under a signal command and emit detector frames."""

    def __init__(self, config: Optional[SimConfig] = None,
                 timing: Optional[TimingPlan] = None) -> None:
        self.config = config or SimConfig()
        self.timing = timing or TimingPlan()
        self.rng = random.Random(self.config.seed)
        # Speed sampling draws from its own generator, deliberately. Sharing
        # ``self.rng`` would couple the arrival stream to the number of
        # departures, so two controllers given the same seed would see
        # different demand — and the benchmark's claim that the only
        # difference between them is timing policy would be false.
        self.speed_rng = random.Random(
            (self.config.seed if self.config.seed is not None else 0) ^ 0x5EED
        )
        self.t = 0.0
        self.start_time = self.config.start_time or datetime.now(timezone.utc).replace(
            hour=7, minute=45, second=0, microsecond=0
        )
        self.state: Dict[str, ApproachState] = {a: ApproachState() for a in APPROACHES}
        self.ped_calls: Dict[str, bool] = {c: False for c in CROSSWALKS}
        self.weather = self.config.weather
        self.environment = ModelledUVSensor(
            latitude=self.config.latitude, longitude=self.config.longitude
        )
        self._emergency_until = 0.0
        self._emergency_approach: Optional[str] = None
        self._emergency_kind: Optional[str] = None
        self._incident_approach: Optional[str] = None
        self._incident_until = 0.0
        self.incidents: list = []

    # ───────────────────────── clock ─────────────────────────

    @property
    def wall_time(self) -> datetime:
        """Local wall clock at the site."""
        return self.start_time + timedelta(seconds=self.t * self.config.time_scale)

    @property
    def solar_time(self) -> datetime:
        """The same instant as a true UTC timestamp, for solar geometry."""
        return self.wall_time - timedelta(hours=self.config.utc_offset_hours)

    def demand_multiplier(self) -> float:
        """Interpolated hour-of-day demand factor."""
        now = self.wall_time
        hour = now.hour + now.minute / 60.0
        lo = HOURLY_PROFILE[int(hour) % 24]
        hi = HOURLY_PROFILE[(int(hour) + 1) % 24]
        frac = hour - int(hour)
        return lo + (hi - lo) * frac

    # ───────────────────────── stepping ─────────────────────────

    def step(self, dt: float, command: Optional[SignalCommand] = None) -> SensorFrame:
        """Advance the simulation and return the resulting detector frame."""
        self.t += dt
        multiplier = self.demand_multiplier()
        env = self.environment.read(self.solar_time, self.weather)
        discharge = self.timing.discharge_rate(self.weather, env.glare)

        for approach in APPROACHES:
            st = self.state[approach]
            demand = self.config.demand[approach]
            rate = demand.vehicles_per_hour * multiplier / 3600.0

            arrivals = self._poisson(rate * dt)
            st.arrivals += arrivals
            st.queue += arrivals
            st.gap = 0.0 if arrivals else st.gap + dt

            signal = command.vehicle[approach] if command else Signal.RED
            serving = signal is Signal.GREEN
            if serving:
                st.green_elapsed += dt
                # Ramp discharge over the start-up lost time.
                ramp = min(1.0, st.green_elapsed / STARTUP_LOST_TIME)
                capacity = discharge * demand.lanes * dt * ramp
                if self.t < st.blocked_until:
                    capacity *= 0.25  # lane blocked by an incident
                served = min(st.queue, capacity)
                st.queue -= served
                st.departures += served
                st.waiting = 0.0 if st.queue <= 0.5 else st.waiting
            else:
                st.green_elapsed = 0.0
                if st.queue > 0.5:
                    st.waiting += dt

            st.delay += st.queue * dt

        self._step_events(dt)
        return self.frame()

    def _step_events(self, dt: float) -> None:
        cfg = self.config

        # Pedestrian pushes.
        for crosswalk in CROSSWALKS:
            if not self.ped_calls[crosswalk]:
                p = cfg.ped_call_rate_per_hour / 4 / 3600.0 * dt
                if self.rng.random() < p:
                    self.ped_calls[crosswalk] = True

        # Emergency vehicles.
        if self.t >= self._emergency_until:
            self._emergency_approach = None
            self._emergency_kind = None
            if self.rng.random() < cfg.emergency_rate_per_hour / 3600.0 * dt:
                self._emergency_approach = self.rng.choice(APPROACHES)
                self._emergency_kind = self.rng.choice(
                    ("ambulance", "fire", "police")
                )
                self._emergency_until = self.t + cfg.emergency_duration

        # Lane blockages.
        if self.t >= self._incident_until:
            if self._incident_approach is not None:
                self._incident_approach = None
            if self.rng.random() < cfg.incident_rate_per_hour / 3600.0 * dt:
                approach = self.rng.choice(APPROACHES)
                self._incident_approach = approach
                self._incident_until = self.t + cfg.incident_duration
                self.state[approach].blocked_until = self._incident_until
                self.incidents.append(
                    {
                        "t": round(self.t, 1),
                        "approach": approach,
                        "kind": "lane_blockage",
                        "duration": cfg.incident_duration,
                    }
                )

    # ───────────────────────── observation ─────────────────────────

    def frame(self) -> SensorFrame:
        approaches = {}
        for approach in APPROACHES:
            st = self.state[approach]
            demand = self.config.demand[approach]
            queue = int(round(st.queue))
            # Occupancy saturates: a 20-vehicle queue and a 40-vehicle queue
            # look much the same to a loop at the stop line.
            occupancy = min(1.0, 1.0 - math.exp(-st.queue / (6.0 * demand.lanes)))
            flow = st.departures  # reset below, so this is per-interval

            # Speed. Free-flow when the approach is empty, falling towards
            # zero as the queue builds, with per-sample dispersion so the
            # 85th percentile is not simply the mean. Stopped vehicles are
            # excluded from the sample, as a radar head or a tracker
            # excludes them: averaging in zeros reports a queue, not a
            # speed, and would make every red interval look like a traffic
            # jam on the open road upstream.
            free_flow = demand.speed_limit_kmh * demand.free_flow_factor
            mean_speed = free_flow * max(0.0, 1.0 - occupancy) ** 0.8
            samples = max(0, int(round(flow)))
            if samples and mean_speed > 1.0:
                observed = [
                    max(3.0, self.speed_rng.gauss(mean_speed, mean_speed * 0.18))
                    for _ in range(samples)
                ]
                speeding = sum(1 for v in observed if v > demand.speed_limit_kmh)
                measured = round(sum(observed) / len(observed), 1)
                p85 = round(percentile(observed, 0.85), 1)
            else:
                observed, speeding, measured, p85 = [], 0, 0.0, 0.0

            approaches[approach] = ApproachReading(
                queue=queue,
                flow=round(flow, 2),
                occupancy=round(occupancy, 3),
                mean_speed=measured,
                speed_85th=p85,
                speed_limit=demand.speed_limit_kmh,
                speeding=speeding,
                speed_samples=len(observed),
                gap=round(st.gap, 2),
                max_wait=round(st.waiting, 1),
            )
            st.departures = 0.0

        return SensorFrame(
            timestamp=self.wall_time,
            approaches=approaches,
            pedestrian_calls=dict(self.ped_calls),
            emergency_approach=self._emergency_approach,
            emergency_kind=self._emergency_kind,
            emergency_eta=(
                None
                if self._emergency_approach is None
                else round(max(0.0, self._emergency_until - self.t), 1)
            ),
            weather=self.weather,
            visibility_km=10.0 if self.weather in ("clear", "cloudy") else 1.8,
            surface_wet=self.weather in ("rain", "snow", "thunderstorm"),
            environment=self.environment.read(self.solar_time, self.weather),
        )

    def clear_ped_calls(self, crosswalks) -> None:
        for c in crosswalks:
            self.ped_calls[c] = False

    # ───────────────────────── external injection ─────────────────────────

    def inject_emergency(self, approach: str, kind: str = "ambulance",
                         duration: Optional[float] = None) -> None:
        self._emergency_approach = approach
        self._emergency_kind = kind
        self._emergency_until = self.t + (duration or self.config.emergency_duration)

    def inject_ped_call(self, crosswalk: str) -> None:
        self.ped_calls[crosswalk] = True

    def set_weather(self, weather: str) -> None:
        self.weather = weather

    # ───────────────────────── metrics ─────────────────────────

    def metrics(self) -> dict:
        total_delay = sum(s.delay for s in self.state.values())
        total_departures = sum(s.arrivals - s.queue for s in self.state.values())
        served = max(total_departures, 1e-9)
        return {
            "elapsed": round(self.t, 1),
            "total_delay_veh_s": round(total_delay, 1),
            "vehicles_served": round(total_departures, 1),
            "mean_delay_s": round(total_delay / served, 2),
            "queue_now": {a: int(round(self.state[a].queue)) for a in APPROACHES},
            "max_wait_now": {a: round(self.state[a].waiting, 1) for a in APPROACHES},
            "incidents": len(self.incidents),
        }

    def _poisson(self, mean: float) -> int:
        """Knuth sampler; means here are well under 1 per tick."""
        if mean <= 0:
            return 0
        if mean < 30:
            limit = math.exp(-mean)
            k, p = 0, 1.0
            while True:
                p *= self.rng.random()
                if p <= limit:
                    return k
                k += 1
        # Normal approximation for large means.
        return max(0, int(round(self.rng.gauss(mean, math.sqrt(mean)))))
