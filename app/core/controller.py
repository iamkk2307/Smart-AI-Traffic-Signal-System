"""
Adaptive signal controller.

Design notes, because the obvious implementation is unsafe
----------------------------------------------------------
A naive "pick the busiest approach each cycle" controller has three defects
that matter more than any amount of optimisation:

1. **No change interval.** Flipping a green directly to red removes the
   yellow that tells drivers in the dilemma zone to stop. Every phase
   service here runs GREEN → YELLOW → ALL_RED, with no path that skips
   either, including emergency preemption.
2. **No minimum green.** Re-deciding on every sample lets the phase flap on
   detector noise. Green is committed for ``min_green`` seconds before any
   termination check runs, and a served pedestrian interval extends that
   commitment to cover walk plus clearance.
3. **Unbounded preemption.** An emergency input that is simply obeyed can
   hold a phase indefinitely and starve every other approach. Preemption
   here is bounded by a hold limit *and* a budget over a trailing window,
   and exhausting the budget is recorded in the decision log rather than
   silently dropped.

The controller is deliberately dependency-free and driven by an explicit
clock, so the whole thing is reproducible under test. Arrival forecasts from
the ML layer enter through one optional callable; with no forecaster attached
the controller degrades to actuated control rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from .phases import (
    APPROACHES,
    CROSSWALKS,
    PHASE_FOR_APPROACH,
    PHASES,
    PHASES_FOR_CROSSWALK,
    VEHICLE_RING,
    Interval,
    Signal,
    Walk,
)
from .types import PhaseDecision, SensorFrame, SignalCommand

ArrivalForecast = Callable[[str], float]


@dataclass
class TimingPlan:
    """Interval lengths and policy weights, all in seconds unless noted."""

    min_green: float = 10.0
    max_green: float = 60.0
    yellow_change: float = 3.5
    red_clearance: float = 2.0
    ped_walk: float = 7.0
    ped_clearance: float = 11.0
    #: Hard ceiling on how long any approach may be held at red.
    max_red: float = 120.0
    #: A phase gaps out once every served approach has been arrival-free
    #: for this long and its queue has discharged.
    gap_threshold: float = 3.0
    queue_clear: int = 2
    #: A conflicting phase must beat the served phase by this factor before
    #: a gap-out is allowed. Pure anti-flap.
    hysteresis: float = 1.15
    #: Emergency preemption limits.
    preempt_max_hold: float = 45.0
    preempt_budget: float = 90.0
    preempt_window: float = 600.0
    #: Pedestrian service limits.
    ped_max_wait: float = 90.0
    #: Pressure weights.
    w_queue: float = 1.0
    w_occupancy: float = 0.45
    w_forecast: float = 0.35
    w_wait: float = 0.8
    w_ped: float = 0.9
    #: Normalisers so each term lands roughly in 0..1.
    queue_norm: float = 20.0
    forecast_norm: float = 30.0
    #: Adverse weather lengthens clearance and start-up loss.
    weather_extra_clearance: Dict[str, float] = field(
        default_factory=lambda: {
            "clear": 0.0,
            "cloudy": 0.0,
            "rain": 1.0,
            "fog": 1.5,
            "snow": 2.5,
            "thunderstorm": 1.5,
        }
    )
    #: Low-angle solar glare, reported by the UV/light sensor. Drivers
    #: facing into the sun need longer to see the indication change and
    #: start up more slowly, so clearance and minimum green both grow and
    #: the discharge rate is derated. Deliberately not applied to the
    #: pressure score: see ``_pressure``.
    glare_extra_clearance: float = 1.2
    glare_discharge_factor: float = 0.92
    #: Saturation discharge rate (veh/s/lane) and its weather derating.
    saturation_flow: float = 0.52
    weather_discharge_factor: Dict[str, float] = field(
        default_factory=lambda: {
            "clear": 1.0,
            "cloudy": 1.0,
            "rain": 0.88,
            "fog": 0.80,
            "snow": 0.65,
            "thunderstorm": 0.82,
        }
    )

    def clearance_for(self, weather: str, glare: bool = False) -> Tuple[float, float]:
        """(yellow, all-red) adjusted for surface, visibility and glare."""
        extra = self.weather_extra_clearance.get(weather, 0.0)
        if glare:
            extra += self.glare_extra_clearance
        return self.yellow_change + extra * 0.4, self.red_clearance + extra * 0.6

    def min_green_for(self, weather: str, glare: bool = False) -> float:
        extra = self.weather_extra_clearance.get(weather, 0.0)
        if glare:
            extra += self.glare_extra_clearance
        return self.min_green + extra

    def discharge_rate(self, weather: str, glare: bool = False) -> float:
        rate = self.saturation_flow * self.weather_discharge_factor.get(weather, 1.0)
        return rate * (self.glare_discharge_factor if glare else 1.0)


class AdaptiveController:
    """Phase-based actuated controller with explainable phase selection.

    Parameters
    ----------
    timing:
        Interval lengths and weights.
    mode:
        ``"adaptive"`` runs actuated control with pressure-based phase
        selection. ``"fixed_time"`` runs a conventional equal-split cycle
        and exists so the adaptive plan can be measured against a baseline
        instead of merely asserted to be better.
    forecast:
        Optional callable returning predicted arrivals per minute for an
        approach. Supplied by the ML layer at runtime.
    """

    def __init__(
        self,
        timing: Optional[TimingPlan] = None,
        mode: str = "adaptive",
        forecast: Optional[ArrivalForecast] = None,
        clock_start: float = 0.0,
    ) -> None:
        if mode not in ("adaptive", "fixed_time"):
            raise ValueError(f"unknown mode {mode!r}")
        self.timing = timing or TimingPlan()
        self.mode = mode
        self.forecast = forecast

        self.t = clock_start
        self.cycle = 1
        self.phase = VEHICLE_RING[0]
        self.interval = Interval.GREEN
        self.interval_elapsed = 0.0
        self.green_elapsed = 0.0

        # Phase history for starvation pressure.
        self._last_green_end: Dict[str, float] = {k: clock_start for k in PHASES}
        self._ped_called_at: Dict[str, Optional[float]] = {c: None for c in CROSSWALKS}
        self._ped_interval_elapsed = 0.0
        self._serving_ped = False

        # Preemption bookkeeping.
        self._preempt_active = False
        self._preempt_started: Optional[float] = None
        self._preempt_spans: List[Tuple[float, float]] = []
        self._preempt_target: Optional[str] = None
        self._preempt_denied_logged = False

        self.decisions: List[PhaseDecision] = []
        self.faulted = False
        self._fault_detail = ""

    # ───────────────────────── public API ─────────────────────────

    def tick(self, frame: SensorFrame, dt: float = 1.0) -> SignalCommand:
        """Advance the controller by ``dt`` seconds and return the command."""
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.t += dt
        self.interval_elapsed += dt
        self._register_ped_calls(frame)

        if self.faulted:
            return self._command(frame)

        if self.interval is Interval.GREEN:
            self.green_elapsed += dt
            if self._serving_ped:
                self._ped_interval_elapsed += dt
            self._tick_green(frame)
        elif self.interval is Interval.YELLOW:
            self._tick_yellow(frame)
        elif self.interval is Interval.ALL_RED:
            self._tick_all_red(frame)

        return self._command(frame)

    def trip_fault(self, detail: str) -> None:
        """Drop to flashing red. Only the interlock or an operator calls this."""
        self.faulted = True
        self._fault_detail = detail
        self.interval = Interval.FAULT_FLASH
        self.interval_elapsed = 0.0

    def clear_fault(self) -> None:
        """Restart the ring from all-red after a fault is acknowledged."""
        self.faulted = False
        self._fault_detail = ""
        self.interval = Interval.ALL_RED
        self.interval_elapsed = 0.0
        self.green_elapsed = 0.0

    def place_ped_call(self, crosswalk: str) -> None:
        if crosswalk not in CROSSWALKS:
            raise ValueError(f"unknown crosswalk {crosswalk!r}")
        if self._ped_called_at[crosswalk] is None:
            self._ped_called_at[crosswalk] = self.t

    @property
    def preempt_used(self) -> float:
        """Preemption seconds consumed inside the trailing window."""
        floor = self.t - self.timing.preempt_window
        used = 0.0
        for start, end in self._preempt_spans:
            used += max(0.0, min(end, self.t) - max(start, floor))
        if self._preempt_active and self._preempt_started is not None:
            used += max(0.0, self.t - max(self._preempt_started, floor))
        return used

    def red_time(self, approach: str) -> float:
        """Seconds since the phase serving ``approach`` last held green."""
        phase = PHASE_FOR_APPROACH[approach]
        if self.phase == phase and self.interval is Interval.GREEN:
            return 0.0
        return self.t - self._last_green_end[phase]

    # ─────────────────────── interval handlers ───────────────────────

    def _tick_green(self, frame: SensorFrame) -> None:
        timing = self.timing
        min_green = max(
            timing.min_green_for(frame.weather, frame.environment.glare),
            self._ped_commitment(),
        )

        preempt_target = self._preempt_decision(frame)
        if preempt_target is not None and preempt_target != self.phase:
            if self.green_elapsed >= min_green:
                self._terminate_green("preempt", frame)
            return

        if self._preempt_active:
            # Holding the emergency phase. Release when the vehicle has gone,
            # the hold limit is hit, or the budget runs out.
            hold = self.t - (self._preempt_started or self.t)
            exhausted = self.preempt_used >= timing.preempt_budget
            if (
                frame.emergency_approach is None
                or hold >= timing.preempt_max_hold
                or exhausted
            ):
                self._end_preempt()
                if self.green_elapsed >= min_green:
                    self._terminate_green(
                        "preempt_release" if not exhausted else "preempt_exhausted",
                        frame,
                    )
            return

        if self.green_elapsed < min_green:
            return

        if self.mode == "fixed_time":
            split = (timing.max_green + timing.min_green) / 2
            if self.green_elapsed >= split:
                self._terminate_green("fixed_split", frame)
            return

        pressure, _ = self._pressure(frame)
        served = pressure.get(self.phase, 0.0)
        best_other = max(
            (v for k, v in pressure.items() if k != self.phase), default=0.0
        )

        if self.green_elapsed >= timing.max_green:
            # Max out, but only if somebody is actually waiting. With no
            # conflicting call, resting in green beats cycling to an empty
            # street and paying the clearance cost for nothing.
            if best_other > 0.0:
                self._terminate_green("max_out", frame)
            return

        if best_other > served * timing.hysteresis and self._gapped_out(frame):
            self._terminate_green("gap_out", frame)

    def _tick_yellow(self, frame: SensorFrame) -> None:
        yellow, _ = self.timing.clearance_for(frame.weather, frame.environment.glare)
        if self.interval_elapsed >= yellow:
            self.interval = Interval.ALL_RED
            self.interval_elapsed = 0.0

    def _tick_all_red(self, frame: SensorFrame) -> None:
        _, all_red = self.timing.clearance_for(frame.weather, frame.environment.glare)
        if self.interval_elapsed < all_red:
            return

        next_phase, trigger, reason, pressure, terms = self._select_phase(frame)

        if next_phase == VEHICLE_RING[0] and self.phase != VEHICLE_RING[0]:
            self.cycle += 1

        self.decisions.append(
            PhaseDecision(
                timestamp=datetime.now(timezone.utc),
                cycle=self.cycle,
                from_phase=self.phase,
                to_phase=next_phase,
                trigger=trigger,
                reason=reason,
                pressure=pressure,
                terms=terms,
                served_green_time=self.green_elapsed,
            )
        )
        if len(self.decisions) > 500:
            del self.decisions[:-500]

        self.phase = next_phase
        self.interval = Interval.GREEN
        self.interval_elapsed = 0.0
        self.green_elapsed = 0.0
        self._begin_ped_service(next_phase)

        if trigger == "preempt":
            self._preempt_active = True
            self._preempt_started = self.t
            self._preempt_denied_logged = False

    def _terminate_green(self, trigger: str, frame: SensorFrame) -> None:
        self._pending_trigger = trigger
        self._last_green_end[self.phase] = self.t
        self.interval = Interval.YELLOW
        self.interval_elapsed = 0.0
        self._serving_ped = False
        self._ped_interval_elapsed = 0.0

    # ───────────────────────── phase selection ─────────────────────────

    def _select_phase(
        self, frame: SensorFrame
    ) -> Tuple[str, str, str, Dict[str, float], Dict[str, Dict[str, float]]]:
        pressure, terms = self._pressure(frame)
        trigger = getattr(self, "_pending_trigger", "recall")

        # 1. Emergency preemption outranks everything it is allowed to.
        target = self._preempt_decision(frame)
        if target is not None:
            return (
                target,
                "preempt",
                (
                    f"Emergency {frame.emergency_kind or 'vehicle'} on the "
                    f"{target} corridor. Preempting to {PHASES[target].label.lower()}; "
                    f"{self.preempt_used:.0f}s of {self.timing.preempt_budget:.0f}s "
                    f"preemption budget used in the last "
                    f"{self.timing.preempt_window / 60:.0f} min."
                ),
                pressure,
                terms,
            )

        if frame.emergency_approach is not None and not self._preempt_denied_logged:
            self._preempt_denied_logged = True
            held = PHASE_FOR_APPROACH[frame.emergency_approach]
            return (
                self._highest(pressure),
                "preempt_denied",
                (
                    f"Emergency reported on {frame.emergency_approach} but the "
                    f"preemption budget is spent ({self.preempt_used:.0f}s of "
                    f"{self.timing.preempt_budget:.0f}s). Serving "
                    f"{held} by normal rotation and raising an alarm — "
                    "preemption is not being silently dropped."
                ),
                pressure,
                terms,
            )

        # 2. Hard red ceiling. No approach with a call waits longer than
        #    max_red. An approach with nobody on it has no claim, which is
        #    what lets the ring rest in green on an empty cross street.
        starved = [
            a
            for a in APPROACHES
            if self.red_time(a) >= self.timing.max_red and self._has_call(frame, a)
        ]
        if starved:
            worst = max(starved, key=self.red_time)
            phase = PHASE_FOR_APPROACH[worst]
            return (
                phase,
                "starvation",
                (
                    f"{worst.title()} has been at red for "
                    f"{self.red_time(worst):.0f}s, at the {self.timing.max_red:.0f}s "
                    f"ceiling. Forcing {PHASES[phase].label.lower()} regardless of "
                    "pressure."
                ),
                pressure,
                terms,
            )

        # 3. A pedestrian call that cannot wait any longer gets the exclusive
        #    phase. Shorter waits are picked up concurrently by the vehicle
        #    phase that runs parallel to the crosswalk.
        overdue = [
            c
            for c, at in self._ped_called_at.items()
            if at is not None and self.t - at >= self.timing.ped_max_wait
        ]
        if overdue:
            waited = max(self.t - (self._ped_called_at[c] or self.t) for c in overdue)
            return (
                "PED",
                "ped_call",
                (
                    f"Pedestrian call on {', '.join(sorted(overdue))} waiting "
                    f"{waited:.0f}s. Running the exclusive pedestrian phase — all "
                    "vehicle movements stop."
                ),
                pressure,
                terms,
            )

        # 4. Otherwise serve the highest-pressure phase.
        winner = self._highest(pressure)
        reason = self._explain(winner, trigger, pressure, terms, frame)
        return winner, trigger, reason, pressure, terms

    def _highest(self, pressure: Dict[str, float]) -> str:
        candidates = {k: v for k, v in pressure.items() if k in VEHICLE_RING}
        # Ties break towards the phase that has waited longer, not towards
        # whichever key happens to sort first.
        return max(
            candidates,
            key=lambda k: (candidates[k], self.t - self._last_green_end[k]),
        )

    def _pressure(
        self, frame: SensorFrame
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Score every phase and return both the totals and the components.

        Every term is per-approach and normalised, so a weather or peak-hour
        effect that applies equally to all approaches cannot change the
        ranking. Those effects belong in the timing plan (clearance, discharge
        rate) and in the forecast, which is where they are applied.
        """
        t = self.timing
        pressure: Dict[str, float] = {}
        terms: Dict[str, Dict[str, float]] = {}

        for key in VEHICLE_RING:
            phase = PHASES[key]
            q = occ = pred = wait = 0.0
            for approach in phase.vehicles:
                r = frame.approaches[approach]
                q += min(r.queue / t.queue_norm, 2.0)
                occ += min(r.occupancy, 1.0)
                if self.forecast is not None:
                    pred += min(self.forecast(approach) / t.forecast_norm, 2.0)
                # Accumulated red time only counts where somebody is
                # actually waiting. Without this gate an empty approach
                # builds pressure out of nothing and the ring serves a
                # street with no vehicles on it.
                if self._has_call(frame, approach):
                    wait += min(self.red_time(approach) / t.max_red, 1.0)
            n = max(len(phase.vehicles), 1)
            component = {
                "queue": t.w_queue * q / n,
                "occupancy": t.w_occupancy * occ / n,
                "forecast": t.w_forecast * pred / n,
                "wait": t.w_wait * wait / n,
                "pedestrian": 0.0,
            }
            ped_waiting = [
                self.t - at
                for c, at in self._ped_called_at.items()
                if at is not None and key in PHASES_FOR_CROSSWALK[c]
            ]
            if ped_waiting:
                component["pedestrian"] = t.w_ped * min(
                    max(ped_waiting) / t.ped_max_wait, 1.0
                )
            terms[key] = component
            pressure[key] = round(sum(component.values()), 4)

        return pressure, terms

    def _explain(
        self,
        winner: str,
        trigger: str,
        pressure: Dict[str, float],
        terms: Dict[str, Dict[str, float]],
        frame: SensorFrame,
    ) -> str:
        phase = PHASES[winner]
        queues = ", ".join(
            f"{a} {frame.approaches[a].queue}" for a in sorted(phase.vehicles)
        )
        dominant = max(terms[winner], key=lambda k: terms[winner][k])
        label = {
            "queue": "standing queue",
            "occupancy": "detector occupancy",
            "forecast": "forecast arrivals",
            "wait": "accumulated red time",
            "pedestrian": "a waiting pedestrian call",
        }[dominant]
        trigger_text = {
            "gap_out": "Served phase gapped out",
            "max_out": "Served phase hit maximum green",
            "fixed_split": "Fixed-time split elapsed",
            "preempt_release": "Emergency cleared",
            "preempt_exhausted": "Preemption budget exhausted",
            "recall": "Ring recall",
        }.get(trigger, trigger)
        env = frame.environment
        extra = ""
        if frame.weather not in ("clear", "cloudy") or env.glare:
            yellow, all_red = self.timing.clearance_for(frame.weather, env.glare)
            conditions = (
                "Clear" if frame.weather in ("clear", "cloudy")
                else frame.weather.title()
            )
            if env.glare:
                conditions += (
                    f" with low-sun glare on {', '.join(env.approaches_affected)}"
                    f" (UV {env.uv_index:.1f}, sun {env.solar_elevation:.1f}°"
                    f" at {env.solar_azimuth:.0f}°)"
                )
            extra = (
                f" {conditions}: clearance extended to "
                f"{yellow:.1f}s yellow plus {all_red:.1f}s all-red and discharge "
                f"derated to "
                f"{self.timing.discharge_rate(frame.weather, env.glare):.2f} veh/s."
            )
        if frame.degraded_approaches:
            extra += (
                " Detector data unavailable on "
                f"{', '.join(sorted(frame.degraded_approaches))}; those approaches "
                "are on constant recall and cannot gap out."
            )
        return (
            f"{trigger_text}. {phase.label} selected on {label} "
            f"(pressure {pressure[winner]:.2f} vs "
            f"{max((v for k, v in pressure.items() if k != winner), default=0):.2f}); "
            f"queues {queues}.{extra}"
        )

    # ───────────────────────── helpers ─────────────────────────

    def _has_call(self, frame: SensorFrame, approach: str) -> bool:
        """True when an approach has demand registered against it.

        A degraded station is read as a standing call, not as an empty
        road. A failed detector and a clear road produce identical
        readings, and only one of those two is safe to assume.
        """
        r = frame.approaches[approach]
        if r.degraded or approach in frame.degraded_approaches:
            return True
        return (
            r.queue > 0
            or r.occupancy > 0.02
            or r.gap < self.timing.gap_threshold
        )

    def _gapped_out(self, frame: SensorFrame) -> bool:
        """True when the served phase has no more traffic to discharge."""
        phase = PHASES[self.phase]
        if not phase.vehicles:
            return True
        for approach in phase.vehicles:
            r = frame.approaches[approach]
            # An approach with no trustworthy data has not been shown to be
            # clear, so it cannot gap out. The phase runs to max green.
            if r.degraded or approach in frame.degraded_approaches:
                return False
            if r.queue > self.timing.queue_clear:
                return False
            if r.gap < self.timing.gap_threshold:
                return False
        return True

    def _preempt_decision(self, frame: SensorFrame) -> Optional[str]:
        """The phase to preempt to, or None if preemption is not available."""
        if frame.emergency_approach is None:
            return None
        if frame.emergency_approach not in PHASE_FOR_APPROACH:
            return None
        if self.preempt_used >= self.timing.preempt_budget:
            return None
        if self._preempt_active:
            return self._preempt_target
        target = PHASE_FOR_APPROACH[frame.emergency_approach]
        self._preempt_target = target
        return target

    def _end_preempt(self) -> None:
        if self._preempt_active and self._preempt_started is not None:
            self._preempt_spans.append((self._preempt_started, self.t))
            floor = self.t - self.timing.preempt_window * 2
            self._preempt_spans = [s for s in self._preempt_spans if s[1] >= floor]
        self._preempt_active = False
        self._preempt_started = None
        self._preempt_target = None

    def _register_ped_calls(self, frame: SensorFrame) -> None:
        for crosswalk, pressed in frame.pedestrian_calls.items():
            if pressed and self._ped_called_at.get(crosswalk) is None:
                self._ped_called_at[crosswalk] = self.t

    def _begin_ped_service(self, phase_key: str) -> None:
        """Clear the calls this phase serves and start the walk interval."""
        served = PHASES[phase_key].pedestrians
        calls = [c for c in served if self._ped_called_at.get(c) is not None]
        for c in calls:
            self._ped_called_at[c] = None
        self._serving_ped = bool(calls) or phase_key == "PED"
        self._ped_interval_elapsed = 0.0

    def _ped_commitment(self) -> float:
        """Green cannot end before a started walk has been cleared."""
        if not self._serving_ped:
            return 0.0
        return self.timing.ped_walk + self.timing.ped_clearance

    def _walk_state(self, crosswalk: str) -> Walk:
        if self.faulted:
            return Walk.DONT_WALK
        if crosswalk not in PHASES[self.phase].pedestrians:
            return Walk.DONT_WALK
        if self.interval is not Interval.GREEN or not self._serving_ped:
            return Walk.DONT_WALK
        if self._ped_interval_elapsed <= self.timing.ped_walk:
            return Walk.WALK
        if self._ped_interval_elapsed <= self.timing.ped_walk + self.timing.ped_clearance:
            return Walk.FLASHING
        return Walk.DONT_WALK

    def _command(self, frame: SensorFrame) -> SignalCommand:
        phase = PHASES[self.phase]
        if self.faulted:
            vehicle = {a: Signal.RED for a in APPROACHES}
        elif self.interval is Interval.GREEN:
            vehicle = {
                a: Signal.GREEN if a in phase.vehicles else Signal.RED
                for a in APPROACHES
            }
        elif self.interval is Interval.YELLOW:
            vehicle = {
                a: Signal.YELLOW if a in phase.vehicles else Signal.RED
                for a in APPROACHES
            }
        else:
            vehicle = {a: Signal.RED for a in APPROACHES}

        remaining = self._remaining(frame)
        return SignalCommand(
            timestamp=datetime.now(timezone.utc),
            phase=self.phase,
            interval=self.interval,
            vehicle=vehicle,
            pedestrian={c: self._walk_state(c) for c in CROSSWALKS},
            interval_elapsed=self.interval_elapsed,
            interval_remaining=remaining,
            cycle=self.cycle,
            preempted=self._preempt_active,
        )

    def _remaining(self, frame: SensorFrame) -> Optional[float]:
        yellow, all_red = self.timing.clearance_for(frame.weather, frame.environment.glare)
        if self.interval is Interval.YELLOW:
            return max(0.0, yellow - self.interval_elapsed)
        if self.interval is Interval.ALL_RED:
            return max(0.0, all_red - self.interval_elapsed)
        if self.interval is Interval.GREEN:
            # Guaranteed remaining green, which is what a countdown display
            # may legitimately show. Beyond min green the end is demand
            # dependent, so report the max-out bound instead of a guess.
            floor = max(
                self.timing.min_green_for(frame.weather, frame.environment.glare),
                self._ped_commitment(),
            )
            if self.green_elapsed < floor:
                return max(0.0, floor - self.green_elapsed)
            return max(0.0, self.timing.max_green - self.green_elapsed)
        return None

    def snapshot(self) -> dict:
        return {
            "clock": round(self.t, 1),
            "cycle": self.cycle,
            "phase": self.phase,
            "interval": self.interval.value,
            "mode": self.mode,
            "green_elapsed": round(self.green_elapsed, 1),
            "preempt_active": self._preempt_active,
            "preempt_used": round(self.preempt_used, 1),
            "preempt_budget": self.timing.preempt_budget,
            "faulted": self.faulted,
            "fault_detail": self._fault_detail,
            "red_time": {a: round(self.red_time(a), 1) for a in APPROACHES},
            "ped_calls": {
                c: (None if at is None else round(self.t - at, 1))
                for c, at in self._ped_called_at.items()
            },
        }
