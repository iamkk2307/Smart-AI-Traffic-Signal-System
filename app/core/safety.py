"""
Safety interlock.

This is a *verifier*, not a second controller. It watches the command stream
and checks it against the physical conflict matrix and the minimum interval
requirements. When a check fails it records the event and trips the
controller to flashing red.

It deliberately does not try to repair a bad command. A controller that
emitted conflicting greens has an unknown internal state; substituting a
plausible-looking green for the illegal one hides the defect and keeps a
broken controller in service. Flashing red is the correct failure mode for a
signalised intersection — drivers know what to do with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .phases import APPROACHES, CROSSWALKS, CONFLICTS, Interval, Signal, Walk
from .types import SafetyEvent, SignalCommand

EPS = 1e-6


@dataclass
class InterlockLimits:
    min_green: float = 8.0  #: absolute floor, below any configured min green
    min_yellow: float = 3.0
    min_all_red: float = 1.0


class SafetyInterlock:
    def __init__(self, limits: Optional[InterlockLimits] = None) -> None:
        self.limits = limits or InterlockLimits()
        self.events: List[SafetyEvent] = []
        self._prev: Optional[SignalCommand] = None
        self._prev_t: float = 0.0
        self._green_since: Dict[str, float] = {}
        self._yellow_since: Dict[str, float] = {}
        #: Last time any movement conflicting with the key showed a
        #: permissive or change indication.
        self._conflict_active_until: Dict[str, float] = {
            m: -1e9 for m in APPROACHES + CROSSWALKS
        }

    # ───────────────────────── checks ─────────────────────────

    def check(self, t: float, command: SignalCommand) -> List[SafetyEvent]:
        """Validate one command. Returns the events raised by this command."""
        raised: List[SafetyEvent] = []

        served = self._served(command)
        raised += self._check_conflicts(t, command, served)
        raised += self._check_change_intervals(t, command)
        raised += self._check_clearance(t, command)

        self._update_history(t, command, served)
        self.events.extend(raised)
        if len(self.events) > 200:
            del self.events[:-200]
        return raised

    def _served(self, command: SignalCommand) -> List[str]:
        """Movements currently given permission to enter the intersection."""
        served = [a for a, s in command.vehicle.items() if s is Signal.GREEN]
        served += [
            c
            for c, w in command.pedestrian.items()
            if w in (Walk.WALK, Walk.FLASHING)
        ]
        return served

    def _check_conflicts(
        self, t: float, command: SignalCommand, served: List[str]
    ) -> List[SafetyEvent]:
        out = []
        for i, a in enumerate(served):
            for b in served[i + 1 :]:
                if b in CONFLICTS.get(a, frozenset()):
                    out.append(
                        self._event(
                            command,
                            "conflicting_movements",
                            f"{a} and {b} were served simultaneously at t={t:.1f}s",
                        )
                    )
        return out

    def _check_change_intervals(
        self, t: float, command: SignalCommand
    ) -> List[SafetyEvent]:
        """A green must become yellow before it becomes red."""
        out = []
        if self._prev is None:
            return out
        for approach in APPROACHES:
            was = self._prev.vehicle[approach]
            now = command.vehicle[approach]
            if was is Signal.GREEN and now is Signal.RED:
                out.append(
                    self._event(
                        command,
                        "missing_change_interval",
                        f"{approach} went GREEN to RED with no yellow at t={t:.1f}s",
                    )
                )
            if was is Signal.YELLOW and now is Signal.GREEN:
                out.append(
                    self._event(
                        command,
                        "yellow_to_green",
                        f"{approach} returned YELLOW to GREEN at t={t:.1f}s",
                    )
                )
            if was is Signal.GREEN and now is Signal.YELLOW:
                held = t - self._green_since.get(approach, t)
                if held + EPS < self.limits.min_green:
                    out.append(
                        self._event(
                            command,
                            "short_green",
                            f"{approach} held green {held:.1f}s, floor is "
                            f"{self.limits.min_green:.1f}s",
                        )
                    )
            if was is Signal.YELLOW and now is Signal.RED:
                held = t - self._yellow_since.get(approach, t)
                if held + EPS < self.limits.min_yellow:
                    out.append(
                        self._event(
                            command,
                            "short_yellow",
                            f"{approach} held yellow {held:.1f}s, floor is "
                            f"{self.limits.min_yellow:.1f}s",
                        )
                    )
        return out

    def _check_clearance(
        self, t: float, command: SignalCommand
    ) -> List[SafetyEvent]:
        """A movement may not start until conflicting traffic has cleared."""
        out = []
        if self._prev is None:
            return out
        for movement in APPROACHES + CROSSWALKS:
            starting = self._is_starting(movement, command)
            if not starting:
                continue
            since = t - self._conflict_active_until[movement]
            if since + EPS < self.limits.min_all_red:
                out.append(
                    self._event(
                        command,
                        "insufficient_clearance",
                        f"{movement} started {since:.1f}s after conflicting "
                        f"traffic lost its indication; floor is "
                        f"{self.limits.min_all_red:.1f}s",
                    )
                )
        return out

    def _is_starting(self, movement: str, command: SignalCommand) -> bool:
        assert self._prev is not None
        if movement in APPROACHES:
            return (
                command.vehicle[movement] is Signal.GREEN
                and self._prev.vehicle[movement] is not Signal.GREEN
            )
        return command.pedestrian[movement] is Walk.WALK and self._prev.pedestrian[
            movement
        ] is not Walk.WALK

    # ───────────────────────── bookkeeping ─────────────────────────

    def _update_history(
        self, t: float, command: SignalCommand, served: List[str]
    ) -> None:
        for approach in APPROACHES:
            now = command.vehicle[approach]
            was = self._prev.vehicle[approach] if self._prev else None
            if now is Signal.GREEN and was is not Signal.GREEN:
                self._green_since[approach] = t
            if now is Signal.YELLOW and was is not Signal.YELLOW:
                self._yellow_since[approach] = t

        # Any movement with a permissive or change indication keeps the
        # clearance clock of everything it conflicts with pinned to now.
        active = list(served)
        active += [a for a, s in command.vehicle.items() if s is Signal.YELLOW]
        for movement in APPROACHES + CROSSWALKS:
            if any(m in CONFLICTS.get(movement, frozenset()) for m in active):
                self._conflict_active_until[movement] = t

        self._prev = command
        self._prev_t = t

    def _event(self, command: SignalCommand, kind: str, detail: str) -> SafetyEvent:
        return SafetyEvent(
            timestamp=datetime.now(timezone.utc),
            kind=kind,
            detail=detail,
            command=command.to_dict(),
        )

    def statistics(self) -> dict:
        kinds: Dict[str, int] = {}
        for e in self.events:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        return {"total": len(self.events), "by_kind": kinds}
