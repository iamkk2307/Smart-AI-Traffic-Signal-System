"""
Movement, phase and conflict definitions for a four-leg intersection.

This module is the single source of truth for *what is physically safe*.
Everything else in the controller is timing and policy; this is geometry.

Movements
---------
Vehicle movements are named by the leg traffic arrives from:
``north`` is the northbound-to-southbound through/right movement, etc.
Left turns are not modelled as separate movements — this controller runs
protected-only phasing with no permissive turns, which is what makes the
conflict matrix below exhaustive rather than approximate.

Crosswalk movements are named by the leg they cross: ``cw_north`` is the
walk across the north leg. A crosswalk is perpendicular to the roadway it
crosses, so pedestrians on ``cw_north`` conflict with vehicles on the
north/south axis and run concurrently with the east/west axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Tuple

APPROACHES: Tuple[str, ...] = ("north", "south", "east", "west")
CROSSWALKS: Tuple[str, ...] = ("cw_north", "cw_south", "cw_east", "cw_west")

NS_AXIS = frozenset({"north", "south"})
EW_AXIS = frozenset({"east", "west"})


class Signal(str, Enum):
    """Vehicle signal indication."""

    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class Walk(str, Enum):
    """Pedestrian signal indication."""

    DONT_WALK = "DONT_WALK"
    FLASHING = "FLASHING"  # clearance interval — finish crossing, do not start
    WALK = "WALK"


class Interval(str, Enum):
    """Where the controller is inside a phase service."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"
    FAULT_FLASH = "FAULT_FLASH"  # interlock tripped: flashing red, fail-safe


@dataclass(frozen=True)
class Phase:
    key: str
    label: str
    vehicles: FrozenSet[str]
    pedestrians: FrozenSet[str]
    #: Phases on the same barrier may follow each other without a full
    #: clearance; phases across a barrier may not. Kept for extensibility —
    #: this ring enforces clearance between every phase regardless.
    barrier: int


PHASE_NS = Phase(
    key="NS",
    label="North–South through",
    vehicles=NS_AXIS,
    pedestrians=frozenset({"cw_east", "cw_west"}),
    barrier=0,
)
PHASE_EW = Phase(
    key="EW",
    label="East–West through",
    vehicles=EW_AXIS,
    pedestrians=frozenset({"cw_north", "cw_south"}),
    barrier=1,
)
PHASE_PED = Phase(
    key="PED",
    label="Exclusive pedestrian",
    vehicles=frozenset(),
    pedestrians=frozenset(CROSSWALKS),
    barrier=2,
)

PHASES: Dict[str, Phase] = {p.key: p for p in (PHASE_NS, PHASE_EW, PHASE_PED)}

#: Phases that serve vehicles, in ring order. PED is demand-only and is
#: never entered by rotation — it has to be called.
VEHICLE_RING: Tuple[str, ...] = ("NS", "EW")

#: Which phase serves a given approach.
PHASE_FOR_APPROACH: Dict[str, str] = {
    "north": "NS",
    "south": "NS",
    "east": "EW",
    "west": "EW",
}

#: Which phases give a crosswalk a walk indication.
PHASES_FOR_CROSSWALK: Dict[str, Tuple[str, ...]] = {
    "cw_north": ("EW", "PED"),
    "cw_south": ("EW", "PED"),
    "cw_east": ("NS", "PED"),
    "cw_west": ("NS", "PED"),
}


def _build_conflicts() -> Dict[str, FrozenSet[str]]:
    """Exhaustive pairwise conflict matrix over all movements."""
    conflicts: Dict[str, set] = {m: set() for m in APPROACHES + CROSSWALKS}

    # Vehicle axes cross each other.
    for a in NS_AXIS:
        conflicts[a] |= set(EW_AXIS)
    for a in EW_AXIS:
        conflicts[a] |= set(NS_AXIS)

    # Vehicles cross the crosswalks perpendicular to their own axis.
    for a in NS_AXIS:
        conflicts[a] |= {"cw_north", "cw_south"}
    for a in EW_AXIS:
        conflicts[a] |= {"cw_east", "cw_west"}
    for cw in ("cw_north", "cw_south"):
        conflicts[cw] |= set(NS_AXIS)
    for cw in ("cw_east", "cw_west"):
        conflicts[cw] |= set(EW_AXIS)

    # Pedestrians do not conflict with pedestrians.
    return {m: frozenset(s) for m, s in conflicts.items()}


CONFLICTS: Dict[str, FrozenSet[str]] = _build_conflicts()


def conflicting(movement_a: str, movement_b: str) -> bool:
    """True when two movements must never be served simultaneously."""
    return movement_b in CONFLICTS.get(movement_a, frozenset())


def opposing_phase(phase_key: str) -> str:
    """The other vehicle phase in the ring."""
    idx = VEHICLE_RING.index(phase_key) if phase_key in VEHICLE_RING else 0
    return VEHICLE_RING[(idx + 1) % len(VEHICLE_RING)]


def validate_phase_definitions() -> None:
    """Assert every declared phase is internally conflict-free.

    Called at import time by the controller. A phase that serves two
    conflicting movements is a defect in this file, not a runtime
    condition, so it should fail loudly and early.
    """
    for phase in PHASES.values():
        served = sorted(phase.vehicles | phase.pedestrians)
        for i, a in enumerate(served):
            for b in served[i + 1 :]:
                if conflicting(a, b):
                    raise AssertionError(
                        f"phase {phase.key!r} serves conflicting movements "
                        f"{a!r} and {b!r}"
                    )


validate_phase_definitions()
