"""The conflict matrix is the safety foundation; it gets its own tests."""

import pytest

from app.core.phases import (
    APPROACHES,
    CROSSWALKS,
    CONFLICTS,
    PHASES,
    conflicting,
    validate_phase_definitions,
)


def test_every_declared_phase_is_internally_conflict_free():
    validate_phase_definitions()  # raises on failure


def test_conflict_matrix_is_symmetric():
    for a in APPROACHES + CROSSWALKS:
        for b in APPROACHES + CROSSWALKS:
            assert conflicting(a, b) == conflicting(b, a), (a, b)


def test_no_movement_conflicts_with_itself():
    for movement in APPROACHES + CROSSWALKS:
        assert not conflicting(movement, movement)


def test_vehicle_axes_conflict():
    assert conflicting("north", "east")
    assert conflicting("south", "west")
    assert not conflicting("north", "south")
    assert not conflicting("east", "west")


def test_crosswalk_conflicts_with_the_axis_it_crosses_only():
    # Walking across the north leg means crossing the north-south roadway.
    assert conflicting("cw_north", "north")
    assert conflicting("cw_north", "south")
    assert not conflicting("cw_north", "east")
    assert not conflicting("cw_north", "west")


def test_pedestrians_never_conflict_with_each_other():
    for a in CROSSWALKS:
        for b in CROSSWALKS:
            assert not conflicting(a, b)


def test_each_phase_serves_the_crosswalks_parallel_to_its_vehicles():
    ns = PHASES["NS"]
    assert ns.pedestrians == {"cw_east", "cw_west"}
    ew = PHASES["EW"]
    assert ew.pedestrians == {"cw_north", "cw_south"}
    assert PHASES["PED"].vehicles == frozenset()
