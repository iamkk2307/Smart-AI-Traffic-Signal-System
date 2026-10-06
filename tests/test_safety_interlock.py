"""
A verifier that never fires has not been shown to work.

Each test here hand-builds a command stream containing one specific defect
and asserts the interlock names it. The v1 codebase this replaces had a
safety module whose docstring promised minimum green and all-red clearance
checks that did not exist; these tests are what makes the promise checkable.
"""

from datetime import datetime, timezone

import pytest

from app.core.phases import APPROACHES, CROSSWALKS, Interval, Signal, Walk
from app.core.safety import SafetyInterlock
from app.core.types import SignalCommand


def command(vehicle=None, pedestrian=None, phase="NS", interval=Interval.GREEN):
    return SignalCommand(
        timestamp=datetime.now(timezone.utc),
        phase=phase,
        interval=interval,
        vehicle={a: (vehicle or {}).get(a, Signal.RED) for a in APPROACHES},
        pedestrian={c: (pedestrian or {}).get(c, Walk.DONT_WALK) for c in CROSSWALKS},
        interval_elapsed=0.0,
        interval_remaining=None,
        cycle=1,
    )


def test_clean_stream_raises_nothing():
    lock = SafetyInterlock()
    t = 0.0
    # NS green 12s, yellow 4s, all-red 2s, then EW green.
    for _ in range(12):
        t += 1
        assert not lock.check(t, command({"north": Signal.GREEN, "south": Signal.GREEN}))
    for _ in range(4):
        t += 1
        assert not lock.check(
            t,
            command({"north": Signal.YELLOW, "south": Signal.YELLOW},
                    interval=Interval.YELLOW),
        )
    for _ in range(2):
        t += 1
        assert not lock.check(t, command(interval=Interval.ALL_RED))
    t += 1
    assert not lock.check(
        t, command({"east": Signal.GREEN, "west": Signal.GREEN}, phase="EW")
    )


def test_catches_conflicting_vehicle_greens():
    lock = SafetyInterlock()
    events = lock.check(1.0, command({"north": Signal.GREEN, "east": Signal.GREEN}))
    assert [e.kind for e in events] == ["conflicting_movements"]


def test_catches_pedestrian_walking_into_a_conflicting_green():
    lock = SafetyInterlock()
    events = lock.check(
        1.0,
        command({"north": Signal.GREEN}, {"cw_north": Walk.WALK}),
    )
    assert any(e.kind == "conflicting_movements" for e in events)


def test_catches_flashing_dont_walk_against_a_conflicting_green():
    # A pedestrian still in the crossing is exposed just as much as one
    # starting out, so FLASHING counts as served.
    lock = SafetyInterlock()
    events = lock.check(
        1.0, command({"north": Signal.GREEN}, {"cw_south": Walk.FLASHING})
    )
    assert any(e.kind == "conflicting_movements" for e in events)


def test_catches_green_to_red_with_no_change_interval():
    lock = SafetyInterlock()
    t = 0.0
    for _ in range(12):
        t += 1
        lock.check(t, command({"north": Signal.GREEN, "south": Signal.GREEN}))
    events = lock.check(t + 1, command(interval=Interval.ALL_RED))
    assert any(e.kind == "missing_change_interval" for e in events)


def test_catches_short_green():
    lock = SafetyInterlock()
    lock.check(1.0, command({"north": Signal.GREEN}))
    lock.check(2.0, command({"north": Signal.GREEN}))
    events = lock.check(
        3.0, command({"north": Signal.YELLOW}, interval=Interval.YELLOW)
    )
    assert any(e.kind == "short_green" for e in events)


def test_catches_short_yellow():
    lock = SafetyInterlock()
    t = 0.0
    for _ in range(10):
        t += 1
        lock.check(t, command({"north": Signal.GREEN}))
    t += 1
    lock.check(t, command({"north": Signal.YELLOW}, interval=Interval.YELLOW))
    events = lock.check(t + 1, command(interval=Interval.ALL_RED))
    assert any(e.kind == "short_yellow" for e in events)


def test_catches_insufficient_clearance_between_conflicting_greens():
    lock = SafetyInterlock()
    t = 0.0
    for _ in range(10):
        t += 1
        lock.check(t, command({"north": Signal.GREEN}))
    for _ in range(8):
        t += 0.5
        lock.check(t, command({"north": Signal.YELLOW}, interval=Interval.YELLOW))
    # Conflicting green half a second after the yellow dropped: the required
    # all-red clearance is 1.0s, so this must be caught.
    events = lock.check(t + 0.5, command({"east": Signal.GREEN}, phase="EW"))
    assert any(e.kind == "insufficient_clearance" for e in events)


def test_catches_yellow_returning_to_green():
    lock = SafetyInterlock()
    lock.check(1.0, command({"north": Signal.YELLOW}, interval=Interval.YELLOW))
    events = lock.check(2.0, command({"north": Signal.GREEN}))
    assert any(e.kind == "yellow_to_green" for e in events)


def test_statistics_group_by_kind():
    lock = SafetyInterlock()
    lock.check(1.0, command({"north": Signal.GREEN, "east": Signal.GREEN}))
    lock.check(2.0, command({"south": Signal.GREEN, "west": Signal.GREEN}))
    stats = lock.statistics()
    # The second command also snaps north and east from green to red, so it
    # raises change-interval findings on top of the conflict finding.
    assert stats["by_kind"]["conflicting_movements"] == 2
    assert stats["by_kind"]["missing_change_interval"] == 2
    assert stats["total"] == 4
