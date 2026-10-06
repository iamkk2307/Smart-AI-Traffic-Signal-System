"""
Simulation tests.

The simulation has to be closed-loop for any of the controller evaluation to
mean anything, so that property is tested directly: red must grow a queue
and green must discharge it.
"""

import pytest

from app.core.controller import AdaptiveController, TimingPlan
from app.core.phases import APPROACHES, Interval, Signal, Walk
from app.core.types import SignalCommand
from app.sim.intersection import ApproachDemand, IntersectionSim, SimConfig


def fixed_command(greens):
    from datetime import datetime, timezone

    from app.core.phases import CROSSWALKS

    return SignalCommand(
        timestamp=datetime.now(timezone.utc),
        phase="NS",
        interval=Interval.GREEN,
        vehicle={
            a: (Signal.GREEN if a in greens else Signal.RED) for a in APPROACHES
        },
        pedestrian={c: Walk.DONT_WALK for c in CROSSWALKS},
        interval_elapsed=0.0,
        interval_remaining=None,
        cycle=1,
    )


def test_a_red_approach_accumulates_queue():
    sim = IntersectionSim(SimConfig(seed=1))
    for _ in range(120):
        frame = sim.step(1.0, fixed_command([]))
    assert frame.approaches["north"].queue > 10


def test_a_green_approach_discharges_its_queue():
    sim = IntersectionSim(SimConfig(seed=1))
    for _ in range(120):
        sim.step(1.0, fixed_command([]))
    queued = sim.state["north"].queue
    for _ in range(90):
        frame = sim.step(1.0, fixed_command(["north", "south"]))
    assert frame.approaches["north"].queue < queued / 2


def test_demand_follows_the_hour_of_day_profile():
    from datetime import datetime, timezone

    peak = IntersectionSim(
        SimConfig(
            seed=1,
            start_time=datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc),
        )
    )
    night = IntersectionSim(
        SimConfig(
            seed=1,
            start_time=datetime(2026, 6, 1, 3, 0, tzinfo=timezone.utc),
        )
    )
    assert peak.demand_multiplier() > 4 * night.demand_multiplier()


def test_delay_accumulates_only_while_vehicles_are_queued():
    sim = IntersectionSim(SimConfig(seed=2))
    for _ in range(60):
        sim.step(1.0, fixed_command([]))
    delayed = sim.metrics()["total_delay_veh_s"]
    assert delayed > 0


def test_adaptive_control_beats_fixed_time_on_the_same_demand():
    """The comparison the architecture exists to justify.

    Both controllers see the same arrivals from the same seed. If adaptive
    control does not reduce mean delay on an asymmetric demand pattern, the
    pressure model is not earning its complexity.
    """
    config = lambda: SimConfig(
        seed=42,
        demand={
            "north": ApproachDemand(980, 2),
            "south": ApproachDemand(900, 2),
            "east": ApproachDemand(260, 1),
            "west": ApproachDemand(240, 1),
        },
        emergency_rate_per_hour=0.0,
        incident_rate_per_hour=0.0,
        ped_call_rate_per_hour=0.0,
    )

    results = {}
    for mode in ("adaptive", "fixed_time"):
        timing = TimingPlan()
        sim = IntersectionSim(config(), timing)
        controller = AdaptiveController(timing, mode=mode)
        command = None
        for _ in range(3600):
            frame = sim.step(1.0, command)
            command = controller.tick(frame, 1.0)
        results[mode] = sim.metrics()["mean_delay_s"]

    assert results["adaptive"] < results["fixed_time"], results


def test_arrival_stream_is_independent_of_the_controller():
    """Two controllers on one seed must see exactly the same demand.

    The benchmark's only claim is that the difference between adaptive and
    fixed-time is the timing policy. That holds only if arrivals are drawn
    from a generator the controller cannot perturb, so the speed sampler
    gets its own. Without this separation a controller that discharges more
    vehicles consumes more draws and silently shifts the arrival stream,
    and every comparison in the README becomes meaningless.
    """
    from app.core.controller import AdaptiveController, TimingPlan
    from app.core.phases import APPROACHES
    from app.sim.intersection import IntersectionSim, SimConfig

    arrivals = {}
    for mode in ("adaptive", "fixed_time"):
        sim = IntersectionSim(SimConfig(seed=4242), TimingPlan())
        controller = AdaptiveController(TimingPlan(), mode=mode)
        command = None
        for _ in range(1800):
            frame = sim.step(1.0, command)
            command = controller.tick(frame, 1.0)
        arrivals[mode] = {a: round(sim.state[a].arrivals, 6) for a in APPROACHES}

    assert arrivals["adaptive"] == arrivals["fixed_time"]
    assert sum(arrivals["adaptive"].values()) > 0
