"""
Controller property tests.

These are the guarantees the controller claims. Each one is checked over a
long closed-loop run against the microsimulation, so a regression in timing
policy shows up as a failing property rather than as a slightly different
number on a dashboard.
"""

import random

import pytest

from app.core.controller import AdaptiveController, TimingPlan
from app.core.phases import APPROACHES, Interval, Signal, Walk
from app.core.safety import SafetyInterlock
from app.sensors import SimulationSensor
from app.sim.intersection import ApproachDemand, SimConfig


def run(
    seconds: int = 2400,
    dt: float = 1.0,
    mode: str = "adaptive",
    timing: TimingPlan | None = None,
    config: SimConfig | None = None,
    forecast=None,
):
    """Drive the controller against the simulation and return the trace."""
    timing = timing or TimingPlan()
    sensor = SimulationSensor(config or SimConfig(seed=11), timing)
    controller = AdaptiveController(timing, mode=mode, forecast=forecast)
    interlock = SafetyInterlock()

    command = None
    trace = []
    for _ in range(int(seconds / dt)):
        frame = sensor.read(dt, command)
        command = controller.tick(frame, dt)
        served = [c for c, w in command.pedestrian.items() if w is not Walk.DONT_WALK]
        if served:
            sensor.sim.clear_ped_calls(served)
        findings = interlock.check(controller.t, command)
        trace.append((controller.t, frame, command, findings))
    return controller, sensor, interlock, trace


# ───────────────────────── safety properties ─────────────────────────


def test_long_run_produces_no_interlock_findings():
    _, _, interlock, _ = run(3600)
    assert interlock.events == [], [e.detail for e in interlock.events]


def test_no_conflicting_greens_ever():
    _, _, _, trace = run(3600)
    for t, _, command, _ in trace:
        greens = set(command.greens())
        assert not (greens & {"north", "south"} and greens & {"east", "west"}), t


def test_every_green_ends_through_yellow_then_all_red():
    _, _, _, trace = run(3600)
    previous = None
    for t, _, command, _ in trace:
        if previous is not None:
            for approach in APPROACHES:
                was, now = previous.vehicle[approach], command.vehicle[approach]
                if was is Signal.GREEN:
                    assert now in (Signal.GREEN, Signal.YELLOW), (approach, t)
                if was is Signal.YELLOW:
                    assert now in (Signal.YELLOW, Signal.RED), (approach, t)
        previous = command


def test_pedestrians_get_a_full_walk_plus_clearance():
    timing = TimingPlan()
    _, _, _, trace = run(3600, timing=timing)
    walking = 0
    runs = []
    for _, _, command, _ in trace:
        if any(w is Walk.WALK for w in command.pedestrian.values()):
            walking += 1
        elif walking:
            runs.append(walking)
            walking = 0
    assert runs, "no pedestrian service occurred in an hour of simulated demand"
    # Walk interval is not truncated: every served call gets the configured
    # walk time, give or take one sample.
    assert min(runs) >= timing.ped_walk - 1


def test_fault_drops_every_approach_to_red():
    controller, sensor, _, _ = run(120)
    frame = sensor.read(1.0, None)
    controller.trip_fault("injected")
    command = controller.tick(frame, 1.0)
    assert command.interval is Interval.FAULT_FLASH
    assert all(s is Signal.RED for s in command.vehicle.values())
    assert all(w is Walk.DONT_WALK for w in command.pedestrian.values())


# ───────────────────────── timing properties ─────────────────────────


def test_minimum_green_is_honoured():
    timing = TimingPlan(min_green=12.0)
    _, _, _, trace = run(3600, timing=timing)
    held = 0
    durations = []
    for _, _, command, _ in trace:
        if command.interval is Interval.GREEN:
            held += 1
        elif held:
            durations.append(held)
            held = 0
    assert durations
    assert min(durations) >= timing.min_green - 1, min(durations)


def test_green_never_exceeds_max_green_while_demand_is_waiting():
    timing = TimingPlan(max_green=40.0)
    config = SimConfig(
        seed=5,
        demand={a: ApproachDemand(850, 2) for a in APPROACHES},  # all four busy
    )
    _, _, _, trace = run(2400, timing=timing, config=config)
    held = 0
    worst = 0
    for _, _, command, _ in trace:
        if command.interval is Interval.GREEN and not command.preempted:
            held += 1
            worst = max(worst, held)
        else:
            held = 0
    # Max out plus one sample of latency.
    assert worst <= timing.max_green + 2, worst


def test_controller_rests_in_green_when_nothing_conflicts():
    """With demand on one axis only, cycling to an empty street is waste."""
    timing = TimingPlan(max_green=30.0)
    config = SimConfig(
        seed=3,
        demand={
            "north": ApproachDemand(900, 2),
            "south": ApproachDemand(900, 2),
            "east": ApproachDemand(0, 2),
            "west": ApproachDemand(0, 2),
        },
        ped_call_rate_per_hour=0.0,
        emergency_rate_per_hour=0.0,
        incident_rate_per_hour=0.0,
    )
    controller, _, _, trace = run(1200, timing=timing, config=config)
    ns_green = sum(
        1 for _, _, c, _ in trace if c.interval is Interval.GREEN and c.phase == "NS"
    )
    assert ns_green / len(trace) > 0.85, ns_green / len(trace)
    assert controller.cycle < 8, controller.cycle


def test_no_approach_is_held_red_past_the_ceiling():
    timing = TimingPlan(max_red=90.0, max_green=60.0)
    config = SimConfig(
        seed=9,
        demand={
            "north": ApproachDemand(1100, 2),
            "south": ApproachDemand(1100, 2),
            "east": ApproachDemand(40, 1),  # almost nobody, easy to starve
            "west": ApproachDemand(40, 1),
        },
        emergency_rate_per_hour=0.0,
    )
    controller, _, _, trace = run(3600, timing=timing, config=config)
    yellow, all_red = timing.clearance_for("clear")
    # Once the ceiling is reached the phase wins the next selection, so the
    # bound includes the remaining green of the phase in service.
    bound = timing.max_red + timing.max_green + yellow + all_red + 2
    worst = max(
        max(controller.red_time(a) for a in APPROACHES)
        for _ in (0,)
    )
    observed = 0.0
    for t, frame, _, _ in trace:
        observed = max(observed, max(frame.approaches[a].max_wait for a in APPROACHES))
    assert observed <= bound, (observed, bound)


# ───────────────────────── preemption properties ─────────────────────────


def test_preemption_is_bounded_by_its_budget():
    timing = TimingPlan(preempt_budget=60.0, preempt_window=600.0,
                        preempt_max_hold=30.0)
    config = SimConfig(
        seed=4,
        emergency_rate_per_hour=400.0,  # relentless: try to lock the ring
        emergency_duration=60.0,
    )
    controller, _, _, trace = run(2400, timing=timing, config=config)
    preempted = [(t, c) for t, _, c, _ in trace if c.preempted]
    assert preempted, "no preemption occurred despite constant emergency input"

    # Check every 600s window: preemption seconds inside it must respect the
    # budget, give or take the hold granularity.
    stamps = [t for t, c in preempted]
    for start in range(0, 2400, 60):
        inside = [t for t in stamps if start <= t < start + timing.preempt_window]
        assert len(inside) <= timing.preempt_budget + timing.preempt_max_hold, (
            start,
            len(inside),
        )


def test_preemption_still_reaches_through_yellow_and_all_red():
    """An emergency must not produce a snap change; it buys priority, not a
    shortcut past the clearance intervals."""
    timing = TimingPlan()
    config = SimConfig(seed=2, emergency_rate_per_hour=250.0,
                       emergency_duration=30.0)
    _, _, interlock, trace = run(2400, timing=timing, config=config)
    assert interlock.events == [], [e.detail for e in interlock.events]
    assert any(c.preempted for _, _, c, _ in trace)


def test_exhausted_budget_is_recorded_not_dropped():
    timing = TimingPlan(preempt_budget=25.0, preempt_window=3600.0)
    config = SimConfig(seed=6, emergency_rate_per_hour=600.0,
                       emergency_duration=90.0)
    controller, _, _, _ = run(1800, timing=timing, config=config)
    triggers = {d.trigger for d in controller.decisions}
    assert "preempt" in triggers
    assert "preempt_denied" in triggers, triggers
    denied = next(d for d in controller.decisions if d.trigger == "preempt_denied")
    assert "budget" in denied.reason.lower()


# ───────────────────────── explainability ─────────────────────────


def test_every_decision_carries_a_reason_and_its_numbers():
    controller, _, _, _ = run(1800)
    assert controller.decisions
    for decision in controller.decisions:
        assert len(decision.reason) > 30
        assert set(decision.pressure) >= {"NS", "EW"}
        for terms in decision.terms.values():
            assert {"queue", "occupancy", "wait"} <= set(terms)


def test_pressure_terms_sum_to_the_reported_pressure():
    controller, sensor, _, _ = run(600)
    frame = sensor.read(1.0, None)
    pressure, terms = controller._pressure(frame)
    for phase, total in pressure.items():
        assert total == pytest.approx(sum(terms[phase].values()), abs=1e-3)


def test_a_uniform_multiplier_cannot_change_the_ranking():
    """The defect this guards against: v1 multiplied every direction's score
    by the same weather and rush-hour factors, then cited them as reasons.
    A monotone transform applied to all candidates cannot move an argmax."""
    controller, sensor, _, _ = run(900)
    frame = sensor.read(1.0, None)
    clear, _ = controller._pressure(frame)
    frame.weather = "rain"
    wet, _ = controller._pressure(frame)
    assert max(clear, key=clear.get) == max(wet, key=wet.get)
    # Weather instead moves the timing, which is where it belongs.
    assert controller.timing.clearance_for("rain") > controller.timing.clearance_for(
        "clear"
    )
    assert controller.timing.discharge_rate("rain") < controller.timing.discharge_rate(
        "clear"
    )


def test_forecast_term_is_optional():
    with_forecast, _, _, _ = run(600, forecast=lambda approach: 12.0)
    without, _, _, _ = run(600, forecast=None)
    assert with_forecast.decisions and without.decisions
    terms = with_forecast.decisions[0].terms["NS"]
    assert terms["forecast"] > 0
    assert without.decisions[0].terms["NS"]["forecast"] == 0.0


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        AdaptiveController(mode="vibes")


# ───────────────────── glare and detector integrity ─────────────────────


def _glare_frame(sensor, glare_legs=("west",)):
    """A frame from the simulation with glare asserted on some legs."""
    frame = sensor.read(1.0, None)
    frame.environment.glare = bool(glare_legs)
    frame.environment.approaches_affected = list(glare_legs)
    frame.environment.uv_index = 0.4
    frame.environment.solar_elevation = 6.0
    frame.environment.illuminance_lux = 24000.0
    return frame


def test_glare_lengthens_clearance_and_raises_minimum_green():
    timing = TimingPlan()
    plain_yellow, plain_red = timing.clearance_for("clear", glare=False)
    glare_yellow, glare_red = timing.clearance_for("clear", glare=True)
    assert glare_yellow > plain_yellow
    assert glare_red > plain_red
    assert timing.min_green_for("clear", glare=True) > timing.min_green_for("clear")
    assert timing.discharge_rate("clear", glare=True) < timing.discharge_rate("clear")


def test_glare_and_adverse_weather_compound():
    timing = TimingPlan()
    rain = timing.clearance_for("rain", glare=False)[1]
    both = timing.clearance_for("rain", glare=True)[1]
    assert both > rain


def test_glare_actually_holds_yellow_longer_in_the_ring():
    """The timing plan change must reach the interval state machine."""
    from app.sensors import SimulationSensor

    # Quarter-second ticks: the glare adjustment is a fraction of a second
    # on the yellow, so a 1 s tick would quantise it away and the test
    # would pass or fail on rounding rather than on behaviour.
    dt = 0.25
    durations = {}
    for glare in (False, True):
        sensor = SimulationSensor(SimConfig(seed=11), TimingPlan())
        controller = AdaptiveController(timing=TimingPlan())
        held = 0.0
        seen_yellow = False
        for _ in range(4000):
            frame = _glare_frame(sensor, ("west",) if glare else ())
            command = controller.tick(frame, dt)
            if command.interval is Interval.YELLOW:
                seen_yellow = True
                held += dt
            elif seen_yellow:
                break
        durations[glare] = held
    assert durations[True] > durations[False]
    expected = TimingPlan().glare_extra_clearance * 0.4
    assert durations[True] - durations[False] == pytest.approx(expected, abs=dt)


def test_glare_is_named_in_the_decision_record():
    """An operator reading the log must see why timing changed."""
    from app.sensors import SimulationSensor

    sensor = SimulationSensor(SimConfig(seed=5), TimingPlan())
    controller = AdaptiveController(timing=TimingPlan())
    for _ in range(600):
        controller.tick(_glare_frame(sensor, ("west",)), 1.0)
        if controller.decisions:
            break
    assert controller.decisions
    reason = controller.decisions[-1].reason
    assert "glare" in reason.lower()
    assert "west" in reason


def test_glare_does_not_change_the_phase_ranking():
    """Glare adjusts intervals, not scores.

    The same argument as the weather multiplier test above: a term applied
    equally to every candidate cannot change an argmax, so glare is kept
    out of the pressure function entirely.
    """
    from app.sensors import SimulationSensor

    sensor = SimulationSensor(SimConfig(seed=3), TimingPlan())
    controller = AdaptiveController(timing=TimingPlan())
    for _ in range(120):
        controller.tick(sensor.read(1.0, None), 1.0)

    frame = sensor.read(1.0, None)
    plain, _ = controller._pressure(frame)
    frame.environment.glare = True
    frame.environment.approaches_affected = ["west", "north"]
    glared, _ = controller._pressure(frame)
    assert plain == glared


def test_degraded_detector_prevents_gap_out_and_forces_a_call():
    """A dead detector must not read as a clear road."""
    from app.sensors import SimulationSensor

    sensor = SimulationSensor(SimConfig(seed=9), TimingPlan())
    controller = AdaptiveController(timing=TimingPlan())
    frame = sensor.read(1.0, None)
    for approach in APPROACHES:
        reading = frame.approaches[approach]
        reading.queue = 0
        reading.occupancy = 0.0
        reading.gap = 99.0
    assert controller._gapped_out(frame) is True
    assert not any(controller._has_call(frame, a) for a in APPROACHES)

    frame.degraded_approaches = list(APPROACHES)
    for approach in APPROACHES:
        frame.approaches[approach].degraded = True
    assert controller._gapped_out(frame) is False
    assert all(controller._has_call(frame, a) for a in APPROACHES)
