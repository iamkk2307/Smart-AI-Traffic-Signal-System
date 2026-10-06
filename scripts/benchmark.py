#!/usr/bin/env python3
"""
Compare adaptive control against a fixed-time plan on identical demand.

Both controllers are driven by the same seeded arrival stream, so the only
difference is the timing policy. Run it before and after any change to the
pressure model — a change that does not move these numbers is not an
improvement, whatever the commit message says.

    python scripts/benchmark.py --hours 2 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.controller import AdaptiveController, TimingPlan
from app.core.phases import APPROACHES, Walk
from app.core.safety import SafetyInterlock
from app.ml.features import Observation, minute_of_week
from app.ml.predictor import ArrivalForecaster
from app.sim.intersection import ApproachDemand, IntersectionSim, SimConfig

DEMAND = {
    "north": ApproachDemand(980, 2),
    "south": ApproachDemand(900, 2),
    "east": ApproachDemand(280, 1),
    "west": ApproachDemand(240, 1),
}


def make_config(seed: int, weather: str) -> SimConfig:
    return SimConfig(seed=seed, demand=dict(DEMAND), weather=weather)


def run(mode: str, seconds: int, seed: int, weather: str, forecast=None) -> dict:
    timing = TimingPlan()
    sim = IntersectionSim(make_config(seed, weather), timing)
    controller = AdaptiveController(timing, mode=mode, forecast=forecast)
    interlock = SafetyInterlock()

    forecaster = forecast.__self__ if hasattr(forecast, "__self__") else None
    last_minute = -1
    command = None

    for _ in range(seconds):
        frame = sim.step(1.0, command)
        command = controller.tick(frame, 1.0)
        served = [c for c, w in command.pedestrian.items() if w is not Walk.DONT_WALK]
        if served:
            sim.clear_ped_calls(served)
        interlock.check(controller.t, command)

        if forecaster is not None:
            minute = minute_of_week(frame.timestamp)
            if minute != last_minute:
                last_minute = minute
                forecaster.observe_frame(frame, minute)
                if controller.t % 900 == 0:
                    forecaster.train()

    metrics = sim.metrics()
    return {
        "mode": mode,
        "mean_delay_s": metrics["mean_delay_s"],
        "vehicles_served": metrics["vehicles_served"],
        "total_delay_veh_h": round(metrics["total_delay_veh_s"] / 3600, 1),
        "cycles": controller.cycle,
        "safety_events": len(interlock.events),
        "triggers": _trigger_mix(controller),
    }


def _trigger_mix(controller) -> dict:
    mix: dict = {}
    for decision in controller.decisions:
        mix[decision.trigger] = mix.get(decision.trigger, 0) + 1
    return dict(sorted(mix.items(), key=lambda kv: -kv[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weather", default="clear")
    parser.add_argument("--with-forecast", action="store_true")
    args = parser.parse_args()

    seconds = int(args.hours * 3600)
    rows = [
        run("fixed_time", seconds, args.seed, args.weather),
        run("adaptive", seconds, args.seed, args.weather),
    ]
    if args.with_forecast:
        forecaster = ArrivalForecaster()
        rows.append(
            run(
                "adaptive",
                seconds,
                args.seed,
                args.weather,
                forecast=forecaster.as_callable(),
            )
            | {"mode": "adaptive+forecast"}
        )

    baseline = rows[0]["mean_delay_s"]
    width = max(len(r["mode"]) for r in rows)
    print(f"\n{args.hours:g}h run, seed {args.seed}, weather {args.weather}\n")
    print(f"{'mode':<{width}}  {'mean delay':>10}  {'vs fixed':>9}  "
          f"{'served':>8}  {'cycles':>7}  {'safety':>7}")
    print("-" * (width + 50))
    for row in rows:
        delta = (row["mean_delay_s"] - baseline) / baseline * 100
        print(
            f"{row['mode']:<{width}}  {row['mean_delay_s']:>9.2f}s  "
            f"{delta:>+8.1f}%  {row['vehicles_served']:>8.0f}  "
            f"{row['cycles']:>7}  {row['safety_events']:>7}"
        )
    print("\ndecision mix (adaptive):")
    for trigger, count in rows[1]["triggers"].items():
        print(f"  {trigger:<18} {count}")
    print()


if __name__ == "__main__":
    main()
