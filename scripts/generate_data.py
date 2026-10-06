#!/usr/bin/env python3
"""
Generate a synthetic dataset for this intersection.

Every number below comes out of the closed-loop simulation being driven by
the real controller, interlock and incident detector. Nothing is drawn from
a distribution chosen to look plausible on a chart: arrivals are random,
but queues, delays, speeds, phase decisions and incidents are consequences
of the control policy acting on those arrivals. A dataset fabricated
independently of the controller would be useless for exactly the thing a
dataset is for — telling you whether the controller is any good.

What it writes, under ``data/synthetic/`` by default:

===========================  ====================================================
``detector_log.csv``         One row per approach per sample interval: queue,
                             flow, occupancy, measured speed, 85th percentile,
                             speeding count, wait, signal indication, plus the
                             environment (weather, UV, sun angle, glare).
``decisions.csv``            Every phase change with its trigger, the reason
                             text, and the pressure terms behind it.
``incidents.csv``            Rule-layer incidents raised during the run.
``iot_messages.jsonl``       The same run re-expressed as MQTT payloads, ready
                             to replay through ``IoTSensor.submit``. Includes
                             deliberate faults — dropouts, malformed bodies,
                             out-of-range values — because a clean feed does
                             not exercise the staleness or validation paths.
``runs.csv``                 One row per simulated day: demand profile,
                             weather, delay, cycles, safety events.
``summary.json``             Run parameters and aggregate figures, including
                             the adaptive-versus-fixed comparison.
===========================  ====================================================

Usage::

    python scripts/generate_data.py                    # 7 days, random seed
    python scripts/generate_data.py --days 3 --seed 42 # reproducible
    python scripts/generate_data.py --sample-seconds 30
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.controller import AdaptiveController, TimingPlan
from app.core.phases import APPROACHES, CROSSWALKS, Walk
from app.core.safety import SafetyInterlock
from app.ml.incidents import IncidentDetector
from app.sim.intersection import ApproachDemand, IntersectionSim, SimConfig

#: Weather is sampled per day from this distribution rather than set to
#: "clear" throughout. A dataset with one condition in it cannot show
#: whether the weather handling does anything.
WEATHER_WEIGHTS = {
    "clear": 0.46,
    "cloudy": 0.22,
    "rain": 0.16,
    "fog": 0.06,
    "thunderstorm": 0.07,
    "snow": 0.03,
}

#: Posted limits and lane counts for the site. The arterial is the
#: north-south axis; the cross street is narrower and slower.
GEOMETRY = {
    "north": dict(lanes=2, speed_limit_kmh=60.0),
    "south": dict(lanes=2, speed_limit_kmh=60.0),
    "east": dict(lanes=1, speed_limit_kmh=40.0),
    "west": dict(lanes=1, speed_limit_kmh=40.0),
}

DETECTOR_FIELDS = [
    "timestamp", "day", "t", "approach", "signal", "phase", "interval", "cycle",
    "queue", "flow", "occupancy", "mean_speed", "speed_85th", "speed_limit",
    "speeding", "speed_samples", "gap", "max_wait", "degraded",
    "weather", "uv_index", "solar_elevation", "solar_azimuth", "glare",
    "glare_on_this_approach", "preempted",
]


def daily_demand(rng: random.Random) -> dict:
    """Random but realistic demand for one day.

    Directional split is drawn too, so the dataset contains days where the
    arterial is tidal and days where it is balanced. A dataset in which
    north and south always match would hide any asymmetry bug.
    """
    arterial = rng.uniform(620, 1080)
    cross = rng.uniform(180, 460)
    tidal = rng.uniform(0.75, 1.25)
    return {
        "north": ApproachDemand(
            round(arterial * tidal), **GEOMETRY["north"]
        ),
        "south": ApproachDemand(
            round(arterial / tidal), **GEOMETRY["south"]
        ),
        "east": ApproachDemand(
            round(cross * rng.uniform(0.85, 1.15)), **GEOMETRY["east"]
        ),
        "west": ApproachDemand(
            round(cross * rng.uniform(0.85, 1.15)), **GEOMETRY["west"]
        ),
    }


def pick_weather(rng: random.Random) -> str:
    return rng.choices(
        list(WEATHER_WEIGHTS), weights=list(WEATHER_WEIGHTS.values())
    )[0]


def run_day(
    day: int,
    seed: int,
    hours: float,
    sample_seconds: float,
    mode: str = "adaptive",
    collect: bool = True,
):
    """Simulate one day and return its rows and metrics.

    The day's randomness is derived from ``seed`` alone, inside this
    function. That matters for the baseline comparison: the fault injection
    below draws from the same generator, and it draws once per published
    message, so a shared generator would be advanced a different number of
    times by the adaptive pass (which collects messages) and the
    fixed-time pass (which does not). The two passes would then see
    different weather and different demand, and the comparison between them
    would be meaningless — which is exactly the bug this split fixes.
    """
    day_rng = random.Random(seed)
    weather = pick_weather(day_rng)
    demand = daily_demand(day_rng)
    config = SimConfig(
        seed=seed,
        demand=demand,
        weather=weather,
        # Start before the morning peak and compress the clock so a day's
        # shape fits in the simulated hours.
        start_time=None,
        time_scale=max(1.0, (24.0 / hours) * 0.6),
    )
    timing = TimingPlan()
    sim = IntersectionSim(config, timing)
    controller = AdaptiveController(timing, mode=mode)
    interlock = SafetyInterlock()
    detector = IncidentDetector()

    detector_rows: list[dict] = []
    decision_rows: list[dict] = []
    incident_rows: list[dict] = []
    messages: list[dict] = []
    seen_decisions = 0
    command = None
    next_sample = 0.0

    # Which stations will drop out today, and when. Real sites lose a loop
    # or a radio link; a dataset without that cannot exercise recall.
    dropout = None
    if day_rng.random() < 0.45:
        dropout = {
            "approach": day_rng.choice(APPROACHES),
            "start": day_rng.uniform(0.15, 0.7) * hours * 3600,
            "length": day_rng.uniform(90, 600),
        }

    for step in range(int(hours * 3600)):
        frame = sim.step(1.0, command)
        command = controller.tick(frame, 1.0)

        served = [c for c, w in command.pedestrian.items() if w is not Walk.DONT_WALK]
        if served:
            sim.clear_ped_calls(served)
        interlock.check(controller.t, command)

        for incident in detector.update(frame, command, 1.0):
            payload = incident.to_dict()
            row = {
                "day": day,
                "timestamp": frame.timestamp.isoformat(),
                **{k: v for k, v in payload.items() if k != "detail"},
                "detail": payload.get("detail", ""),
            }
            if row not in incident_rows:
                incident_rows.append(row)

        while seen_decisions < len(controller.decisions):
            decision = controller.decisions[seen_decisions]
            seen_decisions += 1
            if not collect:
                continue
            terms = decision.terms.get(decision.to_phase, {})
            decision_rows.append(
                {
                    "day": day,
                    "timestamp": decision.timestamp.isoformat(),
                    "cycle": decision.cycle,
                    "from_phase": decision.from_phase,
                    "to_phase": decision.to_phase,
                    "trigger": decision.trigger,
                    "served_green_time": decision.served_green_time,
                    "pressure_to": decision.pressure.get(decision.to_phase, 0.0),
                    "pressure_from": decision.pressure.get(decision.from_phase, 0.0),
                    **{f"term_{k}": round(v, 4) for k, v in terms.items()},
                    "reason": decision.reason,
                }
            )

        if not collect or controller.t < next_sample:
            continue
        next_sample = controller.t + sample_seconds

        env = frame.environment
        in_dropout = bool(
            dropout
            and dropout["start"] <= controller.t < dropout["start"] + dropout["length"]
        )
        for approach in APPROACHES:
            r = frame.approaches[approach]
            degraded = in_dropout and approach == dropout["approach"]
            detector_rows.append(
                {
                    "timestamp": frame.timestamp.isoformat(),
                    "day": day,
                    "t": round(controller.t, 1),
                    "approach": approach,
                    "signal": command.vehicle[approach].value,
                    "phase": command.phase,
                    "interval": command.interval.value,
                    "cycle": command.cycle,
                    "queue": r.queue,
                    "flow": r.flow,
                    "occupancy": r.occupancy,
                    "mean_speed": r.mean_speed,
                    "speed_85th": r.speed_85th,
                    "speed_limit": r.speed_limit,
                    "speeding": r.speeding,
                    "speed_samples": r.speed_samples,
                    "gap": r.gap,
                    "max_wait": r.max_wait,
                    "degraded": degraded,
                    "weather": frame.weather,
                    "uv_index": env.uv_index,
                    "solar_elevation": env.solar_elevation,
                    "solar_azimuth": env.solar_azimuth,
                    "glare": env.glare,
                    "glare_on_this_approach": approach in env.approaches_affected,
                    "preempted": command.preempted,
                }
            )
            messages.extend(
                _messages_for(frame, approach, r, degraded, day_rng)
            )

        if env.uv_index > 0 or env.illuminance_lux > 0:
            messages.append(
                {
                    "topic": "traffic/j1/environment/uv",
                    "payload": {
                        "uv_index": env.uv_index,
                        "illuminance_lux": env.illuminance_lux,
                    },
                    "timestamp": frame.timestamp.isoformat(),
                }
            )
        if frame.emergency_approach:
            messages.append(
                {
                    "topic": "traffic/j1/emergency",
                    "payload": {
                        "approach": frame.emergency_approach,
                        "kind": frame.emergency_kind,
                        "eta": frame.emergency_eta,
                    },
                    "timestamp": frame.timestamp.isoformat(),
                }
            )

    metrics = sim.metrics()
    run_row = {
        "day": day,
        "seed": seed,
        "mode": mode,
        "weather": weather,
        "demand_vph": {a: d.vehicles_per_hour for a, d in demand.items()},
        "mean_delay_s": metrics["mean_delay_s"],
        "vehicles_served": round(metrics["vehicles_served"]),
        "cycles": controller.cycle,
        "safety_events": len(interlock.events),
        "incidents": len(incident_rows),
        "dropout": dropout["approach"] if dropout else "",
        "dropout_seconds": round(dropout["length"], 1) if dropout else 0.0,
    }
    return detector_rows, decision_rows, incident_rows, messages, run_row


def _messages_for(frame, approach, reading, degraded, rng):
    """Express one detector reading as MQTT traffic, faults included."""
    if degraded:
        # A dropped station publishes nothing at all. That silence is the
        # signal, and it is the hardest case for a consumer to handle, so
        # the dataset has to contain it rather than a tidy "status: down".
        return []

    out = []
    payload = {
        "queue": reading.queue,
        "flow": reading.flow,
        "occupancy": reading.occupancy,
        "mean_speed": reading.mean_speed,
        "speed_85th": reading.speed_85th,
        "speeding": reading.speeding,
        "speed_samples": reading.speed_samples,
        "gap": reading.gap,
        "max_wait": reading.max_wait,
        "speed_limit": reading.speed_limit,
    }

    roll = rng.random()
    if roll < 0.004:
        # A firmware bug that sends an occupancy above 1 and a negative
        # queue. The backend must clamp these, so they belong in the data.
        payload = {**payload, "occupancy": round(rng.uniform(1.2, 6.0), 2), "queue": -3}
    elif roll < 0.007:
        out.append(
            {
                "topic": f"traffic/j1/approach/{approach}",
                "payload": "<<truncated frame>>",
                "timestamp": frame.timestamp.isoformat(),
                "malformed": True,
            }
        )
        return out

    out.append(
        {
            "topic": f"traffic/j1/approach/{approach}",
            "payload": payload,
            "timestamp": frame.timestamp.isoformat(),
        }
    )
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        path.write_text("")
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--hours", type=float, default=4.0,
                        help="simulated hours per day (clock is compressed)")
    parser.add_argument("--sample-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=None,
                        help="omit for a different dataset every run")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip the fixed-time comparison run")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(1, 10**6)
    rng = random.Random(seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    detectors, decisions, incidents, messages, runs = [], [], [], [], []
    for day in range(1, args.days + 1):
        day_seed = rng.randrange(1, 10**6)
        d, dec, inc, msg, run = run_day(
            day, day_seed, args.hours, args.sample_seconds
        )
        detectors += d
        decisions += dec
        incidents += inc
        messages += msg
        runs.append(run)
        print(
            f"day {day:>2}  seed {day_seed:<7} {run['weather']:<13}"
            f" delay {run['mean_delay_s']:>6.2f}s  cycles {run['cycles']:>4}"
            f"  safety {run['safety_events']}"
            + (f"  dropout {run['dropout']}" if run["dropout"] else "")
        )

    # The same demand, under fixed-time control, so the dataset carries its
    # own baseline. A delay figure with nothing to compare against says
    # nothing about the controller.
    baseline = None
    if not args.no_baseline:
        fixed_runs = []
        # Re-seeded from the same root, so the fixed-time runs see exactly
        # the same demand and weather days as the adaptive ones.
        rng2 = random.Random(seed)
        for day in range(1, args.days + 1):
            day_seed = rng2.randrange(1, 10**6)
            *_, run = run_day(
                day, day_seed, args.hours, args.sample_seconds,
                mode="fixed_time", collect=False,
            )
            run["mode"] = "fixed_time"
            fixed_runs.append(run)
        runs += fixed_runs
        adaptive_delay = sum(r["mean_delay_s"] for r in runs if r["mode"] == "adaptive")
        fixed_delay = sum(r["mean_delay_s"] for r in fixed_runs)
        baseline = {
            "adaptive_mean_delay_s": round(adaptive_delay / args.days, 2),
            "fixed_time_mean_delay_s": round(fixed_delay / args.days, 2),
            "delay_reduction_pct": round(
                (1 - adaptive_delay / fixed_delay) * 100, 1
            ) if fixed_delay else 0.0,
        }

    write_csv(out / "detector_log.csv", detectors, DETECTOR_FIELDS)
    write_csv(out / "decisions.csv", decisions)
    write_csv(out / "incidents.csv", incidents)
    write_csv(
        out / "runs.csv",
        [{**r, "demand_vph": json.dumps(r["demand_vph"])} for r in runs],
    )
    with (out / "iot_messages.jsonl").open("w") as handle:
        for message in messages:
            handle.write(json.dumps(message) + "\n")

    speeding = sum(r["speeding"] for r in detectors)
    samples = sum(r["speed_samples"] for r in detectors)
    glare_rows = sum(1 for r in detectors if r["glare_on_this_approach"])
    summary = {
        "seed": seed,
        "days": args.days,
        "simulated_hours_per_day": args.hours,
        "sample_seconds": args.sample_seconds,
        "rows": {
            "detector_log": len(detectors),
            "decisions": len(decisions),
            "incidents": len(incidents),
            "iot_messages": len(messages),
        },
        "speed": {
            "measurements": samples,
            "speeding_observations": speeding,
            "speeding_rate": round(speeding / samples, 4) if samples else None,
        },
        "glare_approach_rows": glare_rows,
        "malformed_messages": sum(1 for m in messages if m.get("malformed")),
        "safety_events": sum(r["safety_events"] for r in runs),
        "baseline": baseline,
        "notes": [
            "Arrivals are random; everything else is the controller's "
            "response to them.",
            "Speeds on the simulation backend are modelled from occupancy and "
            "free-flow speed. They are not measured by the tracker in "
            "app/sensors/speed.py, which needs a camera.",
            "The delay reduction here is usually larger than the figure in "
            "scripts/benchmark.py, and should not be quoted in its place. "
            "Demand is randomised, so some days are oversaturated, and a "
            "fixed-time plan degrades much faster than an actuated one once "
            "demand exceeds capacity. benchmark.py holds demand fixed at a "
            "level both controllers can serve, which is the fair comparison.",
            "Detector rows with degraded=true correspond to a station that "
            "publishes nothing at all in iot_messages.jsonl. The silence is "
            "the fault signal.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nwrote {len(detectors)} detector rows, {len(decisions)} decisions, "
          f"{len(incidents)} incidents, {len(messages)} messages to {out}/")
    if baseline:
        print(
            f"adaptive {baseline['adaptive_mean_delay_s']}s vs fixed "
            f"{baseline['fixed_time_mean_delay_s']}s  "
            f"({baseline['delay_reduction_pct']}%)"
        )
    print(f"seed {seed} — pass --seed {seed} to reproduce this dataset")


if __name__ == "__main__":
    main()
