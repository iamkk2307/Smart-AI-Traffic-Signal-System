#!/usr/bin/env python3
"""
Build the standalone simulation viewer.

Takes one day out of the dataset written by ``generate_data.py`` and embeds
it in ``web/demo_template.html`` as a single self-contained HTML file: no
server, no network, no external assets. Open it anywhere.

It is a *playback* viewer and says so on the page. The numbers were produced
by the real controller, interlock and incident detector against the
closed-loop microsimulation, so they are not invented — but nothing in the
page is computing control decisions as you watch, and a demo that implied
otherwise would be worse than no demo.

    python scripts/generate_data.py --seed 7
    python scripts/build_demo.py --day 1 --out dist/simulation.html
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.controller import TimingPlan
from app.core.phases import APPROACHES, PHASES

TEMPLATE = Path(__file__).resolve().parent.parent / "web" / "demo_template.html"


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pick_day(runs: list[dict], detectors: list[dict], requested: int | None) -> str:
    """Choose which day to ship.

    Default is the day that shows the most: a dropout if there is one, then
    whichever has the most glare samples. A viewer that only ever shows the
    nominal case demonstrates nothing about the failure handling.
    """
    days = sorted({row["day"] for row in detectors}, key=int)
    if requested is not None:
        if str(requested) not in days:
            raise SystemExit(f"day {requested} not in dataset; have {days}")
        return str(requested)

    interest: dict[str, tuple[int, int]] = {}
    for day in days:
        rows = [r for r in detectors if r["day"] == day]
        dropout = sum(1 for r in rows if r["degraded"] == "True")
        glare = sum(1 for r in rows if r["glare_on_this_approach"] == "True")
        interest[day] = (1 if dropout else 0, glare)
    return max(days, key=lambda d: interest[d])


def build(day: str, detectors, decisions, incidents, runs, summary) -> dict:
    rows = [r for r in detectors if r["day"] == day]
    by_t: dict[float, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_t[float(row["t"])][row["approach"]] = row
    stamps = sorted(by_t)
    # Only keep samples where every approach reported, so an index into one
    # series always lines up with the same instant in every other.
    stamps = [t for t in stamps if len(by_t[t]) == len(APPROACHES)]

    timing = TimingPlan()
    payload: dict = {
        "approaches": list(APPROACHES),
        "phase_label": {k: p.label for k, p in PHASES.items()},
        "limit": {},
        "t": [], "clock": [], "phase": [], "interval": [], "cycle": [],
        "signal": [], "queue": [], "flow": [], "occ": [], "speed": [],
        "p85": [], "speeding": [], "wait": [],
        "uv": [], "elev": [], "az": [], "glare_legs": [], "degraded": [],
        "clearance": [], "preempted": [],
    }

    first = by_t[stamps[0]][APPROACHES[0]]
    payload["weather"] = first["weather"]
    for approach in APPROACHES:
        payload["limit"][approach] = float(by_t[stamps[0]][approach]["speed_limit"])

    for t in stamps:
        group = by_t[t]
        lead = group[APPROACHES[0]]
        glare_legs = [a for a in APPROACHES if group[a]["glare_on_this_approach"] == "True"]
        glare = lead["glare"] == "True"
        yellow, all_red = timing.clearance_for(lead["weather"], glare)
        payload["t"].append(float(t))
        payload["clock"].append(lead["timestamp"])
        payload["phase"].append(lead["phase"])
        payload["interval"].append(lead["interval"])
        payload["cycle"].append(int(lead["cycle"]))
        payload["preempted"].append(lead["preempted"] == "True")
        payload["uv"].append(float(lead["uv_index"]))
        payload["elev"].append(float(lead["solar_elevation"]))
        payload["az"].append(float(lead["solar_azimuth"]))
        payload["glare_legs"].append(glare_legs)
        payload["degraded"].append(
            [a for a in APPROACHES if group[a]["degraded"] == "True"]
        )
        payload["clearance"].append(
            [
                round(yellow, 2),
                round(all_red, 2),
                round(timing.discharge_rate(lead["weather"], glare), 3),
            ]
        )
        payload["signal"].append([group[a]["signal"] for a in APPROACHES])
        payload["queue"].append([int(group[a]["queue"]) for a in APPROACHES])
        payload["flow"].append([float(group[a]["flow"]) for a in APPROACHES])
        payload["occ"].append([float(group[a]["occupancy"]) for a in APPROACHES])
        payload["speed"].append([float(group[a]["mean_speed"]) for a in APPROACHES])
        payload["p85"].append([float(group[a]["speed_85th"]) for a in APPROACHES])
        payload["speeding"].append([int(group[a]["speeding"]) for a in APPROACHES])
        payload["wait"].append([float(group[a]["max_wait"]) for a in APPROACHES])

    payload["sample_seconds"] = (
        round(stamps[1] - stamps[0], 2) if len(stamps) > 1 else 1.0
    )

    # ── align the decision log to sample indices ──
    # Decision timestamps are generation wall-clock, not simulated time, so
    # they cannot be matched directly. Phase transitions in the sampled
    # series can be, and that is what the log is describing.
    day_decisions = [d for d in decisions if d["day"] == day]
    aligned, cursor = [], 0
    for idx in range(1, len(stamps)):
        prev, current = payload["phase"][idx - 1], payload["phase"][idx]
        if prev == current:
            continue
        while cursor < len(day_decisions) and day_decisions[cursor]["to_phase"] != current:
            cursor += 1
        if cursor >= len(day_decisions):
            break
        d = day_decisions[cursor]
        cursor += 1
        aligned.append(
            {
                "i": idx,
                "from": d["from_phase"],
                "to": d["to_phase"],
                "trigger": d["trigger"],
                "cycle": int(d["cycle"]),
                "green": float(d["served_green_time"]),
                "reason": d["reason"],
                "terms": {
                    key[5:]: float(value)
                    for key, value in d.items()
                    if key.startswith("term_") and value not in ("", None)
                },
            }
        )
    payload["decisions"] = aligned

    # ── incidents, matched on the simulated timestamp ──
    index_of = {stamp: idx for idx, stamp in enumerate(payload["clock"])}
    payload["incidents"] = []
    for row in incidents:
        if row["day"] != day:
            continue
        idx = index_of.get(row["timestamp"])
        if idx is None:
            # Raised between samples; attach it to the next one rather than
            # dropping it.
            idx = next(
                (k for k, c in enumerate(payload["clock"]) if c >= row["timestamp"]),
                None,
            )
        if idx is None:
            continue
        payload["incidents"].append(
            {
                "i": idx,
                "approach": row.get("approach", ""),
                "kind": row.get("kind", ""),
                "severity": row.get("severity", ""),
                "layer": row.get("layer", "rule"),
                "detail": row.get("detail", ""),
            }
        )

    # ── totals across the whole dataset, not just this day ──
    adaptive = [r for r in runs if r["mode"] == "adaptive"]
    fixed = [r for r in runs if r["mode"] == "fixed_time"]
    mean = lambda rs: sum(float(r["mean_delay_s"]) for r in rs) / max(len(rs), 1)
    baseline = (summary or {}).get("baseline") or {}
    payload["totals"] = {
        "adaptive_delay": round(baseline.get("adaptive_mean_delay_s", mean(adaptive)), 2),
        "fixed_delay": round(baseline.get("fixed_time_mean_delay_s", mean(fixed)), 2),
        "reduction": baseline.get("delay_reduction_pct", 0.0),
        "safety_events": sum(int(r["safety_events"]) for r in runs),
    }

    run_row = next((r for r in adaptive if r["day"] == day), {})
    demand = json.loads(run_row.get("demand_vph", "{}") or "{}")
    payload["subtitle"] = (
        f"Day {day} of {len(adaptive)} · {payload['weather']} · demand "
        + ", ".join(f"{a[0].upper()} {int(v)}" for a, v in demand.items())
        + f" vph · {len(stamps)} samples at {payload['sample_seconds']:g}s"
    )
    glare_samples = sum(1 for legs in payload["glare_legs"] if legs)
    down_samples = sum(1 for legs in payload["degraded"] if legs)
    payload["footnote"] = (
        f"Recorded playback of seed {(summary or {}).get('seed', '?')}, day {day}. "
        f"{len(aligned)} phase changes, {len(payload['incidents'])} incidents, "
        f"{glare_samples} samples with low-sun glare, {down_samples} with a "
        f"detector dropout. Nothing on this page is computing control "
        f"decisions live; it is replaying a run produced by the controller in "
        f"app/core."
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--day", type=int, default=None)
    parser.add_argument("--out", default="dist/simulation.html")
    args = parser.parse_args()

    source = Path(args.data)
    detectors = load_rows(source / "detector_log.csv")
    if not detectors:
        raise SystemExit(
            f"no detector_log.csv under {source}/ — run scripts/generate_data.py first"
        )
    decisions = load_rows(source / "decisions.csv")
    incidents = load_rows(source / "incidents.csv")
    runs = load_rows(source / "runs.csv")
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    day = pick_day(runs, detectors, args.day)
    payload = build(day, detectors, decisions, incidents, runs, summary)

    template = TEMPLATE.read_text()
    marker = "/*__DATA__*/ null"
    if marker not in template:
        raise SystemExit(f"{TEMPLATE} is missing the {marker!r} injection point")
    html = template.replace(
        marker, json.dumps(payload, separators=(",", ":"))
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    size_kb = out.stat().st_size / 1024
    print(
        f"day {day}: {len(payload['t'])} samples, {len(payload['decisions'])} "
        f"decisions, {len(payload['incidents'])} incidents"
    )
    print(f"wrote {out} ({size_kb:.0f} kB, self-contained)")


if __name__ == "__main__":
    main()
