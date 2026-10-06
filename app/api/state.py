"""Read-only views of live controller state."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.phases import APPROACHES, CROSSWALKS
from app.services.runtime import get_runtime

router = APIRouter()


@router.get("/state", summary="Everything the dashboard needs for one frame")
def state() -> dict:
    return get_runtime().state()


@router.get("/series", summary="Rolling queue, flow and forecast history")
def series(limit: int = Query(240, ge=1, le=2000)) -> dict:
    runtime = get_runtime()
    return {"samples": runtime.series(limit), "approaches": list(APPROACHES)}


@router.get("/decisions", summary="Recent phase decisions with their reasoning")
def decisions(limit: int = Query(25, ge=1, le=60)) -> dict:
    return {"decisions": list(get_runtime().decision_feed)[:limit]}


@router.get("/incidents", summary="Open and recent incidents")
def incidents() -> dict:
    runtime = get_runtime()
    return {
        "open": [i.to_dict() for i in runtime.detector.open_incidents],
        "recent": list(runtime.incident_feed),
        "detector": runtime.detector.status(),
    }


@router.get("/safety", summary="Interlock findings; an empty list is expected")
def safety() -> dict:
    runtime = get_runtime()
    return {
        "statistics": runtime.interlock.statistics(),
        "events": list(runtime.safety_feed),
        "faulted": runtime.controller.faulted,
    }


@router.get("/sensor", summary="Detector backend health, staleness and UV reading")
def sensor() -> dict:
    """Why the controller is seeing what it is seeing.

    For the MQTT backend this is where a silent roadside station shows up,
    so it is a read-only route rather than something buried in logs.
    """
    runtime = get_runtime()
    frame = runtime.frame
    return {
        "health": runtime.sensor.health(),
        "environment": (
            frame.to_dict()["environment"] if frame is not None else None
        ),
        "degraded_approaches": list(frame.degraded_approaches) if frame else [],
        "speed": (
            {
                a: {
                    "mean": frame.approaches[a].mean_speed,
                    "p85": frame.approaches[a].speed_85th,
                    "limit": frame.approaches[a].speed_limit,
                    "speeding": frame.approaches[a].speeding,
                    "samples": frame.approaches[a].speed_samples,
                }
                for a in APPROACHES
            }
            if frame is not None
            else {}
        ),
    }


@router.get("/layout", summary="Static intersection geometry")
def layout() -> dict:
    from app.core.phases import CONFLICTS, PHASES

    return {
        "approaches": list(APPROACHES),
        "crosswalks": list(CROSSWALKS),
        "phases": {
            k: {
                "label": p.label,
                "vehicles": sorted(p.vehicles),
                "pedestrians": sorted(p.pedestrians),
            }
            for k, p in PHASES.items()
        },
        "conflicts": {k: sorted(v) for k, v in CONFLICTS.items()},
    }
