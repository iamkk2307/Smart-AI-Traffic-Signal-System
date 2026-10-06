"""Historical analytics read from the persisted log."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import IncidentRecord, ModelMetric, PhaseLog, TrafficLog, get_session

router = APIRouter()


@router.get("/summary", summary="Delay, throughput and decision mix over a window")
def summary(
    hours: int = Query(6, ge=1, le=168),
    session: Session = Depends(get_session),
) -> dict:
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

    logs = session.execute(
        select(
            func.count(TrafficLog.id),
            func.avg(TrafficLog.total_queue),
            func.max(TrafficLog.total_queue),
            func.avg(TrafficLog.mean_delay),
        ).where(TrafficLog.timestamp >= since)
    ).one()

    triggers = session.execute(
        select(PhaseLog.trigger, func.count(PhaseLog.id))
        .where(PhaseLog.timestamp >= since)
        .group_by(PhaseLog.trigger)
        .order_by(func.count(PhaseLog.id).desc())
    ).all()

    green = session.execute(
        select(PhaseLog.to_phase, func.avg(PhaseLog.served_green_time))
        .where(PhaseLog.timestamp >= since)
        .group_by(PhaseLog.to_phase)
    ).all()

    incidents = session.execute(
        select(IncidentRecord.kind, func.count(IncidentRecord.id))
        .where(IncidentRecord.detected_at >= since)
        .group_by(IncidentRecord.kind)
    ).all()

    return {
        "window_hours": hours,
        "samples": logs[0] or 0,
        "mean_total_queue": round(logs[1] or 0.0, 2),
        "peak_total_queue": logs[2] or 0,
        "mean_delay_s": round(logs[3] or 0.0, 2),
        "phase_triggers": {t: c for t, c in triggers},
        "mean_green_by_phase": {p: round(v or 0.0, 1) for p, v in green},
        "incidents_by_kind": {k: c for k, c in incidents},
    }


@router.get("/phase-log", summary="The decision audit trail")
def phase_log(
    limit: int = Query(50, ge=1, le=500),
    session: Session = Depends(get_session),
) -> dict:
    rows = session.execute(
        select(PhaseLog).order_by(PhaseLog.timestamp.desc()).limit(limit)
    ).scalars().all()
    return {
        "entries": [
            {
                "timestamp": r.timestamp.isoformat(),
                "cycle": r.cycle,
                "from_phase": r.from_phase,
                "to_phase": r.to_phase,
                "trigger": r.trigger,
                "reason": r.reason,
                "served_green_time": r.served_green_time,
                "pressure": r.pressure,
            }
            for r in rows
        ]
    }


@router.get("/model-metrics", summary="Every forecaster run against its baseline")
def model_metrics(
    limit: int = Query(40, ge=1, le=400),
    session: Session = Depends(get_session),
) -> dict:
    rows = session.execute(
        select(ModelMetric).order_by(ModelMetric.trained_at.desc()).limit(limit)
    ).scalars().all()
    return {
        "runs": [
            {
                "trained_at": r.trained_at.isoformat(),
                "approach": r.approach,
                "estimator": r.estimator,
                "rows": r.rows,
                "model_mae": round(r.model_mae, 3),
                "baseline_mae": round(r.baseline_mae, 3),
                "promoted": r.promoted,
            }
            for r in rows
        ]
    }
