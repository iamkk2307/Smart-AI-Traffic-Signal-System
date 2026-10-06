"""Forecaster and anomaly-model operations."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.phases import APPROACHES
from app.services.runtime import get_runtime

from .deps import require_operator

router = APIRouter()


@router.get("/status", summary="What each approach is currently forecasting with")
def status() -> dict:
    runtime = get_runtime()
    return {
        "forecaster": runtime.forecaster.status(),
        "anomaly": runtime.detector.status(),
    }


@router.get("/forecast", summary="Next-interval arrivals per approach")
def forecast() -> dict:
    runtime = get_runtime()
    return {
        "unit": "vehicles_per_minute",
        "horizon_intervals": runtime.forecaster.horizon,
        "values": {a: round(runtime.forecaster.predict(a), 2) for a in APPROACHES},
        "serving": runtime.forecaster.status()["serving"],
    }


@router.post(
    "/train",
    summary="Retrain now; models are promoted only if they beat the baseline",
    dependencies=[Depends(require_operator)],
)
def train() -> dict:
    return get_runtime().train_now()
