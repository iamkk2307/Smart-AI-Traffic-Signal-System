"""Operator actions. Every route here changes control state and needs a token."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.phases import APPROACHES, CROSSWALKS
from app.services.runtime import get_runtime

from .deps import require_operator

router = APIRouter(dependencies=[Depends(require_operator)])


class PedCall(BaseModel):
    crosswalk: str = Field(description="One of cw_north, cw_south, cw_east, cw_west")


class EmergencyCall(BaseModel):
    approach: str
    kind: str = "ambulance"
    duration: float = Field(25.0, gt=0, le=120)


class WeatherUpdate(BaseModel):
    weather: str = Field(description="clear, cloudy, rain, fog, snow or thunderstorm")


class ClockUpdate(BaseModel):
    hour: float = Field(
        description="Local hour of day, 0-24. Simulation backend only.",
        ge=0,
        lt=24,
    )


class UvReading(BaseModel):
    uv_index: float = Field(ge=0, le=20)
    illuminance_lux: float | None = Field(None, ge=0)


class ModeUpdate(BaseModel):
    mode: str = Field(description="adaptive or fixed_time")


@router.post("/ped-call", summary="Place a pedestrian call")
def ped_call(body: PedCall) -> dict:
    if body.crosswalk not in CROSSWALKS:
        raise HTTPException(422, f"unknown crosswalk {body.crosswalk!r}")
    return get_runtime().place_ped_call(body.crosswalk)


@router.post("/emergency", summary="Raise an emergency-vehicle preemption request")
def emergency(body: EmergencyCall) -> dict:
    if body.approach not in APPROACHES:
        raise HTTPException(422, f"unknown approach {body.approach!r}")
    try:
        return get_runtime().inject_emergency(body.approach, body.kind, body.duration)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/weather", summary="Set conditions; adjusts clearance and discharge")
def weather(body: WeatherUpdate) -> dict:
    return get_runtime().set_weather(body.weather)


@router.post("/clock", summary="Move the simulated clock; changes the sun angle")
def clock(body: ClockUpdate) -> dict:
    try:
        return get_runtime().set_clock(body.hour)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/uv", summary="Push a UV/illuminance reading from roadside hardware")
def uv(body: UvReading) -> dict:
    """Accepts a hardware reading on backends that take one.

    The simulation models UV from solar geometry, so it has nothing to
    accept; saying so is better than pretending the value was stored.
    """
    runtime = get_runtime()
    environment = getattr(runtime.sensor, "environment", None)
    if not hasattr(environment, "submit"):
        raise HTTPException(
            409,
            "this sensor backend models UV from solar geometry and does not "
            "accept pushed readings",
        )
    environment.submit(body.uv_index, body.illuminance_lux)
    return {"uv_index": body.uv_index, "accepted": True}


@router.post("/mode", summary="Switch between adaptive and fixed-time control")
def mode(body: ModeUpdate) -> dict:
    try:
        return get_runtime().set_mode(body.mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/clear-fault", summary="Acknowledge a fault and restart the ring")
def clear_fault() -> dict:
    return get_runtime().clear_fault()
