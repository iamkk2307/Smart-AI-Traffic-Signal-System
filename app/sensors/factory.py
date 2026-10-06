"""Sensor selection."""

from __future__ import annotations

from typing import Optional

from app.core.controller import TimingPlan
from app.sim.intersection import SimConfig

from .base import TrafficSensor

MODES = ("simulation", "camera", "iot")


def create_sensor(
    mode: str = "simulation",
    sim_config: Optional[SimConfig] = None,
    timing: Optional[TimingPlan] = None,
    **kwargs,
) -> TrafficSensor:
    """Build the configured sensor backend.

    ``camera`` and ``iot`` carry optional third-party dependencies
    (ultralytics/opencv and paho-mqtt), so they are imported only when
    selected. A typo in the mode fails here, at start-up, rather than
    silently falling back to the simulation and presenting synthetic
    demand as live detector data.
    """
    if mode == "simulation":
        from .simulation import SimulationSensor

        return SimulationSensor(sim_config, timing)
    if mode == "camera":
        from .camera import CameraSensor

        return CameraSensor(**kwargs)
    if mode == "iot":
        from .iot import IoTSensor

        return IoTSensor(**kwargs)
    raise ValueError(f"unknown sensor mode {mode!r}; expected one of {MODES}")
