"""Simulation-backed sensor: wraps the closed-loop microsimulation."""

from __future__ import annotations

from typing import Optional

from app.core.controller import TimingPlan
from app.core.types import SensorFrame, SignalCommand
from app.sim.intersection import IntersectionSim, SimConfig

from .base import TrafficSensor


class SimulationSensor(TrafficSensor):
    name = "simulation"

    def __init__(self, config: Optional[SimConfig] = None,
                 timing: Optional[TimingPlan] = None) -> None:
        self.sim = IntersectionSim(config, timing)

    def read(self, dt: float, command: Optional[SignalCommand] = None) -> SensorFrame:
        return self.sim.step(dt, command)

    def health(self) -> dict:
        return {"sensor": self.name, "status": "ok", **self.sim.metrics()}
