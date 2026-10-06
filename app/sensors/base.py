"""Sensor abstraction. The controller never learns where a frame came from."""

from __future__ import annotations

import abc
from typing import Optional

from app.core.types import SensorFrame, SignalCommand


class TrafficSensor(abc.ABC):
    """A source of :class:`SensorFrame` observations.

    ``read`` receives the current signal command because real detectors are
    affected by it — a stop-line loop registers departures only on green —
    and because the simulation backend is closed-loop. Implementations that
    do not need it ignore the argument.
    """

    name: str = "sensor"

    @abc.abstractmethod
    def read(self, dt: float, command: Optional[SignalCommand] = None) -> SensorFrame:
        ...

    def health(self) -> dict:
        return {"sensor": self.name, "status": "ok"}

    def close(self) -> None:
        pass
