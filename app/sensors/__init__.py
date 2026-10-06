"""Sensor backends.

Exports are resolved lazily. ``app.sim.intersection`` needs the UV/light
model from ``app.sensors.environment``, while ``app.sensors.simulation``
wraps the simulation — importing both eagerly here would make that a cycle.
Deferring the lookup keeps ``from app.sensors import create_sensor`` working
without ordering the two modules against each other.
"""

from typing import Any

__all__ = [
    "TrafficSensor",
    "create_sensor",
    "SimulationSensor",
    "CameraSensor",
    "IoTSensor",
    "ModelledUVSensor",
    "PushedUVSensor",
    "ApproachSpeedTracker",
]

_LOCATIONS = {
    "TrafficSensor": "base",
    "create_sensor": "factory",
    "SimulationSensor": "simulation",
    "CameraSensor": "camera",
    "IoTSensor": "iot",
    "ModelledUVSensor": "environment",
    "PushedUVSensor": "environment",
    "ApproachSpeedTracker": "speed",
}


def __getattr__(name: str) -> Any:
    module = _LOCATIONS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
