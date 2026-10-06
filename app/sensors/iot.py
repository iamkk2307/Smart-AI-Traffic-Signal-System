"""
IoT detector backend.

Roadside hardware — inductive loops, radar heads, pushbutton controllers,
emergency-vehicle beacons and the UV/light sensor — publishes to an MQTT
broker. This sensor subscribes, keeps the most recent payload per topic, and
snapshots them into a :class:`SensorFrame` when the control loop asks for
one. The loop is never blocked on the network: the broker callback writes
into a cache under a lock and ``read`` only ever reads that cache.

Staleness is the whole problem with field hardware, so it is handled
explicitly rather than left to chance. A detector that has stopped
publishing looks exactly like a detector reporting an empty road, and those
two must not be treated the same way. When a station exceeds ``max_age`` its
last known values are retained, its ``degraded`` flag is set, and the
approach is listed in ``SensorFrame.degraded_approaches``; the controller
responds by placing the approach on constant recall and refusing to gap out
on it. The failure mode is a longer green for a street that may be empty,
which is merely inefficient. The alternative failure mode is never serving a
street that is full.

Topics, under the configured prefix (default ``traffic/j1``)::

    <prefix>/approach/<north|south|east|west>/detector
        {"queue": 7, "flow": 18.4, "occupancy": 0.42, "mean_speed": 37.1,
         "speed_85th": 44.0, "speeding": 1, "gap": 2.1, "max_wait": 31.0,
         "speed_limit": 50}
    <prefix>/crosswalk/<cw_north|cw_south|cw_east|cw_west>/call
        {"call": true}
    <prefix>/emergency
        {"approach": "east", "kind": "ambulance", "eta": 12.0}
    <prefix>/environment/uv
        {"uv_index": 7.4, "illuminance_lux": 94000}
    <prefix>/weather
        {"condition": "rain", "visibility_km": 2.0, "surface_wet": true}

Every field is optional; anything absent keeps its default. Payloads arrive
from field devices, so they are parsed defensively and a malformed message
is counted and dropped rather than allowed to raise inside a broker
callback.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from app.core.phases import APPROACHES, CROSSWALKS
from app.core.types import ApproachReading, SensorFrame, SignalCommand

from .base import TrafficSensor
from .environment import PushedUVSensor

logger = logging.getLogger(__name__)

WEATHER_CONDITIONS = {"clear", "cloudy", "rain", "fog", "snow", "thunderstorm"}
EMERGENCY_KINDS = {"ambulance", "fire", "police"}


def _num(payload: dict, key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out and abs(out) != float("inf") else default


class IoTSensor(TrafficSensor):
    """Detector frames assembled from MQTT messages.

    The transport is optional. Constructed without ``broker_host`` the
    sensor is a pure cache and messages are handed to :meth:`submit`
    directly — which is how the tests drive it, and how a different
    transport (serial, HTTP webhook, NTCIP gateway) can be bolted on
    without touching the parsing or the staleness logic.
    """

    name = "iot"

    def __init__(
        self,
        broker_host: Optional[str] = None,
        broker_port: int = 1883,
        prefix: str = "traffic/j1",
        username: Optional[str] = None,
        password: Optional[str] = None,
        max_age: float = 5.0,
        emergency_ttl: float = 6.0,
        ped_ttl: float = 120.0,
        environment: Optional[PushedUVSensor] = None,
    ) -> None:
        self.prefix = prefix.rstrip("/")
        self.max_age = max_age
        self.emergency_ttl = emergency_ttl
        self.ped_ttl = ped_ttl
        self.environment = environment or PushedUVSensor(max_age=max(max_age, 120.0))

        self._lock = threading.Lock()
        self._approach: Dict[str, Tuple[ApproachReading, datetime]] = {}
        self._ped: Dict[str, datetime] = {}
        self._emergency: Optional[Tuple[dict, datetime]] = None
        self._weather: Tuple[str, float, bool] = ("clear", 10.0, False)
        self._messages = 0
        self._malformed = 0
        self._client = None

        if broker_host:
            self._connect(broker_host, broker_port, username, password)

    # ───────────────────────── transport ─────────────────────────

    def _connect(
        self,
        host: str,
        port: int,
        username: Optional[str],
        password: Optional[str],
    ) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("iot backend needs 'paho-mqtt'") from exc

        client = mqtt.Client()
        if username:
            client.username_pw_set(username, password or "")

        def on_connect(c, _userdata, _flags, rc):  # pragma: no cover - needs broker
            if rc != 0:
                logger.error("mqtt connect refused, rc=%s", rc)
                return
            c.subscribe(f"{self.prefix}/#", qos=1)
            logger.info("subscribed to %s/#", self.prefix)

        def on_message(_c, _userdata, message):  # pragma: no cover - needs broker
            try:
                self.submit(message.topic, message.payload.decode("utf-8"))
            except Exception:
                # A raising callback kills the network thread and takes the
                # whole detector feed down with it.
                with self._lock:
                    self._malformed += 1
                logger.warning("dropped mqtt message on %s", message.topic)

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect_async(host, port, keepalive=30)
        client.loop_start()
        self._client = client

    # ───────────────────────── ingestion ─────────────────────────

    def submit(
        self, topic: str, payload: Any, when: Optional[datetime] = None
    ) -> bool:
        """Accept one message. Returns False if it was unusable."""
        when = when or datetime.now(timezone.utc)
        if isinstance(payload, (str, bytes)):
            try:
                payload = json.loads(payload)
            except (ValueError, TypeError):
                with self._lock:
                    self._malformed += 1
                return False
        if not isinstance(payload, dict):
            with self._lock:
                self._malformed += 1
            return False

        if not topic.startswith(self.prefix):
            with self._lock:
                self._malformed += 1
            return False
        parts = topic[len(self.prefix):].strip("/").split("/")

        with self._lock:
            self._messages += 1

        if parts[:1] == ["approach"] and len(parts) >= 2:
            return self._on_detector(parts[1], payload, when)
        if parts[:1] == ["crosswalk"] and len(parts) >= 2:
            return self._on_ped(parts[1], payload, when)
        if parts[:1] == ["emergency"]:
            return self._on_emergency(payload, when)
        if parts[:2] == ["environment", "uv"]:
            self.environment.submit(
                uv_index=_num(payload, "uv_index", 0.0),
                illuminance_lux=payload.get("illuminance_lux"),
                when=when,
            )
            return True
        if parts[:1] == ["weather"]:
            return self._on_weather(payload)

        with self._lock:
            self._malformed += 1
        return False

    def _on_detector(self, approach: str, payload: dict, when: datetime) -> bool:
        if approach not in APPROACHES:
            with self._lock:
                self._malformed += 1
            return False
        reading = ApproachReading(
            queue=max(0, int(_num(payload, "queue", 0))),
            flow=max(0.0, _num(payload, "flow", 0.0)),
            occupancy=min(1.0, max(0.0, _num(payload, "occupancy", 0.0))),
            mean_speed=max(0.0, _num(payload, "mean_speed", 0.0)),
            speed_85th=max(0.0, _num(payload, "speed_85th", 0.0)),
            speed_limit=max(1.0, _num(payload, "speed_limit", 50.0)),
            speeding=max(0, int(_num(payload, "speeding", 0))),
            speed_samples=max(0, int(_num(payload, "speed_samples", 0))),
            gap=max(0.0, _num(payload, "gap", 99.0)),
            max_wait=max(0.0, _num(payload, "max_wait", 0.0)),
        )
        with self._lock:
            self._approach[approach] = (reading, when)
        return True

    def _on_ped(self, crosswalk: str, payload: dict, when: datetime) -> bool:
        if crosswalk not in CROSSWALKS:
            with self._lock:
                self._malformed += 1
            return False
        with self._lock:
            if payload.get("call", True):
                self._ped[crosswalk] = when
            else:
                self._ped.pop(crosswalk, None)
        return True

    def _on_emergency(self, payload: dict, when: datetime) -> bool:
        approach = payload.get("approach")
        if approach is not None and approach not in APPROACHES:
            with self._lock:
                self._malformed += 1
            return False
        with self._lock:
            if approach is None or payload.get("clear"):
                self._emergency = None
            else:
                kind = payload.get("kind", "ambulance")
                self._emergency = (
                    {
                        "approach": approach,
                        "kind": kind if kind in EMERGENCY_KINDS else "ambulance",
                        "eta": _num(payload, "eta", 0.0),
                    },
                    when,
                )
        return True

    def _on_weather(self, payload: dict) -> bool:
        condition = str(payload.get("condition", "clear"))
        if condition not in WEATHER_CONDITIONS:
            with self._lock:
                self._malformed += 1
            return False
        with self._lock:
            self._weather = (
                condition,
                _num(payload, "visibility_km", 10.0),
                bool(payload.get("surface_wet", condition in ("rain", "snow"))),
            )
        return True

    # ───────────────────────── reading ─────────────────────────

    def read(self, dt: float, command: Optional[SignalCommand] = None) -> SensorFrame:
        now = datetime.now(timezone.utc)
        with self._lock:
            cached = dict(self._approach)
            ped = dict(self._ped)
            emergency = self._emergency
            weather, visibility, wet = self._weather

        approaches: Dict[str, ApproachReading] = {}
        degraded = []
        for approach in APPROACHES:
            entry = cached.get(approach)
            if entry is None:
                # Never published. Unknown, not empty.
                approaches[approach] = ApproachReading(degraded=True)
                degraded.append(approach)
                continue
            reading, at = entry
            age = (now - at).total_seconds()
            if age > self.max_age:
                stale = ApproachReading(**{**reading.__dict__})
                stale.degraded = True
                approaches[approach] = stale
                degraded.append(approach)
            else:
                approaches[approach] = reading

        emergency_approach = emergency_kind = None
        emergency_eta = None
        if emergency is not None:
            payload, at = emergency
            if (now - at).total_seconds() <= self.emergency_ttl:
                emergency_approach = payload["approach"]
                emergency_kind = payload["kind"]
                emergency_eta = payload["eta"]

        return SensorFrame(
            timestamp=now,
            approaches=approaches,
            pedestrian_calls={
                c: (c in ped and (now - ped[c]).total_seconds() <= self.ped_ttl)
                for c in CROSSWALKS
            },
            emergency_approach=emergency_approach,
            emergency_kind=emergency_kind,
            emergency_eta=emergency_eta,
            weather=weather,
            visibility_km=visibility,
            surface_wet=wet,
            environment=self.environment.read(now, weather),
            degraded_approaches=degraded,
        )

    def health(self) -> dict:
        now = datetime.now(timezone.utc)
        with self._lock:
            ages = {
                a: round((now - at).total_seconds(), 1)
                for a, (_, at) in self._approach.items()
            }
            messages, malformed = self._messages, self._malformed
        silent = [a for a in APPROACHES if a not in ages]
        stale = [a for a, age in ages.items() if age > self.max_age]
        return {
            "sensor": self.name,
            "status": "ok" if not (silent or stale) else "degraded",
            "connected": self._client is not None,
            "prefix": self.prefix,
            "messages": messages,
            "malformed": malformed,
            "detector_age_s": ages,
            "never_reported": silent,
            "stale": stale,
            "uv_age_s": (
                None
                if self.environment.age(now) is None
                else round(self.environment.age(now), 1)
            ),
        }

    def close(self) -> None:
        if self._client is not None:  # pragma: no cover - needs broker
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                logger.warning("failed to close mqtt client", exc_info=True)
