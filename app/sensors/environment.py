"""
Roadside UV / ambient-light sensor.

Why a UV sensor is here at all
------------------------------
A UV index on a dashboard is decoration. What makes it a control input is
the mechanism it stands in for: UV irradiance is dominated by the *direct*
solar beam, so a high index means the sun is unobscured, and an unobscured
sun at low elevation is the single most common optical failure mode at a
signalised junction. It washes out the signal head for drivers facing into
it and it saturates the camera sensor that the detection model depends on.

So this module does two things and refuses to do a third:

* it measures UV index and illuminance (or models them from solar geometry
  when no hardware reading is available), and
* it decides whether low-angle glare is present *and which approaches it
  affects*, from the sun's azimuth against each approach's heading.

What it does not do is multiply every approach's pressure score by a
"conditions factor". A factor applied equally to all candidates cannot
change an argmax; it would be dead weight presented as intelligence. Glare
is applied where it has a physical effect: longer change intervals, a
higher minimum green, a derated discharge rate, and a lowered trust in
camera detections on the affected legs.

Backends
--------
``ModelledUVSensor``  clear-sky model from latitude, longitude and the clock.
``PushedUVSensor``    latest reading pushed in from hardware (see ``iot.py``),
                      falling back to the model and flagging ``stale`` when
                      the feed goes quiet.
"""

from __future__ import annotations

import abc
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.core.phases import APPROACHES
from app.core.types import EnvironmentReading

#: Compass heading a driver on each approach is travelling towards, in
#: degrees clockwise from north. Approaches are named by the leg traffic
#: arrives *from* (see ``core.phases``), so the north approach travels
#: southbound and its drivers face 180°.
APPROACH_HEADING: Dict[str, float] = {
    "north": 180.0,
    "south": 0.0,
    "east": 270.0,
    "west": 90.0,
}

#: Direct-beam attenuation by sky condition. UV index is measured, so these
#: are only used by the modelled backend.
WEATHER_BEAM_FACTOR: Dict[str, float] = {
    "clear": 1.0,
    "cloudy": 0.42,
    "rain": 0.25,
    "fog": 0.18,
    "snow": 0.30,
    "thunderstorm": 0.20,
}


def solar_position(when: datetime, latitude: float, longitude: float):
    """Return ``(elevation, azimuth)`` in degrees.

    NOAA's low-precision algorithm. Accurate to a fraction of a degree,
    which is far better than needed: the glare threshold is a band several
    degrees wide and the sun crosses it over several minutes.
    """
    when = when.astimezone(timezone.utc)
    day_of_year = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60.0 + when.second / 3600.0

    gamma = 2 * math.pi / 365.0 * (day_of_year - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    time_offset = eqtime + 4.0 * longitude
    true_solar_minutes = hour * 60.0 + time_offset
    hour_angle = math.radians(true_solar_minutes / 4.0 - 180.0)

    lat = math.radians(latitude)
    cos_zenith = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(
        decl
    ) * math.cos(hour_angle)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    elevation = 90.0 - math.degrees(zenith)

    # Azimuth from the south, positive towards the west, then rotated to
    # compass bearing. atan2 is used rather than acos so the morning and
    # afternoon branches do not need a sign fix-up, which is the usual
    # source of a solar azimuth that is correct for half the day.
    azimuth_from_south = math.atan2(
        math.sin(hour_angle),
        math.cos(hour_angle) * math.sin(lat) - math.tan(decl) * math.cos(lat),
    )
    azimuth = (math.degrees(azimuth_from_south) + 180.0) % 360.0
    return elevation, azimuth


def clear_sky_uv_index(elevation: float, altitude_m: float = 0.0) -> float:
    """Approximate clear-sky UV index for a solar elevation.

    Erythemal irradiance scales roughly with the cosine of the zenith angle
    raised to a power near 2.5 once atmospheric path length is accounted
    for. Below the horizon it is zero, not negative.
    """
    if elevation <= 0.0:
        return 0.0
    mu = math.sin(math.radians(elevation))
    index = 12.5 * (mu ** 2.5) * (1.0 + 0.00008 * altitude_m)
    return round(max(0.0, index), 2)


def clear_sky_illuminance(elevation: float) -> float:
    """Approximate horizontal illuminance in lux."""
    if elevation <= -0.833:
        return 0.0
    if elevation <= 0.0:
        return round(400.0 * (elevation + 0.833) / 0.833, 1)
    return round(128000.0 * math.sin(math.radians(elevation)) ** 1.15, 1)


def angular_separation(a: float, b: float) -> float:
    """Smallest absolute difference between two compass bearings."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


class UVLightSensor(abc.ABC):
    """A source of :class:`EnvironmentReading` observations."""

    def __init__(
        self,
        latitude: float = 17.385,
        longitude: float = 78.4867,
        altitude_m: float = 0.0,
        glare_elevation_max: float = 15.0,
        glare_beam_fraction: float = 0.55,
        glare_lux_min: float = 1200.0,
        glare_azimuth_tolerance: float = 35.0,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.altitude_m = altitude_m
        self.glare_elevation_max = glare_elevation_max
        self.glare_beam_fraction = glare_beam_fraction
        self.glare_lux_min = glare_lux_min
        self.glare_azimuth_tolerance = glare_azimuth_tolerance

    @abc.abstractmethod
    def read(self, when: Optional[datetime] = None, weather: str = "clear") -> EnvironmentReading:
        ...

    # ───────────────────────── shared logic ─────────────────────────

    def _affected(
        self, elevation: float, azimuth: float, uv_index: float, lux: float
    ) -> List[str]:
        """Approaches whose drivers and camera are looking into the sun.

        Four conditions, all necessary.

        1. The sun is up but low.
        2. There is enough visible light to dazzle, measured in lux. This
           is the magnitude test, and it has to be the visible band: UV is
           attenuated far more strongly by the long atmospheric path at low
           sun, so clear-sky UV at 2° elevation is essentially zero while
           the sun is still blinding. A UV magnitude threshold would
           therefore switch glare *off* exactly at sunrise and sunset.
        3. The direct beam is present rather than diffused by cloud. This
           is what UV is good for, as a *ratio* to the clear-sky value at
           the same elevation: the ratio stays meaningful at low sun even
           though both quantities are small, whereas any absolute index
           threshold does not.
        4. The driver's heading is within tolerance of the solar azimuth.
        """
        if not (0.0 < elevation <= self.glare_elevation_max):
            return []
        if lux < self.glare_lux_min:
            return []
        expected = clear_sky_uv_index(elevation, self.altitude_m)
        if expected > 1e-9 and uv_index < self.glare_beam_fraction * expected:
            return []
        return [
            approach
            for approach in APPROACHES
            if angular_separation(APPROACH_HEADING[approach], azimuth)
            <= self.glare_azimuth_tolerance
        ]

    def _build(
        self,
        when: datetime,
        uv_index: float,
        lux: float,
        source: str,
        stale: bool,
    ) -> EnvironmentReading:
        elevation, azimuth = solar_position(when, self.latitude, self.longitude)
        affected = self._affected(elevation, azimuth, uv_index, lux)
        return EnvironmentReading(
            uv_index=round(uv_index, 2),
            solar_elevation=round(elevation, 2),
            solar_azimuth=round(azimuth, 1),
            illuminance_lux=round(lux, 1),
            glare=bool(affected),
            approaches_affected=affected,
            source=source,
            stale=stale,
        )


class ModelledUVSensor(UVLightSensor):
    """Clear-sky model, attenuated by the reported sky condition.

    Used when no hardware sensor is fitted, and as the fallback when a
    hardware feed goes stale. It is labelled ``source="model"`` on every
    reading so nothing downstream can mistake it for a measurement.
    """

    def read(self, when: Optional[datetime] = None, weather: str = "clear") -> EnvironmentReading:
        when = when or datetime.now(timezone.utc)
        elevation, _ = solar_position(when, self.latitude, self.longitude)
        beam = WEATHER_BEAM_FACTOR.get(weather, 1.0)
        uv = clear_sky_uv_index(elevation, self.altitude_m) * beam
        lux = clear_sky_illuminance(elevation) * (0.25 + 0.75 * beam)
        return self._build(when, uv, lux, source="model", stale=False)


class PushedUVSensor(UVLightSensor):
    """Hardware readings pushed in by a transport (MQTT, serial, HTTP).

    ``submit`` is called by the transport; ``read`` is called by the control
    loop. If nothing has arrived within ``max_age`` the reading falls back
    to the model and is flagged stale, because a frozen last-known UV value
    is worse than an honest model — it would hold a glare decision long
    after the sun had moved.
    """

    def __init__(self, *args, max_age: float = 120.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_age = max_age
        self._model = ModelledUVSensor(
            latitude=self.latitude,
            longitude=self.longitude,
            altitude_m=self.altitude_m,
            glare_elevation_max=self.glare_elevation_max,
            glare_beam_fraction=self.glare_beam_fraction,
            glare_lux_min=self.glare_lux_min,
            glare_azimuth_tolerance=self.glare_azimuth_tolerance,
        )
        self._uv: Optional[float] = None
        self._lux: Optional[float] = None
        self._at: Optional[datetime] = None

    def submit(
        self,
        uv_index: float,
        illuminance_lux: Optional[float] = None,
        when: Optional[datetime] = None,
    ) -> None:
        self._uv = max(0.0, float(uv_index))
        if illuminance_lux is not None:
            self._lux = max(0.0, float(illuminance_lux))
        self._at = when or datetime.now(timezone.utc)

    def age(self, when: Optional[datetime] = None) -> Optional[float]:
        if self._at is None:
            return None
        when = when or datetime.now(timezone.utc)
        return (when - self._at).total_seconds()

    def read(self, when: Optional[datetime] = None, weather: str = "clear") -> EnvironmentReading:
        when = when or datetime.now(timezone.utc)
        age = self.age(when)
        if self._uv is None or age is None or age > self.max_age:
            reading = self._model.read(when, weather)
            reading.stale = True
            return reading
        elevation, _ = solar_position(when, self.latitude, self.longitude)
        lux = self._lux if self._lux is not None else clear_sky_illuminance(elevation)
        return self._build(when, self._uv, lux, source="sensor", stale=False)
