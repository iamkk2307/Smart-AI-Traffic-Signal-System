"""
UV / ambient-light sensor tests.

Two things are being checked. The solar geometry has to be right, because
every glare decision rests on it — an azimuth that is correct only in the
morning would put the sun on the wrong approach all afternoon. And glare has
to be *selective*: it must fire on the legs facing the sun, stay off when
cloud removes the direct beam, and stay off at midday.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.sensors.environment import (
    APPROACH_HEADING,
    ModelledUVSensor,
    PushedUVSensor,
    angular_separation,
    clear_sky_illuminance,
    clear_sky_uv_index,
    solar_position,
)

LAT, LON = 17.385, 78.4867  # Hyderabad
OFFSET = timedelta(hours=5.5)


def local(hour: float, day: int = 6, month: int = 10) -> datetime:
    """A local wall-clock hour at the site, as a UTC instant."""
    return datetime(
        2026, month, day, int(hour), int((hour % 1) * 60), tzinfo=timezone.utc
    ) - OFFSET


# ───────────────────────── geometry ─────────────────────────


def test_sun_is_south_at_local_noon():
    elevation, azimuth = solar_position(local(12.1), LAT, LON)
    assert elevation > 60.0
    assert angular_separation(azimuth, 180.0) < 15.0


def test_azimuth_swings_east_to_west_across_the_day():
    morning = solar_position(local(7.0), LAT, LON)[1]
    noon = solar_position(local(12.1), LAT, LON)[1]
    evening = solar_position(local(17.0), LAT, LON)[1]
    assert morning < noon < evening
    assert angular_separation(morning, 90.0) < 35.0
    assert angular_separation(evening, 270.0) < 35.0


def test_sun_is_below_the_horizon_at_night():
    assert solar_position(local(0.5), LAT, LON)[0] < 0.0
    assert solar_position(local(23.0), LAT, LON)[0] < 0.0


def test_uv_and_illuminance_are_zero_below_the_horizon():
    assert clear_sky_uv_index(-5.0) == 0.0
    assert clear_sky_illuminance(-5.0) == 0.0


def test_uv_rises_monotonically_with_elevation():
    values = [clear_sky_uv_index(e) for e in (5, 15, 30, 50, 70, 89)]
    assert values == sorted(values)
    assert values[-1] > 10.0


def test_angular_separation_wraps():
    assert angular_separation(350.0, 10.0) == pytest.approx(20.0)
    assert angular_separation(10.0, 350.0) == pytest.approx(20.0)


# ───────────────────────── glare ─────────────────────────


def test_no_glare_at_midday():
    """The sun is high; nobody is driving into it."""
    reading = ModelledUVSensor(LAT, LON).read(local(12.1), "clear")
    assert reading.uv_index > 8.0
    assert reading.glare is False
    assert reading.approaches_affected == []


def test_morning_glare_hits_the_eastbound_approach_only():
    sensor = ModelledUVSensor(LAT, LON)
    reading = sensor.read(local(6.6), "clear")
    assert reading.glare is True
    # The ``west`` approach is the one traffic arrives from the west on,
    # travelling east — into the rising sun.
    assert reading.approaches_affected == ["west"]
    assert APPROACH_HEADING["west"] == 90.0


def test_evening_glare_hits_the_westbound_approach_only():
    reading = ModelledUVSensor(LAT, LON).read(local(17.6), "clear")
    assert reading.glare is True
    assert reading.approaches_affected == ["east"]


def test_cloud_removes_glare_at_the_same_sun_position():
    """Same geometry, same lux threshold crossing — only the beam differs.

    This is the test that justifies carrying a UV sensor rather than just a
    clock and a compass. Without the beam-fraction check, every clear-sky
    *time* would be reported as glare regardless of the sky.
    """
    sensor = ModelledUVSensor(LAT, LON)
    clear = sensor.read(local(6.6), "clear")
    overcast = sensor.read(local(6.6), "fog")
    assert clear.glare is True
    assert overcast.glare is False
    assert overcast.solar_elevation == clear.solar_elevation


def test_glare_is_not_reported_at_night():
    reading = ModelledUVSensor(LAT, LON).read(local(2.0), "clear")
    assert reading.glare is False
    assert reading.illuminance_lux == 0.0


def test_a_uv_magnitude_threshold_would_have_missed_sunrise():
    """Documents why the magnitude test is lux and not UV index.

    Clear-sky UV at a couple of degrees of elevation is near zero even
    though the sun is blinding, so any absolute UV threshold high enough to
    mean "strong sun" would switch glare off exactly when it matters.
    """
    reading = ModelledUVSensor(LAT, LON).read(local(6.3), "clear")
    assert reading.glare is True
    assert reading.uv_index < 1.0
    assert reading.illuminance_lux > 1200.0


def test_model_readings_are_labelled_as_model():
    assert ModelledUVSensor(LAT, LON).read(local(9.0)).source == "model"


# ───────────────────── pushed hardware readings ─────────────────────


def test_pushed_reading_is_used_and_labelled_sensor():
    sensor = PushedUVSensor(LAT, LON, max_age=60.0)
    when = local(6.6)
    sensor.submit(uv_index=0.4, illuminance_lux=20000.0, when=when)
    reading = sensor.read(when, "clear")
    assert reading.source == "sensor"
    assert reading.stale is False
    assert reading.uv_index == pytest.approx(0.4)
    assert reading.glare is True


def test_hardware_reporting_an_obscured_sky_overrides_the_clear_model():
    """The point of real hardware: the model cannot see a dust storm."""
    sensor = PushedUVSensor(LAT, LON, max_age=60.0)
    when = local(6.6)
    sensor.submit(uv_index=0.001, illuminance_lux=900.0, when=when)
    assert sensor.read(when, "clear").glare is False


def test_stale_feed_falls_back_to_the_model_and_says_so():
    sensor = PushedUVSensor(LAT, LON, max_age=30.0)
    when = local(12.1)
    sensor.submit(uv_index=9.0, when=when)
    fresh = sensor.read(when, "clear")
    assert fresh.stale is False

    later = when + timedelta(minutes=5)
    stale = sensor.read(later, "clear")
    assert stale.stale is True
    assert stale.source == "model"
    assert sensor.age(later) == pytest.approx(300.0, abs=1.0)


def test_never_pushed_sensor_is_stale_from_the_start():
    sensor = PushedUVSensor(LAT, LON)
    reading = sensor.read(local(12.1), "clear")
    assert reading.stale is True
    assert sensor.age() is None
