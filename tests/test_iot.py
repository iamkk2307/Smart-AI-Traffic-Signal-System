"""
IoT backend tests.

The interesting claims are about failure, not about the happy path. Field
hardware goes quiet, sends rubbish, and reports an emergency that has
already passed. A detector that has stopped publishing must not read as an
empty road, and the controller must respond to that by recalling the
approach rather than skipping it.

No broker is involved: messages are handed to ``submit`` directly, which is
also how a non-MQTT transport would feed the same cache.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.core.controller import AdaptiveController, TimingPlan
from app.core.phases import APPROACHES, Interval
from app.sensors.iot import IoTSensor

PREFIX = "traffic/test"


def sensor(**kwargs) -> IoTSensor:
    return IoTSensor(prefix=PREFIX, **kwargs)


def detector(approach: str, **fields) -> str:
    return f"{PREFIX}/approach/{approach}"


def feed_all(s: IoTSensor, when=None, **fields) -> None:
    for approach in APPROACHES:
        s.submit(detector(approach), {"queue": 4, "occupancy": 0.3, **fields}, when)


# ───────────────────────── ingestion ─────────────────────────


def test_detector_payload_reaches_the_frame():
    s = sensor()
    s.submit(
        detector("north"),
        {
            "queue": 9,
            "flow": 18.4,
            "occupancy": 0.42,
            "mean_speed": 37.1,
            "speed_85th": 44.0,
            "speeding": 2,
            "speed_samples": 11,
            "gap": 2.1,
            "max_wait": 31.0,
            "speed_limit": 60,
        },
    )
    reading = s.read(1.0).approaches["north"]
    assert reading.queue == 9
    assert reading.mean_speed == pytest.approx(37.1)
    assert reading.speed_85th == pytest.approx(44.0)
    assert reading.speeding == 2
    assert reading.speed_limit == pytest.approx(60.0)
    assert reading.degraded is False


def test_json_string_payloads_are_accepted():
    s = sensor()
    assert s.submit(detector("east"), '{"queue": 3}') is True
    assert s.read(1.0).approaches["east"].queue == 3


def test_malformed_payloads_are_counted_and_dropped():
    s = sensor()
    assert s.submit(detector("north"), "not json at all") is False
    assert s.submit(detector("north"), "[1, 2, 3]") is False
    assert s.submit(detector("nowhere"), {"queue": 1}) is False
    assert s.submit("other/system/topic", {"queue": 1}) is False
    assert s.submit(f"{PREFIX}/unknown/thing", {"queue": 1}) is False
    assert s.health()["malformed"] == 5
    # And none of it reached the frame.
    assert s.read(1.0).approaches["north"].queue == 0


def test_out_of_range_values_are_clamped_not_trusted():
    s = sensor()
    s.submit(
        detector("west"),
        {"queue": -5, "occupancy": 4.2, "mean_speed": -10, "gap": -1},
    )
    reading = s.read(1.0).approaches["west"]
    assert reading.queue == 0
    assert reading.occupancy == 1.0
    assert reading.mean_speed == 0.0
    assert reading.gap == 0.0


def test_pedestrian_call_latches_and_can_be_cleared():
    s = sensor()
    s.submit(f"{PREFIX}/crosswalk/cw_north", {"call": True})
    assert s.read(1.0).pedestrian_calls["cw_north"] is True
    s.submit(f"{PREFIX}/crosswalk/cw_north", {"call": False})
    assert s.read(1.0).pedestrian_calls["cw_north"] is False


def test_weather_payload_is_validated_against_known_conditions():
    s = sensor()
    assert s.submit(f"{PREFIX}/weather", {"condition": "rain"}) is True
    frame = s.read(1.0)
    assert frame.weather == "rain"
    assert frame.surface_wet is True
    assert s.submit(f"{PREFIX}/weather", {"condition": "locusts"}) is False
    assert s.read(1.0).weather == "rain"


def test_uv_payload_feeds_the_environment_reading():
    s = sensor()
    s.submit(f"{PREFIX}/environment/uv", {"uv_index": 7.4, "illuminance_lux": 94000})
    assert s.health()["uv_age_s"] is not None
    assert s.read(1.0).environment.source == "sensor"


# ───────────────────────── emergency ─────────────────────────


def test_emergency_request_expires_rather_than_latching():
    """A beacon that stops publishing means the vehicle has gone past."""
    now = datetime.now(timezone.utc)
    s = sensor(emergency_ttl=5.0)
    s.submit(f"{PREFIX}/emergency", {"approach": "east", "kind": "ambulance"}, now)
    assert s.read(1.0).emergency_approach == "east"

    # Nothing further published; the request must lapse on its own.
    s._emergency = (s._emergency[0], now - timedelta(seconds=30))
    assert s.read(1.0).emergency_approach is None


def test_emergency_clear_message_releases_immediately():
    s = sensor()
    s.submit(f"{PREFIX}/emergency", {"approach": "north"})
    s.submit(f"{PREFIX}/emergency", {"approach": "north", "clear": True})
    assert s.read(1.0).emergency_approach is None


def test_unknown_emergency_kind_falls_back_rather_than_being_passed_on():
    s = sensor()
    s.submit(f"{PREFIX}/emergency", {"approach": "south", "kind": "spaceship"})
    assert s.read(1.0).emergency_kind == "ambulance"


# ───────────────────────── staleness ─────────────────────────


def test_a_station_that_never_reported_is_degraded_not_empty():
    s = sensor()
    frame = s.read(1.0)
    assert sorted(frame.degraded_approaches) == sorted(APPROACHES)
    assert all(frame.approaches[a].degraded for a in APPROACHES)
    assert s.health()["status"] == "degraded"
    assert sorted(s.health()["never_reported"]) == sorted(APPROACHES)


def test_a_stale_station_keeps_its_last_values_and_is_flagged():
    now = datetime.now(timezone.utc)
    s = sensor(max_age=5.0)
    feed_all(s, when=now, queue=6)
    fresh = s.read(1.0)
    assert fresh.degraded_approaches == []

    old = now - timedelta(seconds=60)
    feed_all(s, when=old, queue=6)
    stale = s.read(1.0)
    assert sorted(stale.degraded_approaches) == sorted(APPROACHES)
    # Last known values are retained; they are just no longer trusted.
    assert stale.approaches["north"].queue == 6
    assert stale.approaches["north"].degraded is True
    assert sorted(s.health()["stale"]) == sorted(APPROACHES)


def test_degraded_approach_is_recalled_by_the_controller():
    """The failure mode must be a wasted green, never a skipped street."""
    controller = AdaptiveController(timing=TimingPlan())
    s = sensor(max_age=5.0)
    old = datetime.now(timezone.utc) - timedelta(seconds=60)
    # Everything quiet and stale: an empty-looking frame that must not be
    # read as empty.
    feed_all(s, when=old, queue=0, occupancy=0.0, gap=99.0)
    frame = s.read(1.0)

    assert all(controller._has_call(frame, a) for a in APPROACHES)
    assert controller._gapped_out(frame) is False


def test_a_healthy_quiet_approach_still_gaps_out():
    """The recall above must come from degradation, not from quietness."""
    controller = AdaptiveController(timing=TimingPlan())
    s = sensor(max_age=60.0)
    feed_all(s, queue=0, occupancy=0.0, gap=99.0)
    frame = s.read(1.0)
    assert frame.degraded_approaches == []
    assert not any(controller._has_call(frame, a) for a in APPROACHES)
    assert controller._gapped_out(frame) is True


def test_health_reports_message_counts_and_ages():
    s = sensor()
    feed_all(s)
    health = s.health()
    assert health["messages"] == len(APPROACHES)
    assert health["connected"] is False
    assert set(health["detector_age_s"]) == set(APPROACHES)
    assert all(age >= 0.0 for age in health["detector_age_s"].values())
