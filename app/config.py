"""Configuration. Every value is overridable by environment variable."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.controller import TimingPlan


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── service ──
    app_name: str = "Smart Traffic Control"
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # CORS is explicit. A wildcard origin combined with credentials is
    # rejected by browsers anyway, and an open control-plane API is not a
    # default worth shipping.
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://localhost:5173"]
    )

    #: Required for any endpoint that changes signal state. Unset means
    #: operator endpoints refuse rather than run unauthenticated.
    operator_token: Optional[str] = None

    # ── storage ──
    database_url: str = "sqlite:///./data/traffic.db"
    model_dir: str = "./data/models"
    #: How often detector frames are written to the log, in seconds.
    log_interval: float = 10.0
    #: Rolling window kept in memory for the dashboard, in samples.
    live_history: int = 720

    # ── control loop ──
    sensor_mode: str = "simulation"  # simulation | camera | iot
    control_mode: str = "adaptive"  # adaptive | fixed_time
    tick_seconds: float = 1.0
    #: Wall-clock seconds per simulated second. Below 1.0 the demo runs fast.
    realtime_factor: float = 1.0

    # ── timing plan ──
    min_green: float = 10.0
    max_green: float = 60.0
    yellow_change: float = 3.5
    red_clearance: float = 2.0
    ped_walk: float = 7.0
    ped_clearance: float = 11.0
    max_red: float = 120.0
    gap_threshold: float = 3.0
    hysteresis: float = 1.15
    preempt_max_hold: float = 45.0
    preempt_budget: float = 90.0
    preempt_window: float = 600.0
    ped_max_wait: float = 90.0
    #: Low-sun glare adjustments, applied only on the approaches the UV
    #: sensor says are affected.
    glare_extra_clearance: float = 1.2
    glare_discharge_factor: float = 0.92

    # ── ml ──
    forecast_enabled: bool = True
    forecast_horizon: int = 1
    train_interval_seconds: float = 900.0

    # ── site, for solar geometry ──
    site_latitude: float = 17.385
    site_longitude: float = 78.4867
    site_altitude_m: float = 0.0
    #: Offset of the site's local wall clock from UTC, used by the
    #: simulation so its local demand profile maps to the right sun angle.
    site_utc_offset_hours: float = 5.5

    # ── UV / ambient light sensor ──
    #: Sun must be above the horizon and no higher than this for glare.
    glare_elevation_max: float = 15.0
    #: Measured UV as a fraction of clear-sky UV at the same elevation,
    #: above which the direct beam is taken to be present.
    glare_beam_fraction: float = 0.55
    #: Visible illuminance, in lux, below which there is not enough light
    #: to dazzle. The magnitude test is lux rather than UV index because
    #: UV is near zero at the low sun angles where glare actually occurs.
    glare_lux_min: float = 1200.0
    #: How far a driver's heading may differ from the solar azimuth and
    #: still be looking into the sun.
    glare_azimuth_tolerance: float = 35.0
    #: Seconds before a pushed hardware UV reading is treated as stale and
    #: the clear-sky model takes over.
    uv_max_age: float = 120.0

    # ── camera backend ──
    camera_source: str = "0"
    camera_weights: str = "yolov8n.pt"
    camera_calibration: Optional[str] = None
    #: Mean detection confidence on a glare-affected approach below which
    #: the approach is reported degraded rather than empty.
    camera_glare_confidence_floor: float = 0.45

    # ── iot backend (MQTT) ──
    mqtt_host: Optional[str] = None
    mqtt_port: int = 1883
    mqtt_prefix: str = "traffic/j1"
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    #: Seconds without a detector message before the station is degraded
    #: and its approach is placed on constant recall.
    iot_detector_max_age: float = 5.0
    #: An emergency beacon that stops publishing means the vehicle has
    #: passed, so preemption requests expire rather than latch.
    iot_emergency_ttl: float = 6.0

    def timing_plan(self) -> TimingPlan:
        return TimingPlan(
            min_green=self.min_green,
            max_green=self.max_green,
            yellow_change=self.yellow_change,
            red_clearance=self.red_clearance,
            ped_walk=self.ped_walk,
            ped_clearance=self.ped_clearance,
            max_red=self.max_red,
            gap_threshold=self.gap_threshold,
            hysteresis=self.hysteresis,
            preempt_max_hold=self.preempt_max_hold,
            preempt_budget=self.preempt_budget,
            preempt_window=self.preempt_window,
            ped_max_wait=self.ped_max_wait,
            glare_extra_clearance=self.glare_extra_clearance,
            glare_discharge_factor=self.glare_discharge_factor,
        )

    def uv_sensor_kwargs(self) -> dict:
        return {
            "latitude": self.site_latitude,
            "longitude": self.site_longitude,
            "altitude_m": self.site_altitude_m,
            "glare_elevation_max": self.glare_elevation_max,
            "glare_beam_fraction": self.glare_beam_fraction,
            "glare_lux_min": self.glare_lux_min,
            "glare_azimuth_tolerance": self.glare_azimuth_tolerance,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
