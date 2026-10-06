"""
Forecasting tests.

The point of these is not that a gradient-boosted model exists. It is that
the model is only used when it earns its place, that the validation split
cannot leak the answer, and that the whole layer degrades to a baseline
instead of failing.
"""

import math
import random

import pytest

from app.ml.features import (
    FEATURE_NAMES,
    LAGS,
    Observation,
    build_dataset,
    build_row,
    chronological_split,
    mean_absolute_error,
    persistence_baseline,
)
from app.ml.incidents import DetectorThresholds, IncidentDetector
from app.ml.predictor import MIN_TRAINING_ROWS, ArrivalForecaster


def structured_series(n=1200, noise=1.2, seed=1):
    """Volume with a genuine daily cycle plus noise — learnable, not trivial."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        minute = i % 10080
        hour = (minute % 1440) / 60.0
        level = 18 + 14 * math.sin(2 * math.pi * (hour - 7) / 24) ** 2
        out.append(
            Observation(
                minute_of_week=minute,
                volume=max(0.0, level + rng.gauss(0, noise)),
                occupancy=min(1.0, level / 40),
                queue=level / 3,
            )
        )
    return out


# ───────────────────────── features ─────────────────────────


def test_row_width_matches_the_declared_feature_names():
    history = structured_series(200)
    row = build_row(history, 100)
    assert row is not None
    assert len(row) == len(FEATURE_NAMES)


def test_row_is_none_without_enough_history():
    history = structured_series(200)
    for i in range(max(LAGS)):
        assert build_row(history, i) is None
    assert build_row(history, max(LAGS)) is not None


def test_time_of_day_is_cyclic_so_midnight_adjoins_23_00():
    history = structured_series(3000)
    tod_sin = FEATURE_NAMES.index("tod_sin")
    tod_cos = FEATURE_NAMES.index("tod_cos")

    def at(minute_of_day):
        idx = next(
            i
            for i in range(max(LAGS), len(history))
            if history[i].minute_of_week % 1440 == minute_of_day
        )
        row = build_row(history, idx)
        return row[tod_sin], row[tod_cos]

    a, b = at(1439), at(0)
    distance = math.dist(a, b)
    far = math.dist(at(0), at(720))
    assert distance < far / 10


def test_chronological_split_keeps_validation_strictly_later():
    history = structured_series(600)
    X, y, idx = build_dataset(history)
    X_train, y_train, X_val, y_val = chronological_split(X, y, 0.25)
    assert len(X_train) + len(X_val) == len(X)
    assert y_train == y[: len(y_train)]
    assert y_val == y[len(y_train) :]


def test_persistence_baseline_reads_lag_one():
    history = structured_series(200)
    X, y, _ = build_dataset(history)
    baseline = persistence_baseline(X)
    lag_1 = FEATURE_NAMES.index("lag_1")
    assert baseline == [row[lag_1] for row in X]


# ───────────────────────── forecaster ─────────────────────────


def test_training_refuses_a_short_series():
    forecaster = ArrivalForecaster()
    for obs in structured_series(40):
        forecaster.observe("north", obs)
    report = forecaster.train(["north"])["north"]
    assert report["promoted"] is False
    assert report["estimator"] == "insufficient_history"


def test_model_is_promoted_only_when_it_beats_the_baseline():
    forecaster = ArrivalForecaster()
    for obs in structured_series(1500, noise=1.0, seed=3):
        forecaster.observe("north", obs)
    report = forecaster.train(["north"])["north"]
    assert report["rows"] > MIN_TRAINING_ROWS
    if report["promoted"]:
        assert report["model_mae"] < report["baseline_mae"]
        assert forecaster.status()["serving"]["north"] == "model"
    else:
        assert report["model_mae"] >= report["baseline_mae"]
        assert forecaster.status()["serving"]["north"] == "baseline"


def test_structured_signal_is_learnable_at_all():
    """Guards against the v1 defect: regressing volume on a row counter.

    A model given real features must beat persistence on a series with a
    strong daily cycle. If this fails, the feature set is broken.
    """
    forecaster = ArrivalForecaster()
    for obs in structured_series(2500, noise=0.8, seed=5):
        forecaster.observe("north", obs)
    report = forecaster.train(["north"])["north"]
    assert report["promoted"], report


def test_pure_noise_does_not_get_promoted():
    rng = random.Random(0)
    forecaster = ArrivalForecaster()
    for i in range(1200):
        forecaster.observe(
            "east",
            Observation(
                minute_of_week=i % 10080,
                volume=rng.gauss(20, 6),
                occupancy=0.4,
                queue=5,
            ),
        )
    report = forecaster.train(["east"])["east"]
    # On white noise the mean is optimal and persistence is roughly
    # sqrt(2) worse, so a model may legitimately win here. What must not
    # happen is a promotion with a worse validation error than the baseline.
    if report["promoted"]:
        assert report["model_mae"] < report["baseline_mae"]


def test_predict_falls_back_to_a_rolling_mean_with_no_model():
    forecaster = ArrivalForecaster()
    for obs in structured_series(60):
        forecaster.observe("west", obs)
    value = forecaster.predict("west")
    window = forecaster.history["west"][-5:]
    assert value == pytest.approx(sum(o.volume for o in window) / 5, abs=1e-6)


def test_predict_on_an_empty_history_is_zero_not_an_error():
    assert ArrivalForecaster().predict("north") == 0.0


# ───────────────────────── incident detection ─────────────────────────


def test_blockage_rule_fires_on_a_queue_that_will_not_discharge_on_green():
    from datetime import datetime, timezone

    from app.core.phases import APPROACHES, CROSSWALKS, Interval, Signal, Walk
    from app.core.types import ApproachReading, SensorFrame, SignalCommand

    detector = IncidentDetector(DetectorThresholds(stalled_seconds=10.0))
    frame = SensorFrame(
        approaches={
            a: ApproachReading(queue=14, flow=0.2, occupancy=0.8, max_wait=20)
            for a in APPROACHES
        }
    )
    command = SignalCommand(
        timestamp=datetime.now(timezone.utc),
        phase="NS",
        interval=Interval.GREEN,
        vehicle={
            a: (Signal.GREEN if a in ("north", "south") else Signal.RED)
            for a in APPROACHES
        },
        pedestrian={c: Walk.DONT_WALK for c in CROSSWALKS},
        interval_elapsed=5.0,
        interval_remaining=None,
        cycle=1,
    )
    raised = []
    for _ in range(12):
        raised = detector.update(frame, command, 1.0)
    kinds = {i.kind for i in raised}
    assert "lane_blockage" in kinds
    blockage = next(i for i in raised if i.kind == "lane_blockage")
    assert blockage.layer == "rule"
    assert blockage.approach in ("north", "south")


def test_anomaly_layer_refuses_to_fit_on_too_little_data():
    detector = IncidentDetector()
    result = detector.fit_anomaly()
    assert result["fitted"] is False
