"""
Feature construction for arrival forecasting.

Why this module exists at all
-----------------------------
The tempting shortcut is to regress volume on a row counter — fit
``density ~ cycle_number`` and call it a prediction. It is worse than
useless: a tree ensemble cannot extrapolate past the largest index it saw
in training, so every forecast collapses to the last leaf value, and a
linear fit on a counter just reports the run's overall drift.

Traffic volume is driven by time of day, day of week, and its own recent
history. So the feature set here is lags of the series itself, a rolling
level and trend, and cyclic encodings of the clock. Hour-of-day is encoded
as a sine/cosine pair rather than an integer so that 23:00 and 00:00 are
adjacent in feature space, which is where the peak-shoulder behaviour
actually lives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: Interval the series is aggregated to, in seconds.
INTERVAL_SECONDS = 60

LAGS: Tuple[int, ...] = (1, 2, 3, 5, 10, 15)
WEATHER_LEVELS: Tuple[str, ...] = (
    "clear",
    "cloudy",
    "rain",
    "fog",
    "snow",
    "thunderstorm",
)

FEATURE_NAMES: Tuple[str, ...] = (
    tuple(f"lag_{k}" for k in LAGS)
    + ("roll_mean_5", "roll_mean_15", "trend_5", "roll_std_15")
    + ("tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend")
    + ("occupancy_lag_1", "queue_lag_1")
    + tuple(f"weather_{w}" for w in WEATHER_LEVELS)
)


@dataclass
class Observation:
    """One aggregated interval for one approach."""

    minute_of_week: int  #: 0 .. 10079
    volume: float  #: vehicles arriving in this interval
    occupancy: float
    queue: float
    weather: str = "clear"


def _cyclic(value: float, period: float) -> Tuple[float, float]:
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def build_row(history: Sequence[Observation], index: int) -> Optional[List[float]]:
    """Features describing the state *before* ``history[index]``.

    Returns ``None`` when there is not enough history behind ``index`` to
    fill the longest lag, so the caller never trains on padded zeros.
    """
    longest = max(LAGS)
    if index < longest:
        return None

    series = [o.volume for o in history]
    window_5 = series[index - 5 : index]
    window_15 = series[index - 15 : index]

    row: List[float] = [series[index - k] for k in LAGS]

    mean_5 = sum(window_5) / len(window_5)
    mean_15 = sum(window_15) / len(window_15)
    row.append(mean_5)
    row.append(mean_15)
    # Simple slope across the last five intervals.
    row.append((window_5[-1] - window_5[0]) / max(len(window_5) - 1, 1))
    var_15 = sum((v - mean_15) ** 2 for v in window_15) / len(window_15)
    row.append(math.sqrt(var_15))

    current = history[index]
    minute_of_day = current.minute_of_week % 1440
    day_of_week = current.minute_of_week // 1440
    tod_sin, tod_cos = _cyclic(minute_of_day, 1440)
    dow_sin, dow_cos = _cyclic(day_of_week, 7)
    row += [tod_sin, tod_cos, dow_sin, dow_cos, 1.0 if day_of_week >= 5 else 0.0]

    previous = history[index - 1]
    row.append(previous.occupancy)
    row.append(previous.queue)

    weather = current.weather if current.weather in WEATHER_LEVELS else "clear"
    row += [1.0 if weather == level else 0.0 for level in WEATHER_LEVELS]

    assert len(row) == len(FEATURE_NAMES), (len(row), len(FEATURE_NAMES))
    return row


def build_dataset(
    history: Sequence[Observation], horizon: int = 1
) -> Tuple[List[List[float]], List[float], List[int]]:
    """Supervised set predicting volume ``horizon`` intervals ahead.

    Returns ``(X, y, indices)``. ``indices`` are the positions in ``history``
    the targets came from, which the caller needs for a chronological split.
    """
    X: List[List[float]] = []
    y: List[float] = []
    idx: List[int] = []
    for i in range(len(history) - horizon):
        row = build_row(history, i)
        if row is None:
            continue
        X.append(row)
        y.append(history[i + horizon].volume)
        idx.append(i + horizon)
    return X, y, idx


def chronological_split(
    X: List[List[float]], y: List[float], validation_fraction: float = 0.25
) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    """Split by time, never at random.

    A random split on a lagged time series leaks the answer: the target for
    interval *t* appears as ``lag_1`` in the row for *t+1*, so a shuffled
    validation set sits on both sides of the fence and the reported error is
    meaningless.
    """
    if not X:
        return [], [], [], []
    cut = max(1, int(len(X) * (1.0 - validation_fraction)))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def minute_of_week(timestamp) -> int:
    return timestamp.weekday() * 1440 + timestamp.hour * 60 + timestamp.minute


def persistence_baseline(X: List[List[float]]) -> List[float]:
    """Predict the previous interval's volume. The bar any model must clear."""
    lag_1 = FEATURE_NAMES.index("lag_1")
    return [row[lag_1] for row in X]


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return float("nan")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)
