"""
Arrival forecaster.

One model per approach, trained on its own recent history. Two properties
matter more than the choice of estimator:

* **It is measured against a persistence baseline.** "Predict what happened
  last interval" is a strong forecaster at one-minute resolution, and a
  model that cannot beat it is adding latency and risk for nothing. The
  trained model is only promoted into service if it wins on a chronological
  validation split; otherwise the forecaster keeps returning the baseline
  and says so in its status.
* **It degrades instead of failing.** If scikit-learn is unavailable, the
  data is too short, or a fit raises, ``predict`` falls back to the rolling
  mean. The controller treats the forecast as one weighted term among
  several, so a missing forecast costs accuracy, not availability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.core.phases import APPROACHES

from .features import (
    FEATURE_NAMES,
    Observation,
    build_dataset,
    build_row,
    chronological_split,
    mean_absolute_error,
    persistence_baseline,
)

logger = logging.getLogger(__name__)

MIN_TRAINING_ROWS = 120


@dataclass
class ModelReport:
    approach: str
    rows: int
    model_mae: float
    baseline_mae: float
    promoted: bool
    estimator: str

    @property
    def improvement(self) -> float:
        if not self.baseline_mae:
            return 0.0
        return (self.baseline_mae - self.model_mae) / self.baseline_mae

    def to_dict(self) -> dict:
        return {
            "approach": self.approach,
            "rows": self.rows,
            "model_mae": round(self.model_mae, 3),
            "baseline_mae": round(self.baseline_mae, 3),
            "improvement_pct": round(self.improvement * 100, 1),
            "promoted": self.promoted,
            "estimator": self.estimator,
        }


class ArrivalForecaster:
    """Predicts next-interval arrivals, in vehicles per minute, per approach."""

    def __init__(self, horizon: int = 1, model_dir: Optional[str] = None) -> None:
        self.horizon = horizon
        self.model_dir = Path(model_dir) if model_dir else None
        self.history: Dict[str, List[Observation]] = {a: [] for a in APPROACHES}
        self.models: Dict[str, object] = {}
        self.reports: Dict[str, ModelReport] = {}
        self._max_history = 20_000

    # ───────────────────────── data ─────────────────────────

    def observe(self, approach: str, observation: Observation) -> None:
        series = self.history[approach]
        series.append(observation)
        if len(series) > self._max_history:
            del series[: len(series) - self._max_history]

    def observe_frame(self, frame, minute_of_week: int) -> None:
        """Record one aggregated interval from a sensor frame."""
        for approach in APPROACHES:
            reading = frame.approaches[approach]
            self.observe(
                approach,
                Observation(
                    minute_of_week=minute_of_week,
                    volume=reading.flow,
                    occupancy=reading.occupancy,
                    queue=float(reading.queue),
                    weather=frame.weather,
                ),
            )

    # ───────────────────────── training ─────────────────────────

    def train(self, approaches: Optional[Sequence[str]] = None) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for approach in approaches or APPROACHES:
            report = self._train_one(approach)
            if report is not None:
                self.reports[approach] = report
                out[approach] = report.to_dict()
            else:
                out[approach] = {
                    "approach": approach,
                    "rows": len(self.history[approach]),
                    "promoted": False,
                    "estimator": "insufficient_history",
                    "note": f"needs at least {MIN_TRAINING_ROWS} usable intervals",
                }
        return out

    def _train_one(self, approach: str) -> Optional[ModelReport]:
        X, y, _ = build_dataset(self.history[approach], self.horizon)
        if len(X) < MIN_TRAINING_ROWS:
            return None

        X_train, y_train, X_val, y_val = chronological_split(X, y)
        if not X_val:
            return None

        baseline_mae = mean_absolute_error(y_val, persistence_baseline(X_val))

        estimator, name = self._make_estimator()
        if estimator is None:
            return ModelReport(
                approach, len(X), baseline_mae, baseline_mae, False, "baseline_only"
            )

        try:
            estimator.fit(X_train, y_train)
            model_mae = mean_absolute_error(y_val, list(estimator.predict(X_val)))
        except Exception:
            logger.warning("fit failed for %s", approach, exc_info=True)
            return ModelReport(
                approach, len(X), baseline_mae, baseline_mae, False, "fit_failed"
            )

        promoted = model_mae < baseline_mae
        if promoted:
            self.models[approach] = estimator
            self._persist(approach, estimator)
        else:
            self.models.pop(approach, None)
            logger.info(
                "%s: model MAE %.3f did not beat baseline %.3f, keeping baseline",
                approach,
                model_mae,
                baseline_mae,
            )

        return ModelReport(approach, len(X), model_mae, baseline_mae, promoted, name)

    def _make_estimator(self):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor

            return (
                HistGradientBoostingRegressor(
                    max_iter=220,
                    learning_rate=0.06,
                    max_depth=6,
                    min_samples_leaf=12,
                    l2_regularization=0.5,
                    random_state=42,
                ),
                "HistGradientBoostingRegressor",
            )
        except ImportError:
            pass
        try:
            from sklearn.linear_model import Ridge

            return Ridge(alpha=1.0), "Ridge"
        except ImportError:
            logger.warning("scikit-learn unavailable; forecaster stays on baseline")
            return None, "unavailable"

    # ───────────────────────── inference ─────────────────────────

    def predict(self, approach: str) -> float:
        """Forecast arrivals per minute for the next interval."""
        series = self.history.get(approach) or []
        if not series:
            return 0.0

        model = self.models.get(approach)
        if model is not None:
            row = build_row(series, len(series) - 1)
            if row is not None:
                try:
                    return max(0.0, float(model.predict([row])[0]))
                except Exception:
                    logger.warning("predict failed for %s", approach, exc_info=True)

        window = series[-5:]
        return max(0.0, sum(o.volume for o in window) / len(window))

    def as_callable(self):
        """Bind to the controller's ``forecast`` hook."""
        return self.predict

    # ───────────────────────── persistence ─────────────────────────

    def _persist(self, approach: str, estimator) -> None:
        if self.model_dir is None:
            return
        try:
            import joblib

            self.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"estimator": estimator, "features": FEATURE_NAMES,
                 "horizon": self.horizon},
                self.model_dir / f"arrivals_{approach}.joblib",
            )
        except Exception:
            logger.warning("could not persist model for %s", approach, exc_info=True)

    def load(self) -> List[str]:
        """Restore promoted models from disk. Returns the approaches loaded."""
        if self.model_dir is None or not self.model_dir.exists():
            return []
        loaded = []
        try:
            import joblib
        except ImportError:
            return []
        for approach in APPROACHES:
            path = self.model_dir / f"arrivals_{approach}.joblib"
            if not path.exists():
                continue
            try:
                bundle = joblib.load(path)
                if tuple(bundle.get("features", ())) != FEATURE_NAMES:
                    logger.warning(
                        "%s: stored model has a stale feature set, skipping", approach
                    )
                    continue
                self.models[approach] = bundle["estimator"]
                loaded.append(approach)
            except Exception:
                logger.warning("could not load model for %s", approach, exc_info=True)
        return loaded

    def status(self) -> dict:
        return {
            "horizon_intervals": self.horizon,
            "history_rows": {a: len(self.history[a]) for a in APPROACHES},
            "serving": {
                a: ("model" if a in self.models else "baseline") for a in APPROACHES
            },
            "reports": {a: r.to_dict() for a, r in self.reports.items()},
        }
