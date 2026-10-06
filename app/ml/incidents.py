"""
Incident detection.

Two layers, in this order, because they fail differently.

The **rule layer** catches the cases with an unambiguous physical signature:
a queue that is not discharging while its own signal is green can only be a
downstream blockage, and occupancy pinned high with flow at zero is a
stoppage. These rules are explainable, need no training window, and are
what an operator will actually act on.

The **statistical layer** is an isolation forest over the joint detector
state. It exists to catch the patterns nobody wrote a rule for, and its
output is labelled as an anomaly rather than a diagnosis — it says "this
intersection is not behaving like itself", which is a different and weaker
claim than "there is a crash on the north approach". Treating an isolation
forest score as an incident classification is how these systems end up
crying wolf; keeping the two layers labelled separately is the fix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.core.phases import APPROACHES

logger = logging.getLogger(__name__)


@dataclass
class Incident:
    timestamp: datetime
    approach: Optional[str]
    kind: str
    severity: str  #: low | medium | high | critical
    confidence: float
    detail: str
    layer: str  #: rule | anomaly

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "approach": self.approach,
            "kind": self.kind,
            "severity": self.severity,
            "confidence": round(self.confidence, 2),
            "detail": self.detail,
            "layer": self.layer,
        }


@dataclass
class DetectorThresholds:
    stalled_queue: int = 8
    stalled_flow: float = 1.5  #: veh/min while green — effectively stopped
    stalled_seconds: float = 20.0
    jam_occupancy: float = 0.9
    jam_seconds: float = 45.0
    wait_critical: float = 150.0
    anomaly_min_rows: int = 300
    anomaly_contamination: float = 0.02


class IncidentDetector:
    def __init__(self, thresholds: Optional[DetectorThresholds] = None) -> None:
        self.thresholds = thresholds or DetectorThresholds()
        self._stalled_for: Dict[str, float] = {a: 0.0 for a in APPROACHES}
        self._jammed_for: Dict[str, float] = {a: 0.0 for a in APPROACHES}
        self._open: Dict[str, Incident] = {}
        self.history: List[List[float]] = []
        self._forest = None
        self._trained_rows = 0

    # ───────────────────────── rules ─────────────────────────

    def update(self, frame, command, dt: float) -> List[Incident]:
        th = self.thresholds
        raised: List[Incident] = []
        greens = set(command.greens()) if command else set()

        for approach in APPROACHES:
            reading = frame.approaches[approach]

            serving = approach in greens
            stalled = (
                serving
                and reading.queue >= th.stalled_queue
                and reading.flow <= th.stalled_flow
            )
            self._stalled_for[approach] = (
                self._stalled_for[approach] + dt if stalled else 0.0
            )
            if self._stalled_for[approach] >= th.stalled_seconds:
                raised.append(
                    self._raise(
                        f"blockage:{approach}",
                        Incident(
                            timestamp=datetime.now(timezone.utc),
                            approach=approach,
                            kind="lane_blockage",
                            severity="high",
                            confidence=0.9,
                            detail=(
                                f"{approach.title()} has held {reading.queue} vehicles "
                                f"with {reading.flow:.1f} veh/min discharge for "
                                f"{self._stalled_for[approach]:.0f}s of its own green. "
                                "Queue is not clearing on green, so the obstruction is "
                                "downstream of the stop line."
                            ),
                            layer="rule",
                        ),
                    )
                )

            jammed = reading.occupancy >= th.jam_occupancy
            self._jammed_for[approach] = (
                self._jammed_for[approach] + dt if jammed else 0.0
            )
            if self._jammed_for[approach] >= th.jam_seconds:
                raised.append(
                    self._raise(
                        f"jam:{approach}",
                        Incident(
                            timestamp=datetime.now(timezone.utc),
                            approach=approach,
                            kind="saturation",
                            severity="medium",
                            confidence=0.75,
                            detail=(
                                f"{approach.title()} detector occupancy has stayed at "
                                f"{reading.occupancy:.2f} for "
                                f"{self._jammed_for[approach]:.0f}s. Demand exceeds the "
                                "capacity this phase can supply."
                            ),
                            layer="rule",
                        ),
                    )
                )

            if reading.max_wait >= th.wait_critical:
                raised.append(
                    self._raise(
                        f"wait:{approach}",
                        Incident(
                            timestamp=datetime.now(timezone.utc),
                            approach=approach,
                            kind="excessive_delay",
                            severity="critical",
                            confidence=1.0,
                            detail=(
                                f"{approach.title()} head of queue has waited "
                                f"{reading.max_wait:.0f}s, past the "
                                f"{th.wait_critical:.0f}s service limit."
                            ),
                            layer="rule",
                        ),
                    )
                )

            if not stalled and not jammed and reading.max_wait < th.wait_critical:
                for prefix in ("blockage", "jam", "wait"):
                    self._open.pop(f"{prefix}:{approach}", None)

        self.history.append(self._vector(frame))
        if len(self.history) > 20_000:
            del self.history[:-20_000]

        anomaly = self._score_anomaly()
        if anomaly is not None:
            raised.append(anomaly)

        return raised

    def _raise(self, key: str, incident: Incident) -> Incident:
        existing = self._open.get(key)
        if existing is None:
            self._open[key] = incident
        return self._open[key]

    @property
    def open_incidents(self) -> List[Incident]:
        return list(self._open.values())

    # ───────────────────────── anomaly layer ─────────────────────────

    def _vector(self, frame) -> List[float]:
        row: List[float] = []
        for approach in APPROACHES:
            r = frame.approaches[approach]
            row += [float(r.queue), r.flow, r.occupancy, r.mean_speed, r.max_wait]
        return row

    def fit_anomaly(self) -> dict:
        th = self.thresholds
        if len(self.history) < th.anomaly_min_rows:
            return {
                "fitted": False,
                "rows": len(self.history),
                "note": f"needs at least {th.anomaly_min_rows} intervals",
            }
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            return {"fitted": False, "note": "scikit-learn unavailable"}

        forest = IsolationForest(
            n_estimators=200,
            contamination=th.anomaly_contamination,
            random_state=42,
        )
        forest.fit(self.history)
        self._forest = forest
        self._trained_rows = len(self.history)
        return {"fitted": True, "rows": self._trained_rows}

    def _score_anomaly(self) -> Optional[Incident]:
        if self._forest is None or not self.history:
            return None
        row = self.history[-1]
        try:
            flagged = self._forest.predict([row])[0] == -1
            score = float(self._forest.score_samples([row])[0])
        except Exception:
            logger.warning("anomaly scoring failed", exc_info=True)
            return None
        if not flagged:
            self._open.pop("anomaly", None)
            return None
        return self._raise(
            "anomaly",
            Incident(
                timestamp=datetime.now(timezone.utc),
                approach=None,
                kind="pattern_anomaly",
                severity="low",
                confidence=min(1.0, abs(score) / 0.7),
                detail=(
                    "Joint detector state is outside the range learned from "
                    f"{self._trained_rows} intervals (isolation score {score:.3f}). "
                    "This flags unfamiliar behaviour, not a diagnosed cause."
                ),
                layer="anomaly",
            ),
        )

    def status(self) -> dict:
        return {
            "rows": len(self.history),
            "anomaly_model": "fitted" if self._forest else "not_fitted",
            "trained_rows": self._trained_rows,
            "open": [i.to_dict() for i in self.open_incidents],
        }
