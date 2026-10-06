"""Persistence schema: detector log, decision audit trail, incidents, metrics."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Intersection(Base):
    __tablename__ = "intersections"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    latitude: Mapped[float] = mapped_column(Float, default=0.0)
    longitude: Mapped[float] = mapped_column(Float, default=0.0)
    control_mode: Mapped[str] = mapped_column(String(16), default="adaptive")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    traffic_logs: Mapped[list["TrafficLog"]] = relationship(back_populates="intersection")


class TrafficLog(Base):
    """One aggregated detector frame."""

    __tablename__ = "traffic_logs"
    __table_args__ = (Index("ix_traffic_logs_lookup", "intersection_id", "timestamp"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    intersection_id: Mapped[int] = mapped_column(ForeignKey("intersections.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    phase: Mapped[str] = mapped_column(String(8))
    interval: Mapped[str] = mapped_column(String(16))
    cycle: Mapped[int] = mapped_column(Integer, default=0)
    weather: Mapped[str] = mapped_column(String(24), default="clear")
    total_queue: Mapped[int] = mapped_column(Integer, default=0)
    mean_delay: Mapped[float] = mapped_column(Float, default=0.0)
    approaches: Mapped[dict] = mapped_column(JSON, default=dict)

    intersection: Mapped[Intersection] = relationship(back_populates="traffic_logs")


class PhaseLog(Base):
    """Audit trail: every phase change, with the numbers behind it."""

    __tablename__ = "phase_logs"
    __table_args__ = (Index("ix_phase_logs_lookup", "intersection_id", "timestamp"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    intersection_id: Mapped[int] = mapped_column(ForeignKey("intersections.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    cycle: Mapped[int] = mapped_column(Integer, default=0)
    from_phase: Mapped[str] = mapped_column(String(8))
    to_phase: Mapped[str] = mapped_column(String(8))
    trigger: Mapped[str] = mapped_column(String(24))
    reason: Mapped[str] = mapped_column(String(640))
    served_green_time: Mapped[float] = mapped_column(Float, default=0.0)
    pressure: Mapped[dict] = mapped_column(JSON, default=dict)
    terms: Mapped[dict] = mapped_column(JSON, default=dict)


class IncidentRecord(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(primary_key=True)
    intersection_id: Mapped[int] = mapped_column(ForeignKey("intersections.id"))
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    approach: Mapped[str | None] = mapped_column(String(16), nullable=True)
    kind: Mapped[str] = mapped_column(String(32))
    severity: Mapped[str] = mapped_column(String(16))
    layer: Mapped[str] = mapped_column(String(16), default="rule")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[str] = mapped_column(String(640), default="")


class SafetyLog(Base):
    """Interlock findings. An empty table is the expected state."""

    __tablename__ = "safety_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    intersection_id: Mapped[int] = mapped_column(ForeignKey("intersections.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    kind: Mapped[str] = mapped_column(String(48))
    detail: Mapped[str] = mapped_column(String(640))
    command: Mapped[dict] = mapped_column(JSON, default=dict)


class ModelMetric(Base):
    """One training run of one forecaster, with its baseline comparison."""

    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    approach: Mapped[str] = mapped_column(String(16))
    estimator: Mapped[str] = mapped_column(String(64))
    rows: Mapped[int] = mapped_column(Integer, default=0)
    model_mae: Mapped[float] = mapped_column(Float, default=0.0)
    baseline_mae: Mapped[float] = mapped_column(Float, default=0.0)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
