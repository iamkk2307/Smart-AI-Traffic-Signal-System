"""Engine and session factory."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

from .models import Base, Intersection

_settings = get_settings()

if _settings.database_url.startswith("sqlite"):
    path = _settings.database_url.split("///")[-1]
    if path not in (":memory:", ""):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    _settings.database_url,
    future=True,
    connect_args=(
        {"check_same_thread": False}
        if _settings.database_url.startswith("sqlite")
        else {}
    ),
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        if session.query(Intersection).count() == 0:
            session.add(
                Intersection(
                    code="INT-001",
                    name="Rajiv Gandhi Chowk",
                    latitude=17.4435,
                    longitude=78.3772,
                    control_mode=_settings.control_mode,
                )
            )
            session.commit()


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session
