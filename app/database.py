"""
Database engine and session management (SQLAlchemy 2.0, SQLite by default).

A single engine is created from settings; `SessionLocal` yields thread-safe
sessions. `init_db` creates tables at startup. Sessions must be created per
thread (the background worker creates its own).
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_settings = get_settings()

# `check_same_thread=False` lets the background worker use its own session on a
# different thread; we still create one session per unit of work.
_connect_args = {"check_same_thread": False} if _settings.sqlalchemy_url.startswith("sqlite") else {}

engine = create_engine(
    _settings.sqlalchemy_url,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


def init_db() -> None:
    """Create all tables. Import models first so they register on Base."""
    from . import models  # noqa: F401  (ensures models are registered)

    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency that yields a database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
