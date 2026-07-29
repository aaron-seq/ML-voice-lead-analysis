"""Async engine and session management."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Base

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _engine_kwargs(url: str, echo: bool) -> dict:
    kwargs: dict = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        # SQLite has no server-side pool to manage, and the default
        # NullPool/StaticPool interplay differs from Postgres. Leave pooling
        # to the driver and only configure what applies.
        kwargs["connect_args"] = {"timeout": 30}
    else:
        kwargs.update(
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # survives a database restart or idle timeout
            pool_recycle=1800,
        )
    return kwargs


def init_engine(url: str, echo: bool = False) -> AsyncEngine:
    """Create the process-wide engine. Idempotent."""
    global _engine, _sessionmaker
    if _engine is not None:
        return _engine

    if url.startswith("sqlite") and ":memory:" not in url:
        # Fail loudly at startup rather than on the first write.
        db_path = Path(url.split("///", 1)[-1])
        db_path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_async_engine(url, **_engine_kwargs(url, echo))
    _sessionmaker = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )
    logger.info("Database engine initialised", extra={"dialect": _engine.dialect.name})
    return _engine


async def create_schema() -> None:
    """Create tables if absent.

    Adequate for SQLite and first-run Postgres. Schema *changes* against a
    populated database should go through Alembic -- see docs/OPERATIONS.md.
    """
    engine = _require_engine()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None


def _require_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database engine not initialised; call init_engine() first")
    return _engine


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session. Commits on success, rolls back on failure."""
    if _sessionmaker is None:
        raise RuntimeError("Database engine not initialised; call init_engine() first")
    session = _sessionmaker()
    try:
        yield session
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with session_scope() as session:
        yield session


async def check_health() -> tuple[bool, str]:
    from sqlalchemy import text

    try:
        engine = _require_engine()
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True, engine.dialect.name
    except Exception as exc:
        return False, str(exc)
