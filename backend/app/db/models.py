"""Database schema.

Two tables. ``calls`` carries the queryable, denormalised facts used for
listing, filtering and aggregation; the full ``CallAnalysis`` is stored beside
it as JSON. Aggregate analytics run as SQL over indexed columns rather than by
loading every analysis into Python, which is what makes the dashboard scale
past a demo dataset.

``audit_events`` records who did what. Deleting a call or exporting customer
conversations are exactly the actions a security review asks about.

Portability note: ``JSON`` and ``String`` are used rather than Postgres-native
``JSONB``/``UUID`` so the same models run on SQLite (local, tests) and Postgres
(production) without a dialect branch.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.domain.models import AnalysisStatus, utcnow


class UTCDateTime(TypeDecorator[datetime]):
    """A timestamp that is timezone-aware UTC on every backend (issue #38).

    ``DateTime(timezone=True)`` is a no-op on SQLite: the driver stores wall
    time with no offset and hands back naive datetimes. That caused two bugs on
    the default local backend, while PostgreSQL's TIMESTAMPTZ hid both:

    * **Reads came back naive.** The API then serialised ``created_at`` without
      an offset while ``updated_at`` (still in memory) carried one, in the same
      response. The dashboard passes ``created_at`` to ``new Date()``, which
      parses an offset-less ISO string as *local* time, so a call created
      seconds ago rendered as "6h ago" in India and "just now" for four hours
      in New York.
    * **Filters compared wall clocks, not instants.** ``created_after`` given
      in +05:30 was compared digit-for-digit against stored UTC wall time, so a
      bound one minute before a call excluded it.

    Normalising at the persistence boundary fixes both where the information
    is actually lost. The DDL type is unchanged, so existing databases need no
    migration: SQLite rows were already UTC wall time by convention, and
    Postgres rows were already TIMESTAMPTZ.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            # Naive values are UTC by this codebase's convention: utcnow() is
            # the only clock, and it is aware. A naive value reaching here came
            # from outside (an API query parameter without an offset).
            value = value.replace(tzinfo=UTC)
        value = value.astimezone(UTC)
        # SQLite has no offset column type; store the UTC wall time explicitly
        # rather than relying on the driver to discard the offset silently.
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class CallRecord(Base):
    """One analysed (or pending) call."""

    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow, nullable=False
    )

    source_format: Mapped[str] = mapped_column(String(32), default="text")
    source_uri: Mapped[str | None] = mapped_column(String(512))
    audio_uri: Mapped[str | None] = mapped_column(String(512))

    status: Mapped[str] = mapped_column(
        String(16), default=AnalysisStatus.PENDING.value, nullable=False
    )
    error: Mapped[str | None] = mapped_column(Text)

    # Denormalised analysis facts. Kept in sync by the repository on every
    # write; never edited independently of `analysis_json`.
    lead_score: Mapped[float | None] = mapped_column(Float)
    lead_band: Mapped[str | None] = mapped_column(String(16))
    outcome: Mapped[str | None] = mapped_column(String(32))
    sentiment: Mapped[float | None] = mapped_column(Float)
    duration_s: Mapped[float | None] = mapped_column(Float)
    word_count: Mapped[int] = mapped_column(Integer, default=0)

    owner: Mapped[str | None] = mapped_column(String(128), index=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)

    engine_version: Mapped[str | None] = mapped_column(String(32))
    analysis_json: Mapped[dict | None] = mapped_column(JSON)

    __table_args__ = (
        # Supports the dashboard's default view: newest first, filtered by band.
        Index("ix_calls_created_at", "created_at"),
        Index("ix_calls_band_created", "lead_band", "created_at"),
        Index("ix_calls_status", "status"),
        Index("ix_calls_owner_created", "owner", "created_at"),
    )


class AuditEvent(Base):
    """Append-only record of consequential actions."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, nullable=False, index=True
    )
    principal: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    call_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("calls.id", ondelete="SET NULL")
    )
    detail: Mapped[dict] = mapped_column(JSON, default=dict)

    call: Mapped[CallRecord | None] = relationship()
