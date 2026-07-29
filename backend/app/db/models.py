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

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.domain.models import AnalysisStatus, utcnow


class Base(DeclarativeBase):
    pass


class CallRecord(Base):
    """One analysed (or pending) call."""

    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
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
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    principal: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    call_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("calls.id", ondelete="SET NULL")
    )
    detail: Mapped[dict] = mapped_column(JSON, default=dict)

    call: Mapped[CallRecord | None] = relationship()
