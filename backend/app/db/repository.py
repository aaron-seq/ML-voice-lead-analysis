"""Data access for calls and audit events.

All SQL lives here. Routers and services deal in domain models and never see a
``select()``, which keeps the query surface small enough to review and means
the storage engine can change without touching the API.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Select, case, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.db.models import AuditEvent, CallRecord
from app.domain.models import (
    AnalysisStatus,
    BandBreakdown,
    Call,
    CallAnalysis,
    CallOutcomeLabel,
    LeadBand,
    PipelineAnalytics,
    TrendPoint,
    utcnow,
)

# Whitelist for ordering. Prevents an arbitrary user string reaching ORDER BY.
SORTABLE: dict[str, Any] = {
    "created_at": CallRecord.created_at,
    "updated_at": CallRecord.updated_at,
    "lead_score": CallRecord.lead_score,
    "sentiment": CallRecord.sentiment,
    "duration_s": CallRecord.duration_s,
    "title": CallRecord.title,
}

MAX_PAGE_SIZE = 100


@dataclass(slots=True)
class CallFilter:
    """Everything the list endpoint can filter by."""

    search: str | None = None
    bands: list[LeadBand] | None = None
    statuses: list[AnalysisStatus] | None = None
    outcomes: list[str] | None = None
    owner: str | None = None
    min_score: float | None = None
    max_score: float | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None


@dataclass(slots=True)
class Page:
    items: list[Call]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        if self.page_size <= 0:
            return 0
        return (self.total + self.page_size - 1) // self.page_size

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_previous(self) -> bool:
        return self.page > 1


class CallRepository:
    """Persistence for calls."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- mapping ----------------------------------------------------------

    @staticmethod
    def _to_domain(record: CallRecord, include_analysis: bool = True) -> Call:
        analysis = None
        if include_analysis and record.analysis_json:
            # Tolerate rows written by an older schema rather than failing the
            # whole list request; the row still has usable denormalised facts.
            try:
                analysis = CallAnalysis.model_validate(record.analysis_json)
            except Exception:
                analysis = None

        return Call(
            id=record.id,
            title=record.title,
            created_at=record.created_at,
            updated_at=record.updated_at,
            source_format=record.source_format,
            source_uri=record.source_uri,
            audio_uri=record.audio_uri,
            status=AnalysisStatus(record.status),
            lead_score=record.lead_score,
            lead_band=LeadBand(record.lead_band) if record.lead_band else None,
            outcome=CallOutcomeLabel(record.outcome) if record.outcome else None,
            sentiment=record.sentiment,
            duration_s=record.duration_s,
            word_count=record.word_count or 0,
            error=record.error,
            owner=record.owner,
            tags=list(record.tags or []),
            analysis=analysis,
        )

    # -- writes -----------------------------------------------------------

    async def create(
        self,
        call_id: str,
        title: str,
        source_format: str,
        source_uri: str | None = None,
        owner: str | None = None,
        tags: list[str] | None = None,
    ) -> Call:
        record = CallRecord(
            id=call_id,
            title=title,
            source_format=source_format,
            source_uri=source_uri,
            owner=owner,
            tags=tags or [],
            status=AnalysisStatus.PENDING.value,
        )
        self._session.add(record)
        await self._session.flush()
        return self._to_domain(record)

    async def attach_analysis(
        self,
        call_id: str,
        analysis: CallAnalysis,
        word_count: int,
        source_format: str | None = None,
    ) -> Call:
        """Store an analysis and refresh the denormalised columns atomically."""
        record = await self._require(call_id)
        record.analysis_json = analysis.model_dump(mode="json")
        if source_format:
            # Only known once the transcript has been parsed, which happens
            # after the row is created.
            record.source_format = source_format
        record.status = AnalysisStatus.COMPLETED.value
        record.error = None
        record.lead_score = analysis.lead_score.score
        record.lead_band = analysis.lead_score.band.value
        record.outcome = analysis.outcome.label.value
        record.sentiment = analysis.sentiment.overall
        record.duration_s = analysis.metadata.duration_s
        record.word_count = word_count
        record.engine_version = analysis.metadata.engine_version
        record.updated_at = utcnow()
        await self._session.flush()
        return self._to_domain(record)

    async def mark_status(
        self,
        call_id: str,
        status: AnalysisStatus,
        error: str | None = None,
        commit: bool = False,
    ) -> None:
        """Update a call's processing status.

        ``commit`` exists for the failure path. The request-scoped session
        rolls back when an exception propagates, which would erase the very
        record that tells the user why their upload was rejected -- so the
        failure state has to be committed before the exception is re-raised.
        """
        record = await self._require(call_id)
        record.status = status.value
        record.error = error
        record.updated_at = utcnow()
        await self._session.flush()
        if commit:
            await self._session.commit()

    async def update_metadata(
        self,
        call_id: str,
        title: str | None = None,
        owner: str | None = None,
        tags: list[str] | None = None,
    ) -> Call:
        record = await self._require(call_id)
        if title is not None:
            record.title = title
        if owner is not None:
            record.owner = owner
        if tags is not None:
            record.tags = tags
        record.updated_at = utcnow()
        await self._session.flush()
        return self._to_domain(record)

    async def delete(self, call_id: str) -> None:
        record = await self._require(call_id)
        await self._session.delete(record)
        await self._session.flush()

    # -- reads ------------------------------------------------------------

    async def _require(self, call_id: str) -> CallRecord:
        record = await self._session.get(CallRecord, call_id)
        if record is None:
            raise NotFoundError(f"Call '{call_id}' not found", call_id=call_id)
        return record

    async def get(self, call_id: str) -> Call:
        return self._to_domain(await self._require(call_id))

    async def exists(self, call_id: str) -> bool:
        return await self._session.get(CallRecord, call_id) is not None

    def _apply_filters(self, stmt: Select, filters: CallFilter) -> Select:
        if filters.search:
            # ILIKE is Postgres-only; lower() works on both supported dialects.
            pattern = f"%{filters.search.lower()}%"
            stmt = stmt.where(func.lower(CallRecord.title).like(pattern))
        if filters.bands:
            stmt = stmt.where(CallRecord.lead_band.in_([b.value for b in filters.bands]))
        if filters.statuses:
            stmt = stmt.where(
                CallRecord.status.in_([s.value for s in filters.statuses])
            )
        if filters.outcomes:
            stmt = stmt.where(CallRecord.outcome.in_(filters.outcomes))
        if filters.owner:
            stmt = stmt.where(CallRecord.owner == filters.owner)
        if filters.min_score is not None:
            stmt = stmt.where(CallRecord.lead_score >= filters.min_score)
        if filters.max_score is not None:
            stmt = stmt.where(CallRecord.lead_score <= filters.max_score)
        if filters.created_after:
            stmt = stmt.where(CallRecord.created_at >= filters.created_after)
        if filters.created_before:
            stmt = stmt.where(CallRecord.created_at <= filters.created_before)
        return stmt

    async def list_page(
        self,
        filters: CallFilter | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        include_analysis: bool = False,
    ) -> Page:
        """Paginated, filtered listing.

        Named ``list_page`` rather than ``list``: a method called ``list``
        shadows the builtin inside the class body, so every ``-> list[Call]``
        annotation in this class would silently resolve to the method.

        ``include_analysis`` is off by default: the full analysis blob is
        typically 20-50 KB, so hydrating it for a 20-row list page would move
        megabytes to render a table of summaries.
        """
        filters = filters or CallFilter()
        page = max(1, page)
        page_size = max(1, min(page_size, MAX_PAGE_SIZE))

        count_stmt = self._apply_filters(
            select(func.count()).select_from(CallRecord), filters
        )
        total = int((await self._session.execute(count_stmt)).scalar_one())

        column = SORTABLE.get(sort_by, CallRecord.created_at)
        order = column.desc() if sort_desc else column.asc()

        stmt = self._apply_filters(select(CallRecord), filters)
        # Tie-break on id so pagination is stable when the sort column has
        # duplicates (e.g. many calls with a null score).
        stmt = stmt.order_by(order, CallRecord.id).limit(page_size).offset(
            (page - 1) * page_size
        )

        records = (await self._session.execute(stmt)).scalars().all()
        return Page(
            items=[self._to_domain(r, include_analysis) for r in records],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def list_for_export(self, filters: CallFilter | None = None) -> list[Call]:
        """Every matching call, analysis included. Used by CSV/JSON export."""
        stmt = self._apply_filters(select(CallRecord), filters or CallFilter())
        stmt = stmt.order_by(CallRecord.created_at.desc())
        records = (await self._session.execute(stmt)).scalars().all()
        return [self._to_domain(r, include_analysis=True) for r in records]

    # -- analytics --------------------------------------------------------

    async def analytics(self, days: int = 30) -> PipelineAnalytics:
        """Portfolio metrics.

        Counts and averages are computed in SQL. Objection/signal/risk
        frequencies require reading the analysis JSON, so those are restricted
        to the requested window -- documented in docs/API.md as a known
        scaling limit with the fix (a normalised signals table) on the roadmap.
        """
        since = utcnow() - timedelta(days=days)

        totals = (
            await self._session.execute(
                select(
                    func.count(CallRecord.id),
                    func.sum(case((CallRecord.status == "completed", 1), else_=0)),
                    func.sum(case((CallRecord.status == "failed", 1), else_=0)),
                    func.avg(CallRecord.lead_score),
                    func.avg(CallRecord.sentiment),
                )
            )
        ).one()
        total_calls, analysed, failed, avg_score, avg_sentiment = totals

        band_rows = (
            await self._session.execute(
                select(CallRecord.lead_band, func.count())
                .where(CallRecord.lead_band.is_not(None))
                .group_by(CallRecord.lead_band)
            )
        ).all()
        band_counts: dict[str, int] = {str(band): int(count) for band, count in band_rows}

        outcome_rows = (
            await self._session.execute(
                select(CallRecord.outcome, func.count())
                .where(CallRecord.outcome.is_not(None))
                .group_by(CallRecord.outcome)
            )
        ).all()

        trend = await self._trend(since)

        recent = (
            (
                await self._session.execute(
                    select(CallRecord.analysis_json).where(
                        CallRecord.created_at >= since,
                        CallRecord.analysis_json.is_not(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        objections, signals, risks = _frequencies(list(recent))

        return PipelineAnalytics(
            total_calls=int(total_calls or 0),
            analysed_calls=int(analysed or 0),
            failed_calls=int(failed or 0),
            avg_lead_score=round(float(avg_score), 2) if avg_score is not None else None,
            avg_sentiment=(
                round(float(avg_sentiment), 4) if avg_sentiment is not None else None
            ),
            band_breakdown=BandBreakdown(
                hot=band_counts.get("hot", 0),
                warm=band_counts.get("warm", 0),
                cold=band_counts.get("cold", 0),
                disqualified=band_counts.get("disqualified", 0),
            ),
            outcome_breakdown={str(o): int(c) for o, c in outcome_rows},
            top_objections=objections,
            top_signals=signals,
            top_risks=risks,
            trend=trend,
        )

    async def _trend(self, since: datetime) -> list[TrendPoint]:
        # date() is supported by both SQLite and Postgres.
        day = func.date(CallRecord.created_at)
        rows = (
            await self._session.execute(
                select(
                    day,
                    func.count(CallRecord.id),
                    func.avg(CallRecord.lead_score),
                    func.avg(CallRecord.sentiment),
                )
                .where(CallRecord.created_at >= since)
                .group_by(day)
                .order_by(day)
            )
        ).all()
        return [
            TrendPoint(
                date=str(date),
                calls=int(count),
                avg_lead_score=round(float(score), 2) if score is not None else None,
                avg_sentiment=round(float(sentiment), 4)
                if sentiment is not None
                else None,
            )
            for date, count, score, sentiment in rows
        ]


def _frequencies(
    analyses: list[dict | None],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Count objection categories, signal types and risks across analyses."""
    objections: Counter[str] = Counter()
    signals: Counter[str] = Counter()
    risks: Counter[str] = Counter()
    unaddressed: Counter[str] = Counter()

    for payload in analyses:
        if not payload:
            continue
        for objection in payload.get("objections") or []:
            category = objection.get("category")
            if category:
                objections[category] += 1
                if not objection.get("addressed"):
                    unaddressed[category] += 1
        for signal in payload.get("signals") or []:
            if signal.get("type"):
                signals[signal["type"]] += 1
        for risk in (payload.get("lead_score") or {}).get("risks") or []:
            risks[risk] += 1

    return (
        [
            {"category": name, "count": count, "unaddressed": unaddressed.get(name, 0)}
            for name, count in objections.most_common(8)
        ],
        [{"type": name, "count": count} for name, count in signals.most_common(8)],
        [{"risk": name, "count": count} for name, count in risks.most_common(8)],
    )


class AuditRepository:
    """Append-only audit log."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        principal: str,
        action: str,
        call_id: str | None = None,
        detail: dict | None = None,
    ) -> None:
        self._session.add(
            AuditEvent(
                principal=principal,
                action=action,
                call_id=call_id,
                detail=detail or {},
            )
        )
        await self._session.flush()

    async def recent(self, limit: int = 100) -> list[AuditEvent]:
        stmt = select(AuditEvent).order_by(AuditEvent.at.desc()).limit(limit)
        return list((await self._session.execute(stmt)).scalars().all())

    async def purge_before(self, cutoff: datetime) -> int:
        result = await self._session.execute(
            delete(AuditEvent).where(AuditEvent.at < cutoff)
        )
        return int(getattr(result, 'rowcount', 0) or 0)
