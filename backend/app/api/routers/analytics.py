"""Pipeline analytics and export.

Everything here is computed from stored analyses. There are no hardcoded
figures: if the database is empty the endpoints return zeros and empty lists,
which is the honest answer. (The previous implementation returned a fixed
``{"Hot": 15, "Warm": 25, "Cold": 10}`` regardless of the data.)
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.api.deps import (
    AuditRepoDep,
    CallRepoDep,
    RequireAdmin,
    RequireAnalyst,
    RequireViewer,
)
from app.db.repository import EXPORT_MAX_ROWS, CallFilter
from app.domain.models import LeadBand, PipelineAnalytics, utcnow

router = APIRouter(tags=["Analytics"])

CSV_COLUMNS = [
    "call_id", "title", "created_at", "owner", "status",
    "lead_score", "lead_band", "confidence", "outcome", "outcome_confidence",
    "sentiment", "sentiment_trend", "duration_s", "word_count",
    "rep_talk_ratio", "prospect_talk_ratio",
    "signals", "objections", "unaddressed_objections", "risks",
    "top_action", "headline",
]


@router.get(
    "/analytics/pipeline",
    response_model=PipelineAnalytics,
    summary="Portfolio-level analytics",
)
async def pipeline_analytics(
    principal: RequireViewer,
    repo: CallRepoDep,
    days: Annotated[int, Query(ge=1, le=365, description="Trend window")] = 30,
) -> PipelineAnalytics:
    """Aggregate metrics across all calls.

    Totals and band/outcome breakdowns cover the whole dataset. Objection,
    signal and risk frequencies are limited to the last `days` because they
    require reading analysis JSON -- see docs/API.md.
    """
    return await repo.analytics(days=days)


@router.get(
    "/export/calls.csv",
    summary="Export calls as CSV",
    response_class=StreamingResponse,
)
async def export_csv(
    principal: RequireAnalyst,
    repo: CallRepoDep,
    audit: AuditRepoDep,
    band: Annotated[list[LeadBand] | None, Query()] = None,
    owner: Annotated[str | None, Query(max_length=128)] = None,
    min_score: Annotated[float | None, Query(ge=0, le=100)] = None,
    created_after: Annotated[datetime | None, Query()] = None,
) -> StreamingResponse:
    """One row per call, flattened for a spreadsheet or CRM import.

    Capped at `EXPORT_MAX_ROWS` newest-first. A truncated export says so in the
    `X-Export-Truncated` header rather than looking like a complete one --
    narrow the filters to get the rest.

    Exporting customer conversation data is audited.
    """
    calls = await repo.list_for_export(
        CallFilter(
            bands=band, owner=owner, min_score=min_score, created_after=created_after
        )
    )
    truncated = len(calls) > EXPORT_MAX_ROWS
    del calls[EXPORT_MAX_ROWS:]

    await audit.record(
        principal.id, "export.csv", None,
        {
            "rows": len(calls),
            "truncated": truncated,
            "filtered_by_band": [b.value for b in band or []],
        },
    )

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()

    for call in calls:
        row = {
            "call_id": call.id,
            "title": call.title,
            "created_at": call.created_at.isoformat(),
            "owner": call.owner or "",
            "status": call.status.value,
            "lead_score": call.lead_score if call.lead_score is not None else "",
            "lead_band": call.lead_band.value if call.lead_band else "",
            "outcome": call.outcome or "",
            "sentiment": call.sentiment if call.sentiment is not None else "",
            "duration_s": call.duration_s if call.duration_s is not None else "",
            "word_count": call.word_count,
        }
        analysis = call.analysis
        if analysis:
            unaddressed = [o for o in analysis.objections if not o.addressed]
            row.update({
                "confidence": analysis.lead_score.confidence,
                "outcome_confidence": analysis.outcome.confidence,
                "sentiment_trend": analysis.sentiment.trend,
                "rep_talk_ratio": analysis.talk_metrics.rep_talk_ratio,
                "prospect_talk_ratio": analysis.talk_metrics.prospect_talk_ratio,
                "signals": "|".join(s.type.value for s in analysis.signals),
                "objections": "|".join(o.category.value for o in analysis.objections),
                "unaddressed_objections": "|".join(
                    o.category.value for o in unaddressed
                ),
                "risks": "|".join(r.value for r in analysis.lead_score.risks),
                "top_action": (
                    analysis.next_actions[0].action if analysis.next_actions else ""
                ),
                "headline": analysis.summary.headline,
            })
        writer.writerow(row)

    buffer.seek(0)
    stamp = utcnow().strftime("%Y%m%d-%H%M%S")
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="callsense-{stamp}.csv"',
            "X-Export-Truncated": "true" if truncated else "false",
        },
    )


@router.get("/audit", summary="Recent audit events (admin)")
async def list_audit_events(
    principal: RequireAdmin,
    audit: AuditRepoDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[dict]:
    """Who did what. Useful for answering compliance questions directly.

    Admin-only: the log names the principal behind every upload, export and
    deletion, so it is exactly the endpoint that must not be readable by the
    analysts it records. This enforces what the role table in docs/API.md and
    ``Role``'s own docstring already promised -- it previously accepted any
    analyst.
    """
    events = await audit.recent(limit)
    return [
        {
            "at": event.at.isoformat(),
            "principal": event.principal,
            "action": event.action,
            "call_id": event.call_id,
            "detail": event.detail,
        }
        for event in events
    ]
