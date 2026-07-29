"""Call ingestion, retrieval and management endpoints."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status

from app.api.deps import (
    AuditRepoDep,
    BlobStoreDep,
    CallRepoDep,
    EngineDep,
    RequireAdmin,
    RequireAnalyst,
    RequireViewer,
    SettingsDep,
)
from app.api.schemas import (
    AnalyseTextRequest,
    AnalyseTextResponse,
    CallDetailResponse,
    CallSummaryResponse,
    CreateCallRequest,
    PageMeta,
    PaginatedResponse,
    UpdateCallRequest,
)
from app.core.errors import NotFoundError, ValidationError
from app.db.repository import SORTABLE, CallFilter
from app.domain.models import AnalysisStatus, CallOutcomeLabel, LeadBand, Transcript
from app.services.ingestion import IngestionService, derive_title

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/calls", tags=["Calls"])


def _page_meta(page) -> PageMeta:
    return PageMeta(
        page=page.page,
        page_size=page.page_size,
        total_items=page.total,
        total_pages=page.total_pages,
        has_next=page.has_next,
        has_previous=page.has_previous,
    )


@router.get(
    "",
    response_model=PaginatedResponse[CallSummaryResponse],
    summary="List calls",
)
async def list_calls(
    principal: RequireViewer,
    repo: CallRepoDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    sort_by: Annotated[str, Query()] = "created_at",
    sort_desc: Annotated[bool, Query()] = True,
    search: Annotated[str | None, Query(max_length=200)] = None,
    band: Annotated[list[LeadBand] | None, Query()] = None,
    call_status: Annotated[list[AnalysisStatus] | None, Query(alias="status")] = None,
    outcome: Annotated[list[CallOutcomeLabel] | None, Query()] = None,
    owner: Annotated[str | None, Query(max_length=128)] = None,
    min_score: Annotated[float | None, Query(ge=0, le=100)] = None,
    max_score: Annotated[float | None, Query(ge=0, le=100)] = None,
    created_after: Annotated[datetime | None, Query()] = None,
    created_before: Annotated[datetime | None, Query()] = None,
) -> PaginatedResponse[CallSummaryResponse]:
    """Paginated, filterable call list.

    Sorting is restricted to a fixed set of columns; an unknown `sort_by` is
    rejected rather than silently ignored, so a broken client surfaces its bug
    instead of quietly showing mis-ordered data.
    """
    if sort_by not in SORTABLE:
        raise ValidationError(
            f"Cannot sort by '{sort_by}'.", allowed=sorted(SORTABLE)
        )
    if min_score is not None and max_score is not None and min_score > max_score:
        raise ValidationError("min_score cannot exceed max_score")

    result = await repo.list_page(
        filters=CallFilter(
            search=search,
            bands=band,
            statuses=call_status,
            outcomes=[o.value for o in outcome] if outcome else None,
            owner=owner,
            min_score=min_score,
            max_score=max_score,
            created_after=created_after,
            created_before=created_before,
        ),
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_desc=sort_desc,
    )
    return PaginatedResponse(
        items=[CallSummaryResponse.from_domain(c) for c in result.items],
        pagination=_page_meta(result),
    )


@router.post(
    "",
    response_model=CallDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a transcript as JSON",
)
async def create_call(
    payload: CreateCallRequest,
    principal: RequireAnalyst,
    repo: CallRepoDep,
    audit: AuditRepoDep,
    engine: EngineDep,
    blobs: BlobStoreDep,
    settings: SettingsDep,
) -> CallDetailResponse:
    """Ingest and analyse a transcript supplied inline."""
    if len(payload.transcript.encode("utf-8")) > settings.max_upload_bytes:
        raise ValidationError(
            "Transcript exceeds the maximum size",
            max_bytes=settings.max_upload_bytes,
        )

    service = IngestionService(repo, blobs, engine, audit)
    call = await service.ingest(
        content=payload.transcript,
        title=payload.title,
        principal_id=principal.id,
        owner=payload.owner,
        tags=payload.tags,
    )
    return CallDetailResponse.model_validate(call, from_attributes=True)


@router.post(
    "/upload",
    response_model=CallDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a transcript file",
)
async def upload_call(
    principal: RequireAnalyst,
    repo: CallRepoDep,
    audit: AuditRepoDep,
    engine: EngineDep,
    blobs: BlobStoreDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Transcript: .txt, .json, .vtt, .srt")],
    title: Annotated[str | None, Form(max_length=200)] = None,
    owner: Annotated[str | None, Form(max_length=128)] = None,
    tags: Annotated[str | None, Form(description="Comma-separated")] = None,
) -> CallDetailResponse:
    """Ingest and analyse an uploaded transcript file.

    The body is read with an explicit size cap rather than trusting the
    Content-Length header, which a client controls.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(64 * 1024):
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise ValidationError(
                "Uploaded file exceeds the maximum size",
                max_bytes=settings.max_upload_bytes,
            )
        chunks.append(chunk)
    content = b"".join(chunks)

    if not content.strip():
        raise ValidationError("Uploaded file is empty")

    service = IngestionService(repo, blobs, engine, audit)
    call = await service.ingest(
        content=content,
        title=title or derive_title(file.filename),
        principal_id=principal.id,
        filename=file.filename,
        owner=owner,
        tags=[t.strip() for t in tags.split(",") if t.strip()] if tags else None,
    )
    return CallDetailResponse.model_validate(call, from_attributes=True)


@router.post(
    "/analyse",
    response_model=AnalyseTextResponse,
    summary="Analyse a transcript without storing it",
)
async def analyse_only(
    payload: AnalyseTextRequest,
    principal: RequireAnalyst,
    engine: EngineDep,
    settings: SettingsDep,
) -> AnalyseTextResponse:
    """Run the engine and return the result without persisting anything.

    Nothing touches the database or blob storage, which makes this safe for
    callers who are not permitted to store customer conversations here.
    """
    if len(payload.transcript.encode("utf-8")) > settings.max_upload_bytes:
        raise ValidationError(
            "Transcript exceeds the maximum size",
            max_bytes=settings.max_upload_bytes,
        )
    transcript, analysis = engine.analyse_text(
        payload.transcript, call_id=payload.call_id
    )
    return AnalyseTextResponse(analysis=analysis, transcript=transcript)


@router.get(
    "/{call_id}",
    response_model=CallDetailResponse,
    summary="Get one call with its full analysis",
)
async def get_call(
    call_id: str,
    principal: RequireViewer,
    repo: CallRepoDep,
    blobs: BlobStoreDep,
    include_transcript: Annotated[bool, Query()] = False,
) -> CallDetailResponse:
    """Retrieve a call.

    `call_id` is a database key, not a path fragment, so there is no path
    traversal surface here (issue #30). Blob keys are derived server-side from
    the validated id and never taken from user input.
    """
    call = await repo.get(call_id)
    response = CallDetailResponse.model_validate(call, from_attributes=True)

    if include_transcript:
        try:
            raw = await blobs.get(f"parsed/{call_id}")
            response.transcript = Transcript.model_validate(json.loads(raw))
        except NotFoundError:
            logger.info("Parsed transcript missing", extra={"call_id": call_id})
        except (ValueError, TypeError):
            logger.warning("Stored transcript is corrupt", extra={"call_id": call_id})

    return response


@router.patch(
    "/{call_id}",
    response_model=CallSummaryResponse,
    summary="Update call metadata",
)
async def update_call(
    call_id: str,
    payload: UpdateCallRequest,
    principal: RequireAnalyst,
    repo: CallRepoDep,
) -> CallSummaryResponse:
    call = await repo.update_metadata(
        call_id, title=payload.title, owner=payload.owner, tags=payload.tags
    )
    return CallSummaryResponse.from_domain(call)


@router.post(
    "/{call_id}/reanalyse",
    response_model=CallDetailResponse,
    summary="Re-run the current engine over a stored call",
)
async def reanalyse_call(
    call_id: str,
    principal: RequireAnalyst,
    repo: CallRepoDep,
    audit: AuditRepoDep,
    engine: EngineDep,
    blobs: BlobStoreDep,
) -> CallDetailResponse:
    """Re-analyse using the transcript already in storage.

    Runs synchronously: analysis is single-digit milliseconds, so a job queue
    would add operational weight for no user-visible gain. If a future analyzer
    makes this slow, the work moves to a worker and this returns 202.
    """
    service = IngestionService(repo, blobs, engine, audit)
    call = await service.reanalyse(call_id, principal.id)
    return CallDetailResponse.model_validate(call, from_attributes=True)


@router.delete(
    "/{call_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a call",
)
async def delete_call(
    call_id: str,
    principal: RequireAdmin,
    repo: CallRepoDep,
    audit: AuditRepoDep,
    engine: EngineDep,
    blobs: BlobStoreDep,
) -> Response:
    """Permanently delete a call and its stored transcript. Admin only, audited."""
    service = IngestionService(repo, blobs, engine, audit)
    await service.delete(call_id, principal.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
