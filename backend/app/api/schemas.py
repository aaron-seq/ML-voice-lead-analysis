"""HTTP request/response schemas.

Thin wrappers over the domain models. They exist so the wire format can carry
pagination envelopes and request validation without the domain contract
growing HTTP concerns.
"""

from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.models import (
    AnalysisStatus,
    Call,
    CallAnalysis,
    CallOutcomeLabel,
    LeadBand,
    Transcript,
)

T = TypeVar("T")

MAX_TITLE_LENGTH = 200
MAX_TAGS = 20


class PageMeta(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    pagination: PageMeta


class CallSummaryResponse(BaseModel):
    """List-view projection. Excludes the full analysis blob by design."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    status: AnalysisStatus
    source_format: str
    lead_score: float | None = None
    lead_band: LeadBand | None = None
    outcome: CallOutcomeLabel | None = None
    sentiment: float | None = None
    duration_s: float | None = None
    word_count: int = 0
    owner: str | None = None
    tags: list[str] = Field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_domain(cls, call: Call) -> CallSummaryResponse:
        return cls.model_validate(call, from_attributes=True)


class CallDetailResponse(BaseModel):
    """Full call view, including the analysis and optionally the transcript."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    status: AnalysisStatus
    source_format: str
    source_uri: str | None = None
    lead_score: float | None = None
    lead_band: LeadBand | None = None
    outcome: CallOutcomeLabel | None = None
    sentiment: float | None = None
    duration_s: float | None = None
    word_count: int = 0
    owner: str | None = None
    tags: list[str] = Field(default_factory=list)
    error: str | None = None
    analysis: CallAnalysis | None = None
    transcript: Transcript | None = None


class CreateCallRequest(BaseModel):
    """Submit a transcript as JSON. For files, use the multipart endpoint."""

    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    transcript: str = Field(min_length=1, description="Raw transcript in any "
                                                      "supported format")
    owner: str | None = Field(None, max_length=128)
    tags: list[str] = Field(default_factory=list, max_length=MAX_TAGS)

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, tags: list[str]) -> list[str]:
        cleaned = [t.strip()[:40] for t in tags if t and t.strip()]
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(cleaned))


class UpdateCallRequest(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=MAX_TITLE_LENGTH)
    owner: str | None = Field(None, max_length=128)
    tags: list[str] | None = Field(None, max_length=MAX_TAGS)


class AnalyseTextRequest(BaseModel):
    """Stateless analysis: run the engine without persisting anything.

    Useful for trying the product, for CI checks against a golden transcript,
    and for callers who cannot store customer conversations on this system.
    """

    transcript: str = Field(min_length=1)
    call_id: str = Field("adhoc", max_length=64)


class AnalyseTextResponse(BaseModel):
    analysis: CallAnalysis
    transcript: Transcript


class ErrorResponse(BaseModel):
    code: str
    message: str
    context: dict = Field(default_factory=dict)
    request_id: str | None = None


class HealthComponent(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: str = Field(description="'healthy', 'degraded' or 'unhealthy'")
    version: str
    environment: str
    uptime_seconds: float
    components: dict[str, HealthComponent]
