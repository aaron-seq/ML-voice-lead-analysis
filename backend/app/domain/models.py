"""Canonical domain contract for CallSense.

Every layer -- analysis engine, persistence, HTTP API, CLI -- speaks these
models. There is exactly one field-naming convention (snake_case) and one
serialisation format, which is what makes an analysis produced by the CLI
readable by the API without a translation step.

Historically the pipeline emitted ``analyzed_file_name``/``lead_scoring_results``
while the API deserialised ``fileName``/``leadScore``, so no real analysis was
ever readable through the API (issue #23). These models are the fix: they are
the only place a call analysis is defined.

All timestamps are timezone-aware UTC (issue #26).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

# Bumped whenever a change to the engine alters scores for identical input.
# Persisted with every analysis so results can be compared like-for-like and
# stale rows can be selectively re-analysed.
#
# 1.1.0: engagement and the rep_monologue risk now measure talk balance in
# seconds when the transcript carries timings, having previously used word
# share as a proxy. Scores for timed transcripts move; scores for untimed ones
# are unchanged.
#
# 1.2.0: a rejection the prospect later reverses by agreeing a concrete meeting
# no longer disqualifies the call (issue #40). Calls with that pattern move from
# DISQUALIFIED to their computed band; nothing else moves.
ENGINE_VERSION = "1.2.0"

# Bumped whenever the shape of these models changes incompatibly. Adding
# `CallAnalysis.dynamics` with a default is additive -- rows written under
# version 1 still deserialise -- so this stays put.
SCHEMA_VERSION = 1


def utcnow() -> datetime:
    """Timezone-aware UTC now. The only clock the codebase should use."""
    return datetime.now(UTC)


# Note: ``enum.StrEnum`` (Python 3.11+) already serialises as its value and
# compares equal to a plain string, which is exactly what the JSON contract
# and the TypeScript literal unions on the frontend rely on.


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class SpeakerRole(StrEnum):
    """Who is talking.

    Assigned heuristically by the engine (see analysis.diarization). ``UNKNOWN``
    is used rather than guessing when the evidence is weak.
    """

    REP = "rep"
    PROSPECT = "prospect"
    UNKNOWN = "unknown"


class LeadBand(StrEnum):
    """Coarse bucket derived from the numeric lead score."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    DISQUALIFIED = "disqualified"


class SignalType(StrEnum):
    """Categories of buying signal the engine detects.

    The first six map to the classic BANT/MEDDIC qualification dimensions,
    which is what makes the lead score explainable to a sales manager rather
    than an opaque number.
    """

    BUDGET = "budget"
    AUTHORITY = "authority"
    NEED = "need"
    TIMELINE = "timeline"
    URGENCY = "urgency"
    COMPETITOR = "competitor"
    ENTHUSIASM = "enthusiasm"
    NEXT_STEP = "next_step"
    TECHNICAL_FIT = "technical_fit"
    SOCIAL_PROOF = "social_proof"


class ObjectionCategory(StrEnum):
    """Objection taxonomy. Each has a documented handling playbook."""

    PRICE = "price"
    TIMING = "timing"
    AUTHORITY = "authority"
    COMPETITOR = "competitor"
    NEED = "need"
    TRUST = "trust"
    COMPLEXITY = "complexity"


class RiskFlag(StrEnum):
    """Deal risks that should suppress an otherwise-high score."""

    NO_DECISION_MAKER = "no_decision_maker"
    BUDGET_UNCONFIRMED = "budget_unconfirmed"
    NO_NEXT_STEP = "no_next_step"
    COMPETITOR_INCUMBENT = "competitor_incumbent"
    LOW_ENGAGEMENT = "low_engagement"
    REP_MONOLOGUE = "rep_monologue"
    NEGATIVE_SENTIMENT_TREND = "negative_sentiment_trend"


class CallOutcomeLabel(StrEnum):
    """What actually happened on the call."""

    MEETING_BOOKED = "meeting_booked"
    FOLLOW_UP_AGREED = "follow_up_agreed"
    INFORMATION_REQUESTED = "information_requested"
    NOT_INTERESTED = "not_interested"
    NOT_A_FIT = "not_a_fit"
    GATEKEEPER_ONLY = "gatekeeper_only"
    NO_CLEAR_OUTCOME = "no_clear_outcome"


class ActionPriority(StrEnum):
    P0 = "p0"
    P1 = "p1"
    P2 = "p2"


class AnalysisStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# --------------------------------------------------------------------------
# Transcript primitives
# --------------------------------------------------------------------------


class DomainModel(BaseModel):
    """Base with strict-ish validation and stable serialisation."""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=False,
        populate_by_name=True,
        ser_json_timedelta="float",
    )


class TranscriptSegment(DomainModel):
    """One contiguous utterance by one speaker."""

    index: int = Field(ge=0, description="Position in the transcript, 0-based")
    speaker: str = Field(description="Raw speaker label from the source, e.g. 'spk_0'")
    role: SpeakerRole = SpeakerRole.UNKNOWN
    text: str
    start_s: float | None = Field(
        None, ge=0, description="Offset from call start, seconds. None if untimed."
    )
    end_s: float | None = Field(None, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_s(self) -> float | None:
        if self.start_s is None or self.end_s is None:
            return None
        return max(0.0, self.end_s - self.start_s)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def word_count(self) -> int:
        return len(self.text.split())


class Transcript(DomainModel):
    """A parsed call transcript, normalised from any supported source format."""

    source_format: str = Field(description="Detected input format, e.g. 'aws_transcribe'")
    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: str = "en"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def text(self) -> str:
        return "\n".join(seg.text for seg in self.segments)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def word_count(self) -> int:
        return sum(seg.word_count for seg in self.segments)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_s(self) -> float | None:
        """Wall-clock duration if the source carried timings, else None.

        Deliberately returns None rather than estimating from word count --
        a fabricated duration silently corrupts every downstream per-minute
        metric. Callers that want an estimate must ask for one explicitly.
        """
        ends = [s.end_s for s in self.segments if s.end_s is not None]
        starts = [s.start_s for s in self.segments if s.start_s is not None]
        if not ends or not starts:
            return None
        return max(ends) - min(starts)

    def estimated_duration_s(self, words_per_minute: float = 150.0) -> float:
        """Duration estimate for untimed transcripts. Explicitly opt-in."""
        actual = self.duration_s
        if actual is not None:
            return actual
        return (self.word_count / words_per_minute) * 60.0


class Evidence(DomainModel):
    """A pointer back into the transcript justifying a finding.

    Every signal, objection and risk carries evidence. This is what separates
    a defensible analysis from a plausible-looking guess: a user can always
    ask "why did you say that?" and get the exact quote.
    """

    segment_index: int = Field(ge=0)
    quote: str = Field(description="The matched text, trimmed to a readable span")
    speaker: str
    role: SpeakerRole = SpeakerRole.UNKNOWN
    start_s: float | None = None
    matched_phrase: str | None = Field(
        None, description="The lexicon entry or pattern that fired"
    )


# --------------------------------------------------------------------------
# Analysis components
# --------------------------------------------------------------------------


class SentimentPoint(DomainModel):
    """Sentiment at one point on the call timeline."""

    position: float = Field(
        ge=0.0, le=1.0, description="Fractional position through the call (0=start)"
    )
    at_s: float | None = None
    score: float = Field(ge=-1.0, le=1.0)
    segment_index: int = Field(ge=0)


class SentimentAnalysis(DomainModel):
    overall: float = Field(ge=-1.0, le=1.0)
    prospect: float | None = Field(
        None, ge=-1.0, le=1.0, description="Sentiment of prospect turns only"
    )
    rep: float | None = Field(None, ge=-1.0, le=1.0)
    timeline: list[SentimentPoint] = Field(default_factory=list)
    trend: float = Field(
        0.0,
        description="Slope of prospect sentiment across the call. "
        "Positive = warming up, negative = cooling off.",
    )
    volatility: float = Field(0.0, ge=0.0, description="Std-dev of the timeline")
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class Signal(DomainModel):
    """A detected buying signal with its supporting evidence."""

    type: SignalType
    label: str = Field(description="Human-readable one-liner")
    strength: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)


class Objection(DomainModel):
    """A prospect concern, with a suggested way to handle it."""

    category: ObjectionCategory
    label: str
    severity: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    suggested_response: str = Field(
        description="Playbook guidance for the rep. Static per category -- "
        "this is a prompt for the human, not a generated answer."
    )
    addressed: bool = Field(
        False,
        description="True when the rep visibly responded to this objection "
        "later in the call.",
    )


class ScoreComponent(DomainModel):
    """One weighted term in the lead score.

    Exposing the arithmetic is the point: a rep who disagrees with a score can
    see exactly which component drove it and check the evidence behind it.
    """

    name: str
    raw: float = Field(ge=0.0, le=1.0, description="Normalised component value")
    weight: float = Field(description="Points this component can contribute")
    contribution: float = Field(description="raw * weight, in score points")
    detail: str = Field("", description="Why this component scored as it did")


class LeadScore(DomainModel):
    score: float = Field(ge=0.0, le=100.0)
    band: LeadBand
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How much evidence the score rests on. Low confidence on a "
        "high score means 'promising but thin' -- not 'probably hot'.",
    )
    components: list[ScoreComponent] = Field(default_factory=list)
    penalties: list[ScoreComponent] = Field(default_factory=list)
    rationale: list[str] = Field(
        default_factory=list, description="Plain-English drivers, highest impact first"
    )
    risks: list[RiskFlag] = Field(default_factory=list)


class CallOutcome(DomainModel):
    label: CallOutcomeLabel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    superseded_rejection: Evidence | None = Field(
        None,
        description="A rejection the prospect voiced and then reversed by "
        "agreeing a concrete meeting later in the call. Set only when that "
        "override decided the outcome, so a reviewer can check the call.",
    )


class NextAction(DomainModel):
    """A concrete thing the rep should do next."""

    action: str
    priority: ActionPriority
    rationale: str
    due_within_hours: int = Field(gt=0)


class CallSummary(DomainModel):
    headline: str = Field(description="One sentence a manager can skim")
    bullets: list[str] = Field(default_factory=list)
    key_quotes: list[Evidence] = Field(default_factory=list)


class TalkMetrics(DomainModel):
    """Who spoke, how much, and how inquisitively.

    Ratios here are shares of *words*, which is all an untimed transcript
    supports. When the source carries timings, ``ConversationDynamics`` reports
    the same balance in seconds -- the quantity the industry actually means by
    "talk-to-listen ratio" -- and the scoring rubric prefers it.
    """

    rep_talk_ratio: float = Field(
        0.0, ge=0.0, le=1.0, description="Share of words spoken by the rep"
    )
    prospect_talk_ratio: float = Field(0.0, ge=0.0, le=1.0)
    longest_rep_monologue_words: int = Field(0, ge=0)
    question_count_rep: int = Field(0, ge=0)
    question_count_prospect: int = Field(0, ge=0)
    turn_count: int = Field(0, ge=0)
    segments_analysed: int = Field(0, ge=0)


class ConversationDynamics(DomainModel):
    """Timing-derived conversation dynamics.

    Every field is ``None`` when the transcript carried no usable timings;
    ``available`` says which case you are in and ``unavailable_reason`` says
    why. Nothing here is estimated from word counts -- a fabricated duration
    would corrupt every per-minute figure that reads it.

    None of these feed the lead score except the two with published outcome
    associations (talk-time ratio, monologue length), which refine the
    ``engagement`` component and the ``rep_monologue`` risk rather than adding
    weight of their own. See ``app/analysis/dynamics.py`` for why.
    """

    available: bool = False
    unavailable_reason: str | None = None

    call_span_s: float | None = Field(None, ge=0)
    rep_talk_seconds: float | None = Field(None, ge=0)
    prospect_talk_seconds: float | None = Field(None, ge=0)
    rep_talk_time_ratio: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Rep's share of attributed speaking time, not of words",
    )
    longest_rep_monologue_s: float | None = Field(None, ge=0)
    longest_prospect_turn_s: float | None = Field(None, ge=0)

    turn_count: int = Field(0, ge=0, description="Speaker turns, not subtitle cues")
    interruptions_by_rep: int = Field(
        0, ge=0, description="Times the rep began speaking while the prospect was"
    )
    interruptions_by_prospect: int = Field(0, ge=0)
    overtalk_seconds: float = Field(
        0.0, ge=0, description="Total airtime with both parties speaking"
    )

    rep_speech_rate_wpm: float | None = Field(None, ge=0)
    prospect_speech_rate_wpm: float | None = Field(None, ge=0)
    rep_response_latency_s: float | None = Field(
        None, ge=0,
        description="Mean silence before the rep replies -- how much room they "
        "leave for an answer",
    )
    prospect_response_latency_s: float | None = Field(
        None, ge=0, description="Mean silence before the prospect replies"
    )
    dead_air_ratio: float | None = Field(
        None, ge=0.0, le=1.0, description="Share of the call with nobody speaking"
    )
    prospect_share_by_third: list[float] | None = Field(
        None,
        description="Prospect's share of talk time in each third of the call, "
        "for reading engagement trajectory rather than its average",
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Plain-language notes, each naming the number behind it",
    )


class AnalysisMetadata(DomainModel):
    engine_version: str = ENGINE_VERSION
    schema_version: int = SCHEMA_VERSION
    analyzers: dict[str, str] = Field(
        default_factory=dict, description="analyzer name -> implementation used"
    )
    processing_ms: float = Field(0.0, ge=0.0)
    word_count: int = Field(0, ge=0)
    duration_s: float | None = None
    duration_is_estimated: bool = False
    warnings: list[str] = Field(default_factory=list)


class CallAnalysis(DomainModel):
    """The complete result of analysing one call. The product's core artifact."""

    call_id: str
    analyzed_at: datetime = Field(default_factory=utcnow)

    summary: CallSummary
    sentiment: SentimentAnalysis
    signals: list[Signal] = Field(default_factory=list)
    objections: list[Objection] = Field(default_factory=list)
    lead_score: LeadScore
    outcome: CallOutcome
    next_actions: list[NextAction] = Field(default_factory=list)
    talk_metrics: TalkMetrics
    dynamics: ConversationDynamics = Field(
        default_factory=ConversationDynamics,
        description="Timing-derived dynamics; `available` is false for untimed "
        "transcripts. Defaulted so analyses stored under schema 1 still load.",
    )
    topics: list[str] = Field(default_factory=list)
    metadata: AnalysisMetadata

    def signal_types(self) -> set[SignalType]:
        return {s.type for s in self.signals}


class Call(DomainModel):
    """A call record: source material plus its latest analysis."""

    id: str
    title: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    source_format: str = "text"
    source_uri: str | None = Field(
        None, description="Where the raw transcript lives in blob storage"
    )
    audio_uri: str | None = None

    # Denormalised for cheap list/filter queries. Always derived from `analysis`.
    status: AnalysisStatus = AnalysisStatus.PENDING
    lead_score: float | None = None
    lead_band: LeadBand | None = None
    outcome: CallOutcomeLabel | None = None
    sentiment: float | None = None
    duration_s: float | None = None
    word_count: int = 0
    error: str | None = None

    owner: str | None = Field(None, description="Rep who owns the call")
    tags: list[str] = Field(default_factory=list)

    analysis: CallAnalysis | None = None


# --------------------------------------------------------------------------
# Aggregate analytics
# --------------------------------------------------------------------------


class BandBreakdown(DomainModel):
    hot: int = 0
    warm: int = 0
    cold: int = 0
    disqualified: int = 0


class TrendPoint(DomainModel):
    date: str = Field(description="ISO date, UTC")
    calls: int = 0
    avg_lead_score: float | None = None
    avg_sentiment: float | None = None


class PipelineAnalytics(DomainModel):
    """Portfolio-level view across many calls -- the manager's screen."""

    total_calls: int = 0
    analysed_calls: int = 0
    failed_calls: int = 0
    avg_lead_score: float | None = None
    avg_sentiment: float | None = None
    band_breakdown: BandBreakdown = Field(default_factory=BandBreakdown)
    outcome_breakdown: dict[str, int] = Field(default_factory=dict)
    top_objections: list[dict[str, Any]] = Field(default_factory=list)
    top_signals: list[dict[str, Any]] = Field(default_factory=list)
    top_risks: list[dict[str, Any]] = Field(default_factory=list)
    trend: list[TrendPoint] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)
