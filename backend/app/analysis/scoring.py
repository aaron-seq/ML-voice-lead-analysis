"""Explainable lead scoring.

The score is a transparent weighted sum over qualification dimensions, with
explicit penalties for deal risks. Every component reports its raw value, its
weight and its contribution in points, so a rep who disagrees with a score can
see precisely which term produced it and inspect the evidence behind that term.

Why a linear model rather than a learned classifier
---------------------------------------------------
A supervised model needs labelled outcomes (did this lead convert?), which this
project has no dataset for. Shipping a black box trained on synthetic data --
as the previous ``ml/train.py`` did -- produces confident numbers with no
relationship to reality. A documented rubric is honest about being a rubric,
is auditable, and gives a sales team something they can argue with and tune.

``docs/SCORING.md`` documents the weights and the intended calibration.
``app/analysis/evaluation.py`` provides the harness for validating the rubric
against real labelled outcomes once a team has them.
"""

from __future__ import annotations

from app.domain.models import (
    ConversationDynamics,
    LeadBand,
    LeadScore,
    Objection,
    ObjectionCategory,
    RiskFlag,
    ScoreComponent,
    SentimentAnalysis,
    Signal,
    SignalType,
    TalkMetrics,
    Transcript,
)

# Positive components. Weights are points out of 100 and sum to 100.
# Ordered by how predictive each dimension is of a deal progressing, which is
# why an agreed next step outweighs enthusiasm by a factor of two.
WEIGHTS: dict[str, float] = {
    "need": 20.0,
    "next_step": 18.0,
    "budget": 15.0,
    "authority": 13.0,
    "timeline": 12.0,
    "engagement": 12.0,
    "sentiment": 10.0,
}

# Penalties, in points subtracted. Capped in aggregate by _MAX_PENALTY so a
# call cannot be driven to zero by risk flags alone.
PENALTIES: dict[RiskFlag, float] = {
    RiskFlag.NO_DECISION_MAKER: 10.0,
    RiskFlag.BUDGET_UNCONFIRMED: 6.0,
    RiskFlag.NO_NEXT_STEP: 12.0,
    RiskFlag.COMPETITOR_INCUMBENT: 8.0,
    RiskFlag.LOW_ENGAGEMENT: 10.0,
    RiskFlag.REP_MONOLOGUE: 6.0,
    RiskFlag.NEGATIVE_SENTIMENT_TREND: 8.0,
}

_MAX_PENALTY = 35.0

# Band thresholds. Documented in docs/SCORING.md; tuning these is the primary
# knob a sales team has for matching the score to their own funnel.
#
# COLD is the floor of the numeric scale. DISQUALIFIED is deliberately NOT a
# threshold band -- it is only ever set by an explicit rejection (see
# `_has_hard_disqualifier` and `outcome_is_rejection`). A call that simply
# produced few signals is a cold lead, not a dead one, and collapsing the two
# would make "show me disqualified leads" return every thin transcript.
BAND_THRESHOLDS: list[tuple[float, LeadBand]] = [
    (72.0, LeadBand.HOT),
    (48.0, LeadBand.WARM),
    (0.0, LeadBand.COLD),
]

# Signals that feed each qualification component.
_COMPONENT_SIGNALS: dict[str, tuple[SignalType, ...]] = {
    "need": (SignalType.NEED, SignalType.TECHNICAL_FIT),
    "next_step": (SignalType.NEXT_STEP,),
    "budget": (SignalType.BUDGET,),
    "authority": (SignalType.AUTHORITY,),
    "timeline": (SignalType.TIMELINE, SignalType.URGENCY),
}

# A hard disqualification overrides the arithmetic entirely: a prospect who
# said "not interested" is not a 40-point warm lead because they were polite.
_DISQUALIFYING = {ObjectionCategory.NEED, ObjectionCategory.COMPETITOR}


def score_lead(
    transcript: Transcript,
    signals: list[Signal],
    objections: list[Objection],
    sentiment: SentimentAnalysis,
    talk: TalkMetrics,
    outcome_is_rejection: bool = False,
    dynamics: ConversationDynamics | None = None,
) -> LeadScore:
    """Compute the lead score with a full audit trail.

    ``dynamics`` is optional so a caller with only a transcript still gets a
    score; when it is present and timed, talk balance is measured in seconds
    rather than words.
    """
    by_type = {s.type: s for s in signals}
    dynamics = dynamics or ConversationDynamics()

    components: list[ScoreComponent] = []
    for name, signal_types in _COMPONENT_SIGNALS.items():
        strengths = [by_type[t].strength for t in signal_types if t in by_type]
        raw = max(strengths) if strengths else 0.0
        components.append(
            _component(name, raw, _signal_detail(name, signal_types, by_type))
        )

    components.append(_component("engagement", *_engagement(talk, dynamics)))
    components.append(_component("sentiment", *_sentiment_component(sentiment)))

    risks = _detect_risks(by_type, objections, sentiment, talk, dynamics)
    penalties = [
        ScoreComponent(
            name=risk.value,
            raw=1.0,
            weight=-PENALTIES[risk],
            contribution=-PENALTIES[risk],
            detail=_RISK_DETAIL[risk],
        )
        for risk in risks
    ]

    positive = sum(c.contribution for c in components)
    penalty_total = min(_MAX_PENALTY, sum(-p.contribution for p in penalties))
    score = max(0.0, min(100.0, positive - penalty_total))

    band = _band(score)
    if outcome_is_rejection or _has_hard_disqualifier(objections):
        band = LeadBand.DISQUALIFIED

    return LeadScore(
        score=round(score, 1),
        band=band,
        confidence=_confidence(transcript, signals, sentiment),
        components=components,
        penalties=penalties,
        rationale=_rationale(components, penalties, band),
        risks=risks,
    )


def _component(name: str, raw: float, detail: str) -> ScoreComponent:
    weight = WEIGHTS[name]
    raw = max(0.0, min(1.0, raw))
    return ScoreComponent(
        name=name,
        raw=round(raw, 3),
        weight=weight,
        contribution=round(raw * weight, 2),
        detail=detail,
    )


def _signal_detail(
    name: str, signal_types: tuple[SignalType, ...], by_type: dict[SignalType, Signal]
) -> str:
    present = [by_type[t] for t in signal_types if t in by_type]
    if not present:
        return f"No {name.replace('_', ' ')} signal detected"
    best = max(present, key=lambda s: s.strength)
    if best.evidence:
        return f'Matched "{best.evidence[0].matched_phrase}" ({best.evidence[0].role})'
    return best.label


def _engagement(
    talk: TalkMetrics, dynamics: ConversationDynamics
) -> tuple[float, str]:
    """Engagement from how much, and how inquisitively, the prospect spoke.

    A prospect who talks a third to two-thirds of the time and asks questions
    is engaged. Both extremes are bad: silence means no buy-in, and a prospect
    who never lets the rep speak is usually venting, not buying.

    Measured in seconds when the transcript is timed, words otherwise. The
    distinction is not pedantry: word share and time share diverge whenever the
    two speakers talk at different rates, and published samples separate won
    from lost calls by around five points of *time* share -- a margin a biased
    proxy can easily swallow.
    """
    if dynamics.available and dynamics.rep_talk_time_ratio is not None:
        ratio = 1.0 - dynamics.rep_talk_time_ratio
        basis = "talk time"
    else:
        ratio = talk.prospect_talk_ratio
        basis = "words"

    if ratio <= 0.0:
        # Undiarised transcript -- no basis to judge, so score neutrally
        # rather than penalising a call for its file format.
        return 0.5, "Speaker roles unavailable; engagement scored neutrally"

    # Peak at 0.45 prospect share, falling off symmetrically.
    balance = max(0.0, 1.0 - abs(ratio - 0.45) / 0.45)
    inquisitiveness = min(1.0, talk.question_count_prospect / 5.0)
    raw = 0.65 * balance + 0.35 * inquisitiveness
    return raw, (
        f"Prospect spoke {ratio:.0%} of {basis} and asked "
        f"{talk.question_count_prospect} question(s)"
    )


def _sentiment_component(sentiment: SentimentAnalysis) -> tuple[float, str]:
    """Prospect sentiment, with the call's trajectory folded in.

    Trend matters as much as level: a call that starts sceptical and ends
    positive is a better sign than one that starts warm and cools.
    """
    level = sentiment.prospect if sentiment.prospect is not None else sentiment.overall
    raw = (level + 1.0) / 2.0
    raw += sentiment.trend * 0.15
    detail = f"Prospect sentiment {level:+.2f}, trend {sentiment.trend:+.2f}"
    return max(0.0, min(1.0, raw)), detail


# Seconds. 2.5 minutes is the stretch coaching guidance commonly names as the
# point at which a rep should stop and leave room for a reply. It is a
# guideline, not a validated predictor, and cold calls may well run the other
# way: the published cold-call research finds that *successful* cold calls
# contain longer rep bursts, not shorter. Flagged rather than heavily
# penalised, and called out in docs/SCORING.md as the next thing to calibrate.
_MONOLOGUE_SECONDS = 150.0

# Words. The fallback for untimed transcripts, at roughly 150 wpm the same
# 2.5 minutes. Kept so behaviour is unchanged for plain-text input.
_MONOLOGUE_WORDS = 400


def _is_monologue(talk: TalkMetrics, dynamics: ConversationDynamics) -> bool:
    """Whether the rep ran on long enough to flag it.

    Prefers seconds, because a 400-word stretch is 2 minutes from a fast
    talker and over 3 from a slow one, and it is the duration that costs the
    prospect their turn.
    """
    if dynamics.available and dynamics.longest_rep_monologue_s is not None:
        return dynamics.longest_rep_monologue_s > _MONOLOGUE_SECONDS
    return talk.longest_rep_monologue_words > _MONOLOGUE_WORDS


def _detect_risks(
    by_type: dict[SignalType, Signal],
    objections: list[Objection],
    sentiment: SentimentAnalysis,
    talk: TalkMetrics,
    dynamics: ConversationDynamics,
) -> list[RiskFlag]:
    risks: list[RiskFlag] = []

    if SignalType.NEXT_STEP not in by_type:
        risks.append(RiskFlag.NO_NEXT_STEP)

    authority = by_type.get(SignalType.AUTHORITY)
    has_authority_objection = any(
        o.category is ObjectionCategory.AUTHORITY for o in objections
    )
    if has_authority_objection or authority is None or authority.strength < 0.4:
        risks.append(RiskFlag.NO_DECISION_MAKER)

    budget = by_type.get(SignalType.BUDGET)
    if budget is None or budget.strength < 0.5:
        risks.append(RiskFlag.BUDGET_UNCONFIRMED)

    if any(o.category is ObjectionCategory.COMPETITOR and o.severity >= 0.6
           for o in objections):
        risks.append(RiskFlag.COMPETITOR_INCUMBENT)

    # Only judge talk balance when roles were actually inferred.
    if talk.prospect_talk_ratio > 0.0:
        if talk.prospect_talk_ratio < 0.15:
            risks.append(RiskFlag.LOW_ENGAGEMENT)
        if _is_monologue(talk, dynamics):
            risks.append(RiskFlag.REP_MONOLOGUE)

    if sentiment.trend < -0.25:
        risks.append(RiskFlag.NEGATIVE_SENTIMENT_TREND)

    return risks


_RISK_DETAIL: dict[RiskFlag, str] = {
    RiskFlag.NO_NEXT_STEP: "No concrete next step was agreed on the call",
    RiskFlag.NO_DECISION_MAKER: "Decision-making authority was not established",
    RiskFlag.BUDGET_UNCONFIRMED: "Budget was not confirmed",
    RiskFlag.COMPETITOR_INCUMBENT: "An incumbent competitor is in place",
    RiskFlag.LOW_ENGAGEMENT: "Prospect contributed very little to the conversation",
    RiskFlag.REP_MONOLOGUE: "Rep delivered an extended monologue",
    RiskFlag.NEGATIVE_SENTIMENT_TREND: "Prospect sentiment declined over the call",
}


def _has_hard_disqualifier(objections: list[Objection]) -> bool:
    return any(
        o.category in _DISQUALIFYING and o.severity >= 0.85 for o in objections
    )


def _band(score: float) -> LeadBand:
    for threshold, band in BAND_THRESHOLDS:
        if score >= threshold:
            return band
    return LeadBand.COLD


def _confidence(
    transcript: Transcript, signals: list[Signal], sentiment: SentimentAnalysis
) -> float:
    """How much the score can be trusted.

    Kept deliberately separate from the score itself. A 20-word voicemail can
    produce a high score from one lucky phrase; reporting that at confidence
    0.3 is honest, whereas folding it into the score would hide the problem.
    """
    volume = min(1.0, transcript.word_count / 600.0)
    breadth = min(1.0, len(signals) / 6.0)
    diarised = 1.0 if any(s.role.value != "unknown" for s in transcript.segments) else 0.4
    timed = 1.0 if transcript.duration_s is not None else 0.8

    raw = (
        0.35 * volume
        + 0.30 * breadth
        + 0.20 * diarised
        + 0.10 * sentiment.confidence
        + 0.05 * timed
    )
    return round(max(0.05, min(1.0, raw)), 3)


def _rationale(
    components: list[ScoreComponent],
    penalties: list[ScoreComponent],
    band: LeadBand,
) -> list[str]:
    """Plain-English drivers, highest absolute impact first."""
    lines: list[str] = []
    if band is LeadBand.DISQUALIFIED:
        lines.append("Disqualified: the prospect rejected the offering on this call.")

    ranked = sorted(components, key=lambda c: c.contribution, reverse=True)
    for component in ranked[:3]:
        if component.contribution <= 0.5:
            continue
        lines.append(
            f"{component.name.replace('_', ' ').capitalize()} contributed "
            f"{component.contribution:.1f} of {component.weight:.0f} points — "
            f"{component.detail}."
        )

    for penalty in sorted(penalties, key=lambda p: p.contribution)[:2]:
        lines.append(f"{penalty.detail} ({penalty.contribution:.0f} points).")

    if not lines:
        lines.append("No meaningful qualification signals were detected on this call.")
    return lines
