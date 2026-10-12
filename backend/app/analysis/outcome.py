"""Call outcome classification, summarisation and next-best-action.

These three are grouped because they all answer "so what?" rather than "what
was said": they turn the raw analysis into something a rep can act on before
their next call.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import pairwise

from app.analysis.lexicons import OUTCOME_PHRASES, STOPWORDS
from app.analysis.text import PhraseMatcher, is_question, tokenize, truncate_quote
from app.domain.models import (
    ActionPriority,
    CallOutcome,
    CallOutcomeLabel,
    CallSummary,
    Evidence,
    LeadBand,
    LeadScore,
    NextAction,
    Objection,
    ObjectionCategory,
    RiskFlag,
    Signal,
    SignalType,
    SpeakerRole,
    TalkMetrics,
    Transcript,
)

_OUTCOME_MATCHER = PhraseMatcher([
    (phrase, weight, label)
    for label, entries in OUTCOME_PHRASES.items()
    for phrase, weight in entries
])

# Rejections are decisive regardless of anything else agreed on the call, so
# they outrank the positive outcomes when both fire.
_OUTCOME_PRECEDENCE: dict[str, int] = {
    "not_interested": 5,
    "not_a_fit": 5,
    "gatekeeper_only": 4,
    "meeting_booked": 3,
    "follow_up_agreed": 2,
    "information_requested": 1,
}

# The closing portion of a call is where outcomes are actually agreed; matches
# there are weighted more heavily than the same phrase mid-discovery.
_CLOSING_FRACTION = 0.35
_CLOSING_BOOST = 1.35

REJECTION_OUTCOMES = frozenset({
    CallOutcomeLabel.NOT_INTERESTED,
    CallOutcomeLabel.NOT_A_FIT,
})

# "Not a fit" and "doesn't fit" are transitive: they need an object, and when
# the object is elided ("it doesn't fit", "we're not a fit") that object is the
# offering, which is a rejection. When an explicit object follows naming a date
# or a schedule, the speaker is rejecting a *timeline*, and that is frequently
# the opposite of a rejection.
#
# Found by a real transcript: "The other vendor wanted six weeks for a POC,
# which doesn't fit February at all" classified the call as NOT_A_FIT, and
# because a rejection hard-overrides the band, a call with approved budget, a
# named deadline and a booked next step came out DISQUALIFIED. The prospect was
# saying a competitor was too slow.
_SCHEDULE_OBJECTS: frozenset[str] = frozenset({
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "q1", "q2", "q3", "q4",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "timeline", "timelines", "timescale", "timescales", "schedule", "schedules",
    "deadline", "deadlines", "date", "dates", "window", "timeframe",
    "quarter", "month", "week", "sprint",
})

# How far past the match to look for that object. Enough for "doesn't fit our
# timeline" and "doesn't fit the February deadline", not enough to reach into
# the next clause.
_OBJECT_LOOKAHEAD = 3


def _rejection_targets_a_schedule(tokens: list[str], end: int) -> bool:
    """Whether a rejection phrase is aimed at a date rather than the offering."""
    return any(
        token in _SCHEDULE_OBJECTS
        for token in tokens[end:end + _OBJECT_LOOKAHEAD]
    )


def classify_outcome(transcript: Transcript) -> CallOutcome:
    """Determine what the call actually resulted in."""
    segments = transcript.segments
    if not segments:
        return CallOutcome(label=CallOutcomeLabel.NO_CLEAR_OUTCOME, confidence=0.0)

    closing_start = int(len(segments) * (1.0 - _CLOSING_FRACTION))
    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[Evidence]] = defaultdict(list)

    for seg in segments:
        tokens = tokenize(seg.text)
        for match in _OUTCOME_MATCHER.find(tokens):
            label: str = match.key  # type: ignore[assignment]
            weight = match.weight

            # A rejection aimed at a date is not a rejection of the offering,
            # and a rejection outranks every positive outcome, so letting one
            # through costs the whole call's classification.
            if (
                CallOutcomeLabel(label) in REJECTION_OUTCOMES
                and _rejection_targets_a_schedule(tokens, match.end)
            ):
                continue

            if seg.index >= closing_start:
                weight *= _CLOSING_BOOST
            # "Send me the deck" from the prospect is a real request; the rep
            # offering to send something is not an outcome on its own.
            if label == "information_requested" and seg.role is SpeakerRole.REP:
                weight *= 0.5
            scores[label] += weight
            evidence[label].append(
                Evidence(
                    segment_index=seg.index,
                    quote=truncate_quote(seg.text),
                    speaker=seg.speaker,
                    role=seg.role,
                    start_s=seg.start_s,
                    matched_phrase=match.phrase,
                )
            )

    if not scores:
        return CallOutcome(label=CallOutcomeLabel.NO_CLEAR_OUTCOME, confidence=0.35)

    best = max(
        scores,
        key=lambda label: (_OUTCOME_PRECEDENCE.get(label, 0), scores[label]),
    )
    total = sum(scores.values())
    # Confidence blends share-of-evidence with absolute strength, so a single
    # weak match does not report 100% just for being unopposed.
    share = scores[best] / total if total else 0.0
    confidence = min(0.97, 0.45 * share + 0.55 * min(1.0, scores[best] / 1.5))

    return CallOutcome(
        label=CallOutcomeLabel(best),
        confidence=round(confidence, 3),
        evidence=evidence[best][:3],
    )


def summarise(
    transcript: Transcript,
    signals: list[Signal],
    objections: list[Objection],
    outcome: CallOutcome,
    lead_score: LeadScore,
    talk: TalkMetrics,
) -> CallSummary:
    """Build a skimmable summary.

    Extractive, not generative: every quote is verbatim from the transcript.
    A hallucinated summary of a sales call is worse than no summary, because
    a rep will act on it.
    """
    band = lead_score.band.value
    outcome_text = outcome.label.value.replace("_", " ")
    headline = (
        f"{band.capitalize()} lead ({lead_score.score:.0f}/100) — {outcome_text}."
    )

    bullets: list[str] = []

    top_signals = [s for s in signals if s.strength >= 0.5][:3]
    if top_signals:
        bullets.append(
            "Qualification signals: "
            + ", ".join(s.label.lower() for s in top_signals)
            + "."
        )
    else:
        bullets.append("No strong qualification signals were detected.")

    unaddressed = [o for o in objections if not o.addressed]
    if unaddressed:
        bullets.append(
            f"{len(unaddressed)} unaddressed objection(s): "
            + ", ".join(o.category.value for o in unaddressed)
            + "."
        )
    elif objections:
        bullets.append(
            f"{len(objections)} objection(s) raised, all addressed on the call."
        )

    if talk.prospect_talk_ratio > 0:
        bullets.append(
            f"Talk balance: rep {talk.rep_talk_ratio:.0%} / prospect "
            f"{talk.prospect_talk_ratio:.0%}, with "
            f"{talk.question_count_rep} discovery question(s) from the rep."
        )

    if lead_score.risks:
        bullets.append(
            "Risks: "
            + ", ".join(r.value.replace("_", " ") for r in lead_score.risks[:3])
            + "."
        )

    return CallSummary(
        headline=headline,
        bullets=bullets,
        key_quotes=_key_quotes(signals, objections, outcome),
    )


def _key_quotes(
    signals: list[Signal], objections: list[Objection], outcome: CallOutcome
) -> list[Evidence]:
    """The handful of lines that most justify the analysis, deduplicated."""
    candidates: list[tuple[float, Evidence]] = []
    for signal in signals[:4]:
        candidates.extend((signal.strength, ev) for ev in signal.evidence[:1])
    for objection in objections[:3]:
        candidates.extend((objection.severity, ev) for ev in objection.evidence[:1])
    candidates.extend((0.95, ev) for ev in outcome.evidence[:1])

    seen: set[int] = set()
    quotes: list[Evidence] = []
    for _, evidence in sorted(candidates, key=lambda item: item[0], reverse=True):
        if evidence.segment_index in seen:
            continue
        seen.add(evidence.segment_index)
        quotes.append(evidence)
        if len(quotes) >= 5:
            break
    return quotes


def recommend_actions(
    lead_score: LeadScore,
    signals: list[Signal],
    objections: list[Objection],
    outcome: CallOutcome,
) -> list[NextAction]:
    """Rules that convert the analysis into a prioritised to-do list.

    Ordering is P0 first, then by urgency. The rules are intentionally few and
    legible -- a rep should be able to predict what the system will tell them,
    which is what makes them trust it when it says something unexpected.
    """
    actions: list[NextAction] = []
    signal_types = {s.type for s in signals}

    if outcome.label in REJECTION_OUTCOMES:
        actions.append(NextAction(
            action="Mark the lead closed-lost and record the disqualification reason",
            priority=ActionPriority.P2,
            rationale="The prospect explicitly rejected the offering on this call.",
            due_within_hours=72,
        ))
        return actions

    if outcome.label is CallOutcomeLabel.GATEKEEPER_ONLY:
        actions.append(NextAction(
            action="Identify and route to the actual decision maker",
            priority=ActionPriority.P1,
            rationale="The call reached a gatekeeper rather than a buyer.",
            due_within_hours=48,
        ))
        return actions

    if outcome.label is CallOutcomeLabel.MEETING_BOOKED:
        actions.append(NextAction(
            action="Send the calendar invite with a written agenda",
            priority=ActionPriority.P0,
            rationale="A meeting was agreed; confirming it in writing protects "
                      "against no-shows.",
            due_within_hours=4,
        ))
    elif RiskFlag.NO_NEXT_STEP in lead_score.risks and lead_score.band in (
        LeadBand.HOT, LeadBand.WARM
    ):
        actions.append(NextAction(
            action="Propose a specific next meeting time by email today",
            priority=ActionPriority.P0,
            rationale="A qualified call ended without an agreed next step, which "
                      "is the most common way warm leads go cold.",
            due_within_hours=8,
        ))

    if outcome.label is CallOutcomeLabel.INFORMATION_REQUESTED:
        actions.append(NextAction(
            action="Send the requested material with a summary of their stated needs",
            priority=ActionPriority.P0,
            rationale="The prospect asked for information; responding same-day "
                      "keeps momentum.",
            due_within_hours=6,
        ))

    for objection in objections:
        if objection.addressed or objection.severity < 0.55:
            continue
        actions.append(NextAction(
            action=f"Follow up on the unaddressed {objection.category.value} objection",
            priority=(
                ActionPriority.P0 if objection.severity >= 0.8 else ActionPriority.P1
            ),
            rationale=objection.suggested_response,
            due_within_hours=24 if objection.severity >= 0.8 else 48,
        ))

    if RiskFlag.NO_DECISION_MAKER in lead_score.risks:
        actions.append(NextAction(
            action="Map the buying committee and request an introduction to the "
                   "economic buyer",
            priority=ActionPriority.P1,
            rationale="No decision-making authority was established on this call.",
            due_within_hours=48,
        ))

    if RiskFlag.BUDGET_UNCONFIRMED in lead_score.risks and lead_score.band in (
        LeadBand.HOT, LeadBand.WARM
    ):
        actions.append(NextAction(
            action="Confirm budget ownership and approval process",
            priority=ActionPriority.P1,
            rationale="The lead looks qualified on need but budget is unverified.",
            due_within_hours=72,
        ))

    if SignalType.COMPETITOR in signal_types and not any(
        o.category is ObjectionCategory.COMPETITOR for o in objections
    ):
        actions.append(NextAction(
            action="Send a targeted comparison against the incumbent",
            priority=ActionPriority.P2,
            rationale="A competitor was referenced but not raised as a blocker.",
            due_within_hours=72,
        ))

    if not actions:
        actions.append(NextAction(
            action="Log the call and schedule a nurture touch",
            priority=ActionPriority.P2,
            rationale="No urgent follow-up was identified from this conversation.",
            due_within_hours=120,
        ))

    order = {ActionPriority.P0: 0, ActionPriority.P1: 1, ActionPriority.P2: 2}
    actions.sort(key=lambda a: (order[a.priority], a.due_within_hours))
    return actions[:6]


def compute_talk_metrics(transcript: Transcript) -> TalkMetrics:
    """Conversation dynamics, computed from words rather than time.

    Word counts are used because most transcripts lack reliable per-segment
    timings, and word share is a good proxy for time share at conversational
    speeds.
    """
    rep_words = prospect_words = 0
    rep_questions = prospect_questions = 0
    longest_monologue = 0
    current_monologue = 0
    turns = 0
    previous_speaker: str | None = None

    for seg in transcript.segments:
        if seg.speaker != previous_speaker:
            turns += 1
            previous_speaker = seg.speaker

        question = is_question(seg.text)
        if seg.role is SpeakerRole.REP:
            rep_words += seg.word_count
            rep_questions += question
            current_monologue += seg.word_count
            longest_monologue = max(longest_monologue, current_monologue)
        else:
            current_monologue = 0
            if seg.role is SpeakerRole.PROSPECT:
                prospect_words += seg.word_count
                prospect_questions += question

    total = rep_words + prospect_words
    return TalkMetrics(
        rep_talk_ratio=round(rep_words / total, 4) if total else 0.0,
        prospect_talk_ratio=round(prospect_words / total, 4) if total else 0.0,
        longest_rep_monologue_words=longest_monologue,
        question_count_rep=rep_questions,
        question_count_prospect=prospect_questions,
        turn_count=turns,
        segments_analysed=len(transcript.segments),
    )


def extract_topics(transcript: Transcript, limit: int = 10) -> list[str]:
    """Salient topics via stopword-filtered bigram and unigram frequency.

    Bigrams are preferred because "data migration" is a topic and "data" is
    not. A unigram only survives if it is frequent and not already covered by
    a selected bigram.
    """
    tokens = [t for t in tokenize(transcript.text) if t not in STOPWORDS and len(t) > 2]
    if not tokens:
        return []

    # Thresholds scale with transcript length. A fixed "appears 3+ times" bar
    # returns almost nothing on a five-minute call and floods a long one.
    unigram_floor = max(2, len(tokens) // 120)
    bigram_floor = max(2, len(tokens) // 300)

    bigrams = Counter(
        f"{a} {b}" for a, b in pairwise(tokens)
    )
    unigrams = Counter(tokens)

    topics: list[str] = []
    covered: set[str] = set()

    for phrase, count in bigrams.most_common(limit * 3):
        if count < bigram_floor:
            break
        topics.append(phrase)
        covered.update(phrase.split())
        if len(topics) >= limit:
            return topics

    for word, count in unigrams.most_common(limit * 3):
        if len(topics) >= limit:
            break
        if count < unigram_floor or word in covered:
            continue
        topics.append(word)

    return topics[:limit]
