"""Buying-signal and objection detection.

Both work the same way: match reviewed lexicon phrases against the token
stream, keep the evidence, and weight by *who said it*. A prospect saying "we
have budget" is a qualification signal; a rep saying it is a script line. That
attribution is the main thing separating this from keyword counting.
"""

from __future__ import annotations

from collections import defaultdict

from app.analysis.lexicons import (
    NEGATION_WINDOW,
    NEGATORS,
    OBJECTION_PHRASES,
    OBJECTION_PLAYBOOK,
    SIGNAL_PHRASES,
)
from app.analysis.text import PhraseMatch, PhraseMatcher, tokenize, truncate_quote
from app.domain.models import (
    Evidence,
    Objection,
    ObjectionCategory,
    Signal,
    SignalType,
    SpeakerRole,
    Transcript,
    TranscriptSegment,
)

# How much a match counts for, by who said it. A rep mentioning "pricing" is
# doing their job; a prospect raising it unprompted is information.
_ROLE_WEIGHT: dict[SpeakerRole, float] = {
    SpeakerRole.PROSPECT: 1.0,
    SpeakerRole.UNKNOWN: 0.7,
    SpeakerRole.REP: 0.35,
}

# Objections only count from the buying side; a rep pre-empting an objection
# ("I know the price looks high") is not the prospect objecting.
_OBJECTION_ROLE_WEIGHT: dict[SpeakerRole, float] = {
    SpeakerRole.PROSPECT: 1.0,
    SpeakerRole.UNKNOWN: 0.75,
    SpeakerRole.REP: 0.15,
}

_MAX_EVIDENCE_PER_ITEM = 4

_SIGNAL_MATCHER = PhraseMatcher([
    (phrase, weight, signal_type)
    for signal_type, entries in SIGNAL_PHRASES.items()
    for phrase, weight in entries
])

_OBJECTION_MATCHER = PhraseMatcher([
    (phrase, weight, category)
    for category, entries in OBJECTION_PHRASES.items()
    for phrase, weight in entries
])

_SIGNAL_LABELS: dict[SignalType, str] = {
    SignalType.BUDGET: "Budget discussed",
    SignalType.AUTHORITY: "Decision-making authority indicated",
    SignalType.NEED: "Explicit need or pain described",
    SignalType.TIMELINE: "Timeline referenced",
    SignalType.URGENCY: "Urgency expressed",
    SignalType.COMPETITOR: "Competitor or incumbent mentioned",
    SignalType.ENTHUSIASM: "Strong positive reaction",
    SignalType.NEXT_STEP: "Next step proposed",
    SignalType.TECHNICAL_FIT: "Technical fit explored",
    SignalType.SOCIAL_PROOF: "Proof or references requested",
}


def _is_negated(tokens: list[str], match_start: int) -> bool:
    """True when a negator precedes the match and flips its meaning.

    Without this, "it's not entirely my decision" fires the AUTHORITY phrase
    "my decision" and *adds* qualification points to a call where the prospect
    just told you they cannot buy. Only tokens strictly before the match are
    inspected, so objection phrases that legitimately contain a negator
    ("can't afford", "not interested") are unaffected.
    """
    window = tokens[max(0, match_start - NEGATION_WINDOW):match_start]
    return any(token in NEGATORS for token in window)


def _evidence_for(
    segment: TranscriptSegment, match: PhraseMatch, quote_source: str
) -> Evidence:
    return Evidence(
        segment_index=segment.index,
        quote=truncate_quote(quote_source),
        speaker=segment.speaker,
        role=segment.role,
        start_s=segment.start_s,
        matched_phrase=match.phrase,
    )


def detect_signals(transcript: Transcript) -> list[Signal]:
    """Extract buying signals, strongest first.

    A signal's strength combines the best single match with a smaller
    contribution from corroborating matches, so three weak mentions of budget
    outrank one, but never outrank an explicit "budget is approved".
    """
    hits: dict[SignalType, list[tuple[float, Evidence]]] = defaultdict(list)

    for seg in transcript.segments:
        role_weight = _ROLE_WEIGHT[seg.role]
        tokens = tokenize(seg.text)
        for match in _SIGNAL_MATCHER.find(tokens):
            if _is_negated(tokens, match.start):
                continue
            signal_type: SignalType = match.key  # type: ignore[assignment]
            strength = match.weight * role_weight
            hits[signal_type].append((strength, _evidence_for(seg, match, seg.text)))

    signals: list[Signal] = []
    for signal_type, entries in hits.items():
        entries.sort(key=lambda item: item[0], reverse=True)
        best = entries[0][0]
        corroboration = sum(score for score, _ in entries[1:]) * 0.15
        strength = min(1.0, best + corroboration)
        signals.append(
            Signal(
                type=signal_type,
                label=_SIGNAL_LABELS[signal_type],
                strength=round(strength, 3),
                evidence=[ev for _, ev in entries[:_MAX_EVIDENCE_PER_ITEM]],
            )
        )

    signals.sort(key=lambda s: s.strength, reverse=True)
    return signals


def detect_objections(transcript: Transcript) -> list[Objection]:
    """Extract prospect objections with playbook guidance.

    Also marks whether the rep visibly responded. An unaddressed objection is
    the single most actionable thing a call review can surface, so it is
    tracked explicitly rather than left for the reader to notice.
    """
    hits: dict[ObjectionCategory, list[tuple[float, Evidence, int]]] = defaultdict(list)

    for seg in transcript.segments:
        role_weight = _OBJECTION_ROLE_WEIGHT[seg.role]
        tokens = tokenize(seg.text)
        for match in _OBJECTION_MATCHER.find(tokens):
            category: ObjectionCategory = match.key  # type: ignore[assignment]
            severity = match.weight * role_weight
            hits[category].append(
                (severity, _evidence_for(seg, match, seg.text), seg.index)
            )

    objections: list[Objection] = []
    for category, entries in hits.items():
        entries.sort(key=lambda item: item[0], reverse=True)
        severity = min(1.0, entries[0][0] + sum(s for s, _, _ in entries[1:]) * 0.1)
        # Below this the "objection" is usually a passing mention, not a blocker.
        if severity < 0.3:
            continue
        last_raised = max(index for _, _, index in entries)
        objections.append(
            Objection(
                category=category,
                label=f"{category.value.replace('_', ' ').title()} objection",
                severity=round(severity, 3),
                evidence=[ev for _, ev, _ in entries[:_MAX_EVIDENCE_PER_ITEM]],
                suggested_response=OBJECTION_PLAYBOOK[category],
                addressed=_was_addressed(transcript, last_raised),
            )
        )

    objections.sort(key=lambda o: o.severity, reverse=True)
    return objections


def _was_addressed(transcript: Transcript, raised_at: int, window: int = 4) -> bool:
    """True when the rep gave a substantive reply within the next few turns.

    "Substantive" is a length threshold rather than a content check: a one-word
    acknowledgement does not handle an objection, and anything longer is at
    least an attempt. This is intentionally a low bar -- the field flags
    *silence*, and the reviewer judges quality.
    """
    for seg in transcript.segments[raised_at + 1: raised_at + 1 + window]:
        if seg.role is SpeakerRole.REP and seg.word_count >= 15:
            return True
    return False
