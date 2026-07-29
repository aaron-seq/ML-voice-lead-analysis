"""Speaker role inference.

Transcripts arrive with labels like ``spk_0`` that say nothing about who is
selling. Almost every downstream metric depends on knowing which side is which:
"the prospect asked about pricing" and "the rep asked about pricing" mean
opposite things.

This assigns roles per *speaker* (not per segment) using discourse markers and
question/answer asymmetry. When the evidence is weak it returns ``UNKNOWN``
rather than guessing -- the scorer then falls back to speaker-agnostic
behaviour instead of confidently mis-attributing every signal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.analysis.lexicons import PROSPECT_MARKERS, REP_MARKERS
from app.analysis.text import PhraseMatcher, is_question, tokenize
from app.domain.models import SpeakerRole, Transcript

# Below this margin between the two sides' scores, the call is left unlabelled.
_DECISION_MARGIN = 0.15

_REP_MATCHER = PhraseMatcher([(p, 1.0, "rep") for p in REP_MARKERS])
_PROSPECT_MATCHER = PhraseMatcher([(p, 1.0, "prospect") for p in PROSPECT_MARKERS])

# Explicit labels some tools emit. Checked before any heuristic.
_EXPLICIT: dict[str, SpeakerRole] = {
    "rep": SpeakerRole.REP, "sales": SpeakerRole.REP, "agent": SpeakerRole.REP,
    "seller": SpeakerRole.REP, "salesperson": SpeakerRole.REP,
    "prospect": SpeakerRole.PROSPECT, "customer": SpeakerRole.PROSPECT,
    "client": SpeakerRole.PROSPECT, "buyer": SpeakerRole.PROSPECT,
    "lead": SpeakerRole.PROSPECT,
}


@dataclass
class _SpeakerStats:
    words: int = 0
    segments: int = 0
    questions: int = 0
    rep_markers: int = 0
    prospect_markers: int = 0
    first_index: int = field(default=10**9)


def infer_roles(transcript: Transcript) -> Transcript:
    """Return a transcript with ``role`` populated on every segment.

    The input is not mutated; segments are rebuilt with roles attached.
    """
    stats: dict[str, _SpeakerStats] = defaultdict(_SpeakerStats)

    for seg in transcript.segments:
        key = seg.speaker.lower()
        entry = stats[key]
        entry.words += seg.word_count
        entry.segments += 1
        entry.first_index = min(entry.first_index, seg.index)
        if is_question(seg.text):
            entry.questions += 1
        tokens = tokenize(seg.text)
        entry.rep_markers += len(_REP_MATCHER.find(tokens))
        entry.prospect_markers += len(_PROSPECT_MATCHER.find(tokens))

    roles = _assign(stats)
    return transcript.model_copy(
        update={
            "segments": [
                seg.model_copy(update={"role": roles.get(seg.speaker.lower(),
                                                         SpeakerRole.UNKNOWN)})
                for seg in transcript.segments
            ]
        }
    )


def _assign(stats: dict[str, _SpeakerStats]) -> dict[str, SpeakerRole]:
    # An explicit label anywhere in the transcript is authoritative.
    explicit = {
        speaker: role
        for speaker in stats
        for token, role in _EXPLICIT.items()
        if token in speaker
    }
    if explicit:
        resolved = dict.fromkeys(stats, SpeakerRole.UNKNOWN)
        resolved.update(explicit)
        # With one side named, the other side of a two-party call follows.
        if len(stats) == 2 and len(explicit) == 1:
            named, named_role = next(iter(explicit.items()))
            other = next(s for s in stats if s != named)
            resolved[other] = (
                SpeakerRole.PROSPECT if named_role is SpeakerRole.REP
                else SpeakerRole.REP
            )
        return resolved

    # A single speaker (undiarised transcript) cannot be split by role.
    if len(stats) < 2:
        return dict.fromkeys(stats, SpeakerRole.UNKNOWN)

    # Rank the two most substantial speakers; extras stay unknown.
    ranked = sorted(stats.items(), key=lambda kv: kv[1].words, reverse=True)[:2]
    (a_name, a), (b_name, b) = ranked

    score_a = _rep_likelihood(a, b)
    score_b = _rep_likelihood(b, a)

    resolved = dict.fromkeys(stats, SpeakerRole.UNKNOWN)
    if abs(score_a - score_b) < _DECISION_MARGIN:
        return resolved

    if score_a > score_b:
        resolved[a_name], resolved[b_name] = SpeakerRole.REP, SpeakerRole.PROSPECT
    else:
        resolved[a_name], resolved[b_name] = SpeakerRole.PROSPECT, SpeakerRole.REP
    return resolved


def _rep_likelihood(subject: _SpeakerStats, other: _SpeakerStats) -> float:
    """Weighted evidence that ``subject`` is the sales rep, on [0, 1].

    Three independent cues, in descending reliability:
      * discourse markers ("reaching out", "let me walk you through")
      * question rate -- reps run discovery, so they ask proportionally more
      * who spoke first -- on a cold call, the rep opens
    """
    marker_total = subject.rep_markers + subject.prospect_markers
    marker_score = (
        subject.rep_markers / marker_total if marker_total else 0.5
    )

    subject_rate = subject.questions / max(1, subject.segments)
    other_rate = other.questions / max(1, other.segments)
    rate_total = subject_rate + other_rate
    question_score = subject_rate / rate_total if rate_total else 0.5

    opener_score = 1.0 if subject.first_index < other.first_index else 0.0

    return 0.55 * marker_score + 0.30 * question_score + 0.15 * opener_score
