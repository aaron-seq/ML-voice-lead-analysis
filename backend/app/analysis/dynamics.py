"""Conversation dynamics derived from transcript timings.

This repository is named for voice analysis, but until now every conversational
metric it produced was computed from *word counts*: talk ratio was a share of
words, and a monologue was measured in words. Segment timings were parsed out
of WebVTT/SRT and AWS Transcribe payloads and then used for exactly one thing,
the call's total duration.

Word share is a biased proxy for time share. Two speakers with different speech
rates produce different word counts for the same airtime, and speech rate is
itself the thing a sales team wants to know about. The industry metric -- the
"talk-to-listen ratio" every conversation-intelligence product reports -- is a
ratio of *seconds*. Gong's 2025 analysis of 326,000 calls puts closed-won deals
at 57% seller talk time against 62% for lost ones, which is a 5-point margin:
small enough that measuring the wrong quantity can easily erase it.

So this module computes, from timings only:

* talk time per role, and the resulting time-based talk ratio
* the longest uninterrupted stretch by the rep, in seconds
* interruptions and overtalk, from genuinely overlapping turns
* response latency -- how long each side waits before replying
* speech rate in words per minute, per role
* dead air, as the share of the call nobody was speaking
* the prospect's share of each third of the call, i.e. whether they opened up
  or shut down as it went on

What this module deliberately does **not** do is feed those numbers into the
lead score. Talk ratio and monologue length have published associations with
outcomes, so they refine the existing `engagement` component and the
`rep_monologue` risk flag. Interruptions, latency and speech rate do not: a
search of the published conversation-intelligence research turns up guidance
and vendor blog posts but no outcome study, and this project has no labelled
conversion data of its own to calibrate against (ADR 0003). Inventing weights
for them would manufacture exactly the false precision the rest of this
codebase was rewritten to remove. They are reported, charted, and left out of
the score until someone can calibrate them.

Everything here returns ``None`` for an untimed transcript rather than
substituting an estimate. A fabricated duration silently corrupts every
per-minute figure downstream, and most of the transcripts this product sees in
practice are untimed plain text.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from app.domain.models import (
    ConversationDynamics,
    SpeakerRole,
    Transcript,
    TranscriptSegment,
)

# Below this, overlapping timestamps are an artifact of the diarizer rounding
# cue boundaries rather than two people genuinely talking over each other.
# Subtitle timings are routinely quantised to 10ms and cues often abut exactly.
_MIN_OVERLAP_S = 0.25

# A gap longer than this is not a turn-taking pause -- it is a hold, a mute, or
# a transcript with a chunk missing. Including it would swamp the latency mean.
_MAX_PLAUSIBLE_GAP_S = 30.0

# Speech rates outside this range mean the timings are wrong, not that someone
# spoke unusually. Conversational English runs roughly 120-180 wpm; the bounds
# are deliberately wide so only nonsense is dropped.
_MIN_PLAUSIBLE_WPM = 40.0
_MAX_PLAUSIBLE_WPM = 400.0


@dataclass(frozen=True)
class Turn:
    """Consecutive segments from one speaker, collapsed into a single turn.

    Subtitle formats split one utterance across many short cues, so a raw
    segment is not a turn. Without this grouping, `turn_count` counts cues,
    "interruptions" fire between a speaker and themselves, and response
    latency measures the gap between two halves of the same sentence.
    """

    role: SpeakerRole
    speaker: str
    start_s: float
    end_s: float
    word_count: int

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def build_turns(transcript: Transcript) -> list[Turn]:
    """Group consecutive same-speaker segments that carry timings.

    Segments without timings are skipped rather than guessed at, so a partially
    timed transcript yields turns for the part that is timed.

    Sorted by start time before grouping. Nothing guarantees a diariser emits
    cues in time order, and both things this function feeds assume it: merging
    "consecutive" cues from one speaker is only meaningful for cues that are
    adjacent in time, and the overlap scan stops at the first turn starting
    after the current one ends.
    """
    # Each timed segment is narrowed to a plain (start, end, segment) tuple as
    # it is collected, so the floats are non-optional from here on without an
    # `assert` -- which would be stripped under `python -O` and is flagged by
    # bandit as B101 precisely because it can vanish.
    timed: list[tuple[float, float, TranscriptSegment]] = sorted(
        (seg.start_s, seg.end_s, seg)
        for seg in transcript.segments
        if seg.start_s is not None and seg.end_s is not None
    )
    turns: list[Turn] = []
    for start_s, end_s, seg in timed:
        if turns and turns[-1].speaker == seg.speaker:
            previous = turns[-1]
            turns[-1] = Turn(
                role=previous.role,
                speaker=previous.speaker,
                start_s=previous.start_s,
                end_s=max(previous.end_s, end_s),
                word_count=previous.word_count + seg.word_count,
            )
            continue
        turns.append(
            Turn(
                role=seg.role,
                speaker=seg.speaker,
                start_s=start_s,
                end_s=end_s,
                word_count=seg.word_count,
            )
        )
    return turns


def has_usable_timings(transcript: Transcript) -> bool:
    """Whether enough of the transcript is timed to say anything about it.

    One stray timestamp in an otherwise plain-text file is not a timed
    transcript, and metrics computed from it would look as confident as any
    other. Requiring a majority of segments is a proportional test rather than
    an absolute floor, so that "this transcript is barely timed" and "this
    transcript has only one turn" stay distinguishable -- the caller reports a
    different reason for each.
    """
    if not transcript.segments:
        return False
    timed = sum(
        1
        for seg in transcript.segments
        if seg.start_s is not None and seg.end_s is not None
    )
    return timed * 2 >= len(transcript.segments)


def analyse_dynamics(transcript: Transcript) -> ConversationDynamics:
    """Compute what the timings support, and report what they do not.

    Role inference is assumed to have run already (the engine does it before
    calling this). Where roles are unknown the per-role figures stay ``None``
    while the whole-call ones -- dead air, overtalk -- are still computed,
    since those do not depend on knowing who is who.
    """
    if not has_usable_timings(transcript):
        return ConversationDynamics(
            available=False,
            unavailable_reason=(
                "Transcript carries no usable segment timings. Export as WebVTT, "
                "SRT or AWS Transcribe JSON to get conversation dynamics."
            ),
        )

    turns = build_turns(transcript)
    if len(turns) < 2:
        return ConversationDynamics(
            available=False,
            unavailable_reason=(
                "Only one speaker turn is timed; conversation dynamics need at "
                "least two to measure anything."
            ),
        )

    # `build_turns` returns turns in start order, which `_overlaps` relies on.

    call_start = min(t.start_s for t in turns)
    call_end = max(t.end_s for t in turns)
    span = max(0.0, call_end - call_start)

    rep_turns = [t for t in turns if t.role is SpeakerRole.REP]
    prospect_turns = [t for t in turns if t.role is SpeakerRole.PROSPECT]

    rep_time = sum(t.duration_s for t in rep_turns)
    prospect_time = sum(t.duration_s for t in prospect_turns)
    attributed = rep_time + prospect_time

    interruptions_by_rep, interruptions_by_prospect, overtalk = _overlaps(turns)
    rep_latency, prospect_latency = _latencies(turns)

    return ConversationDynamics(
        available=True,
        call_span_s=round(span, 2),
        rep_talk_seconds=round(rep_time, 2) if rep_turns else None,
        prospect_talk_seconds=round(prospect_time, 2) if prospect_turns else None,
        rep_talk_time_ratio=(
            round(rep_time / attributed, 4) if attributed > 0 and rep_turns else None
        ),
        longest_rep_monologue_s=(
            round(max(t.duration_s for t in rep_turns), 2) if rep_turns else None
        ),
        longest_prospect_turn_s=(
            round(max(t.duration_s for t in prospect_turns), 2)
            if prospect_turns
            else None
        ),
        turn_count=len(turns),
        interruptions_by_rep=interruptions_by_rep,
        interruptions_by_prospect=interruptions_by_prospect,
        overtalk_seconds=round(overtalk, 2),
        rep_speech_rate_wpm=_speech_rate(rep_turns),
        prospect_speech_rate_wpm=_speech_rate(prospect_turns),
        rep_response_latency_s=rep_latency,
        prospect_response_latency_s=prospect_latency,
        # Time nobody is recorded as speaking. Overtalk is subtracted because
        # overlapping turns double-count that airtime, which would otherwise
        # let attributed speech exceed the call span and drive this negative.
        dead_air_ratio=(
            round(max(0.0, (span - (attributed - overtalk))) / span, 4)
            if span > 0
            else None
        ),
        prospect_share_by_third=_share_by_third(turns, call_start, call_end),
    )


def _overlaps(turns: list[Turn]) -> tuple[int, int, float]:
    """Count interruptions and total overlapping airtime.

    An interruption is attributed to whoever *started* speaking while the other
    was still going, which is the only reading the timings support. Note the
    asymmetry this cannot see: a rep who talks over a prospect to close a deal
    and a rep who says "mm-hm" in agreement look identical here. That is why
    this number is reported and not scored.
    """
    by_rep = 0
    by_prospect = 0
    overtalk = 0.0

    # Pairwise over a window rather than only adjacent turns: in a three-way
    # overlap the interrupting turn can start before the turn two back ends.
    for i, turn in enumerate(turns):
        for other in turns[i + 1:]:
            if other.start_s >= turn.end_s:
                break  # turns are start-ordered, so nothing later overlaps
            overlap = min(turn.end_s, other.end_s) - other.start_s
            if overlap < _MIN_OVERLAP_S:
                continue
            overtalk += overlap
            if other.role is SpeakerRole.REP and turn.role is SpeakerRole.PROSPECT:
                by_rep += 1
            elif other.role is SpeakerRole.PROSPECT and turn.role is SpeakerRole.REP:
                by_prospect += 1

    return by_rep, by_prospect, overtalk


def _latencies(turns: list[Turn]) -> tuple[float | None, float | None]:
    """Mean gap before each side starts talking, in seconds.

    Reported per role because the two mean different things: the rep's latency
    is how much silence they leave for an answer, the prospect's is how long
    they take to think before giving one.
    """
    rep_gaps: list[float] = []
    prospect_gaps: list[float] = []

    for previous, current in pairwise(turns):
        if previous.role is current.role:
            continue
        gap = current.start_s - previous.end_s
        # Negative gaps are overlaps, counted separately; long ones are holds.
        if gap < 0 or gap > _MAX_PLAUSIBLE_GAP_S:
            continue
        if current.role is SpeakerRole.REP:
            rep_gaps.append(gap)
        elif current.role is SpeakerRole.PROSPECT:
            prospect_gaps.append(gap)

    return _mean(rep_gaps), _mean(prospect_gaps)


def _speech_rate(turns: list[Turn]) -> float | None:
    """Words per minute over time actually spent speaking.

    Computed over summed turn durations rather than the call span, so one
    speaker's rate does not drop simply because the other side talked a lot.
    """
    speaking = sum(t.duration_s for t in turns)
    words = sum(t.word_count for t in turns)
    if speaking <= 0 or words <= 0:
        return None
    wpm = words / (speaking / 60.0)
    if not _MIN_PLAUSIBLE_WPM <= wpm <= _MAX_PLAUSIBLE_WPM:
        # Timings disagree with the text badly enough that the rate is noise.
        return None
    return round(wpm, 1)


def _share_by_third(
    turns: list[Turn], call_start: float, call_end: float
) -> list[float] | None:
    """Prospect's share of attributed talk time in each third of the call.

    The trajectory is the point. A prospect whose share climbs from 20% to 55%
    warmed up; one who falls from 50% to 10% disengaged, and both can average
    to the same unremarkable 35%.
    """
    span = call_end - call_start
    if span <= 0:
        return None

    boundaries = [call_start + span * i / 3.0 for i in range(4)]
    shares: list[float] = []
    for lower, upper in pairwise(boundaries):
        rep = prospect = 0.0
        for turn in turns:
            # Turns straddling a boundary contribute to each third in
            # proportion to the time they actually spend in it.
            inside = max(0.0, min(turn.end_s, upper) - max(turn.start_s, lower))
            if inside <= 0:
                continue
            if turn.role is SpeakerRole.REP:
                rep += inside
            elif turn.role is SpeakerRole.PROSPECT:
                prospect += inside
        total = rep + prospect
        shares.append(round(prospect / total, 4) if total > 0 else 0.0)
    return shares


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def describe(dynamics: ConversationDynamics) -> list[str]:
    """Plain-language observations a rep can act on.

    Only statements the measurement actually supports. Each one names the
    number behind it so the reader can disagree with the threshold rather than
    having to trust it.
    """
    if not dynamics.available:
        return []

    notes: list[str] = []
    ratio = dynamics.rep_talk_time_ratio
    if ratio is not None:
        # 57% closed-won vs 62% closed-lost in Gong's 2025 sample of 326k
        # calls. Narrow margins, so this is phrased as an observation.
        if ratio > 0.65:
            notes.append(
                f"Rep held {ratio:.0%} of the talk time. Published samples put "
                "closed-won calls nearer 57%."
            )
        elif ratio < 0.35:
            notes.append(
                f"Rep held only {ratio:.0%} of the talk time, which is unusually "
                "low for a qualifying call."
            )

    longest = dynamics.longest_rep_monologue_s
    if longest is not None and longest > 150.0:
        notes.append(
            f"Longest rep stretch ran {longest / 60:.1f} minutes without a break; "
            "the common coaching guideline is 2.5."
        )

    if dynamics.interruptions_by_rep >= 3:
        notes.append(
            f"Rep started speaking over the prospect {dynamics.interruptions_by_rep} "
            "times."
        )

    thirds = dynamics.prospect_share_by_third
    if thirds and len(thirds) == 3:
        drift = thirds[2] - thirds[0]
        if drift <= -0.15:
            notes.append(
                f"Prospect's share of the conversation fell from {thirds[0]:.0%} to "
                f"{thirds[2]:.0%} across the call."
            )
        elif drift >= 0.15:
            notes.append(
                f"Prospect opened up as the call went on, from {thirds[0]:.0%} to "
                f"{thirds[2]:.0%} of the talk time."
            )

    if dynamics.dead_air_ratio is not None and dynamics.dead_air_ratio > 0.35:
        notes.append(
            f"{dynamics.dead_air_ratio:.0%} of the call was silence, which usually "
            "means the recording includes hold time."
        )

    return notes
