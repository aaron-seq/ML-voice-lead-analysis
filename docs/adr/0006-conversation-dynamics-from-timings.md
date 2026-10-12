# ADR 0006 — Conversation dynamics from timings, before audio

Status: Accepted
Date: 2026-10-11
Supersedes: nothing. Extends [ADR 0002](0002-pure-python-analysis-engine.md) and
inherits the calibration position of [ADR 0003](0003-explainable-rubric-over-learned-model.md).

## Context

This repository is named for voice analysis and its description promises a
pipeline that "transcribes and analyzes sales cold calls". Neither has been
true of the code for some time. There is no audio path at all, and every
conversational metric the engine produced was derived from **word counts**:

- `rep_talk_ratio` — a share of *words*
- `longest_rep_monologue_words` — a monologue measured in *words*
- `turn_count` — a count of *segments*, which for subtitle input is a count of
  cues rather than turns

Meanwhile `TranscriptSegment` carries `start_s` and `end_s`, the parser
populates them from WebVTT, SRT and AWS Transcribe payloads, and the engine
used them for exactly one thing: the call's total duration.

Two problems follow.

**Word share is a biased proxy for time share.** Two speakers at different
speech rates produce different word counts for identical airtime. The industry
metric — the talk-to-listen ratio every conversation-intelligence product
reports — is a ratio of seconds. Published vendor analyses put closed-won calls
around 57% seller talk time against 62% for lost ones. That is a five-point
margin, comfortably inside the error a word-count proxy introduces.

**The interesting signals were being discarded.** Interruptions, overtalk,
response latency, speech rate, dead air and engagement trajectory are all
computable from timings alone. None of them existed.

## Decision

Add `app/analysis/dynamics.py`, which computes conversation dynamics from
segment timings, and a `ConversationDynamics` block on every `CallAnalysis`.

Three rules govern it.

**1. Timings or nothing.** Every field is `None` when the transcript is
untimed, and the block reports `available: false` with a reason naming the fix.
Nothing is estimated from word counts. A fabricated duration silently corrupts
every per-minute figure that reads it, and most transcripts this product sees
in practice are untimed plain text.

**2. Turns, not segments.** Subtitle formats split one utterance across many
short cues. `build_turns` groups consecutive same-speaker segments, sorted by
start time, before anything is measured. Without it, `turn_count` counts cues,
"interruptions" fire between a speaker and themselves, and response latency
measures the gap between two halves of one sentence.

**3. Only two of these touch the score.** Talk-time ratio and monologue length
have published associations with call outcomes, so they *refine existing terms*
— the `engagement` component and the `rep_monologue` risk flag now measure
seconds when seconds are available, words otherwise. They add no new weight.

Interruptions, response latency, speech rate, dead air and engagement
trajectory are **reported and charted but deliberately unscored**. A search of
the published conversation-intelligence literature turns up coaching guidance
and vendor blog posts for these, but no outcome study, and this project still
has no labelled conversion data of its own (ADR 0003). Assigning them weights
would manufacture exactly the false precision the rest of this codebase was
rewritten to remove.

`ENGINE_VERSION` moves to 1.1.0 because scores for timed transcripts change.
`SCHEMA_VERSION` stays at 1: the new block has a default, so analyses stored
under the previous schema still deserialise.

## Alternatives considered

**Add a weighted `dynamics` component to the rubric.** Rejected. The weights
sum to 100 and are already uncalibrated against outcomes; adding an eighth term
would require rebalancing the other seven on no evidence, moving every stored
score for a change nobody could justify number-by-number.

**Build the audio path first — ASR plus diarization.** Rejected for now, though
it is the obvious next step and is filed as its own issue. Whisper-family
models, WhisperX and NVIDIA Parakeet all now produce word-level timestamps, and
pyannote handles diarization, but the combination adds gigabytes of model
weights, a GPU story, a Hugging Face token for the diarization models, and a
whole new class of failure. More to the point, it would be *redundant work*:
the output of that pipeline is a timed, diarised transcript, which is exactly
the input this module already consumes. Building the consumer first means the
audio path has somewhere to land and a tested contract to land on.

**Keep word-based metrics and add time-based ones alongside.** Rejected as a
permanent answer; it is what the code now does transitionally. Reporting two
talk ratios that disagree, with no statement of which one the score used, is
worse than picking one — so the `engagement` component names its basis in its
`detail` string, and the dashboard labels the figure "of airtime" or "of words".

## Consequences

**Good.**
- The product measures the thing its category actually measures.
- Five metrics that no part of the system previously produced are now
  available, charted, and covered by tests.
- `callsense_conversation_dynamics_total{available}` tells an operator what
  share of their corpus carries timings — a property of *their* transcript
  export that nothing else in the system would have surfaced.
- A timed WebVTT example ships in `examples/transcripts`, so the path is
  exercised by the end-to-end job rather than only by unit tests.

**Costs and risks.**
- Scores for timed transcripts move. That is the point of the change, but it
  means a team re-analysing an existing corpus will see numbers shift; the
  engine version stamp is how they tell which is which.
- The monologue threshold (150s) is coaching guidance, not a validated
  predictor — and the published *cold-call* research points the other way,
  finding that successful cold calls contain **longer** rep bursts. This
  product is aimed at cold calls. The flag is kept because it predates this
  change and carries only a 6-point penalty, but it is the single most
  suspicious number in the rubric and is called out in `docs/SCORING.md`.
- Interruption counts cannot distinguish a rep talking over a prospect from a
  rep saying "mm-hm" in agreement. Overlap alone does not carry that meaning.
  Reported, not scored, for that reason.
- `_overlaps` is O(turns × overlapping-neighbours). Bounded in practice by the
  early exit on start-ordered turns; benchmarks show no measurable change.
