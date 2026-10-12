# The lead score

## What it is, and what it is not

The score is a **transparent weighted rubric** encoding standard sales
qualification practice (BANT/MEDDIC). It is not a model trained on conversion
outcomes.

That distinction is the point. A supervised model needs labelled outcomes — did
this lead convert? — which this project ships without. The previous codebase
trained a classifier on **synthetic data** and reported "85%+ accuracy" in its
README; that number described how well a model reproduced random numbers it had
generated itself. A documented rubric is honest about being a rubric, is
auditable line by line, and gives a sales team something they can argue with.

**Use the score to prioritise a call list consistently. Do not read it as a
probability of closing.**

## The arithmetic

Positive components, weighted out of 100:

| Component | Weight | Source |
|---|---:|---|
| Need | 20 | `NEED`, `TECHNICAL_FIT` signals |
| Next step | 18 | `NEXT_STEP` signals |
| Budget | 15 | `BUDGET` signals |
| Authority | 13 | `AUTHORITY` signals |
| Timeline | 12 | `TIMELINE`, `URGENCY` signals |
| Engagement | 12 | Prospect talk share + questions asked |
| Sentiment | 10 | Prospect sentiment level and trajectory |

Each component contributes `raw × weight`, where `raw ∈ [0,1]` is the strength of
the strongest matching signal plus a small corroboration bonus.

**Talk share is measured in seconds when the transcript is timed**, and in words
otherwise. These are not the same quantity: two speakers at different speech
rates produce different word counts for identical airtime, and published vendor
samples separate won from lost calls by around five points of *time* share — a
margin a word-count proxy can swallow. The `engagement` component names which
basis it used in its `detail` string, and the dashboard labels the figure "of
airtime" or "of words". See [ADR 0006](adr/0006-conversation-dynamics-from-timings.md).

**Why these weights.** An agreed next step is the strongest single predictor that
a deal advances, which is why it outranks enthusiasm by nearly two to one — a
prospect who says "this is amazing" and books nothing is a worse lead than one
who is lukewarm and puts time in the diary. Need is weighted highest because a
deal without a problem to solve does not close regardless of the other four.

### Penalties

Subtracted, capped at 35 points in aggregate:

| Risk | Penalty | Fires when |
|---|---:|---|
| No next step | 12 | No `NEXT_STEP` signal anywhere |
| No decision maker | 10 | Authority signal absent/weak, or an authority objection |
| Low engagement | 10 | Prospect spoke < 15% of words |
| Negative sentiment trend | 8 | Prospect sentiment declined > 0.25 across the call |
| Competitor incumbent | 8 | Competitor objection at severity ≥ 0.6 |
| Budget unconfirmed | 6 | Budget signal absent or weak |
| Rep monologue | 6 | An uninterrupted rep stretch > 150s (timed) or > 400 words (untimed) |

The cap exists so risk flags cannot drive an otherwise-strong call to zero.

> **The rep-monologue flag is the least defensible number in this rubric, and
> it may well be backwards for this product.** 2.5 minutes is widely repeated
> coaching guidance for discovery calls, not a validated predictor. The
> published *cold-call* research points the other way: successful cold calls
> are reported to contain **longer** rep bursts, not shorter — one analysis puts
> the average longest burst in a successful cold call at around 37 seconds,
> with longer being better. This product is aimed at cold calls. The flag is
> retained because it predates the move to timings and costs only 6 points, but
> it is the first thing to calibrate once labelled outcomes exist, and the
> honest answer today is that we do not know its sign. Tracked as a GitHub
> issue.

### Bands

| Band | Range |
|---|---|
| Hot | ≥ 72 |
| Warm | ≥ 48 |
| Cold | < 48 |
| Disqualified | — set only by explicit rejection |

**`DISQUALIFIED` is not a score threshold.** It is applied only when the prospect
actually rejected the offering (outcome `not_interested` / `not_a_fit`, or a
need/competitor objection at severity ≥ 0.85). A call that simply produced few
signals is a *cold* lead, not a dead one. Collapsing the two would make "show me
disqualified leads" return every thin transcript.

## Confidence

Reported separately from the score, on [0,1]:

```
0.35 × volume     (word count, saturating at 600)
0.30 × breadth    (distinct signals found, saturating at 6)
0.20 × diarised   (1.0 if speaker roles were inferred, else 0.4)
0.10 × sentiment confidence
0.05 × timed      (1.0 if the transcript carried timestamps)
```

A high score at low confidence means *promising but thin*, not *probably hot*.
The dashboard shows a warning below 0.45.

## Tuning it for your funnel

The weights are a starting point, not a law. To adapt them:

1. Assemble a CSV of past calls with known outcomes:

   ```csv
   transcript_path,converted
   calls/acme-2024-03.txt,1
   calls/globex-2024-03.txt,0
   ```

2. Run the evaluation harness:

   ```bash
   python -m app.cli.main evaluate labelled_calls.csv
   ```

   It reports ROC AUC, the mean score separation between won and lost deals, and
   the conversion rate per band.

3. **A calibrated rubric shows monotonically decreasing conversion from hot to
   cold.** If yours does not, adjust `WEIGHTS` and `BAND_THRESHOLDS` in
   `backend/app/analysis/scoring.py` and re-run.

4. Bump `ENGINE_VERSION` in `domain/models.py`, then re-analyse stored calls so
   old and new scores are comparable:

   ```bash
   curl -X POST localhost:8000/api/v1/calls/<id>/reanalyse
   ```

   Every analysis records the engine version that produced it.

## Extending the lexicons

`backend/app/analysis/lexicons.py` is deliberately data, not code, so it can be
reviewed by someone who knows sales but not Python.

```python
SignalType.BUDGET: [
    ("we have budget", 0.95),   # near-certain
    ("pricing", 0.50),          # suggestive
],
```

Weights reflect **how diagnostic a phrase is of genuine intent**, not how
enthusiastic it sounds. "What's the pricing?" is a stronger buying signal than
"this is great."

Guidelines:

- Prefer precision. A false "budget confirmed" costs far more trust than a miss.
- Phrases match on token sequences with word boundaries, so `cost` will not fire
  inside `costume`.
- One skipped function word is tolerated: `send over` matches "send it over".
- Negation is handled automatically — do not add negated variants as signals.
- Add a test in `tests/unit/test_analysis.py` alongside any new phrase.
- `TestLexiconHygiene` fails the build if a phrase appears under two signal
  types, because the resulting strength would depend on dictionary ordering.

## Conversation dynamics

A separate, mostly **unscored** block on every analysis, computed from segment
timings. Full rationale in
[ADR 0006](adr/0006-conversation-dynamics-from-timings.md).

| Field | Meaning |
|---|---|
| `rep_talk_time_ratio` | Rep's share of attributed speaking **time** |
| `rep_talk_seconds`, `prospect_talk_seconds` | Summed turn durations |
| `longest_rep_monologue_s` | Longest uninterrupted rep stretch, in seconds |
| `interruptions_by_rep` / `_by_prospect` | Times one side began speaking while the other was |
| `overtalk_seconds` | Total airtime with both parties speaking |
| `rep_response_latency_s` / `prospect_…` | Mean silence before each side replies |
| `rep_speech_rate_wpm` / `prospect_…` | Words per minute over time actually spent speaking |
| `dead_air_ratio` | Share of the call nobody was speaking |
| `prospect_share_by_third` | Prospect's share of talk time in each third of the call |
| `observations` | Plain-language notes, each naming the number behind it |

**Only two of these reach the score.** Talk-time ratio feeds the `engagement`
component and monologue length feeds the `rep_monologue` risk, because both
have published associations with call outcomes. The rest — interruptions,
latency, speech rate, dead air, trajectory — are reported and charted but carry
no weight. There is coaching guidance for them and no outcome study, and this
project has no labelled conversion data to calibrate against
([ADR 0003](adr/0003-explainable-rubric-over-learned-model.md)). Giving them
weights would invent precision.

Two honest limits:

- **Interruptions cannot tell intent.** A rep talking over a prospect to close
  and a rep saying "mm-hm" in agreement are the same overlap. That ambiguity is
  why the number is displayed, not scored.
- **`prospect_share_by_third` is the useful one.** A prospect whose share climbs
  from 20% to 55% warmed up; one who falls from 50% to 10% disengaged. Both
  average to an unremarkable 35%, which is all a single talk ratio would show.

### Getting timings

Every field is `null` and `available` is `false` unless the transcript carries
per-segment timings. Nothing is estimated from word counts — a fabricated
duration corrupts every per-minute figure downstream.

Supported timed formats: **WebVTT**, **SRT**, **AWS Transcribe JSON**. Plain
text `Rep:` / `Prospect:` transcripts are untimed by nature.

`examples/transcripts/stark-timed-discovery.vtt` is a worked example.
`callsense_conversation_dynamics_total{available="false"}` tracks how much of
your corpus is missing timings, which is a property of your transcript export
rather than of this service.
