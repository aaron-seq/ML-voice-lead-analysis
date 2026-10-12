# The trained sentiment model

## Summary

A TF-IDF + logistic-regression sentiment classifier is trained on **real public
data** (Stanford Sentiment Treebank v2) and evaluated against the lexicon
analyzer it would replace.

**It is not the default.** The measurements say it should not be:

| Evaluation set | Trained model | Lexicon baseline | Winner |
|---|---:|---:|---|
| SST-2 dev (n=872) | **0.803** | 0.593 | model, +21.0 |
| SST-2 test, held out (n=1,821) | **0.814** | 0.568 | model, +24.6 |
| **Sales domain set (n=55)** | 0.600 | **0.618** | **lexicon, +1.8** |

The model is dramatically better at the thing it was trained on — film
criticism — and slightly *worse* at the thing this product actually does.

That is textbook **domain shift**, and it is the single most useful result this
exercise produced. Shipping the model as the default while advertising "81%
accuracy" would have been true-but-misleading in exactly the way the original
repository's "85%+ lead scoring accuracy" was.

Reproduce it yourself:

```bash
pip install -r requirements-ml.txt
python -m app.cli.main train
```

The comparison table prints on every run. There is no way to train this model
without seeing how it compares to the incumbent.

## Why it loses on sales language

Look at the precision/recall split rather than accuracy alone:

| | Precision | Recall |
|---|---:|---:|
| Model, sales set | 0.571 | 0.480 |
| Lexicon, sales set | 0.548 | **0.920** |

The lexicon is a high-recall, low-precision detector: it fires positive on
almost every genuinely positive utterance and over-fires on neutral ones. The
model is balanced but misses over half the positives.

For this product, recall is the more valuable error profile. A missed buying
signal is an unworked lead; a false positive costs a rep thirty seconds of
reading the evidence quote that the UI puts right next to the score.

The vocabulary gap is the mechanism. SST-2 never contains "we have budget
allocated", "run it by procurement" or "we're locked into a contract" — the
exact phrases that carry polarity on a sales call.

## What would actually make it win

In rough order of expected value:

1. **Train on in-domain data.** A few thousand labelled sales utterances would
   almost certainly beat the lexicon outright. This is the real blocker, and it
   is a data problem rather than a modelling one.
2. **Fine-tune rather than train from scratch.** Starting from a sentence
   encoder and adapting on a small in-domain set needs far fewer labels than
   training a bag-of-n-grams from zero.
3. **Ensemble.** Take the lexicon's recall and the model's precision — union
   for candidate detection, model for ranking.
4. **Threshold tuning.** The model's decision boundary is at 0.5; on a
   recall-preferring domain it should not be.

Option 1 is Phase 1 of [ROADMAP.md](ROADMAP.md), and it is the same labelled
data the lead-score rubric needs for calibration.

## The evaluation sets

**SST-2** — Socher et al. 2013, 6,920 training sentences with the published
train/dev/test partition. Fetched from a public mirror, cached under
`backend/data/corpora`, content-fingerprinted so a model artifact can be tied
to the exact data it saw.

**Sales domain set** (`backend/data/eval/sales_sentiment.tsv`, n=55) —
hand-labelled by the authors from realistic sales-call language. This is
**weaker evidence than SST-2** and is reported as such: it is a domain smoke
test, not a benchmark. It is small enough that a couple of flipped labels move
the number by several points, which is precisely why the conclusion drawn from
it is the conservative one ("do not switch the default") rather than an
aggressive one.

No public sales-call sentiment corpus exists, which is why this set had to be
written rather than downloaded.

## What this does NOT tell you

Neither evaluation set contains **conversion outcomes**. This work says nothing
about whether the lead score predicts revenue — that remains unvalidated and is
documented in [SCORING.md](SCORING.md) and
[ADR 0003](adr/0003-explainable-rubric-over-learned-model.md).

Sentiment is one component worth 10 of the 100 available points. Improving it
would not, on its own, make the lead score predictive.

## Enabling it anyway

If your calls look more like general conversation than the sales language in
the domain set, the model may well be the better choice for you. Measure first,
then:

```bash
pip install -r requirements-ml.txt
python -m app.cli.main train
export ENABLE_TRAINED_SENTIMENT=true
```

The engine records which analyzer produced each result in
`analysis.metadata.analyzers.sentiment`, so stored analyses always say how they
were scored. If the model fails to load, the engine logs a warning and falls
back to the lexicon rather than failing the analysis.

## Design notes

**Why a linear model and not a transformer.** It trains in ~1.6 seconds on CPU,
adds ~200 MB rather than ~2.5 GB, keeps inference inside the engine's
microsecond budget, and its decision traces to specific weighted n-grams —
the explainability property the whole product is built around. See
[ADR 0002](adr/0002-pure-python-analysis-engine.md).

**Determinism.** Fixed seed, fixed splits, recorded corpus fingerprint. Two
training runs on the same data produce the same model.

**Class balance.** `class_weight="balanced"`, because sales transcripts skew
positive and the majority class must not quietly become the default answer.
