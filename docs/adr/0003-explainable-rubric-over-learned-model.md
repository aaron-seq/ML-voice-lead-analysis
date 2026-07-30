# ADR 0003 — Explainable weighted rubric instead of a trained classifier

**Status:** Accepted

## Context

The previous implementation trained a `RandomForestClassifier` in
`backend/ml/train.py` on data produced by its own
`create_synthetic_training_data()`, then reported "85%+ lead scoring accuracy"
in the README. That number described how well a model reproduced random values
it had generated itself. It carried no information about real leads.

This project ships with no labelled conversion outcomes, and none can be
fabricated honestly.

## Decision

Lead scoring is a documented weighted rubric over BANT/MEDDIC qualification
dimensions (`backend/app/analysis/scoring.py`). Every component reports its raw
value, weight, and contribution in points. Penalties are explicit. Confidence is
reported separately from the score.

The README and `docs/SCORING.md` state plainly that the rubric is **not
validated against conversion outcomes**. `app/analysis/evaluation.py` plus
`callsense evaluate` provide the harness to validate and retune it once a team
has labelled data, reporting ROC AUC and per-band conversion rates.

## Consequences

**Good.** Nothing is overclaimed. A rep who disagrees with a score sees the
arithmetic and the quotes behind it. A sales leader can tune weights directly.

**Good.** Debuggable — a wrong score traces to a specific lexicon entry.

**Cost.** The weights are informed judgement, not fitted parameters. They may
be poorly calibrated for a given funnel. That is why the evaluation harness
ships in the box and is the top roadmap item.

**Cost.** A linear rubric cannot represent interactions (e.g. budget matters
more when urgency is high).

## Revisit if

An evaluation on ≥200 labelled calls exists. Then ship a learned model
*alongside* the rubric with both scores visible, rather than replacing it —
losing explainability is a product regression, not just a technical one.
