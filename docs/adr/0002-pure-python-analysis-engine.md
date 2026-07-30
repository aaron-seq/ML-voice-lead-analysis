# ADR 0002 — No heavyweight ML on the default analysis path

**Status:** Accepted

## Context

Analysing a text file required spaCy plus a model download, TensorFlow, and
transformers: roughly 2.5 GB. Consequences observed in the repository:

- A fresh clone could not run the pipeline without a model download step.
- `pipeline/run_analysis.py` imported `tensorflow` at module scope, so the file
  failed to import at all in most environments.
- The AWS Lambda deployment documented in the README could not have fit.
- CI installed spaCy and a model on every run.

The analysis actually performed — lexicon matching, sentiment, frequency
counting — needed none of it.

## Decision

The engine is pure Python and the standard library. The transformer sentiment
analyzer remains available behind `ENABLE_TRANSFORMER_SENTIMENT=true` and
`requirements-ml.txt`, implementing the same `SentimentAnalyzer` protocol.
If it fails to load, the engine logs a warning and falls back to the lexicon
analyzer rather than failing the analysis.

## Consequences

**Good.** Install is ~15 MB. Measured performance (see
`tests/test_benchmarks.py`): 2.2 ms for a 40-turn call, 472 calls/s
single-threaded. Runs in CI, a Lambda, or a notebook with no setup. Fully
deterministic, which a test asserts.

**Good.** Every score traces to a matched lexicon phrase, which is what makes
the product's explainability claim true rather than aspirational.

**Cost.** Lower recall than an embedding model. Novel phrasing is missed. The
mitigation is a reviewable lexicon plus a documented contribution path, not a
model.

**Cost.** English only.

## Revisit if

Lexicon coverage measurement (roadmap Phase 1) shows unacceptable recall on
real transcripts, or multi-language support becomes a requirement. The
`SentimentAnalyzer` protocol and the `analyzers` metadata field already exist to
make a swap observable.
