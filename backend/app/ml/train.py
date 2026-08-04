"""Train and honestly evaluate a sentiment classifier on real data.

Design constraints, in order of importance:

1. **Real data only.** SST-2 (Socher et al. 2013) with its published
   train/dev/test partition. No synthetic data, ever -- see docs/adr/0003.
2. **Always report the baseline.** Every metric is printed next to the existing
   lexicon analyzer measured on the identical split. A model that does not beat
   the thing it replaces is a regression, and the only way to know is to run
   both.
3. **Always report the domain gap.** SST-2 is film criticism; this product
   reads sales calls. Test accuracy on SST-2 alone would be a misleading
   headline, so the sales evaluation set is scored in the same run.
4. **Deterministic.** Fixed seed, fixed splits, recorded corpus fingerprint.

A linear model over TF-IDF n-grams is used rather than a transformer because
it trains in seconds on CPU, adds ~200 MB rather than ~2.5 GB, keeps inference
in the microseconds the engine budget assumes, and -- unlike a transformer --
its decision can be traced to specific weighted n-grams, which is the property
the whole product is built around.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.ml.datasets import LabelledText, corpus_fingerprint, load_domain_eval, load_sst2

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_VERSION = "sentiment-tfidf-logreg-v1"
RANDOM_SEED = 20260729


@dataclass
class SplitMetrics:
    """Metrics for one model on one split."""

    name: str
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float

    def render(self) -> str:
        return (
            f"    {self.name:<28} n={self.n:<6} acc={self.accuracy:.3f}  "
            f"P={self.precision:.3f}  R={self.recall:.3f}  F1={self.f1:.3f}"
        )


@dataclass
class TrainingReport:
    model_version: str = MODEL_VERSION
    trained_at: str = ""
    train_size: int = 0
    corpus_fingerprint: str = ""
    train_seconds: float = 0.0
    metrics: list[SplitMetrics] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "",
            "  Sentiment model training report",
            "  " + "=" * 70,
            f"  Model      {self.model_version}",
            f"  Trained    {self.trained_at} in {self.train_seconds:.2f}s "
            f"on {self.train_size} examples",
            f"  Corpus     SST-2 fingerprint {self.corpus_fingerprint}",
            "",
            "  RESULTS (trained model vs. the lexicon it would replace)",
        ]
        lines.extend(m.render() for m in self.metrics)
        if self.notes:
            lines += ["", "  READ THIS BEFORE TRUSTING THE NUMBERS"]
            lines += [f"    - {n}" for n in self.notes]
        lines.append("")
        return "\n".join(lines)


def _score(y_true: list[int], y_pred: list[int], name: str) -> SplitMetrics:
    """Accuracy plus precision/recall/F1 for the positive class.

    Implemented directly rather than via sklearn.metrics so this function also
    works when scoring the lexicon baseline, which is not an sklearn estimator.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 0)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return SplitMetrics(
        name=name,
        n=len(y_true),
        accuracy=correct / len(y_true) if y_true else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _lexicon_predictions(rows: list[LabelledText]) -> list[int]:
    """The incumbent, scored on exactly the same examples."""
    from app.analysis.sentiment import LexiconSentimentAnalyzer

    analyzer = LexiconSentimentAnalyzer()
    # The lexicon returns a continuous score; anything at or above zero is the
    # positive call. Ties go to positive, matching how the engine reads it.
    return [1 if analyzer.score_text(row.text) >= 0 else 0 for row in rows]


def build_pipeline():
    """TF-IDF n-grams into a calibrated linear classifier."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    # Bigrams matter here: "not good" and "good" must not share
                    # a representation, which is the same reason the lexicon
                    # analyzer handles negation explicitly.
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=4.0,
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                    # Sales transcripts skew positive; do not let the majority
                    # class quietly become the model's default answer.
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train(output_dir: Path | None = None) -> TrainingReport:
    """Fetch real data, train, evaluate against the baseline, persist."""
    output_dir = output_dir or MODEL_DIR

    train_rows = load_sst2("train")
    dev_rows = load_sst2("dev")
    test_rows = load_sst2("test")
    domain_rows = load_domain_eval()

    pipeline = build_pipeline()

    started = time.perf_counter()
    pipeline.fit([r.text for r in train_rows], [r.label for r in train_rows])
    elapsed = time.perf_counter() - started

    report = TrainingReport(
        trained_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        train_size=len(train_rows),
        corpus_fingerprint=corpus_fingerprint(train_rows),
        train_seconds=elapsed,
    )

    for label, rows in (
        ("SST-2 dev", dev_rows),
        ("SST-2 test (held out)", test_rows),
        ("Sales domain set", domain_rows),
    ):
        truth = [r.label for r in rows]
        texts = [r.text for r in rows]
        report.metrics.append(
            _score(truth, list(pipeline.predict(texts)), f"model / {label}")
        )
        report.metrics.append(
            _score(truth, _lexicon_predictions(rows), f"lexicon / {label}")
        )

    report.notes = [
        "SST-2 is movie reviews. High test accuracy there does NOT transfer "
        "automatically to sales calls; compare the two 'Sales domain set' rows.",
        "The sales set is hand-labelled by the authors (n=55), not a public "
        "corpus. Treat it as a domain smoke test, not a benchmark.",
        "Neither set contains conversion outcomes, so this says nothing about "
        "whether the LEAD SCORE predicts revenue. See docs/SCORING.md.",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _persist(pipeline, report, output_dir)
    return report


def _persist(pipeline: Any, report: TrainingReport, output_dir: Path) -> None:
    import joblib

    model_path = output_dir / f"{MODEL_VERSION}.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    payload = asdict(report)
    (output_dir / f"{MODEL_VERSION}.metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info("Saved model to %s", model_path)


def load_model(model_dir: Path | None = None):
    """Load the trained pipeline, or None when it has not been trained yet."""
    import joblib

    path = (model_dir or MODEL_DIR) / f"{MODEL_VERSION}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def load_metrics(model_dir: Path | None = None) -> dict | None:
    path = (model_dir or MODEL_DIR) / f"{MODEL_VERSION}.metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
