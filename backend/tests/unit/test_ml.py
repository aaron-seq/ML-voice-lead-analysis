"""Trained sentiment model: contract, determinism and the domain-gap guard.

These tests do not require network access or a trained artifact — they skip
when the optional extras or the model are absent, so the default install still
runs the full suite.
"""

from __future__ import annotations

import pytest

sklearn = pytest.importorskip("sklearn", reason="requirements-ml.txt not installed")

# Imported after importorskip on purpose: app.ml pulls in sklearn, so these
# must not execute when the optional extras are absent.
from app.ml.datasets import (  # noqa: E402
    LabelledText,
    corpus_fingerprint,
    load_domain_eval,
)
from app.ml.train import (  # noqa: E402
    MODEL_VERSION,
    _score,
    build_pipeline,
    load_model,
)


class TestScoring:
    def test_perfect_predictions(self):
        m = _score([1, 0, 1, 0], [1, 0, 1, 0], "x")
        assert m.accuracy == 1.0 and m.precision == 1.0 and m.recall == 1.0

    def test_all_wrong(self):
        m = _score([1, 0], [0, 1], "x")
        assert m.accuracy == 0.0

    def test_precision_and_recall_are_distinguished(self):
        """A model that predicts positive for everything has recall 1 and poor
        precision. Reporting only accuracy would hide that."""
        m = _score([1, 0, 0, 0], [1, 1, 1, 1], "x")
        assert m.recall == 1.0
        assert m.precision == pytest.approx(0.25)

    def test_no_positive_predictions_does_not_divide_by_zero(self):
        m = _score([1, 1], [0, 0], "x")
        assert m.precision == 0.0 and m.f1 == 0.0


class TestDomainEvalSet:
    def test_ships_with_the_repository(self):
        rows = load_domain_eval()
        assert len(rows) >= 40

    def test_is_reasonably_balanced(self):
        """A skewed eval set makes accuracy meaningless."""
        rows = load_domain_eval()
        positives = sum(r.label for r in rows)
        ratio = positives / len(rows)
        assert 0.35 <= ratio <= 0.65, f"eval set is {ratio:.0%} positive"

    def test_contains_sales_language_not_generic_text(self):
        joined = " ".join(r.text for r in load_domain_eval()).lower()
        for term in ("budget", "pricing", "vendor", "contract"):
            assert term in joined


class TestDownloadGuard:
    def test_non_https_urls_are_refused(self, tmp_path):
        """urlopen honours file://; a corpus URL must never become a file read."""
        from app.ml.datasets import _download

        for url in ("file:///etc/passwd", "http://example.test/x.tsv"):
            with pytest.raises(ValueError, match="non-HTTPS"):
                _download(url, tmp_path / "out.tsv")


class TestCorpusFingerprint:
    def test_is_stable_for_identical_data(self):
        rows = [LabelledText("a", 1), LabelledText("b", 0)]
        assert corpus_fingerprint(rows) == corpus_fingerprint(list(rows))

    def test_changes_when_data_changes(self):
        a = [LabelledText("a", 1)]
        b = [LabelledText("a", 0)]
        assert corpus_fingerprint(a) != corpus_fingerprint(b)


class TestPipeline:
    def test_uses_bigrams(self):
        """Unigrams alone cannot separate 'good' from 'not good'."""
        assert build_pipeline().named_steps["tfidf"].ngram_range == (1, 2)

    def test_is_class_balanced(self):
        """Sales transcripts skew positive; the majority class must not become
        the model's default answer."""
        assert build_pipeline().named_steps["clf"].class_weight == "balanced"

    def test_seed_is_fixed(self):
        assert build_pipeline().named_steps["clf"].random_state is not None

    def test_trains_and_predicts_on_a_tiny_corpus(self):
        pipeline = build_pipeline()
        # min_df=2 requires each retained term to appear twice.
        texts = ["good great", "good great", "bad awful", "bad awful"]
        pipeline.fit(texts, [1, 1, 0, 0])
        assert len(pipeline.predict(["good great", "bad awful"])) == 2


@pytest.mark.skipif(load_model() is None, reason="no trained model artifact")
class TestTrainedAnalyzer:
    @pytest.fixture
    def analyzer(self):
        from app.analysis.sentiment import TrainedSentimentAnalyzer

        return TrainedSentimentAnalyzer()

    def test_satisfies_the_sentiment_analyzer_protocol(self, analyzer):
        """It must be a drop-in for the lexicon, or the engine cannot swap it."""
        assert analyzer.name == MODEL_VERSION.replace("sentiment-", "trained-")
        assert isinstance(analyzer.score_text("this is great"), float)

    def test_scores_stay_in_the_engine_contract_range(self, analyzer):
        for text in ["fantastic", "terrible", "", "the meeting is on tuesday"]:
            assert -1.0 <= analyzer.score_text(text) <= 1.0

    def test_separates_clear_polarity(self, analyzer):
        assert analyzer.score_text("this is wonderful") > analyzer.score_text(
            "this is awful"
        )

    def test_is_deterministic(self, analyzer):
        scores = {analyzer.score_text("we have budget for this") for _ in range(5)}
        assert len(scores) == 1

    def test_empty_text_is_neutral(self, analyzer):
        assert analyzer.score_text("   ") == 0.0


class TestAnalyzerSelection:
    def test_missing_model_degrades_to_lexicon_instead_of_raising(self, tmp_path):
        """A missing artifact must never take call analysis down."""
        from app.analysis.sentiment import build_analyzer

        analyzer = build_analyzer(use_trained=True)
        assert analyzer.name in {"lexicon-v1", "trained-tfidf-logreg-v1"}
        assert isinstance(analyzer.score_text("great"), float)

    def test_lexicon_remains_the_default(self):
        """The trained model lost on the sales domain set, so it is not the
        default. If that changes, this test should change with it."""
        from app.analysis.sentiment import build_analyzer

        assert build_analyzer().name == "lexicon-v1"
