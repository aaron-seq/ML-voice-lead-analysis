"""Sentiment analysis.

A lexicon analyzer with negation, intensifier and downtoner handling is the
default. It is deterministic, needs no model download, runs in microseconds
and -- importantly for a sales product -- can always point at the words that
produced a score.

An optional transformer analyzer can be enabled via
``ENABLE_TRANSFORMER_SENTIMENT``. It is strictly an upgrade path: the interface
and the resulting ``SentimentAnalysis`` are identical, so nothing downstream
changes shape when it is switched on.
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

from app.analysis import lexicons
from app.analysis.text import tokenize
from app.domain.models import (
    SentimentAnalysis,
    SentimentPoint,
    SpeakerRole,
    Transcript,
)

logger = logging.getLogger(__name__)

# Sentiment terms are sparse, so the mean term weight is scaled up to a usable
# range. The result is squashed with tanh rather than clipped: a hard clip made
# every mildly positive phrase score exactly 1.0, so "good" and "very good"
# were indistinguishable and intensifiers had no observable effect.
_SEGMENT_GAIN = 1.8


class SentimentAnalyzer(Protocol):
    """Anything that can score a piece of text on [-1, 1]."""

    name: str

    def score_text(self, text: str) -> float: ...


class LexiconSentimentAnalyzer:
    """Weighted-lexicon scorer with local syntactic context.

    For each sentiment-bearing token, the preceding three tokens are inspected:
    a negator flips and dampens the polarity (``not great`` is mildly negative,
    not strongly negative), an intensifier amplifies it, a downtoner reduces it.
    This is the standard VADER-style heuristic, implemented directly so the
    lexicon can be sales-specific and reviewable.
    """

    name = "lexicon-v1"

    def __init__(self) -> None:
        self._terms: dict[str, float] = {
            **lexicons.POSITIVE_TERMS,
            **lexicons.NEGATIVE_TERMS,
        }
        # Multi-word entries ("makes sense", "sounds good") need phrase matching.
        self._phrases: dict[tuple[str, ...], float] = {
            tuple(tokenize(term)): weight
            for term, weight in self._terms.items()
            if " " in term
        }
        self._max_phrase = max((len(p) for p in self._phrases), default=1)

    def score_text(self, text: str) -> float:
        tokens = tokenize(text)
        if not tokens:
            return 0.0

        scores: list[float] = []
        i = 0
        while i < len(tokens):
            weight, span = self._lookup(tokens, i)
            if weight is None:
                i += 1
                continue
            scores.append(self._apply_context(tokens, i, weight))
            i += span

        if not scores:
            return 0.0

        mean = sum(scores) / len(scores)
        # tanh keeps the result inside (-1, 1) while preserving ordering all
        # the way up, so stronger wording always yields a strictly higher score.
        return math.tanh(mean * _SEGMENT_GAIN)

    def _lookup(self, tokens: list[str], i: int) -> tuple[float | None, int]:
        """Longest phrase match at position ``i``, else the single token."""
        for length in range(min(self._max_phrase, len(tokens) - i), 1, -1):
            weight = self._phrases.get(tuple(tokens[i:i + length]))
            if weight is not None:
                return weight, length
        return self._terms.get(tokens[i]), 1

    def _apply_context(self, tokens: list[str], i: int, weight: float) -> float:
        window = tokens[max(0, i - lexicons.NEGATION_WINDOW):i]

        for token in window:
            if token in lexicons.NEGATORS:
                # Negation weakens as well as flips: "not bad" is mildly
                # positive, not strongly positive.
                weight = -weight * 0.65
                break

        for token in reversed(window):
            if token in lexicons.INTENSIFIERS:
                weight *= lexicons.INTENSIFIERS[token]
                break
            if token in lexicons.DOWNTONERS:
                weight *= lexicons.DOWNTONERS[token]
                break

        return _clamp(weight)


class TransformerSentimentAnalyzer:
    """Optional transformer-backed scorer.

    Imports are deferred to first use so the dependency stays genuinely
    optional. Construction raises if transformers is unavailable, and the
    engine falls back to the lexicon analyzer rather than failing the analysis.
    """

    name = "transformer-v1"

    def __init__(self, model_name: str) -> None:
        from transformers import pipeline

        self._model_name = model_name
        self._pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            top_k=None,
            device=-1,
            truncation=True,
            max_length=512,
        )

    def score_text(self, text: str) -> float:
        if not text.strip():
            return 0.0
        try:
            results = self._pipe(text)[0]
        except Exception:  # pragma: no cover - model runtime failure
            logger.warning("Transformer sentiment failed; returning neutral",
                           exc_info=True)
            return 0.0
        score = 0.0
        for entry in results:
            label = entry["label"].upper()
            if label.startswith(("POS", "LABEL_2")):
                score += entry["score"]
            elif label.startswith(("NEG", "LABEL_0")):
                score -= entry["score"]
        return _clamp(score)


def build_analyzer(
    use_transformer: bool = False, model_name: str = ""
) -> SentimentAnalyzer:
    """Return the configured analyzer, degrading to lexicon on any failure."""
    if use_transformer:
        try:
            analyzer = TransformerSentimentAnalyzer(model_name)
            logger.info("Using transformer sentiment analyzer: %s", model_name)
            return analyzer
        except Exception as exc:
            logger.warning(
                "Transformer sentiment unavailable (%s); using lexicon analyzer",
                exc,
            )
    return LexiconSentimentAnalyzer()


def analyse_sentiment(
    transcript: Transcript, analyzer: SentimentAnalyzer
) -> SentimentAnalysis:
    """Score the whole call, per speaker role, and over time."""
    timeline: list[SentimentPoint] = []
    rep_scores: list[float] = []
    prospect_scores: list[float] = []
    weighted_sum = 0.0
    total_weight = 0.0

    n = max(1, len(transcript.segments))
    for seg in transcript.segments:
        score = analyzer.score_text(seg.text)
        # Longer turns carry more of the call's sentiment than a "mhm".
        weight = max(1.0, float(seg.word_count))
        weighted_sum += score * weight
        total_weight += weight

        timeline.append(
            SentimentPoint(
                position=seg.index / n,
                at_s=seg.start_s,
                score=score,
                segment_index=seg.index,
            )
        )
        if seg.role is SpeakerRole.REP:
            rep_scores.append(score)
        elif seg.role is SpeakerRole.PROSPECT:
            prospect_scores.append(score)

    overall = _clamp(weighted_sum / total_weight) if total_weight else 0.0

    # The prospect's trajectory is the signal that matters; fall back to the
    # whole-call timeline when roles could not be inferred.
    trend_source = prospect_scores if len(prospect_scores) >= 4 else [
        p.score for p in timeline
    ]
    trend = _linear_trend(trend_source)
    volatility = _stdev([p.score for p in timeline])

    # Confidence reflects how much text the estimate rests on and how many
    # segments actually carried sentiment terms.
    non_neutral = sum(1 for p in timeline if abs(p.score) > 0.05)
    coverage = non_neutral / n
    volume = min(1.0, transcript.word_count / 400.0)
    confidence = round(_clamp01(0.35 + 0.4 * coverage + 0.25 * volume), 3)

    return SentimentAnalysis(
        overall=round(overall, 4),
        rep=round(sum(rep_scores) / len(rep_scores), 4) if rep_scores else None,
        prospect=(
            round(sum(prospect_scores) / len(prospect_scores), 4)
            if prospect_scores else None
        ),
        timeline=timeline,
        trend=round(trend, 4),
        volatility=round(volatility, 4),
        confidence=confidence,
    )


def _linear_trend(values: list[float]) -> float:
    """Least-squares slope over evenly spaced points, scaled to [-1, 1].

    Multiplied by the series length so the result reads as "total drift across
    the call" rather than "drift per segment", which would be near zero for
    long calls regardless of what happened.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    numerator = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return _clamp((numerator / denominator) * n)


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return float(variance**0.5)


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
