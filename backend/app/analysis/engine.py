"""The analysis engine.

Single entry point: give it a transcript, get a ``CallAnalysis``. Pure and
synchronous -- no I/O, no database, no network -- which is what makes it
trivially testable and safe to run in a worker, a Lambda or a CLI.

The engine is deterministic: the same transcript always produces the same
analysis for a given ``ENGINE_VERSION``. Tests assert this directly, because a
scoring system that drifts silently between runs cannot be trusted or debugged.
"""

from __future__ import annotations

import logging
import time

from app.analysis import diarization, parsing, scoring, signals
from app.analysis import outcome as outcome_mod
from app.analysis.sentiment import (
    LexiconSentimentAnalyzer,
    SentimentAnalyzer,
    analyse_sentiment,
    build_analyzer,
)
from app.core.config import Settings
from app.domain.models import (
    ENGINE_VERSION,
    AnalysisMetadata,
    CallAnalysis,
    Transcript,
    utcnow,
)

logger = logging.getLogger(__name__)

# Below this, a "call" is a voicemail or a fragment. Still analysed, but the
# result carries a warning so nobody builds a forecast on it.
_MIN_MEANINGFUL_WORDS = 40


class AnalysisEngine:
    """Orchestrates the analyzers into a complete call analysis.

    Construct once and reuse: building the sentiment analyzer may load a model,
    and the phrase matchers are compiled at import time.
    """

    def __init__(self, sentiment_analyzer: SentimentAnalyzer | None = None) -> None:
        self._sentiment = sentiment_analyzer or LexiconSentimentAnalyzer()

    @classmethod
    def from_settings(cls, settings: Settings) -> AnalysisEngine:
        return cls(
            build_analyzer(
                use_transformer=settings.enable_transformer_sentiment,
                model_name=settings.transformer_model,
            )
        )

    def analyse_text(
        self, content: str | bytes, call_id: str, filename: str | None = None
    ) -> tuple[Transcript, CallAnalysis]:
        """Parse raw transcript content and analyse it in one step.

        Returns the *role-enriched* transcript -- the same one the analysis was
        computed from -- so that a caller persisting both cannot end up with a
        stored transcript whose speaker roles disagree with the analysis that
        cites them.
        """
        transcript = parsing.parse_transcript(content, filename=filename)
        transcript = diarization.infer_roles(transcript)
        return transcript, self.analyse(transcript, call_id)

    def analyse(self, transcript: Transcript, call_id: str) -> CallAnalysis:
        """Run every analyzer over an already-parsed transcript.

        Role inference is idempotent, so this is safe to call with either a
        raw or an already-enriched transcript.
        """
        started = time.perf_counter()
        warnings: list[str] = []

        # Roles first: signal attribution, talk metrics and per-role sentiment
        # all depend on knowing who is who.
        transcript = diarization.infer_roles(transcript)
        if all(seg.role.value == "unknown" for seg in transcript.segments):
            warnings.append(
                "Speaker roles could not be inferred. Signals are attributed "
                "conservatively and engagement is scored neutrally. Supply a "
                "diarised transcript for a materially better analysis."
            )

        if transcript.word_count < _MIN_MEANINGFUL_WORDS:
            warnings.append(
                f"Transcript is very short ({transcript.word_count} words); "
                "treat the score as indicative only."
            )

        sentiment = analyse_sentiment(transcript, self._sentiment)
        detected_signals = signals.detect_signals(transcript)
        objections = signals.detect_objections(transcript)
        talk = outcome_mod.compute_talk_metrics(transcript)
        call_outcome = outcome_mod.classify_outcome(transcript)

        lead_score = scoring.score_lead(
            transcript=transcript,
            signals=detected_signals,
            objections=objections,
            sentiment=sentiment,
            talk=talk,
            outcome_is_rejection=call_outcome.label in outcome_mod.REJECTION_OUTCOMES,
        )

        summary = outcome_mod.summarise(
            transcript, detected_signals, objections, call_outcome, lead_score, talk
        )
        next_actions = outcome_mod.recommend_actions(
            lead_score, detected_signals, objections, call_outcome
        )
        topics = outcome_mod.extract_topics(transcript)

        actual_duration = transcript.duration_s
        duration = (
            actual_duration
            if actual_duration is not None
            else transcript.estimated_duration_s()
        )
        if actual_duration is None:
            warnings.append(
                "Transcript carried no timings; duration is estimated from word "
                "count at 150 wpm."
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Analysed call",
            extra={
                "call_id": call_id,
                "words": transcript.word_count,
                "lead_score": lead_score.score,
                "band": lead_score.band.value,
                "outcome": call_outcome.label.value,
                "processing_ms": round(elapsed_ms, 2),
            },
        )

        return CallAnalysis(
            call_id=call_id,
            analyzed_at=utcnow(),
            summary=summary,
            sentiment=sentiment,
            signals=detected_signals,
            objections=objections,
            lead_score=lead_score,
            outcome=call_outcome,
            next_actions=next_actions,
            talk_metrics=talk,
            topics=topics,
            metadata=AnalysisMetadata(
                engine_version=ENGINE_VERSION,
                analyzers={
                    "sentiment": self._sentiment.name,
                    "signals": "lexicon-v1",
                    "objections": "lexicon-v1",
                    "scoring": "weighted-rubric-v1",
                    "outcome": "lexicon-v1",
                    "diarization": "heuristic-v1",
                },
                processing_ms=round(elapsed_ms, 2),
                word_count=transcript.word_count,
                duration_s=round(duration, 2),
                duration_is_estimated=actual_duration is None,
                warnings=warnings,
            ),
        )


_default_engine: AnalysisEngine | None = None


def get_engine() -> AnalysisEngine:
    """Process-wide default engine, built lazily from settings."""
    global _default_engine
    if _default_engine is None:
        from app.core.config import get_settings

        _default_engine = AnalysisEngine.from_settings(get_settings())
    return _default_engine
