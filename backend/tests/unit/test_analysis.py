"""Engine behaviour: sentiment, signals, objections, scoring and outcomes.

These tests encode the product's judgements. When a weight changes, the ones
that fail should be the ones describing behaviour that genuinely changed.
"""

from __future__ import annotations

import json

import pytest

from app.analysis import lexicons
from app.analysis.diarization import infer_roles
from app.analysis.engine import AnalysisEngine
from app.analysis.parsing import parse_transcript
from app.analysis.scoring import WEIGHTS
from app.analysis.sentiment import LexiconSentimentAnalyzer
from app.analysis.signals import detect_objections, detect_signals
from app.analysis.text import PhraseMatcher, is_question, tokenize
from app.domain.models import (
    CallOutcomeLabel,
    LeadBand,
    ObjectionCategory,
    RiskFlag,
    SignalType,
    SpeakerRole,
)
from tests.conftest import OBJECTION_CALL, QUALIFIED_CALL, REJECTION_CALL, SHORT_CALL


def analyse(engine: AnalysisEngine, text: str):
    return engine.analyse_text(text, call_id="t")[1]


# -- Text primitives ------------------------------------------------------


class TestPhraseMatcher:
    def test_matches_multi_word_phrases(self):
        matcher = PhraseMatcher([("we have budget", 1.0, "budget")])
        assert len(matcher.find(tokenize("so we have budget for this"))) == 1

    def test_prefers_the_longest_phrase(self):
        matcher = PhraseMatcher([
            ("interested", 0.5, "short"),
            ("not interested", 0.9, "long"),
        ])
        matches = matcher.find(tokenize("I am not interested"))
        assert len(matches) == 1
        assert matches[0].key == "long"

    def test_skips_a_single_function_word(self):
        """'send it over' is the same commitment as 'send over'."""
        matcher = PhraseMatcher([("send over", 1.0, "next_step")])
        matches = matcher.find(tokenize("I'll send it over tomorrow"))
        assert len(matches) == 1
        assert matches[0].skips == 1

    def test_does_not_skip_two_tokens(self):
        matcher = PhraseMatcher([("send over", 1.0, "next_step")])
        assert matcher.find(tokenize("send the signed contract over")) == []

    def test_does_not_skip_content_words(self):
        matcher = PhraseMatcher([("budget approved", 1.0, "budget")])
        assert matcher.find(tokenize("budget not approved")) == []

    def test_respects_word_boundaries(self):
        matcher = PhraseMatcher([("cost", 1.0, "budget")])
        assert matcher.find(tokenize("he wore a costume")) == []


class TestQuestionDetection:
    @pytest.mark.parametrize("text", [
        "What does it cost?", "how much is it", "Can it integrate with Snowflake",
        "Is that right?",
    ])
    def test_detects_questions(self, text):
        assert is_question(text)

    @pytest.mark.parametrize("text", ["That sounds good.", "We use Airflow."])
    def test_ignores_statements(self, text):
        assert not is_question(text)


# -- Sentiment ------------------------------------------------------------


class TestSentiment:
    @pytest.fixture
    def analyzer(self):
        return LexiconSentimentAnalyzer()

    def test_positive_and_negative_text_separate(self, analyzer):
        assert analyzer.score_text("This is excellent, I love it") > 0.3
        assert analyzer.score_text("This is terrible and frustrating") < -0.3

    def test_negation_flips_polarity(self, analyzer):
        assert analyzer.score_text("this is great") > 0
        assert analyzer.score_text("this is not great") < 0

    def test_negation_dampens_as_well_as_flips(self, analyzer):
        """'not bad' is mildly positive, not as strong as 'good'."""
        assert 0 < analyzer.score_text("not bad") < analyzer.score_text("good")

    def test_intensifiers_amplify(self, analyzer):
        assert analyzer.score_text("very good") > analyzer.score_text("good")

    def test_downtoners_reduce(self, analyzer):
        assert analyzer.score_text("slightly good") < analyzer.score_text("good")

    def test_neutral_text_scores_near_zero(self, analyzer):
        assert abs(analyzer.score_text("The meeting is on Tuesday")) < 0.2

    def test_scores_stay_in_range(self, analyzer):
        extreme = " ".join(["amazing excellent perfect fantastic"] * 40)
        assert -1.0 <= analyzer.score_text(extreme) <= 1.0

    def test_per_role_sentiment_is_separated(self, engine):
        analysis = analyse(engine, QUALIFIED_CALL)
        assert analysis.sentiment.prospect is not None
        assert analysis.sentiment.rep is not None


# -- Diarisation ----------------------------------------------------------


class TestRoleInference:
    def test_explicit_labels_are_honoured(self):
        transcript = infer_roles(parse_transcript(QUALIFIED_CALL))
        roles = {s.speaker.lower(): s.role for s in transcript.segments}
        assert roles["rep"] is SpeakerRole.REP
        assert roles["prospect"] is SpeakerRole.PROSPECT

    def test_one_named_side_implies_the_other(self):
        transcript = infer_roles(parse_transcript(
            "Sales: Hi, I'm calling from Northwind about your data stack.\n"
            "Dana: We already use something for that.\n"
            "Sales: Understood, what are you running?\n"
            "Dana: Airflow, mostly.\n"
        ))
        roles = {s.speaker.lower(): s.role for s in transcript.segments}
        assert roles["sales"] is SpeakerRole.REP
        assert roles["dana"] is SpeakerRole.PROSPECT

    def test_undiarised_transcript_yields_unknown_roles(self):
        transcript = infer_roles(parse_transcript(
            "We talked about pricing. It went well. They seemed interested."
        ))
        assert all(s.role is SpeakerRole.UNKNOWN for s in transcript.segments)


# -- Signals and objections ----------------------------------------------


class TestSignals:
    def test_detects_core_qualification_signals(self, qualified_transcript):
        found = {s.type for s in detect_signals(qualified_transcript)}
        assert {SignalType.BUDGET, SignalType.NEED, SignalType.NEXT_STEP} <= found

    def test_negated_signals_are_suppressed(self):
        """'not entirely my decision' must not count as authority."""
        transcript = infer_roles(parse_transcript(
            "Rep: Are you able to sign this off?\n"
            "Prospect: It's not entirely my decision, I need approval.\n"
        ))
        signals = {s.type: s for s in detect_signals(transcript)}
        authority = signals.get(SignalType.AUTHORITY)
        assert authority is None or authority.strength < 0.6

    def test_prospect_statements_outweigh_rep_statements(self):
        """The same phrase means more from the buyer than from the seller."""
        prospect_says = infer_roles(parse_transcript(
            "Rep: How are things?\nProspect: We have budget for this.\n"
            "Rep: Good to hear.\nProspect: Yes we do.\n"
        ))
        rep_says = infer_roles(parse_transcript(
            "Rep: Most clients tell me we have budget for this.\n"
            "Prospect: Mm.\nRep: Anyway.\nProspect: Right.\n"
        ))
        p = {s.type: s.strength for s in detect_signals(prospect_says)}
        r = {s.type: s.strength for s in detect_signals(rep_says)}
        assert p.get(SignalType.BUDGET, 0) > r.get(SignalType.BUDGET, 0)

    def test_every_signal_carries_evidence(self, qualified_transcript):
        for signal in detect_signals(qualified_transcript):
            assert signal.evidence, f"{signal.type} has no evidence"
            assert signal.evidence[0].quote


class TestObjections:
    def test_detects_price_objection(self):
        transcript = infer_roles(parse_transcript(OBJECTION_CALL))
        categories = {o.category for o in detect_objections(transcript)}
        assert ObjectionCategory.PRICE in categories

    def test_addressed_flag_set_when_rep_responds(self):
        transcript = infer_roles(parse_transcript(OBJECTION_CALL))
        price = next(
            o for o in detect_objections(transcript)
            if o.category is ObjectionCategory.PRICE
        )
        assert price.addressed is True

    def test_unaddressed_when_rep_stays_silent(self):
        transcript = infer_roles(parse_transcript(
            "Rep: So that's the platform.\n"
            "Prospect: It's too expensive, that's out of our budget.\n"
            "Rep: Ok.\n"
        ))
        price = next(
            o for o in detect_objections(transcript)
            if o.category is ObjectionCategory.PRICE
        )
        assert price.addressed is False

    def test_objections_carry_playbook_guidance(self):
        transcript = infer_roles(parse_transcript(OBJECTION_CALL))
        for objection in detect_objections(transcript):
            assert len(objection.suggested_response) > 40


# -- Scoring --------------------------------------------------------------


class TestScoring:
    def test_weights_sum_to_one_hundred(self):
        """The score must be interpretable as 'out of 100'."""
        assert sum(WEIGHTS.values()) == pytest.approx(100.0)

    def test_qualified_call_outranks_rejection(self, engine):
        good = analyse(engine, QUALIFIED_CALL)
        bad = analyse(engine, REJECTION_CALL)
        assert good.lead_score.score > bad.lead_score.score
        assert good.lead_score.band is LeadBand.HOT
        assert bad.lead_score.band is LeadBand.DISQUALIFIED

    def test_rejection_is_disqualified_regardless_of_arithmetic(self, engine):
        analysis = analyse(engine, REJECTION_CALL)
        assert analysis.lead_score.band is LeadBand.DISQUALIFIED

    def test_score_is_bounded(self, engine):
        for text in (QUALIFIED_CALL, REJECTION_CALL, OBJECTION_CALL, SHORT_CALL):
            score = analyse(engine, text).lead_score.score
            assert 0.0 <= score <= 100.0

    def test_components_reconcile_with_the_total(self, engine):
        """The published arithmetic must actually produce the number."""
        score = analyse(engine, QUALIFIED_CALL).lead_score
        positive = sum(c.contribution for c in score.components)
        penalty = min(35.0, sum(-p.contribution for p in score.penalties))
        assert score.score == pytest.approx(max(0.0, positive - penalty), abs=0.15)

    def test_every_component_is_reported(self, engine):
        names = {c.name for c in analyse(engine, QUALIFIED_CALL).lead_score.components}
        assert names == set(WEIGHTS)

    def test_missing_next_step_is_flagged_as_risk(self, engine):
        analysis = analyse(engine, (
            "Rep: Tell me about your setup.\n"
            "Prospect: We're struggling with a manual process, it costs us a lot.\n"
            "Rep: Understood.\n"
            "Prospect: Anyway, I have to run.\n"
        ))
        assert RiskFlag.NO_NEXT_STEP in analysis.lead_score.risks

    def test_short_call_reports_low_confidence(self, engine):
        assert analyse(engine, SHORT_CALL).lead_score.confidence < 0.45

    def test_rationale_is_populated(self, engine):
        assert analyse(engine, QUALIFIED_CALL).lead_score.rationale


# -- Outcome, actions and summary ----------------------------------------


class TestOutcome:
    @pytest.mark.parametrize("text,expected", [
        (QUALIFIED_CALL, CallOutcomeLabel.MEETING_BOOKED),
        (REJECTION_CALL, CallOutcomeLabel.NOT_INTERESTED),
    ])
    def test_classifies_outcomes(self, engine, text, expected):
        assert analyse(engine, text).outcome.label is expected

    def test_rejection_beats_a_positive_phrase(self, engine):
        """Precedence: a refusal outranks a politeness elsewhere in the call."""
        analysis = analyse(engine, (
            "Rep: Could we schedule a demo next week?\n"
            "Prospect: That sounds good in principle, but I'm not interested. "
            "Please take me off your list.\n"
        ))
        assert analysis.outcome.label is CallOutcomeLabel.NOT_INTERESTED

    def test_no_signal_yields_no_clear_outcome(self, engine):
        analysis = analyse(engine, (
            "Rep: The weather has been strange this week.\n"
            "Prospect: It really has been unusual lately.\n"
        ))
        assert analysis.outcome.label is CallOutcomeLabel.NO_CLEAR_OUTCOME


class TestRejectionTargeting:
    """A rejection phrase aimed at a date is not a rejection of the offering.

    Found by a real transcript: "the other vendor wanted six weeks for a POC,
    which doesn't fit February at all" was classified NOT_A_FIT. Because a
    rejection hard-overrides the band, a call with approved budget, a named
    deadline and a booked next step came out DISQUALIFIED -- from a sentence
    complaining that a *competitor* was too slow.
    """

    def test_a_schedule_complaint_is_not_a_rejection(self, engine):
        analysis = analyse(engine, (
            "Rep: What would you need to see to move forward?\n"
            "Prospect: A proof of concept. The other vendor wanted six weeks, "
            "which doesn't fit February at all.\n"
            "Rep: We can do it in a week. Shall we start Tuesday?\n"
            "Prospect: Tuesday works, send the invite.\n"
        ))
        assert analysis.outcome.label is not CallOutcomeLabel.NOT_A_FIT
        assert analysis.lead_score.band is not LeadBand.DISQUALIFIED

    @pytest.mark.parametrize("object_phrase", [
        "February", "our timeline", "the deadline", "that quarter", "Q1",
    ])
    def test_schedule_objects_are_recognised(self, engine, object_phrase):
        analysis = analyse(engine, (
            "Rep: How does the timing look on your side?\n"
            f"Prospect: Their plan doesn't fit {object_phrase}, so we need "
            "something quicker.\n"
        ))
        assert analysis.outcome.label is not CallOutcomeLabel.NOT_A_FIT

    def test_a_genuine_fit_rejection_still_fires(self):
        """The guard must not swallow the case it was carved out of."""
        engine = AnalysisEngine()
        for text in (
            "Prospect: Honestly this is not a fit for us at all.",
            "Prospect: It doesn't fit what we need, sorry.",
            "Prospect: You're too small for us, we need an enterprise vendor.",
        ):
            analysis = analyse(engine, f"Rep: Thanks for the time.\n{text}\n")
            assert analysis.outcome.label is CallOutcomeLabel.NOT_A_FIT, text

    def test_the_shipped_rejection_example_is_still_disqualified(self, engine):
        """Guard against over-correcting into a system that never rejects."""
        analysis = analyse(engine, REJECTION_CALL)
        assert analysis.lead_score.band is LeadBand.DISQUALIFIED


class TestNextActions:
    def test_rejection_produces_a_closing_action_only(self, engine):
        actions = analyse(engine, REJECTION_CALL).next_actions
        assert len(actions) == 1
        assert "closed-lost" in actions[0].action

    def test_booked_meeting_produces_a_p0_confirmation(self, engine):
        actions = analyse(engine, QUALIFIED_CALL).next_actions
        assert actions[0].priority.value == "p0"
        assert "invite" in actions[0].action.lower()

    def test_actions_are_always_produced(self, engine):
        for text in (QUALIFIED_CALL, REJECTION_CALL, OBJECTION_CALL, SHORT_CALL):
            assert analyse(engine, text).next_actions


class TestTalkMetrics:
    def test_ratios_sum_to_one(self, engine):
        talk = analyse(engine, QUALIFIED_CALL).talk_metrics
        assert talk.rep_talk_ratio + talk.prospect_talk_ratio == pytest.approx(1.0)

    def test_counts_rep_questions(self, engine):
        assert analyse(engine, QUALIFIED_CALL).talk_metrics.question_count_rep >= 1


class TestSummary:
    def test_quotes_are_verbatim_from_the_transcript(self, engine):
        """Extractive, never generative — a hallucinated quote is unusable."""
        transcript, analysis = engine.analyse_text(QUALIFIED_CALL, call_id="t")
        haystack = " ".join(s.text for s in transcript.segments).lower()
        for quote in analysis.summary.key_quotes:
            assert quote.quote.rstrip("…").lower()[:60] in haystack

    def test_topics_exclude_filler_words(self, engine):
        topics = analyse(engine, QUALIFIED_CALL).topics
        assert not ({"thats", "ill", "yeah", "the"} & set(topics))


# -- Engine contract ------------------------------------------------------


class TestEngineContract:
    def test_analysis_is_deterministic(self, engine):
        """Same input, same output — a drifting score cannot be debugged."""
        def normalise(analysis):
            payload = analysis.model_dump(mode="json")
            payload.pop("analyzed_at")
            payload["metadata"].pop("processing_ms")
            return json.dumps(payload, sort_keys=True)

        transcript, _ = engine.analyse_text(QUALIFIED_CALL, call_id="t")
        results = {normalise(engine.analyse(transcript, "t")) for _ in range(3)}
        assert len(results) == 1

    def test_analyse_text_returns_roles_attached(self, engine):
        """The returned transcript must match the one the analysis used."""
        transcript, _ = engine.analyse_text(QUALIFIED_CALL, call_id="t")
        assert any(s.role is not SpeakerRole.UNKNOWN for s in transcript.segments)

    def test_undiarised_input_produces_a_warning(self, engine):
        analysis = analyse(engine, "We discussed pricing and it went reasonably well.")
        assert any("role" in w.lower() for w in analysis.metadata.warnings)

    def test_metadata_records_the_analyzers_used(self, engine):
        metadata = analyse(engine, QUALIFIED_CALL).metadata
        assert metadata.engine_version
        assert metadata.analyzers["sentiment"] == "lexicon-v1"


class TestLexiconHygiene:
    def test_no_phrase_appears_in_two_signal_types(self):
        """A duplicated phrase makes strengths depend on dict ordering."""
        seen: dict[str, str] = {}
        duplicates = []
        for signal_type, entries in lexicons.SIGNAL_PHRASES.items():
            for phrase, _ in entries:
                if phrase in seen:
                    duplicates.append((phrase, seen[phrase], signal_type.value))
                seen[phrase] = signal_type.value
        assert not duplicates, f"duplicate signal phrases: {duplicates}"

    def test_weights_are_in_range(self):
        for entries in lexicons.SIGNAL_PHRASES.values():
            for phrase, weight in entries:
                assert 0.0 < weight <= 1.0, phrase

    def test_every_objection_category_has_a_playbook(self):
        assert set(lexicons.OBJECTION_PHRASES) <= set(lexicons.OBJECTION_PLAYBOOK)


class TestScoringUsesTimingsWhenAvailable:
    """Talk balance drives two scored terms. When the transcript is timed,
    both must measure seconds rather than word share.

    Word share is a biased proxy: two speakers at different speech rates
    produce different word counts for the same airtime. Published samples
    separate won from lost calls by roughly five points of *time* share, a
    margin the proxy can easily swallow.
    """

    TIMED_PROSPECT_HEAVY = (
        "WEBVTT\n\n"
        "1\n00:00:00.000 --> 00:00:20.000\nRep: So. Tell me. Everything.\n\n"
        "2\n00:00:20.000 --> 00:00:25.000\nProspect: We have eighty pipelines "
        "and two engineers babysitting them every week, which is exactly why "
        "we have budget approved to fix it now.\n"
    )

    def test_engagement_detail_names_its_basis(self, engine):
        """A reader must be able to tell which quantity produced the number."""
        timed = analyse(engine, self.TIMED_PROSPECT_HEAVY)
        engagement = next(
            c for c in timed.lead_score.components if c.name == "engagement"
        )
        assert "talk time" in engagement.detail

        untimed = analyse(engine, QUALIFIED_CALL)
        engagement = next(
            c for c in untimed.lead_score.components if c.name == "engagement"
        )
        assert "words" in engagement.detail

    def test_the_two_bases_disagree_on_this_transcript(self, engine):
        """If they always agreed the change would be pointless. Here the rep
        holds 80% of the airtime while speaking a fifth of the words."""
        analysis = analyse(engine, self.TIMED_PROSPECT_HEAVY)
        assert analysis.dynamics.rep_talk_time_ratio is not None
        assert analysis.dynamics.rep_talk_time_ratio > 0.75
        assert analysis.talk_metrics.rep_talk_ratio < 0.25

    def test_monologue_risk_uses_seconds_when_timed(self, engine):
        """160 words is under the 400-word fallback, but spoken over four
        minutes it is exactly the monologue the flag exists to catch."""
        slow = (
            "WEBVTT\n\n"
            "1\n00:00:00.000 --> 00:04:00.000\nRep: "
            + " ".join(["so"] * 160)
            + "\n\n2\n00:04:00.000 --> 00:04:20.000\nProspect: Right, I see, "
            "that is interesting, tell me about the pricing please.\n"
        )
        analysis = analyse(engine, slow)
        assert analysis.talk_metrics.longest_rep_monologue_words < 400
        assert RiskFlag.REP_MONOLOGUE in analysis.lead_score.risks

    def test_untimed_scores_are_unchanged_by_the_new_code_path(self, engine):
        """Plain text is most of what this product sees; it must be unaffected."""
        analysis = analyse(engine, QUALIFIED_CALL)
        assert analysis.dynamics.available is False
        assert analysis.lead_score.band is LeadBand.HOT


class TestDynamicsReachTheAnalysis:
    def test_analysis_carries_a_dynamics_block(self, engine):
        analysis = analyse(engine, QUALIFIED_CALL)
        assert analysis.dynamics is not None
        assert analysis.dynamics.available is False

    def test_the_dynamics_analyzer_is_named_in_metadata(self, engine):
        """Every analyzer that contributed must be identifiable after the fact."""
        analysis = analyse(engine, QUALIFIED_CALL)
        assert analysis.metadata.analyzers["dynamics"] == "timings-v1"

    def test_engine_version_moved_with_the_scoring_change(self):
        """Scores for timed transcripts changed, so stored results must be
        distinguishable from ones produced by the previous engine."""
        from app.domain.models import ENGINE_VERSION

        assert ENGINE_VERSION != "1.0.0"
