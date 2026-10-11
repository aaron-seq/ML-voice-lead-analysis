"""Cross-layer contract tests.

The bug these exist to prevent (issue #23): the pipeline serialised analyses
with one set of field names while the API deserialised a different set, so no
real analysis was ever readable through the API. The whole system passed its
tests because those tests only ever exercised mock data.

The rule enforced here is that an analysis produced anywhere -- CLI, engine,
HTTP -- round-trips through storage and comes back identical.
"""

from __future__ import annotations

import json

import pytest

from app.analysis.engine import AnalysisEngine
from app.domain.models import CallAnalysis
from tests.conftest import QUALIFIED_CALL

API = "/api/v1"

# Every field the dashboard reads. If the backend renames one of these, this
# list fails before the UI silently renders blanks.
UI_CONTRACT: dict[str, list[str]] = {
    "call": [
        "id", "title", "created_at", "status", "source_format", "lead_score",
        "lead_band", "outcome", "sentiment", "duration_s", "word_count",
        "owner", "tags", "error", "analysis",
    ],
    "analysis": [
        "call_id", "summary", "sentiment", "signals", "objections",
        "lead_score", "outcome", "next_actions", "talk_metrics", "dynamics",
        "topics", "metadata",
    ],
    "lead_score": [
        "score", "band", "confidence", "components", "penalties", "rationale",
        "risks",
    ],
    "component": ["name", "raw", "weight", "contribution", "detail"],
    "objection": [
        "category", "label", "severity", "evidence", "suggested_response",
        "addressed",
    ],
    "evidence": ["segment_index", "quote", "speaker", "role", "start_s",
                 "matched_phrase"],
    "action": ["action", "priority", "rationale", "due_within_hours"],
    "talk_metrics": [
        "rep_talk_ratio", "prospect_talk_ratio", "question_count_rep",
        "question_count_prospect", "turn_count",
    ],
    "dynamics": [
        "available", "unavailable_reason", "call_span_s", "rep_talk_seconds",
        "prospect_talk_seconds", "rep_talk_time_ratio",
        "longest_rep_monologue_s", "turn_count", "interruptions_by_rep",
        "interruptions_by_prospect", "overtalk_seconds",
        "rep_speech_rate_wpm", "prospect_speech_rate_wpm",
        "rep_response_latency_s", "prospect_response_latency_s",
        "dead_air_ratio", "prospect_share_by_third", "observations",
    ],
    "metadata": [
        "engine_version", "schema_version", "analyzers", "processing_ms",
        "word_count", "duration_s", "duration_is_estimated", "warnings",
    ],
}


class TestSerialisationRoundTrip:
    def test_engine_output_reparses_as_itself(self, engine: AnalysisEngine):
        """The single most important invariant in the codebase."""
        _, analysis = engine.analyse_text(QUALIFIED_CALL, call_id="c1")

        serialised = analysis.model_dump(mode="json")
        restored = CallAnalysis.model_validate(serialised)

        assert restored.lead_score.score == analysis.lead_score.score
        assert restored.lead_score.band is analysis.lead_score.band
        assert restored.outcome.label is analysis.outcome.label
        assert len(restored.signals) == len(analysis.signals)
        assert len(restored.lead_score.components) == len(
            analysis.lead_score.components
        )

    def test_json_survives_a_string_round_trip(self, engine: AnalysisEngine):
        _, analysis = engine.analyse_text(QUALIFIED_CALL, call_id="c1")
        restored = CallAnalysis.model_validate(json.loads(analysis.model_dump_json()))
        assert restored.model_dump(mode="json") == analysis.model_dump(mode="json")

    def test_api_returns_what_the_engine_produced(self, client, engine):
        """The API must not reshape, rename or drop engine fields."""
        direct = engine.analyse_text(QUALIFIED_CALL, call_id="x")[1]

        response = client.post(
            f"{API}/calls", json={"title": "t", "transcript": QUALIFIED_CALL}
        )
        via_api = response.json()["analysis"]

        assert via_api["lead_score"]["score"] == direct.lead_score.score
        assert via_api["outcome"]["label"] == direct.outcome.label.value
        assert [s["type"] for s in via_api["signals"]] == [
            s.type.value for s in direct.signals
        ]

    def test_analysis_survives_storage_and_retrieval(self, client):
        """Create, then read back: the stored blob must deserialise fully."""
        created = client.post(
            f"{API}/calls", json={"title": "t", "transcript": QUALIFIED_CALL}
        ).json()
        fetched = client.get(f"{API}/calls/{created['id']}").json()
        assert fetched["analysis"] == created["analysis"]


class TestUiContract:
    """Guards the fields frontend/src/lib/types.ts depends on."""

    @pytest.fixture
    def payload(self, client):
        return client.post(
            f"{API}/calls", json={"title": "t", "transcript": QUALIFIED_CALL}
        ).json()

    def test_call_fields_present(self, payload):
        missing = set(UI_CONTRACT["call"]) - set(payload)
        assert not missing, f"API response is missing: {missing}"

    def test_analysis_fields_present(self, payload):
        missing = set(UI_CONTRACT["analysis"]) - set(payload["analysis"])
        assert not missing, f"analysis is missing: {missing}"

    def test_lead_score_fields_present(self, payload):
        score = payload["analysis"]["lead_score"]
        assert not set(UI_CONTRACT["lead_score"]) - set(score)
        assert not set(UI_CONTRACT["component"]) - set(score["components"][0])

    def test_action_fields_present(self, payload):
        action = payload["analysis"]["next_actions"][0]
        assert not set(UI_CONTRACT["action"]) - set(action)

    def test_talk_metrics_fields_present(self, payload):
        assert not set(UI_CONTRACT["talk_metrics"]) - set(
            payload["analysis"]["talk_metrics"]
        )

    def test_dynamics_fields_present(self, payload):
        missing = set(UI_CONTRACT["dynamics"]) - set(payload["analysis"]["dynamics"])
        assert not missing, f"dynamics is missing: {missing}"

    def test_dynamics_is_present_even_for_an_untimed_transcript(self, payload):
        """The block must always exist so the UI can branch on `available`
        rather than on the key being absent."""
        dynamics = payload["analysis"]["dynamics"]
        assert dynamics["available"] is False
        assert dynamics["unavailable_reason"]
        assert dynamics["rep_talk_time_ratio"] is None

    def test_dynamics_is_populated_for_a_timed_transcript(self, client):
        timed = (
            "WEBVTT\n\n"
            "1\n00:00:00.000 --> 00:00:10.000\nRep: How do you handle "
            "pipelines today, out of interest?\n\n"
            "2\n00:00:11.000 --> 00:00:25.000\nProspect: Badly. We have "
            "budget approved and need this live by March.\n\n"
            "3\n00:00:26.000 --> 00:00:30.000\nRep: Shall we book Tuesday?\n\n"
            "4\n00:00:31.000 --> 00:00:34.000\nProspect: Tuesday works fine.\n"
        )
        dynamics = client.post(
            f"{API}/calls", json={"title": "timed", "transcript": timed}
        ).json()["analysis"]["dynamics"]

        assert dynamics["available"] is True
        assert dynamics["rep_talk_time_ratio"] is not None
        assert dynamics["call_span_s"] == pytest.approx(34.0)
        assert len(dynamics["prospect_share_by_third"]) == 3

    def test_metadata_fields_present(self, payload):
        assert not set(UI_CONTRACT["metadata"]) - set(payload["analysis"]["metadata"])

    def test_evidence_fields_present(self, payload):
        evidence = payload["analysis"]["signals"][0]["evidence"][0]
        assert not set(UI_CONTRACT["evidence"]) - set(evidence)

    def test_enums_serialise_as_strings(self, payload):
        """The UI compares against string literals, not objects."""
        assert isinstance(payload["lead_band"], str)
        assert isinstance(payload["analysis"]["outcome"]["label"], str)
        assert isinstance(payload["analysis"]["signals"][0]["type"], str)


class TestOpenApi:
    def test_schema_is_generated(self, client):
        response = client.get(f"{API}/openapi.json")
        assert response.status_code == 200
        assert response.json()["info"]["title"] == "CallSense"

    def test_every_route_is_documented(self, client):
        paths = client.get(f"{API}/openapi.json").json()["paths"]
        for expected in (
            f"{API}/calls",
            f"{API}/calls/{{call_id}}",
            f"{API}/calls/upload",
            f"{API}/calls/analyse",
            f"{API}/analytics/pipeline",
            f"{API}/export/calls.csv",
        ):
            assert expected in paths, f"undocumented route: {expected}"

    def test_operations_carry_summaries(self, client):
        paths = client.get(f"{API}/openapi.json").json()["paths"]
        for path, methods in paths.items():
            for method, spec in methods.items():
                assert spec.get("summary"), f"{method.upper()} {path} has no summary"
