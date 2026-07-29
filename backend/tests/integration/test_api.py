"""API integration tests.

These drive the real application: real database, real storage, real engine.
Nothing is stubbed, so a passing run means the product actually works.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import OBJECTION_CALL, QUALIFIED_CALL, REJECTION_CALL

API = "/api/v1"


def create_call(client, transcript=QUALIFIED_CALL, title="Test call", **kwargs):
    response = client.post(
        f"{API}/calls", json={"title": title, "transcript": transcript, **kwargs}
    )
    assert response.status_code == 201, response.text
    return response.json()


class TestHealth:
    def test_liveness_touches_no_dependencies(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_reports_each_component(self, client):
        response = client.get("/health/ready")
        assert response.status_code == 200
        components = response.json()["components"]
        assert components["database"]["ok"] is True
        assert components["storage"]["ok"] is True

    def test_every_response_carries_a_request_id(self, client):
        assert client.get("/health/live").headers["X-Request-ID"]

    def test_security_headers_are_present(self, client):
        headers = client.get("/health/live").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"

    def test_hsts_is_absent_outside_production(self, client):
        """Setting HSTS over plain HTTP poisons a developer's browser."""
        assert "Strict-Transport-Security" not in client.get("/health/live").headers


class TestCreateAndRead:
    def test_create_returns_a_full_analysis(self, client):
        call = create_call(client)
        assert call["status"] == "completed"
        assert call["lead_band"] == "hot"
        assert call["analysis"]["lead_score"]["components"]
        assert call["analysis"]["next_actions"]

    def test_source_format_is_recorded(self, client):
        assert create_call(client)["source_format"] == "diarized_text"

    def test_get_by_id(self, client):
        call_id = create_call(client)["id"]
        fetched = client.get(f"{API}/calls/{call_id}").json()
        assert fetched["id"] == call_id
        assert fetched["analysis"] is not None

    def test_transcript_is_returned_on_request(self, client):
        call_id = create_call(client)["id"]
        detail = client.get(f"{API}/calls/{call_id}?include_transcript=true").json()
        assert detail["transcript"] is not None
        roles = {s["role"] for s in detail["transcript"]["segments"]}
        assert roles == {"rep", "prospect"}

    def test_transcript_is_omitted_by_default(self, client):
        call_id = create_call(client)["id"]
        assert client.get(f"{API}/calls/{call_id}").json()["transcript"] is None

    def test_unknown_id_returns_404(self, client):
        response = client.get(f"{API}/calls/does-not-exist")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_upload_endpoint_derives_a_title(self, client):
        response = client.post(
            f"{API}/calls/upload",
            files={"file": ("acme-discovery-call.txt", QUALIFIED_CALL.encode(),
                            "text/plain")},
        )
        assert response.status_code == 201
        assert response.json()["title"] == "acme discovery call"

    def test_empty_transcript_is_rejected_with_a_clear_code(self, client):
        response = client.post(
            f"{API}/calls", json={"title": "x", "transcript": "   "}
        )
        assert response.status_code == 400
        assert response.json()["code"] == "empty_transcript"

    def test_unparseable_json_is_rejected(self, client):
        response = client.post(
            f"{API}/calls",
            json={"title": "x", "transcript": json.dumps({"unexpected": True})},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "unsupported_format"

    def test_failed_ingest_still_leaves_a_recoverable_record(self, client):
        """A rejected upload must not vanish — the user needs to see why."""
        client.post(f"{API}/calls", json={"title": "bad", "transcript": "[inaudible]"})
        rows = client.get(f"{API}/calls?status=failed").json()["items"]
        assert len(rows) == 1
        assert rows[0]["error"]


class TestListingAndFiltering:
    @pytest.fixture
    def seeded(self, client):
        create_call(client, QUALIFIED_CALL, "Qualified call", owner="alice")
        create_call(client, REJECTION_CALL, "Rejected call", owner="bob")
        create_call(client, OBJECTION_CALL, "Objection call", owner="alice")
        return client

    def test_lists_all_calls(self, seeded):
        body = seeded.get(f"{API}/calls").json()
        assert body["pagination"]["total_items"] == 3

    def test_filters_by_band(self, seeded):
        body = seeded.get(f"{API}/calls?band=hot").json()
        assert body["pagination"]["total_items"] == 1
        assert body["items"][0]["title"] == "Qualified call"

    def test_filters_by_multiple_bands(self, seeded):
        body = seeded.get(f"{API}/calls?band=hot&band=cold").json()
        assert body["pagination"]["total_items"] == 2

    def test_filters_by_owner(self, seeded):
        assert seeded.get(f"{API}/calls?owner=alice").json()[
            "pagination"]["total_items"] == 2

    def test_search_matches_title(self, seeded):
        body = seeded.get(f"{API}/calls?search=rejected").json()
        assert body["pagination"]["total_items"] == 1

    def test_filters_by_minimum_score(self, seeded):
        body = seeded.get(f"{API}/calls?min_score=50").json()
        assert all(item["lead_score"] >= 50 for item in body["items"])

    def test_pagination_metadata_is_consistent(self, seeded):
        body = seeded.get(f"{API}/calls?page=1&page_size=2").json()
        assert body["pagination"]["total_pages"] == 2
        assert body["pagination"]["has_next"] is True
        assert len(body["items"]) == 2

    def test_sorting_by_score(self, seeded):
        items = seeded.get(f"{API}/calls?sort_by=lead_score&sort_desc=true").json()[
            "items"]
        scores = [i["lead_score"] for i in items]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_sort_field_is_rejected(self, seeded):
        """An arbitrary string must never reach ORDER BY."""
        response = seeded.get(f"{API}/calls?sort_by=lead_score; DROP TABLE calls")
        assert response.status_code == 400
        assert "allowed" in response.json()["context"]

    def test_page_size_cap_is_enforced(self, seeded):
        assert seeded.get(f"{API}/calls?page_size=500").status_code == 422

    def test_contradictory_score_range_is_rejected(self, seeded):
        assert seeded.get(f"{API}/calls?min_score=80&max_score=20").status_code == 400


class TestMutation:
    def test_patch_updates_metadata(self, client):
        call_id = create_call(client)["id"]
        response = client.patch(
            f"{API}/calls/{call_id}",
            json={"title": "Renamed", "owner": "carol", "tags": ["q3"]},
        )
        assert response.status_code == 200
        assert response.json()["title"] == "Renamed"
        assert response.json()["tags"] == ["q3"]

    def test_reanalyse_reproduces_the_same_score(self, client):
        """Determinism, verified through the full storage round trip."""
        call = create_call(client)
        before = call["analysis"]["lead_score"]["score"]
        after = client.post(f"{API}/calls/{call['id']}/reanalyse").json()
        assert after["analysis"]["lead_score"]["score"] == before

    def test_delete_removes_the_call(self, client):
        call_id = create_call(client)["id"]
        assert client.delete(f"{API}/calls/{call_id}").status_code == 204
        assert client.get(f"{API}/calls/{call_id}").status_code == 404

    def test_delete_of_missing_call_returns_404(self, client):
        assert client.delete(f"{API}/calls/nope").status_code == 404


class TestStatelessAnalyse:
    def test_returns_analysis_without_persisting(self, client):
        response = client.post(f"{API}/calls/analyse", json={"transcript": QUALIFIED_CALL})
        assert response.status_code == 200
        assert response.json()["analysis"]["lead_score"]["band"] == "hot"
        # Nothing was stored.
        assert client.get(f"{API}/calls").json()["pagination"]["total_items"] == 0


class TestAnalytics:
    def test_empty_dataset_reports_zeros_not_placeholders(self, client):
        """The old API returned invented figures; this must return real ones."""
        body = client.get(f"{API}/analytics/pipeline").json()
        assert body["total_calls"] == 0
        assert body["avg_lead_score"] is None
        assert body["band_breakdown"] == {
            "hot": 0, "warm": 0, "cold": 0, "disqualified": 0
        }

    def test_aggregates_reflect_stored_calls(self, client):
        create_call(client, QUALIFIED_CALL, "A")
        create_call(client, REJECTION_CALL, "B")
        body = client.get(f"{API}/analytics/pipeline").json()
        assert body["total_calls"] == 2
        assert body["band_breakdown"]["hot"] == 1
        assert body["band_breakdown"]["disqualified"] == 1
        assert body["outcome_breakdown"]["meeting_booked"] == 1

    def test_reports_objection_and_risk_frequencies(self, client):
        create_call(client, OBJECTION_CALL, "A")
        body = client.get(f"{API}/analytics/pipeline").json()
        assert any(o["category"] == "price" for o in body["top_objections"])
        assert body["top_risks"]


class TestExport:
    def test_csv_has_a_header_and_one_row_per_call(self, client):
        create_call(client, QUALIFIED_CALL, "A")
        create_call(client, REJECTION_CALL, "B")
        response = client.get(f"{API}/export/calls.csv")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        lines = response.text.strip().splitlines()
        assert len(lines) == 3
        assert "lead_score" in lines[0]

    def test_csv_respects_filters(self, client):
        create_call(client, QUALIFIED_CALL, "A")
        create_call(client, REJECTION_CALL, "B")
        response = client.get(f"{API}/export/calls.csv?band=hot")
        assert len(response.text.strip().splitlines()) == 2

    def test_export_is_audited(self, client):
        create_call(client)
        client.get(f"{API}/export/calls.csv")
        actions = [e["action"] for e in client.get(f"{API}/audit").json()]
        assert "export.csv" in actions


class TestAudit:
    def test_ingestion_and_deletion_are_recorded(self, client):
        call_id = create_call(client)["id"]
        client.delete(f"{API}/calls/{call_id}")
        actions = [e["action"] for e in client.get(f"{API}/audit").json()]
        assert "call.ingest" in actions
        assert "call.delete" in actions
