"""API integration tests.

These drive the real application: real database, real storage, real engine.
Nothing is stubbed, so a passing run means the product actually works.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import (
    ADMIN_KEY,
    ANALYST_KEY,
    OBJECTION_CALL,
    QUALIFIED_CALL,
    REJECTION_CALL,
    VIEWER_KEY,
)

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

    def test_export_is_bounded_and_says_when_it_truncates(self, client, monkeypatch):
        """Each row carries a 20-50 KB analysis blob, so an unlimited export
        would materialise the whole table in the worker. A truncated export has
        to announce itself rather than look complete."""
        monkeypatch.setattr("app.api.routers.analytics.EXPORT_MAX_ROWS", 2)
        monkeypatch.setattr("app.db.repository.EXPORT_MAX_ROWS", 2)
        for index in range(3):
            create_call(client, title=f"Call {index}")

        response = client.get(f"{API}/export/calls.csv")
        assert response.status_code == 200
        assert response.headers["X-Export-Truncated"] == "true"
        # header row + the capped number of data rows
        assert len(response.text.strip().splitlines()) == 3


class TestRoleEnforcement:
    """Roles as actually enforced over HTTP, with auth switched on."""

    def test_the_audit_log_is_admin_only(self, auth_client):
        """It names the principal behind every upload, export and deletion, so
        the analysts it records must not be able to read it. The endpoint was
        documented admin-only but accepted any analyst."""
        assert auth_client.get(
            f"{API}/audit", headers={"X-API-Key": ANALYST_KEY}
        ).status_code == 403
        assert auth_client.get(
            f"{API}/audit", headers={"X-API-Key": ADMIN_KEY}
        ).status_code == 200

    def test_a_viewer_cannot_write_or_export(self, auth_client):
        headers = {"X-API-Key": VIEWER_KEY}
        assert auth_client.get(f"{API}/calls", headers=headers).status_code == 200
        assert auth_client.post(
            f"{API}/calls",
            json={"title": "t", "transcript": QUALIFIED_CALL},
            headers=headers,
        ).status_code == 403
        assert auth_client.get(
            f"{API}/export/calls.csv", headers=headers
        ).status_code == 403

    def test_only_an_admin_may_delete(self, auth_client):
        created = auth_client.post(
            f"{API}/calls",
            json={"title": "t", "transcript": QUALIFIED_CALL},
            headers={"X-API-Key": ANALYST_KEY},
        )
        call_id = created.json()["id"]
        assert auth_client.delete(
            f"{API}/calls/{call_id}", headers={"X-API-Key": ANALYST_KEY}
        ).status_code == 403
        assert auth_client.delete(
            f"{API}/calls/{call_id}", headers={"X-API-Key": ADMIN_KEY}
        ).status_code == 204

    def test_a_missing_or_bad_key_is_rejected(self, auth_client):
        assert auth_client.get(f"{API}/calls").status_code == 401
        assert auth_client.get(
            f"{API}/calls", headers={"X-API-Key": "not-a-real-key"}
        ).status_code == 401


class TestAudit:
    def test_ingestion_and_deletion_are_recorded(self, client):
        call_id = create_call(client)["id"]
        client.delete(f"{API}/calls/{call_id}")
        actions = [e["action"] for e in client.get(f"{API}/audit").json()]
        assert "call.ingest" in actions
        assert "call.delete" in actions


class TestDatabaseUnderTest:
    """Guards that the `Backend (PostgreSQL)` CI job tests PostgreSQL.

    It did not. The workflow exported TEST_DATABASE_URL, the `settings`
    fixture hardcoded a SQLite file, and the job ran SQLite while its own
    comment claimed it proved "the same schema and queries work on the
    database production actually uses". Nothing read the variable.

    A green job asserting something it never exercised is worse than no job:
    it is the reason a SQLAlchemy major upgrade looked validated against
    Postgres when it had never touched it. These tests fail loudly rather
    than let that recur.
    """

    def test_the_configured_url_is_the_one_actually_used(self, settings):
        from tests.conftest import TEST_DATABASE_URL

        if TEST_DATABASE_URL:
            assert settings.database_url == TEST_DATABASE_URL
        else:
            assert settings.database_url.startswith("sqlite+aiosqlite://")

    def test_the_live_engine_speaks_the_expected_dialect(self, client, settings):
        """Asserted against the running engine, not the setting, so a URL that
        silently fails over to a different backend is still caught."""
        from app.db import session as session_module

        dialect = session_module._engine.dialect.name
        expected = "postgresql" if "postgresql" in settings.database_url else "sqlite"
        assert dialect == expected

    def test_a_write_round_trips_on_whichever_database_is_configured(self, client):
        """The point of running twice: the same query path on both backends."""
        created = create_call(client, title="dialect round trip")
        fetched = client.get(f"{API}/calls/{created['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["title"] == "dialect round trip"


class TestTimestampsAreTimezoneAware:
    """Issue #38: SQLite drops tzinfo, so timestamps came back naive.

    Not cosmetic. The dashboard passes `created_at` to `new Date()`, and
    JavaScript parses an ISO string without an offset as *local* time. On the
    default SQLite backend a call created thirty seconds ago rendered as
    "6h ago" in India and "just now" for four hours in New York; only UTC
    users saw the truth. These run on both backends via TEST_DATABASE_URL.
    """

    @staticmethod
    def _parse(value: str):
        from datetime import datetime

        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def test_read_back_timestamps_carry_a_utc_offset(self, client):
        created = create_call(client, title="tz round trip")
        fetched = client.get(f"{API}/calls/{created['id']}").json()

        for field in ("created_at", "updated_at"):
            stamp = self._parse(fetched[field])
            assert stamp.tzinfo is not None, f"{field} came back naive: {fetched[field]}"
            assert stamp.utcoffset().total_seconds() == 0, f"{field} is not UTC"

    def test_list_timestamps_carry_a_utc_offset(self, client):
        create_call(client, title="tz list")
        item = client.get(f"{API}/calls").json()["items"][0]
        assert self._parse(item["created_at"]).tzinfo is not None

    def test_filter_with_an_offset_compares_instants_not_wall_clocks(self, client):
        """`created_after` in +05:30 must mean the same instant as in UTC.

        Before the fix SQLite compared the stored UTC wall time against the
        filter's *local* wall time, so a bound one minute before creation,
        written in IST, sat five and a half hours in the future and excluded
        the call it should have matched.
        """
        from datetime import timedelta, timezone

        created = create_call(client, title="tz filter")
        made = self._parse(created["created_at"])
        ist = timezone(timedelta(hours=5, minutes=30))

        just_before = (made - timedelta(minutes=1)).astimezone(ist).isoformat()
        hour_after = (made + timedelta(hours=1)).astimezone(ist).isoformat()

        included = client.get(f"{API}/calls", params={"created_after": just_before})
        excluded = client.get(f"{API}/calls", params={"created_after": hour_after})

        assert created["id"] in {c["id"] for c in included.json()["items"]}
        assert created["id"] not in {c["id"] for c in excluded.json()["items"]}
