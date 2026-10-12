"""Metrics registry and Prometheus exposition."""

from __future__ import annotations

import pytest

from app.core.metrics import (
    LATENCY_BUCKETS,
    MetricsRegistry,
    record_analysis,
    record_request,
    registry,
)


@pytest.fixture
def reg():
    return MetricsRegistry()


class TestCounters:
    def test_counter_accumulates(self, reg):
        reg.increment("hits_total", {"route": "/a"})
        reg.increment("hits_total", {"route": "/a"})
        assert 'hits_total{route="/a"} 2' in reg.render()

    def test_label_sets_are_separate_series(self, reg):
        reg.increment("hits_total", {"route": "/a"})
        reg.increment("hits_total", {"route": "/b"})
        output = reg.render()
        assert 'hits_total{route="/a"} 1' in output
        assert 'hits_total{route="/b"} 1' in output

    def test_label_order_does_not_split_a_series(self, reg):
        """Same labels in a different order must be one time series."""
        reg.increment("hits_total", {"a": "1", "b": "2"})
        reg.increment("hits_total", {"b": "2", "a": "1"})
        assert 'hits_total{a="1",b="2"} 2' in reg.render()

    def test_type_and_help_are_emitted(self, reg):
        reg.increment("hits_total", help_text="Total hits.")
        output = reg.render()
        assert "# HELP hits_total Total hits." in output
        assert "# TYPE hits_total counter" in output

    def test_label_values_are_escaped(self, reg):
        """An unescaped quote produces a malformed scrape."""
        reg.increment("hits_total", {"route": 'we"ird'})
        assert r'route="we\"ird"' in reg.render()


class TestHistograms:
    def test_buckets_are_cumulative(self, reg):
        reg.observe("latency_seconds", 0.003)
        reg.observe("latency_seconds", 0.03)
        output = reg.render()
        # 0.003 falls in the first bucket; both fall at or below 0.05.
        assert 'latency_seconds_bucket{le="0.005"} 1' in output
        assert 'latency_seconds_bucket{le="0.05"} 2' in output
        assert 'latency_seconds_bucket{le="+Inf"} 2' in output

    def test_count_and_sum_are_reported(self, reg):
        reg.observe("latency_seconds", 0.1)
        reg.observe("latency_seconds", 0.2)
        output = reg.render()
        assert "latency_seconds_count 2" in output
        assert "latency_seconds_sum 0.3" in output

    def test_observations_above_the_last_bucket_land_in_inf(self, reg):
        reg.observe("latency_seconds", LATENCY_BUCKETS[-1] + 100)
        output = reg.render()
        assert f'latency_seconds_bucket{{le="{LATENCY_BUCKETS[-1]!r}"}} 0' in output
        assert 'latency_seconds_bucket{le="+Inf"} 1' in output

    def test_histogram_type_is_declared(self, reg):
        reg.observe("latency_seconds", 0.01)
        assert "# TYPE latency_seconds histogram" in reg.render()


class TestExposition:
    def test_uptime_gauge_is_always_present(self, reg):
        assert "callsense_uptime_seconds" in reg.render()

    def test_output_ends_with_a_newline(self, reg):
        """Prometheus requires a trailing newline."""
        assert reg.render().endswith("\n")

    def test_empty_registry_still_renders(self, reg):
        assert reg.render().strip()


class TestHelpers:
    def setup_method(self):
        registry.reset()

    def teardown_method(self):
        registry.reset()

    def test_record_request_emits_counter_and_histogram(self):
        record_request("GET", "/api/v1/calls", 200, 0.012)
        output = registry.render()
        assert "callsense_http_requests_total" in output
        assert 'status="200"' in output
        assert "callsense_http_request_duration_seconds_count" in output

    def test_record_analysis_labels_band_and_outcome(self):
        record_analysis("hot", "meeting_booked", 0.003)
        output = registry.render()
        assert 'band="hot"' in output
        assert 'outcome="meeting_booked"' in output


class TestDeclaredMetrics:
    """A declared metric must appear before anything increments it.

    This is the contract the end-to-end smoke job relied on and the code did
    not provide: it boots a fresh server, which has analysed nothing, and
    asserted `callsense_analyses_total` was present. Lazily-registered metrics
    made that assertion fail against a perfectly healthy server.
    """

    def test_declared_counter_renders_with_no_observations(self, reg):
        reg.declare_counter("widgets_total", "Widgets seen.")
        output = reg.render()
        assert "# HELP widgets_total Widgets seen." in output
        assert "# TYPE widgets_total counter" in output

    def test_declared_histogram_renders_with_no_observations(self, reg):
        reg.declare_histogram("widget_seconds", "Widget latency.")
        assert "# TYPE widget_seconds histogram" in reg.render()

    def test_declaring_does_not_invent_a_zero_series(self, reg):
        """Prometheus has no 'zero' for an unlabelled-yet counter; emitting
        `widgets_total 0` would be a series nobody observed."""
        reg.declare_counter("widgets_total", "Widgets seen.")
        assert "widgets_total 0" not in reg.render()

    def test_declaration_survives_reset(self, reg):
        reg.declare_counter("widgets_total", "Widgets seen.")
        reg.increment("widgets_total", {"kind": "a"})
        reg.reset()
        output = reg.render()
        assert "# TYPE widgets_total counter" in output
        assert 'kind="a"' not in output

    @pytest.mark.parametrize(
        "name",
        [
            "callsense_http_requests_total",
            "callsense_http_request_duration_seconds",
            "callsense_analyses_total",
            "callsense_analysis_duration_seconds",
            "callsense_ingestion_failures_total",
        ],
    )
    def test_every_documented_metric_is_present_on_a_cold_registry(self, name):
        """docs/OPERATIONS.md lists these; a scrape must never omit one."""
        registry.reset()
        assert f"# TYPE {name}" in registry.render()


class TestMetricsEndpoint:
    def test_endpoint_serves_prometheus_text(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "# TYPE" in response.text

    def test_requests_are_recorded(self, client):
        client.get("/health/live")
        assert "callsense_http_requests_total" in client.get("/metrics").text

    def test_route_template_is_used_not_the_raw_path(self, client):
        """A raw path label would mint a series per call id."""
        created = client.post(
            "/api/v1/calls",
            json={"title": "t", "transcript": "Rep: Hi.\nProspect: We have budget."},
        )
        call_id = created.json()["id"]
        client.get(f"/api/v1/calls/{call_id}")

        body = client.get("/metrics").text
        assert "{call_id}" in body
        assert call_id not in body

    def test_analysis_metrics_reflect_real_work(self, client):
        client.post(
            "/api/v1/calls",
            json={"title": "t", "transcript": "Rep: Hi.\nProspect: Not interested."},
        )
        assert "callsense_analyses_total" in client.get("/metrics").text

    def test_disabled_setting_actually_disables_the_endpoint(self, settings):
        """The setting must do something -- it previously did not."""
        from fastapi.testclient import TestClient

        from app.db import session as session_module
        from app.main import create_app

        session_module._engine = None
        session_module._sessionmaker = None
        disabled = settings.model_copy(update={"metrics_enabled": False})
        with TestClient(create_app(disabled)) as no_metrics:
            assert no_metrics.get("/metrics").status_code == 404
        session_module._engine = None
        session_module._sessionmaker = None
