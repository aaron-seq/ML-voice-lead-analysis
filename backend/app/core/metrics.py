"""In-process metrics with Prometheus text exposition.

Deliberately dependency-free. ``prometheus-client`` would work, but this needs
three metric types and about eighty lines, and the runtime dependency list is
short on purpose (see docs/ARCHITECTURE.md).

Scope and honesty: these counters live in the worker process. With N uvicorn
workers a scrape hits one of them, so values are per-worker, exactly like the
rate limiter. That is fine for Prometheus (each worker is its own target if you
expose them separately) and misleading if you expect a global total from a
single load-balanced scrape. Documented rather than papered over.

Label cardinality is controlled by using the *route template*
(``/api/v1/calls/{call_id}``) rather than the request path, which would mint a
new time series per call id.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

# Seconds. Chosen around the observed shape of this service: analysis is
# single-digit milliseconds, so the interesting resolution is below 100ms, and
# anything past a second means a dependency is misbehaving.
LATENCY_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)

_LabelKey = tuple[tuple[str, str], ...]


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: _LabelKey) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
    return "{" + inner + "}"


class MetricsRegistry:
    """Counters and histograms, safe for concurrent use."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[_LabelKey, float]] = defaultdict(dict)
        self._counter_help: dict[str, str] = {}
        # Per-bucket observation counts, one slot per bound plus a final
        # overflow slot. Cumulative sums are computed at render time.
        self._histograms: dict[str, dict[_LabelKey, list[int]]] = defaultdict(dict)
        self._histogram_sums: dict[str, dict[_LabelKey, float]] = defaultdict(dict)
        self._histogram_help: dict[str, str] = {}
        # Names declared up front. A declared metric renders its HELP and TYPE
        # headers even with no observations, so a scrape of an idle worker says
        # "zero so far" instead of omitting the metric entirely.
        self._declared_counters: set[str] = set()
        self._declared_histograms: set[str] = set()
        self._started = time.monotonic()

    def declare_counter(self, name: str, help_text: str) -> None:
        """Register a counter before it is first incremented.

        Without this, a metric only appears once something increments it. That
        breaks two things. Operationally, `rate(...[5m])` over an absent series
        returns no data rather than 0, so a quiet worker is indistinguishable
        from a broken scrape and alerts on it behave differently. Practically,
        the end-to-end smoke test asserted on `callsense_analyses_total` and
        failed against a freshly booted server that had not yet analysed
        anything -- which is the correct server behaviour and a wrong metric
        contract.
        """
        with self._lock:
            self._declared_counters.add(name)
            self._counter_help.setdefault(name, help_text)

    def declare_histogram(self, name: str, help_text: str) -> None:
        with self._lock:
            self._declared_histograms.add(name)
            self._histogram_help.setdefault(name, help_text)

    @staticmethod
    def _key(labels: dict[str, str] | None) -> _LabelKey:
        if not labels:
            return ()
        # Sorted so the same label set always maps to one series.
        return tuple(sorted(labels.items()))

    def increment(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        value: float = 1.0,
        help_text: str = "",
    ) -> None:
        key = self._key(labels)
        with self._lock:
            if help_text:
                self._counter_help.setdefault(name, help_text)
            series = self._counters[name]
            series[key] = series.get(key, 0.0) + value

    def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        key = self._key(labels)
        with self._lock:
            if help_text:
                self._histogram_help.setdefault(name, help_text)
            counts = self._histograms[name].setdefault(
                key, [0] * (len(LATENCY_BUCKETS) + 1)
            )
            # Cumulative buckets are computed at render time; here we only
            # record which single bucket the observation fell into.
            index = len(LATENCY_BUCKETS)
            for i, bound in enumerate(LATENCY_BUCKETS):
                if value <= bound:
                    index = i
                    break
            counts[index] += 1
            sums = self._histogram_sums[name]
            sums[key] = sums.get(key, 0.0) + value

    def render(self) -> str:
        """Prometheus text exposition format (version 0.0.4)."""
        lines: list[str] = []
        with self._lock:
            counter_names = sorted(self._declared_counters | self._counters.keys())
            for name in counter_names:
                if help_text := self._counter_help.get(name):
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} counter")
                for labels, value in sorted(self._counters.get(name, {}).items()):
                    lines.append(f"{name}{_render_labels(labels)} {value:g}")

            histogram_names = sorted(
                self._declared_histograms | self._histograms.keys()
            )
            for name in histogram_names:
                if help_text := self._histogram_help.get(name):
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} histogram")
                for labels, counts in sorted(self._histograms.get(name, {}).items()):
                    cumulative = 0
                    for bound, count in zip(LATENCY_BUCKETS, counts, strict=False):
                        cumulative += count
                        bucket_labels = (*labels, ("le", repr(bound)))
                        lines.append(
                            f"{name}_bucket{_render_labels(bucket_labels)} {cumulative}"
                        )
                    total = cumulative + counts[-1]
                    inf_labels = (*labels, ("le", "+Inf"))
                    lines.append(f"{name}_bucket{_render_labels(inf_labels)} {total}")
                    lines.append(f"{name}_count{_render_labels(labels)} {total}")
                    total_sum = self._histogram_sums[name].get(labels, 0.0)
                    lines.append(f"{name}_sum{_render_labels(labels)} {total_sum:g}")

        uptime = time.monotonic() - self._started
        lines.append("# HELP callsense_uptime_seconds Process uptime in seconds.")
        lines.append("# TYPE callsense_uptime_seconds gauge")
        lines.append(f"callsense_uptime_seconds {uptime:g}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Clear all observations. Test-only.

        Declarations survive: they are part of the metric contract, not data.
        """
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._histogram_sums.clear()


# Process-wide registry.
registry = MetricsRegistry()

# Every metric this service exposes, declared at import so the exposition is a
# stable contract rather than a function of what happened to have been
# exercised. Scrapers and dashboards can then rely on the names existing.
registry.declare_counter(
    "callsense_http_requests_total",
    "Total HTTP requests by method, route template and status.",
)
registry.declare_histogram(
    "callsense_http_request_duration_seconds",
    "HTTP request latency in seconds.",
)
registry.declare_counter(
    "callsense_analyses_total",
    "Completed call analyses by lead band and outcome.",
)
registry.declare_histogram(
    "callsense_analysis_duration_seconds",
    "Analysis engine duration in seconds.",
)
registry.declare_counter(
    "callsense_ingestion_failures_total",
    "Transcript ingestions that failed, by error code.",
)
registry.declare_counter(
    "callsense_conversation_dynamics_total",
    "Analyses by whether the transcript carried usable segment timings.",
)


def record_request(method: str, route: str, status: int, duration_s: float) -> None:
    labels = {"method": method, "route": route, "status": str(status)}
    registry.increment("callsense_http_requests_total", labels)
    registry.observe(
        "callsense_http_request_duration_seconds",
        duration_s,
        {"method": method, "route": route},
    )


def record_analysis(band: str, outcome: str, duration_s: float) -> None:
    """Business-level metrics. Band/outcome are bounded enums, so cardinality
    stays small and these are safe to label on."""
    registry.increment("callsense_analyses_total", {"band": band, "outcome": outcome})
    registry.observe("callsense_analysis_duration_seconds", duration_s)


def record_ingestion_failure(reason: str) -> None:
    registry.increment("callsense_ingestion_failures_total", {"reason": reason})


def record_dynamics(available: bool) -> None:
    """Whether a transcript carried the timings conversation dynamics need.

    Worth a metric because it is a property of the customer's transcript
    pipeline, not of this service: a team seeing `available="false"` on every
    call should fix their export, and nothing else in the system would tell
    them that.
    """
    registry.increment(
        "callsense_conversation_dynamics_total",
        {"available": "true" if available else "false"},
    )
