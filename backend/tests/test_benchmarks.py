"""Performance characterisation for the analysis engine.

The README and docs/ARCHITECTURE.md claim the engine runs in "~3 ms per call".
That claim was originally made from a single ad-hoc timing, which is not
evidence. This module measures it properly and asserts a ceiling loose enough
not to flake on shared CI runners but tight enough to catch a real regression
-- for example, someone making the phrase matcher O(tokens x phrases) again.

Run the numbers on demand:

    pytest tests/test_benchmarks.py -s -m slow
"""

from __future__ import annotations

import statistics
import time

import pytest

from app.analysis.engine import AnalysisEngine
from app.analysis.parsing import parse_transcript
from app.analysis.text import PhraseMatcher, tokenize

# Ceilings are deliberately ~10x the measured local figures. A shared CI runner
# is easily 3-5x slower than a laptop, so a tight bound would fail for reasons
# that have nothing to do with the code.
MAX_MEAN_MS_PER_CALL = 40.0
MAX_MEAN_MS_PER_1K_WORDS = 60.0


def _synthetic_transcript(turns: int) -> str:
    """Build a realistic diarised transcript of a given length."""
    rep_lines = [
        "Thanks for taking the time today, I appreciate it.",
        "Can I ask what your current process looks like?",
        "How long has that been a problem for the team?",
        "Let me walk you through how we would approach that.",
        "Does that make sense so far?",
    ]
    prospect_lines = [
        "We are struggling with a completely manual process right now.",
        "What is the pricing like? We have budget allocated this quarter.",
        "This is exactly what we need, I love the automation.",
        "I will need to run it by our VP of Engineering first.",
        "Can we set up a call next week to go through it?",
    ]
    parts = []
    for i in range(turns):
        speaker, pool = (
            ("Rep", rep_lines) if i % 2 == 0 else ("Prospect", prospect_lines)
        )
        parts.append(f"{speaker}: {pool[i % len(pool)]}")
    return "\n".join(parts)


def _time_ms(fn, repeats: int) -> list[float]:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _report(label: str, timings: list[float], extra: str = "") -> None:
    ordered = sorted(timings)
    p95 = ordered[int(len(ordered) * 0.95) - 1] if len(ordered) > 1 else ordered[0]
    print(
        f"\n  {label:<34} mean {statistics.mean(timings):7.2f} ms   "
        f"median {statistics.median(timings):7.2f} ms   "
        f"p95 {p95:7.2f} ms   min {ordered[0]:7.2f} ms {extra}"
    )


class TestEnginePerformance:
    def test_typical_call_analysis_speed(self, engine: AnalysisEngine):
        """A ~40-turn call is the common case."""
        transcript = parse_transcript(_synthetic_transcript(40))
        words = transcript.word_count

        _time_ms(lambda: engine.analyse(transcript, "warmup"), 3)  # warm caches
        timings = _time_ms(lambda: engine.analyse(transcript, "bench"), 40)

        _report("analyse (40 turns)", timings, f"[{words} words]")
        assert statistics.mean(timings) < MAX_MEAN_MS_PER_CALL

    def test_scales_roughly_linearly_with_length(self, engine: AnalysisEngine):
        """Guards against an accidentally quadratic analyzer.

        Asserting a ratio rather than absolute times makes this robust to how
        fast the machine is.
        """
        short = parse_transcript(_synthetic_transcript(20))
        long = parse_transcript(_synthetic_transcript(200))

        _time_ms(lambda: engine.analyse(short, "w"), 3)
        short_ms = statistics.median(_time_ms(lambda: engine.analyse(short, "b"), 20))
        long_ms = statistics.median(_time_ms(lambda: engine.analyse(long, "b"), 20))

        words_ratio = long.word_count / short.word_count
        time_ratio = long_ms / max(short_ms, 1e-6)
        print(
            f"\n  scaling: {words_ratio:.1f}x words -> {time_ratio:.1f}x time "
            f"({short_ms:.2f} ms -> {long_ms:.2f} ms)"
        )

        # Linear would be ~10x. Allow 3x headroom for fixed overheads and
        # noise, but fail well before quadratic (~100x).
        assert time_ratio < words_ratio * 3

    def test_throughput_per_thousand_words(self, engine: AnalysisEngine):
        """The figure quoted in the docs, normalised by length."""
        transcript = parse_transcript(_synthetic_transcript(120))
        _time_ms(lambda: engine.analyse(transcript, "w"), 3)
        timings = _time_ms(lambda: engine.analyse(transcript, "b"), 20)

        per_1k = statistics.mean(timings) / (transcript.word_count / 1000.0)
        _report("analyse per 1k words", timings, f"= {per_1k:.2f} ms/1k words")
        assert per_1k < MAX_MEAN_MS_PER_1K_WORDS

    def test_parsing_is_not_the_bottleneck(self):
        content = _synthetic_transcript(120)
        timings = _time_ms(lambda: parse_transcript(content), 30)
        _report("parse (120 turns)", timings)
        assert statistics.mean(timings) < MAX_MEAN_MS_PER_CALL


class TestPhraseMatcherPerformance:
    def test_matching_is_sublinear_in_phrase_count(self):
        """The matcher indexes by first token, so adding phrases that never
        appear must not slow a pass down. If this regresses, someone replaced
        the index with a scan over all phrases."""
        tokens = tokenize(_synthetic_transcript(100))

        small = PhraseMatcher([("we have budget", 1.0, "b")])
        large = PhraseMatcher(
            [("we have budget", 1.0, "b")]
            + [(f"zzz{i} qqq{i}", 0.5, "noise") for i in range(2000)]
        )

        small_ms = statistics.median(_time_ms(lambda: small.find(tokens), 20))
        large_ms = statistics.median(_time_ms(lambda: large.find(tokens), 20))
        print(
            f"\n  matcher: 1 phrase {small_ms:.3f} ms vs 2001 phrases "
            f"{large_ms:.3f} ms ({large_ms / max(small_ms, 1e-6):.2f}x)"
        )

        # 2000x the phrases must not cost anywhere near 2000x the time.
        assert large_ms < max(small_ms, 0.05) * 5


@pytest.mark.slow
class TestBatchThroughput:
    def test_sustained_batch_rate(self, engine: AnalysisEngine):
        """Sizing figure for docs/OPERATIONS.md: calls analysed per second."""
        transcripts = [parse_transcript(_synthetic_transcript(40)) for _ in range(50)]

        start = time.perf_counter()
        for i, transcript in enumerate(transcripts):
            engine.analyse(transcript, f"call-{i}")
        elapsed = time.perf_counter() - start

        rate = len(transcripts) / elapsed
        total_words = sum(t.word_count for t in transcripts)
        print(
            f"\n  batch: {len(transcripts)} calls ({total_words} words) in "
            f"{elapsed:.2f}s = {rate:.0f} calls/s"
        )
        assert rate > 5
