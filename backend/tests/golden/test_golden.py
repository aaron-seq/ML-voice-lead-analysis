"""Golden-transcript regression suite (issue #45).

The rest of the suite asserts *properties*: a rejection outranks a positive
phrase, weights sum to 100. Properties survive a weight change that quietly
moves every call in a corpus by eight points. This file asserts *values*: the
band, outcome and score each pinned transcript produces today, so that a change
to the engine's judgement shows up as a diff a reviewer reads rather than as a
number someone notices is wrong a month later.

It does not claim the pinned scores are *correct* -- that needs labelled
outcomes (#42). It claims they do not change without someone meaning it to.

Updating
--------
When a change is supposed to move scores::

    pytest tests/golden --update-golden

That rewrites ``expected.json``, and it refuses to record moved scores unless
``ENGINE_VERSION`` was bumped in the same change. Stored analyses carry the
engine version that produced them, so a scoring change that keeps the old
version number makes old and new results indistinguishable. The refusal turns
that bump from something remembered into something enforced.

Adding a case needs no bump: drop a transcript into ``cases/`` (or
``examples/transcripts/``) and run the update.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.engine import AnalysisEngine
from app.domain.models import ENGINE_VERSION

GOLDEN_DIR = Path(__file__).resolve().parent
REPO_ROOT = GOLDEN_DIR.parents[2]
EXPECTED = GOLDEN_DIR / "expected.json"

# Every shipped example is pinned, plus targeted cases for behaviour that is
# easy to break without noticing. Examples are referenced in place rather than
# copied, so editing one fails this suite until the change is deliberate.
SOURCES = (REPO_ROOT / "examples" / "transcripts", GOLDEN_DIR / "cases")
SUFFIXES = {".txt", ".vtt", ".srt", ".json"}

# The engine is deterministic and rounds scores to one decimal place, so this
# is not slack for "close enough": it absorbs a last-digit rounding flip across
# platforms and nothing more. A real scoring change moves a score by far more.
SCORE_TOLERANCE = 0.1


def _case_paths() -> list[Path]:
    return sorted(
        path
        for source in SOURCES
        for path in source.iterdir()
        if path.is_file() and path.suffix in SUFFIXES
    )


def _key(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _analyse(path: Path) -> dict[str, object]:
    _, analysis = AnalysisEngine().analyse_text(
        path.read_bytes(), call_id=path.stem, filename=path.name
    )
    return {
        "band": analysis.lead_score.band.value,
        "outcome": analysis.outcome.label.value,
        "score": analysis.lead_score.score,
    }


def _differs(old: dict, new: dict) -> bool:
    return (
        old["band"] != new["band"]
        or old["outcome"] != new["outcome"]
        or abs(float(old["score"]) - float(new["score"])) > SCORE_TOLERANCE
    )


def _regenerate() -> None:
    fresh = {_key(path): _analyse(path) for path in _case_paths()}

    if EXPECTED.exists():
        recorded = json.loads(EXPECTED.read_text(encoding="utf-8"))
        moved = sorted(
            key
            for key, result in fresh.items()
            if key in recorded["cases"] and _differs(recorded["cases"][key], result)
        )
        if moved and recorded["engine_version"] == ENGINE_VERSION:
            pytest.fail(
                f"Scores moved for {len(moved)} pinned transcript(s) but "
                f"ENGINE_VERSION is still {ENGINE_VERSION}:\n  "
                + "\n  ".join(moved)
                + "\nBump ENGINE_VERSION in app/domain/models.py so stored "
                "analyses from the old engine stay distinguishable, then rerun "
                "with --update-golden.",
                pytrace=False,
            )

    EXPECTED.write_text(
        json.dumps(
            {"engine_version": ENGINE_VERSION, "cases": fresh},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def expected(request: pytest.FixtureRequest) -> dict:
    if request.config.getoption("--update-golden"):
        _regenerate()
    if not EXPECTED.exists():
        pytest.fail("expected.json is missing; run with --update-golden", pytrace=False)
    return json.loads(EXPECTED.read_text(encoding="utf-8"))


def test_every_transcript_on_disk_is_pinned(expected: dict) -> None:
    """A case added or deleted without updating the fixtures is a silent gap."""
    on_disk = {_key(path) for path in _case_paths()}
    pinned = set(expected["cases"])
    assert on_disk == pinned, (
        f"Unpinned: {sorted(on_disk - pinned)}; "
        f"stale: {sorted(pinned - on_disk)}. Run pytest tests/golden --update-golden."
    )


def test_fixtures_were_recorded_by_this_engine_version(expected: dict) -> None:
    """ENGINE_VERSION moved without regenerating, so the fixtures are vouching
    for an engine that no longer exists."""
    assert expected["engine_version"] == ENGINE_VERSION, (
        f"Fixtures were recorded by engine {expected['engine_version']}, this is "
        f"{ENGINE_VERSION}. Run pytest tests/golden --update-golden."
    )


@pytest.mark.parametrize("path", _case_paths(), ids=_key)
def test_pinned_result_is_unchanged(path: Path, expected: dict) -> None:
    pinned = expected["cases"].get(_key(path))
    if pinned is None:
        pytest.fail(f"{_key(path)} is not pinned; run --update-golden", pytrace=False)

    actual = _analyse(path)
    assert not _differs(pinned, actual), (
        f"{_key(path)} changed: pinned {pinned}, now {actual}. If this change is "
        "intended, bump ENGINE_VERSION and run pytest tests/golden --update-golden "
        "so the movement appears in the diff."
    )
