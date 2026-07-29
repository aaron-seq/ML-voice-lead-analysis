"""Evaluation harness for the scoring rubric.

The rubric in ``scoring.py`` is a documented set of weights, not a model fitted
to data. That is the honest position given this project ships with no labelled
outcomes -- but it means nobody, including the authors, currently knows how
well the score predicts conversion.

This module is the path out of that. Point it at a CSV of calls with known
outcomes and it reports the metrics that would justify (or refute) the rubric.
Until someone runs it on real data, the README says the score is unvalidated,
because claiming "85% accuracy" without an evaluation set is how a credible
project becomes an incredible one.

Usage::

    python -m app.cli.main evaluate labelled_calls.csv

CSV columns: ``transcript_path``, ``converted`` (0/1), optional ``band``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from app.analysis.engine import AnalysisEngine
from app.domain.models import LeadBand


@dataclass
class BandStats:
    total: int = 0
    converted: int = 0

    @property
    def conversion_rate(self) -> float | None:
        return self.converted / self.total if self.total else None


@dataclass
class EvaluationReport:
    """What the rubric actually achieves on a labelled set."""

    n: int = 0
    by_band: dict[str, BandStats] = field(default_factory=dict)
    auc: float | None = None
    mean_score_converted: float | None = None
    mean_score_lost: float | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def separation(self) -> float | None:
        """Mean score gap between converted and lost.

        A rubric with no separation is decoration. Positive and large is good;
        near zero means the weights carry no signal for this team's funnel.
        """
        if self.mean_score_converted is None or self.mean_score_lost is None:
            return None
        return self.mean_score_converted - self.mean_score_lost

    def render(self) -> str:
        lines = [
            "Lead-score evaluation",
            "=" * 60,
            f"Calls evaluated: {self.n}",
        ]
        if self.auc is not None:
            lines.append(f"ROC AUC (score vs. conversion): {self.auc:.3f}")
        if self.separation is not None:
            lines.append(
                f"Mean score — converted: {self.mean_score_converted:.1f}, "
                f"lost: {self.mean_score_lost:.1f} "
                f"(separation {self.separation:+.1f})"
            )
        lines.append("")
        lines.append(f"{'Band':<14}{'Calls':>8}{'Converted':>12}{'Rate':>10}")
        lines.append("-" * 60)
        for band in (LeadBand.HOT, LeadBand.WARM, LeadBand.COLD,
                     LeadBand.DISQUALIFIED):
            stats = self.by_band.get(band.value, BandStats())
            rate = stats.conversion_rate
            rate_text = f"{rate:.1%}" if rate is not None else "—"
            lines.append(
                f"{band.value:<14}{stats.total:>8}{stats.converted:>12}{rate_text:>10}"
            )
        lines.append("")
        lines.append(
            "A calibrated rubric shows monotonically decreasing conversion rates "
            "from hot to cold. If it does not, retune WEIGHTS and "
            "BAND_THRESHOLDS in app/analysis/scoring.py."
        )
        if self.errors:
            lines.append("")
            lines.append(f"{len(self.errors)} row(s) failed:")
            lines.extend(f"  - {err}" for err in self.errors[:10])
        return "\n".join(lines)


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """AUC via the rank-sum identity. No numpy dependency.

    Returns None when one class is absent, since AUC is undefined there.
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    # Average ranks so ties contribute 0.5, matching the probabilistic reading
    # of AUC as P(score_pos > score_neg) + 0.5 * P(equal).
    ordered = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and scores[ordered[j + 1]] == scores[ordered[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[ordered[k]] = average_rank
        i = j + 1

    rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def evaluate(csv_path: Path, engine: AnalysisEngine | None = None) -> EvaluationReport:
    """Score a labelled set of transcripts and report calibration."""
    engine = engine or AnalysisEngine()
    report = EvaluationReport()
    scores: list[float] = []
    labels: list[int] = []
    base = csv_path.parent

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            raw_path = (row.get("transcript_path") or "").strip()
            if not raw_path:
                report.errors.append(f"row {row_number}: missing transcript_path")
                continue

            path = Path(raw_path)
            if not path.is_absolute():
                path = base / path

            try:
                content = path.read_text(encoding="utf-8")
                _, analysis = engine.analyse_text(content, call_id=path.stem,
                                                  filename=path.name)
                converted = int(str(row.get("converted", "0")).strip() in {"1", "true",
                                                                          "True", "yes"})
            except Exception as exc:
                report.errors.append(f"row {row_number} ({raw_path}): {exc}")
                continue

            band = analysis.lead_score.band.value
            stats = report.by_band.setdefault(band, BandStats())
            stats.total += 1
            stats.converted += converted

            scores.append(analysis.lead_score.score)
            labels.append(converted)

    report.n = len(scores)
    if report.n:
        report.auc = roc_auc(scores, labels)
        won = [s for s, y in zip(scores, labels, strict=True) if y]
        lost = [s for s, y in zip(scores, labels, strict=True) if not y]
        report.mean_score_converted = sum(won) / len(won) if won else None
        report.mean_score_lost = sum(lost) / len(lost) if lost else None
    return report
