"""CLI commands and the scoring-evaluation harness.

The CLI is a first-class entry point, not a convenience script: it runs the
same engine the API does, and `import` is the documented way to load a backlog
of transcripts. It gets tested like one.
"""

from __future__ import annotations

import json

import pytest

from app.analysis.evaluation import EvaluationReport, evaluate, roc_auc
from app.cli.main import main, render_analysis
from tests.conftest import QUALIFIED_CALL, REJECTION_CALL


@pytest.fixture
def transcript_file(tmp_path):
    path = tmp_path / "acme-call.txt"
    path.write_text(QUALIFIED_CALL, encoding="utf-8")
    return path


class TestAnalyseCommand:
    def test_prints_a_readable_report(self, transcript_file, capsys):
        assert main(["analyse", str(transcript_file), "--no-colour"]) == 0
        out = capsys.readouterr().out
        assert "SCORE BREAKDOWN" in out
        assert "NEXT ACTIONS" in out
        assert "hot" in out.lower()

    def test_json_output_is_machine_readable(self, transcript_file, capsys):
        assert main(["analyse", str(transcript_file), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["lead_score"]["band"] == "hot"
        assert payload["lead_score"]["components"]

    def test_writes_json_to_a_file(self, transcript_file, tmp_path, capsys):
        out_path = tmp_path / "result.json"
        assert main([
            "analyse", str(transcript_file), "--no-colour", "--out", str(out_path)
        ]) == 0
        assert json.loads(out_path.read_text())["call_id"] == "acme-call"

    def test_missing_file_exits_non_zero(self, tmp_path, capsys):
        assert main(["analyse", str(tmp_path / "nope.txt")]) == 2
        assert "no such file" in capsys.readouterr().err

    def test_unparseable_transcript_exits_non_zero(self, tmp_path, capsys):
        bad = tmp_path / "bad.txt"
        bad.write_text("[inaudible] [crosstalk]", encoding="utf-8")
        assert main(["analyse", str(bad), "--no-colour"]) == 1
        assert "error:" in capsys.readouterr().err

    def test_analyze_spelling_is_accepted(self, transcript_file, capsys):
        assert main(["analyze", str(transcript_file), "--no-colour"]) == 0

    def test_render_includes_warnings(self, engine):
        _, analysis = engine.analyse_text(QUALIFIED_CALL, call_id="t")
        rendered = render_analysis(analysis, colour=False)
        assert "WARNINGS" in rendered
        assert "%" in rendered  # confidence is shown


class TestImportCommand:
    def test_imports_a_directory(self, tmp_path, monkeypatch, capsys):
        source = tmp_path / "transcripts"
        source.mkdir()
        (source / "good.txt").write_text(QUALIFIED_CALL, encoding="utf-8")
        (source / "lost.txt").write_text(REJECTION_CALL, encoding="utf-8")

        monkeypatch.setenv(
            "DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'cli.db'}"
        )
        monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path / "blobs"))
        from app.core.config import get_settings

        get_settings.cache_clear()

        assert main(["import", str(source)]) == 0
        out = capsys.readouterr().out
        assert "Imported 2 call(s), 0 failed." in out
        assert "hot" in out
        get_settings.cache_clear()

    def test_missing_directory_exits_non_zero(self, tmp_path, capsys):
        assert main(["import", str(tmp_path / "nope")]) == 2

    def test_empty_directory_is_not_an_error(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert main(["import", str(empty)]) == 0
        assert "No transcripts found" in capsys.readouterr().out


class TestKeygenCommand:
    def test_prints_a_usable_key(self, capsys):
        assert main(["keygen", "--role", "admin"]) == 0
        out = capsys.readouterr().out
        assert "cs_" in out
        assert "AUTH_ENABLED=true" in out

    def test_generated_key_is_accepted_by_the_registry(self, capsys):
        main(["keygen"])
        key = next(
            line.strip()
            for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith("cs_")
        )
        from app.core.security import ApiKeyRegistry, Role

        assert ApiKeyRegistry([f"{key}:viewer"]).resolve(key).role is Role.VIEWER


class TestRocAuc:
    def test_perfect_separation_scores_one(self):
        assert roc_auc([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1]) == 1.0

    def test_inverted_separation_scores_zero(self):
        assert roc_auc([1.0, 2.0, 3.0, 4.0], [1, 1, 0, 0]) == 0.0

    def test_ties_score_one_half(self):
        assert roc_auc([1.0, 1.0, 1.0, 1.0], [0, 1, 0, 1]) == 0.5

    def test_single_class_is_undefined(self):
        """AUC has no meaning without both classes present."""
        assert roc_auc([1.0, 2.0], [1, 1]) is None
        assert roc_auc([1.0, 2.0], [0, 0]) is None


class TestEvaluationHarness:
    @pytest.fixture
    def labelled_csv(self, tmp_path):
        (tmp_path / "won.txt").write_text(QUALIFIED_CALL, encoding="utf-8")
        (tmp_path / "lost.txt").write_text(REJECTION_CALL, encoding="utf-8")
        csv_path = tmp_path / "labelled.csv"
        csv_path.write_text(
            "transcript_path,converted\nwon.txt,1\nlost.txt,0\n", encoding="utf-8"
        )
        return csv_path

    def test_evaluates_a_labelled_set(self, labelled_csv):
        report = evaluate(labelled_csv)
        assert report.n == 2
        assert report.by_band["hot"].converted == 1
        assert report.by_band["disqualified"].converted == 0

    def test_reports_separation_between_won_and_lost(self, labelled_csv):
        report = evaluate(labelled_csv)
        assert report.separation is not None
        assert report.separation > 0

    def test_renders_a_readable_report(self, labelled_csv):
        rendered = evaluate(labelled_csv).render()
        assert "Lead-score evaluation" in rendered
        assert "ROC AUC" in rendered
        assert "monotonically decreasing" in rendered

    def test_bad_rows_are_reported_not_fatal(self, tmp_path):
        csv_path = tmp_path / "labelled.csv"
        csv_path.write_text(
            "transcript_path,converted\nmissing.txt,1\n,0\n", encoding="utf-8"
        )
        report = evaluate(csv_path)
        assert report.n == 0
        assert len(report.errors) == 2

    def test_empty_report_renders_without_crashing(self):
        assert "Calls evaluated: 0" in EvaluationReport().render()

    def test_cli_evaluate_command(self, labelled_csv, capsys):
        assert main(["evaluate", str(labelled_csv)]) == 0
        assert "Lead-score evaluation" in capsys.readouterr().out

    def test_cli_evaluate_missing_file(self, tmp_path):
        assert main(["evaluate", str(tmp_path / "nope.csv")]) == 2
