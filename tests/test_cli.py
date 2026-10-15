"""CLI tests: exit codes and artefacts.

Exit codes are the contract CI depends on, so they are asserted explicitly rather
than inferred from the printed output.
"""

from __future__ import annotations

import json

import pytest

from geq.cli import build_parser, main


class TestRun:
    def test_writes_a_json_and_a_markdown_report(self, tmp_path, repo_root):
        out = tmp_path / "reports" / "v3"
        code = main(
            [
                "run",
                "--corpus",
                str(repo_root / "data" / "golden"),
                "--extractor",
                "rules:v3",
                "--out",
                str(out),
            ]
        )
        assert code == 0
        payload = json.loads(out.with_suffix(".json").read_text())
        assert payload["metrics"]["grounded_recall"] == pytest.approx(1.0)
        assert "Grounded recall" in out.with_suffix(".md").read_text()

    def test_runs_without_writing_anything(self, tmp_path, repo_root, capsys):
        code = main(
            ["run", "--corpus", str(repo_root / "data" / "golden"), "--extractor", "rules:v1"]
        )
        assert code == 0
        assert "rules:v1" in capsys.readouterr().out


class TestCompare:
    @staticmethod
    def _prepare(tmp_path, repo_root, extractor: str) -> str:
        out = tmp_path / f"{extractor.replace(':', '-')}.json"
        main(
            [
                "run",
                "--corpus",
                str(repo_root / "data" / "golden"),
                "--extractor",
                extractor,
                "--out",
                str(out.with_suffix("")),
            ]
        )
        return str(out)

    def test_an_improvement_exits_zero(self, tmp_path, repo_root):
        baseline = self._prepare(tmp_path, repo_root, "rules:v2")
        candidate = self._prepare(tmp_path, repo_root, "rules:v3")
        assert main(["compare", "--baseline", baseline, "--candidate", candidate]) == 0

    def test_a_regression_exits_non_zero(self, tmp_path, repo_root):
        baseline = self._prepare(tmp_path, repo_root, "rules:v3")
        candidate = self._prepare(tmp_path, repo_root, "rules:v2")
        assert main(["compare", "--baseline", baseline, "--candidate", candidate]) == 1

    def test_the_verdict_is_written_to_a_file_when_asked(self, tmp_path, repo_root):
        baseline = self._prepare(tmp_path, repo_root, "rules:v2")
        candidate = self._prepare(tmp_path, repo_root, "rules:v3")
        out = tmp_path / "gate.md"
        main(["compare", "--baseline", baseline, "--candidate", candidate, "--out", str(out)])
        assert "Regression gate: PASS" in out.read_text()


class TestSqlqa:
    def test_the_fixture_database_reports_its_planted_defects(self, repo_root, capsys):
        code = main(
            [
                "sqlqa",
                "--schema",
                str(repo_root / "sql" / "schema.sql"),
                "--seed",
                str(repo_root / "sql" / "seed.sql"),
                "--checks",
                str(repo_root / "sql" / "checks"),
            ]
        )
        assert code == 0, "the fixture is supposed to be dirty; the CLI reports, it does not gate"
        assert "7 offending row(s)" in capsys.readouterr().out

    def test_expect_clean_turns_findings_into_a_failure(self, repo_root):
        code = main(
            [
                "sqlqa",
                "--schema",
                str(repo_root / "sql" / "schema.sql"),
                "--seed",
                str(repo_root / "sql" / "seed.sql"),
                "--checks",
                str(repo_root / "sql" / "checks"),
                "--expect-clean",
            ]
        )
        assert code == 1

    def test_wihtout_a_seed_there_is_nothing_to_find(self, repo_root, capsys):
        code = main(
            [
                "sqlqa",
                "--schema",
                str(repo_root / "sql" / "schema.sql"),
                "--seed",
                "",
                "--checks",
                str(repo_root / "sql" / "checks"),
                "--expect-clean",
            ]
        )
        assert code == 0
        assert "6/6 checks clean" in capsys.readouterr().out


class TestSummaryAndExtract:
    def test_summary_covers_every_run_in_the_directory(self, tmp_path, repo_root, capsys):
        reports = tmp_path / "reports"
        for extractor in ("rules:v1", "rules:v2", "rules:v3"):
            main(
                [
                    "run",
                    "--corpus",
                    str(repo_root / "data" / "golden"),
                    "--extractor",
                    extractor,
                    "--out",
                    str(reports / extractor.replace(":", "-")),
                ]
            )
        assert main(["summary", "--reports", str(reports)]) == 0
        out = capsys.readouterr().out
        assert "`rules:v1`" in out and "`rules:v3`" in out

    def test_summary_with_no_runs_reports_an_error(self, tmp_path):
        assert main(["summary", "--reports", str(tmp_path)]) == 2

    def test_extract_prints_one_document_with_citations(self, repo_root, capsys):
        code = main(
            [
                "extract",
                "--corpus",
                str(repo_root / "data" / "golden"),
                "--doc",
                "case_01_northgate_2024",
                "--extractor",
                "rules:v3",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "case_01_northgate_2024" in out
        assert "13,504,500" in out
        assert "supported" in out


class TestParser:
    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_unknown_arguments_are_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--nope"])

    def test_defaults_are_repo_relative(self):
        args = build_parser().parse_args(["run"])
        assert args.corpus == "data/golden"
        assert args.extractor == "rules:v3"
