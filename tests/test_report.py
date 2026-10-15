"""Report rendering.

Reports are the interface a reviewer reads, so the things that make a report
unusable get tests: an unescaped pipe in a table, a missing headline figure, a
gate verdict that does not say which cells moved.
"""

from __future__ import annotations

from geq.models import CellResult, EvidenceCheck, EvidenceStatus, Verdict, build_scorecard
from geq.regression import compare_runs
from geq.report import (
    render_comparison,
    render_failures,
    render_scorecard,
    render_summary,
    render_summary_table,
)
from geq.runner import RunResult


class TestScorecardRendering:
    def test_the_headline_leads_with_grounded_recall(self, v3_run):
        markdown = render_scorecard(v3_run)
        assert "Grounded recall: 100.0%" in markdown
        assert markdown.index("Grounded recall") < markdown.index("## Per field")

    def test_every_field_appears_in_the_per_field_table(self, v3_run):
        markdown = render_scorecard(v3_run)
        for name in v3_run.scorecard.per_field:
            assert f"`{name}`" in markdown

    def test_arithmetic_results_are_reported_per_document(self, v3_run):
        markdown = render_scorecard(v3_run)
        assert "balance_sheet_identity" in markdown
        assert "out by 100000" in markdown

    def test_a_clean_run_says_there_is_nothing_to_show(self, v3_run):
        assert "_None._" in render_failures(v3_run)

    def test_extractor_notes_are_carried_through(self, v1_run):
        markdown = render_scorecard(v1_run)
        assert "Extractor notes" in markdown
        assert "without citations" in markdown

    def test_the_failure_limit_is_respected_and_disclosed(self, v1_run):
        markdown = render_scorecard(v1_run, failure_limit=3)
        assert "further failing cell(s) omitted" in markdown
        section = markdown.split("## Cells that are not both right and traceable", 1)[1]
        section = section.split("## Extractor notes", 1)[0]
        assert section.count("\n| `case_") == 3

    def test_a_pipe_in_a_value_cannot_break_the_table(self):
        cell = CellResult(
            doc_id="doc",
            field="entity_name",
            verdict=Verdict.WRONG,
            gold="a|b",
            claimed="a | b",
            check=EvidenceCheck(
                field="entity_name",
                status=EvidenceStatus.CONTRADICTED,
                detail="value | mismatch",
            ),
        )
        card = build_scorecard(extractor="t", corpus="c", cells=(cell,))
        markdown = render_failures(RunResult(scorecard=card))
        for line in markdown.splitlines():
            if line.startswith("| `doc`"):
                assert line.count("|") >= 6
                assert "\\|" in line


class TestSummaryRendering:
    def test_one_row_per_scorecard(self, runs):
        table = render_summary_table([run.scorecard for run in runs.values()])
        lines = [line for line in table.splitlines() if line.startswith("| `rules:")]
        assert len(lines) == 3

    def test_the_table_explains_the_headline_column(self, runs):
        markdown = render_summary(list(runs.values()))
        assert "grounded recall" in markdown
        assert "**100.0%**" in markdown

    def test_the_best_revision_is_the_one_with_failures_listed(self, runs):
        markdown = render_summary(list(runs.values()))
        assert "Failing cells, best revision (`rules:v3`)" in markdown
        assert "Arithmetic failures, best revision" in markdown

    def test_an_empty_summary_does_not_crash(self):
        assert "Extractor revision summary" in render_summary([])


class TestComparisonRendering:
    def test_a_passing_gate_says_pass(self, v2_run, v3_run):
        markdown = render_comparison(compare_runs(v2_run, v3_run))
        assert "Regression gate: PASS" in markdown
        assert "## Regressions" not in markdown

    def test_a_failing_gate_states_its_reasons_first(self, v3_run, v2_run):
        markdown = render_comparison(compare_runs(v3_run, v2_run))
        assert "Regression gate: FAIL" in markdown
        assert "## Why this fails" in markdown
        assert markdown.index("Why this fails") < markdown.index("## Metrics")

    def test_regressed_cells_are_named(self, v3_run, v2_run):
        markdown = render_comparison(compare_runs(v3_run, v2_run))
        assert "## Regressions" in markdown
        assert "case_01_northgate_2024" in markdown

    def test_metric_deltas_are_signed(self, v2_run, v3_run):
        markdown = render_comparison(compare_runs(v2_run, v3_run))
        assert "+58.3%" in markdown
