"""Regression gate behaviour.

A gate is only useful if it can be shown to fire. These tests do that from both
directions: a genuine improvement passes, the same change run backwards fails,
and a changed golden set fails even when the metrics look identical.
"""

from __future__ import annotations

import pytest

from geq.models import CellResult, Verdict, build_scorecard
from geq.regression import (
    CellState,
    RegressionPolicy,
    cell_state,
    compare_runs,
)
from geq.runner import RunResult


class TestCellState:
    @pytest.mark.parametrize(
        ("verdict", "grounded", "expected"),
        [
            (Verdict.RIGHT, True, "right_and_grounded"),
            (Verdict.RIGHT, False, "right_ungrounded"),
            (Verdict.WRONG, True, "wrong_but_grounded"),
            (Verdict.WRONG, False, "not_established"),
            (Verdict.MISSING, False, "not_established"),
        ],
    )
    def test_labels_the_pair_of_facts(self, verdict, grounded, expected):
        cell = CellResult(
            doc_id="d",
            field="f",
            verdict=verdict,
            check=None if not grounded else _supported_check("f"),
        )
        assert cell_state(cell).label() == expected

    def test_stars_count_value_and_grounding_separately(self):
        assert CellState(value_ok=True, grounded=True).stars == 2
        assert CellState(value_ok=True, grounded=False).stars == 1
        assert CellState(value_ok=False, grounded=True).stars == 1
        assert CellState(value_ok=False, grounded=False).stars == 0


def _supported_check(field: str):
    from geq.models import EvidenceCheck, EvidenceStatus

    return EvidenceCheck(field=field, status=EvidenceStatus.SUPPORTED, detail="ok")


class TestRealRevisionComparison:
    def test_v2_to_v3_passes_the_gate(self, v2_run, v3_run):
        report = compare_runs(v2_run, v3_run)
        assert report.passed
        assert report.failures() == ()
        assert len(report.fixes) == 28
        assert report.regressions == ()

    def test_v2_to_v3_improves_the_headline_metric(self, v2_run, v3_run):
        report = compare_runs(v2_run, v3_run)
        grounded = report.delta("grounded_recall")
        assert grounded is not None
        assert grounded.baseline == pytest.approx(0.4167, abs=1e-3)
        assert grounded.candidate == pytest.approx(1.0)
        assert grounded.improved

    def test_v3_to_v2_fails_the_gate(self, v3_run, v2_run):
        report = compare_runs(v3_run, v2_run)
        assert not report.passed
        assert len(report.regressions) == 28
        assert any("28 cell(s) regressed" in reason for reason in report.failures())
        assert any("grounded_recall fell" in reason for reason in report.failures())

    def test_a_reversed_change_is_caught_by_the_arithmetic_checks_too(self, v3_run, v2_run):
        report = compare_runs(v3_run, v2_run)
        assert report.arithmetic_regressions
        assert any("balance_sheet_identity" in name for name in report.arithmetic_regressions)

    def test_identical_runs_report_no_change(self, v3_run):
        report = compare_runs(v3_run, v3_run)
        assert report.passed
        assert report.fixes == ()
        assert report.regressions == ()
        assert all(d.delta == pytest.approx(0.0) for d in report.deltas)

    def test_every_change_is_classified(self, v2_run, v3_run):
        report = compare_runs(v2_run, v3_run)
        assert len(report.changes) == 48
        assert {c.kind for c in report.changes} <= {
            "fixed",
            "regressed",
            "unchanged_broken",
            "unchanged_right",
        }
        assert all(c.reason for c in report.changes)

    def test_cell_counts_add_up(self, v2_run, v3_run):
        report = compare_runs(v2_run, v3_run)
        counts = report.to_dict()["counts"]
        assert counts["cells_compared"] == 48
        assert (
            counts["fixed"]
            + counts["regressed"]
            + counts["unchanged_right"]
            + counts["unchanged_broken"]
            == 48
        )

    def test_the_report_is_serialisable(self, v3_run, v2_run):
        payload = compare_runs(v3_run, v2_run).to_dict()
        assert payload["passed"] is False
        assert payload["failures"]
        assert len(payload["changes"]) == 48


class TestCorpusChangeDetection:
    """The gate must refuse to compare two differently-shaped runs."""

    @staticmethod
    def run_from_cells(extractor: str, cells) -> RunResult:
        return RunResult(
            scorecard=build_scorecard(extractor=extractor, corpus="golden", cells=cells)
        )

    def test_a_shrinking_corpus_fails_even_with_identical_metrics(self, v3_run):
        subset = tuple(
            c for c in v3_run.scorecard.cells if c.doc_id != "case_06_harbour_point_2025h1"
        )
        shrunk = self.run_from_cells("rules:v3", subset)
        report = compare_runs(v3_run, shrunk)
        assert report.corpus_changed
        assert not report.passed
        assert any("corpus changed" in reason for reason in report.failures())

    def test_a_new_cell_appearing_is_also_a_corpus_change(self, v3_run):
        extra = (
            *v3_run.scorecard.cells,
            CellResult(doc_id="new", field="total_assets", verdict=Verdict.MISSING),
        )
        grown = self.run_from_cells("rules:v3", extra)
        report = compare_runs(v3_run, grown)
        assert report.corpus_changed
        assert report.unexpected_cells == ("new/total_assets",)
        assert not report.passed

    def test_corpus_change_can_be_permitted_explicitly(self, v3_run):
        subset = tuple(
            c for c in v3_run.scorecard.cells if c.doc_id != "case_06_harbour_point_2025h1"
        )
        shrunk = self.run_from_cells("rules:v3", subset)
        report = compare_runs(v3_run, shrunk, policy=RegressionPolicy(fail_on_corpus_change=False))
        assert not report.corpus_changed or report.passed


class TestPolicyKnobs:
    def test_the_gate_can_be_told_to_tolerate_a_known_regression(self, v3_run, v2_run):
        lenient = RegressionPolicy(
            max_regressions=100,
            max_grounded_recall_drop=1.0,
            max_grounding_rate_drop=1.0,
            max_recall_drop=1.0,
            fail_on_arithmetic_regression=False,
        )
        report = compare_runs(v3_run, v2_run, policy=lenient)
        assert report.passed
        assert report.failures() == ()

    def test_the_default_policy_is_strict(self):
        policy = RegressionPolicy()
        assert policy.max_regressions == 0
        assert policy.max_grounded_recall_drop == 0.0
        assert policy.fail_on_corpus_change
        assert policy.fail_on_arithmetic_regression

    def test_a_single_tolerated_regression_still_fails_on_the_metric_drop(self, v3_run, v2_run):
        report = compare_runs(v3_run, v2_run, policy=RegressionPolicy(max_regressions=100))
        assert not report.passed, "one regressed cell is well within tolerance; the drop is not"
