"""Regression gating between two runs.

The rule this module enforces: **no cell may get worse.** Aggregate metrics can
improve while individual documents break, and a prompt revision that lifts macro
recall by dropping the citations on three statements is not an improvement.

Two details that make the gate hard to game:

* It compares cell sets before comparing values. If the golden set changed
  between runs, the runs are not comparable and the gate fails outright rather
  than reporting a meaningless improvement.
* Value accuracy and grounding regress independently. Losing a citation is a
  regression even when the number stays right, because the number is no longer
  auditable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .models import CellResult, Scorecard, Verdict
from .runner import RunResult

__all__ = ["CellChange", "RegressionPolicy", "RegressionReport", "compare_runs"]

ChangeKind = Literal["fixed", "regressed", "unchanged_broken", "unchanged_right"]


@dataclass(frozen=True, slots=True)
class RegressionPolicy:
    """What the gate tolerates. Defaults are deliberately strict."""

    max_regressions: int = 0
    max_grounded_recall_drop: float = 0.0
    max_grounding_rate_drop: float = 0.0
    max_recall_drop: float = 0.0
    fail_on_corpus_change: bool = True
    fail_on_arithmetic_regression: bool = True


@dataclass(frozen=True, slots=True)
class CellState:
    """The two independent facts about one cell."""

    value_ok: bool
    grounded: bool

    def label(self) -> str:
        if self.value_ok and self.grounded:
            return "right_and_grounded"
        if self.value_ok:
            return "right_ungrounded"
        if self.grounded:
            return "wrong_but_grounded"
        return "not_established"

    @property
    def stars(self) -> int:
        return int(self.value_ok) + int(self.grounded)


def cell_state(cell: CellResult) -> CellState:
    return CellState(value_ok=cell.verdict is Verdict.RIGHT, grounded=cell.grounded)


@dataclass(frozen=True, slots=True)
class CellChange:
    doc_id: str
    field: str
    baseline: str
    candidate: str
    kind: ChangeKind
    reason: str = ""

    @property
    def improved(self) -> bool:
        return self.kind == "fixed"


@dataclass(frozen=True, slots=True)
class MetricDelta:
    name: str
    baseline: float
    candidate: float

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def improved(self) -> bool:
        return self.delta > 0

    @property
    def regressed(self) -> bool:
        return self.delta < 0


@dataclass(frozen=True, slots=True)
class RegressionReport:
    baseline_extractor: str
    candidate_extractor: str
    changes: tuple[CellChange, ...]
    deltas: tuple[MetricDelta, ...]
    missing_cells: tuple[str, ...] = ()
    unexpected_cells: tuple[str, ...] = ()
    arithmetic_regressions: tuple[str, ...] = ()
    policy: RegressionPolicy = RegressionPolicy()

    # -- selection ------------------------------------------------------

    @property
    def regressions(self) -> tuple[CellChange, ...]:
        return tuple(c for c in self.changes if c.kind == "regressed")

    @property
    def fixes(self) -> tuple[CellChange, ...]:
        return tuple(c for c in self.changes if c.kind == "fixed")

    @property
    def corpus_changed(self) -> bool:
        return bool(self.missing_cells or self.unexpected_cells)

    def delta(self, name: str) -> MetricDelta | None:
        return next((d for d in self.deltas if d.name == name), None)

    # -- verdict --------------------------------------------------------

    def failures(self) -> tuple[str, ...]:
        """Every reason the gate refuses this change, in a stable order."""
        reasons: list[str] = []
        if self.corpus_changed and self.policy.fail_on_corpus_change:
            if self.missing_cells:
                reasons.append(
                    f"corpus changed: {len(self.missing_cells)} cell(s) present in the "
                    f"baseline are absent from the candidate ({_preview(self.missing_cells)})"
                )
            if self.unexpected_cells:
                reasons.append(
                    f"corpus changed: {len(self.unexpected_cells)} new cell(s) appeared "
                    f"({_preview(self.unexpected_cells)})"
                )
        if len(self.regressions) > self.policy.max_regressions:
            names = [f"{c.doc_id}/{c.field}" for c in self.regressions]
            reasons.append(
                f"{len(self.regressions)} cell(s) regressed (allowed: "
                f"{self.policy.max_regressions}): {_preview(names)}"
            )
        for name, allowance in (
            ("grounded_recall", self.policy.max_grounded_recall_drop),
            ("grounding_rate", self.policy.max_grounding_rate_drop),
            ("micro_recall", self.policy.max_recall_drop),
        ):
            d = self.delta(name)
            if d is not None and d.delta < -abs(allowance) - 1e-9:
                reasons.append(
                    f"{name} fell by {abs(d.delta):.4f} ({d.baseline:.4f} -> {d.candidate:.4f})"
                )
        if self.arithmetic_regressions and self.policy.fail_on_arithmetic_regression:
            reasons.append(
                "arithmetic checks regressed: " + _preview(list(self.arithmetic_regressions))
            )
        return tuple(reasons)

    @property
    def passed(self) -> bool:
        return not self.failures()

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline_extractor,
            "candidate": self.candidate_extractor,
            "passed": self.passed,
            "failures": list(self.failures()),
            "corpus_changed": self.corpus_changed,
            "counts": {
                "cells_compared": len(self.changes),
                "fixed": len(self.fixes),
                "regressed": len(self.regressions),
                "unchanged_right": sum(1 for c in self.changes if c.kind == "unchanged_right"),
                "unchanged_broken": sum(1 for c in self.changes if c.kind == "unchanged_broken"),
            },
            "deltas": [
                {
                    "name": d.name,
                    "baseline": round(d.baseline, 4),
                    "candidate": round(d.candidate, 4),
                    "delta": round(d.delta, 4),
                }
                for d in self.deltas
            ],
            "changes": [
                {
                    "doc_id": c.doc_id,
                    "field": c.field,
                    "baseline": c.baseline,
                    "candidate": c.candidate,
                    "kind": c.kind,
                    "reason": c.reason,
                }
                for c in self.changes
            ],
            "arithmetic_regressions": list(self.arithmetic_regressions),
        }


def _preview(items: Sequence[str], limit: int = 5) -> str:
    shown = ", ".join(items[:limit])
    return f"{shown}, ..." if len(items) > limit else shown


METRIC_NAMES = (
    "micro_recall",
    "micro_precision",
    "micro_f1",
    "grounding_rate",
    "grounded_recall",
    "citation_rate",
)


def _metrics(scorecard: Scorecard) -> dict[str, float]:
    return {
        "micro_recall": scorecard.micro_recall,
        "micro_precision": scorecard.micro_precision,
        "micro_f1": scorecard.micro_f1,
        "grounding_rate": scorecard.grounding_rate,
        "grounded_recall": scorecard.grounded_recall,
        "citation_rate": scorecard.citation_rate,
    }


def _comparable_cells(scorecard: Scorecard) -> dict[tuple[str, str], CellResult]:
    """Cells that exist because the golden set says so, not because the extractor did."""
    return {
        (cell.doc_id, cell.field): cell
        for cell in scorecard.cells
        if cell.verdict is not Verdict.EXTRA
    }


def _classify(baseline: CellState, candidate: CellState) -> tuple[ChangeKind, str]:
    if candidate.stars > baseline.stars:
        return "fixed", f"{baseline.label()} -> {candidate.label()}"
    if candidate.stars < baseline.stars:
        lost: list[str] = []
        if baseline.value_ok and not candidate.value_ok:
            lost.append("value")
        if baseline.grounded and not candidate.grounded:
            lost.append("grounding")
        return "regressed", f"{baseline.label()} -> {candidate.label()} (lost {' and '.join(lost)})"
    if candidate.stars == 2:
        return "unchanged_right", candidate.label()
    return "unchanged_broken", candidate.label()


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    *,
    policy: RegressionPolicy | None = None,
) -> RegressionReport:
    """Compare two runs cell by cell and decide whether the change is safe."""
    policy = policy or RegressionPolicy()
    base_cells = _comparable_cells(baseline.scorecard)
    cand_cells = _comparable_cells(candidate.scorecard)

    missing = tuple(
        sorted(f"{doc}/{field}" for doc, field in base_cells.keys() - cand_cells.keys())
    )
    unexpected = tuple(
        sorted(f"{doc}/{field}" for doc, field in cand_cells.keys() - base_cells.keys())
    )

    changes: list[CellChange] = []
    for key in sorted(base_cells.keys() & cand_cells.keys()):
        doc_id, field = key
        before = cell_state(base_cells[key])
        after = cell_state(cand_cells[key])
        kind, reason = _classify(before, after)
        changes.append(
            CellChange(
                doc_id=doc_id,
                field=field,
                baseline=before.label(),
                candidate=after.label(),
                kind=kind,
                reason=reason,
            )
        )

    base_metrics = _metrics(baseline.scorecard)
    cand_metrics = _metrics(candidate.scorecard)
    deltas = tuple(
        MetricDelta(name=name, baseline=base_metrics[name], candidate=cand_metrics[name])
        for name in METRIC_NAMES
    )

    arithmetic_regressions = _arithmetic_regressions(baseline, candidate)

    return RegressionReport(
        baseline_extractor=baseline.extractor,
        candidate_extractor=candidate.extractor,
        changes=tuple(changes),
        deltas=deltas,
        missing_cells=missing,
        unexpected_cells=unexpected,
        arithmetic_regressions=arithmetic_regressions,
        policy=policy,
    )


def _arithmetic_regressions(baseline: RunResult, candidate: RunResult) -> tuple[str, ...]:
    """Cases whose arithmetic checks passed before and fail now."""

    def status_map(run: RunResult) -> dict[tuple[str, str], str]:
        return {
            (case.doc_id, result.name): result.status
            for case in run.reconciliations
            for result in case.results
        }

    before = status_map(baseline)
    after = status_map(candidate)
    out: list[str] = []
    for key in sorted(before.keys() | after.keys()):
        if before.get(key) == "pass" and after.get(key) == "fail":
            out.append(f"{key[0]}/{key[1]}")
    return tuple(out)
