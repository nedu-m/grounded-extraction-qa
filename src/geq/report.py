"""Markdown rendering.

Reports are written in the form a reviewer actually reads: the decision first
(did this pass?), then the metric that justifies it, then the individual cells
that failed with their citations, so a disagreement can be checked by hand
without re-running anything.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import CellResult, Scorecard, Verdict
from .regression import RegressionReport
from .runner import RunResult

__all__ = [
    "render_comparison",
    "render_failures",
    "render_scorecard",
    "render_summary",
    "render_summary_table",
]

_DASH = "-"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_summary_table(scorecards: Sequence[Scorecard]) -> str:
    """The revision-over-revision table: the whole story on one screen."""
    header = (
        "| extractor | recall | precision | F1 | citing | grounding | **grounded recall** |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows: list[str] = []
    for card in scorecards:
        rows.append(
            f"| `{card.extractor}` | {_pct(card.micro_recall)} | {_pct(card.micro_precision)} | "
            f"{_pct(card.micro_f1)} | {_pct(card.citation_rate)} | {_pct(card.grounding_rate)} | "
            f"**{_pct(card.grounded_recall)}** |"
        )
    return header + "\n" + "\n".join(rows)


def render_scorecard(run: RunResult, *, failure_limit: int = 12) -> str:
    card = run.scorecard
    lines: list[str] = []
    lines.append(f"# Scorecard: `{card.extractor}`")
    lines.append("")
    lines.append(
        f"Corpus `{card.corpus}` | {len(card.cells)} cells | generated {card.generated_at}"
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- **Grounded recall: {_pct(card.grounded_recall)}** "
        f"({card.right_and_grounded}/{card.gold} true values returned *and* traceable)"
    )
    lines.append(f"- Recall: {_pct(card.micro_recall)} ({card.right}/{card.gold})")
    lines.append(f"- Precision: {_pct(card.micro_precision)} ({card.right}/{card.claims} claims)")
    lines.append(
        f"- Grounding rate: {_pct(card.grounding_rate)} ({card.grounded}/{card.claims} claims "
        f"whose citation contains the claimed value)"
    )
    lines.append(f"- Citation rate: {_pct(card.citation_rate)}")
    gap = card.micro_recall - card.grounded_recall
    lines.append(
        f"- Recall not traceable: {_pct(gap)} of the corpus is asserted without usable evidence"
    )
    lines.append("")
    lines.append("## Arithmetic checks")
    lines.append("")
    lines.append("| document | check | status | detail |")
    lines.append("| --- | --- | --- | --- |")
    for case in run.reconciliations:
        for result in case.results:
            lines.append(
                f"| `{case.doc_id}` | `{result.name}` | {result.status} | {result.detail} |"
            )
    lines.append("")
    lines.append("## Per field")
    lines.append("")
    lines.append(
        "| field | gold | right | wrong | missing | extra | claims | grounded "
        "| precision | recall | grounded recall |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, score in card.per_field.items():
        lines.append(
            f"| `{name}` | {score.gold} | {score.right} | {score.wrong} | {score.missing} | "
            f"{score.extra} | {score.claims} | {score.grounded} | {_pct(score.precision)} | "
            f"{_pct(score.recall)} | {_pct(score.grounded_recall)} |"
        )
    lines.append("")
    lines.append(f"## Cells that are not both right and traceable ({len(card.failures())})")
    lines.append("")
    lines.append(render_failures(run, limit=failure_limit))
    lines.append("")
    if run.extractions:
        notes = {
            doc_id: extraction.notes
            for doc_id, extraction in run.extractions.items()
            if extraction.notes
        }
        if notes:
            lines.append("## Extractor notes")
            lines.append("")
            for doc_id in sorted(notes):
                for note in notes[doc_id]:
                    lines.append(f"- `{doc_id}`: {note}")
            lines.append("")
    return "\n".join(lines)


def render_failures(run: RunResult, *, limit: int | None = None) -> str:
    card = run.scorecard
    failures = card.failures(limit=limit)
    if not failures:
        return "_None._"

    lines = [
        "| document | field | verdict | gold | claimed | evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell in failures:
        lines.append(
            f"| `{cell.doc_id}` | `{cell.field}` | {cell.verdict.value} | "
            f"{_cell(cell.gold)} | {_cell(cell.claimed)} | {_evidence(cell)} |"
        )
    remaining = len(card.failures()) - len(failures)
    if remaining > 0:
        lines.append("")
        lines.append(f"_{remaining} further failing cell(s) omitted._")
    return "\n".join(lines)


def _cell(value: str | None) -> str:
    if value is None:
        return _DASH
    text = value.replace("|", "\\|")
    return f"`{text}`" if text else _DASH


def _evidence(cell: CellResult) -> str:
    if cell.check is None:
        return f"{Verdict.MISSING.value}: nothing claimed"
    status = cell.check.status.value
    detail = cell.check.detail.replace("|", "\\|")
    if len(detail) > 110:
        detail = detail[:107] + "..."
    return f"**{status}**: {detail}"


def render_comparison(report: RegressionReport) -> str:
    lines: list[str] = []
    verdict = "PASS" if report.passed else "FAIL"
    lines.append(f"# Regression gate: {verdict}")
    lines.append("")
    lines.append(
        f"`{report.baseline_extractor}` (baseline) -> `{report.candidate_extractor}` (candidate)"
    )
    lines.append("")
    counts = {
        "fixed": len(report.fixes),
        "regressed": len(report.regressions),
        "unchanged_right": sum(1 for c in report.changes if c.kind == "unchanged_right"),
        "unchanged_broken": sum(1 for c in report.changes if c.kind == "unchanged_broken"),
    }
    lines.append(
        "Cells compared: {cells} ({fixed} fixed, {regressed} regressed, "
        "{unchanged_right} unchanged and correct, {unchanged_broken} still broken)".format(
            cells=len(report.changes),
            fixed=counts["fixed"],
            regressed=counts["regressed"],
            unchanged_right=counts["unchanged_right"],
            unchanged_broken=counts["unchanged_broken"],
        )
    )
    lines.append("")
    if reasons := report.failures():
        lines.append("## Why this fails")
        lines.append("")
        for reason in reasons:
            lines.append(f"- {reason}")
        lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | baseline | candidate | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for d in report.deltas:
        arrow = "+" if d.delta > 0 else ""
        lines.append(
            f"| `{d.name}` | {_pct(d.baseline)} | {_pct(d.candidate)} | {arrow}{_pct(d.delta)} |"
        )
    lines.append("")

    if report.regressions:
        lines.append("## Regressions")
        lines.append("")
        lines.append("| document | field | baseline | candidate |")
        lines.append("| --- | --- | --- | --- |")
        for change in report.regressions:
            lines.append(
                f"| `{change.doc_id}` | `{change.field}` | {change.baseline} | {change.candidate} |"
            )
        lines.append("")

    if report.fixes:
        lines.append(f"## Fixes ({len(report.fixes)}), first 25")
        lines.append("")
        lines.append("| document | field | baseline | candidate |")
        lines.append("| --- | --- | --- | --- |")
        for change in report.fixes[:25]:
            lines.append(
                f"| `{change.doc_id}` | `{change.field}` | {change.baseline} | {change.candidate} |"
            )
        lines.append("")
        if len(report.fixes) > 25:
            lines.append(f"_{len(report.fixes) - 25} further fix(es) omitted._")
            lines.append("")

    still_broken = [f"{c.doc_id}/{c.field}" for c in report.changes if c.kind == "unchanged_broken"]
    if still_broken:
        lines.append("## Still broken after this change")
        lines.append("")
        for name in still_broken[:40]:
            lines.append(f"- `{name}`")
        lines.append("")

    if report.arithmetic_regressions:
        lines.append("## Arithmetic regressions")
        lines.append("")
        for name in report.arithmetic_regressions:
            lines.append(f"- `{name}`")
        lines.append("")

    return "\n".join(lines)


def render_summary(runs: Sequence[RunResult]) -> str:
    """A cross-revision summary, including the failing cells of the best run."""
    cards = [run.scorecard for run in runs]
    lines = [
        "# Extractor revision summary",
        "",
        render_summary_table(cards),
        "",
        "`grounded recall` is the share of true values that were returned **and** backed by a",
        "citation whose text contains that value. It is the only column that licenses putting a",
        "number in front of a reviewer.",
        "",
    ]
    if runs:
        best = max(runs, key=lambda run: run.scorecard.grounded_recall)
        lines.append(f"## Failing cells, best revision (`{best.extractor}`)")
        lines.append("")
        lines.append(render_failures(best))
        lines.append("")
        if best.reconciliation_failures:
            lines.append("## Arithmetic failures, best revision")
            lines.append("")
            for doc_id, result in best.reconciliation_failures:
                lines.append(f"- `{doc_id}` `{result.name}`: {result.detail}")
            lines.append("")
    return "\n".join(lines)
