"""Scoring: turning cells into the numbers a decision can be made on.

The headline metric is **grounded recall** - the share of true values that were
both returned *and* backed by a real citation. Plain recall is reported
alongside it because the gap between the two is the whole point: an extractor
with 90% recall and 40% grounding is not 90% done, it is unusable for anything
that ends up in front of a deal team.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime

from .models import (
    CellResult,
    Claim,
    Extraction,
    FieldKind,
    FieldSpec,
    GoldenCase,
    Scorecard,
    Verdict,
    build_scorecard,
)
from .normalize import apply_scale, canonical
from .verify import ReconciliationResult, check_evidence, reconcile

__all__ = [
    "CaseReconciliation",
    "corpus_grounding_summary",
    "evaluate_case",
    "evaluate_corpus",
    "reconciliation_report",
    "scaled_claim_value",
]


def scaled_claim_value(claim: Claim, spec: FieldSpec) -> str | None:
    """The claim's value at document magnitude, i.e. what a reader would report."""
    base = canonical(claim.raw_value, spec.kind)
    return apply_scale(base, claim.scale, spec.kind)


def _verdict_for(gold_canonical: str | None, claimed: str | None) -> Verdict:
    if claimed is None:
        # A value that cannot be canonicalised cannot be verified as correct.
        return Verdict.WRONG
    return Verdict.RIGHT if claimed == gold_canonical else Verdict.WRONG


def evaluate_case(
    case: GoldenCase,
    extraction: Extraction,
    specs: Mapping[str, FieldSpec],
) -> tuple[CellResult, ...]:
    """Score one extraction against one hand-labelled document."""
    cells: list[CellResult] = []

    for name, spec in specs.items():
        gold = case.label(name)
        claim = extraction.claim(name)

        if claim is None:
            if spec.required and gold is not None:
                cells.append(
                    CellResult(
                        doc_id=case.doc_id,
                        field=name,
                        verdict=Verdict.MISSING,
                        gold=gold,
                    )
                )
            continue

        check = check_evidence(case.text, claim, spec)
        gold_canonical = canonical(gold, spec.kind) if gold is not None else None

        if gold is None:
            verdict = Verdict.EXTRA
        else:
            verdict = _verdict_for(gold_canonical, scaled_claim_value(claim, spec))

        cells.append(
            CellResult(
                doc_id=case.doc_id,
                field=name,
                verdict=verdict,
                gold=gold,
                claimed=claim.raw_value,
                check=check,
            )
        )

    known = set(specs)
    for name, claim in extraction.claims.items():
        if name in known:
            continue
        cells.append(
            CellResult(
                doc_id=case.doc_id,
                field=name,
                verdict=Verdict.EXTRA,
                claimed=claim.raw_value,
                check=check_evidence(
                    case.text, claim, FieldSpec(name=name, kind=FieldKind.TEXT, label=name)
                ),
            )
        )

    return tuple(cells)


def evaluate_corpus(
    cases: Sequence[GoldenCase],
    extractions: Mapping[str, Extraction],
    specs: Mapping[str, FieldSpec],
    *,
    corpus_name: str = "golden",
    extractor: str | None = None,
    generated_at: str | None = None,
) -> Scorecard:
    """Score every case in ``cases`` and aggregate into a :class:`Scorecard`."""
    name = extractor or next(iter(extractions.values())).extractor if extractions else "unknown"
    cells: list[CellResult] = []
    for case in cases:
        extraction = extractions.get(case.doc_id)
        if extraction is None:
            # A document the extractor never even attempted: every required
            # field is a miss, and the scorecard must not quietly shrink.
            cells.extend(
                CellResult(
                    doc_id=case.doc_id,
                    field=field,
                    verdict=Verdict.MISSING,
                    gold=case.labels.get(field),
                )
                for field, spec in specs.items()
                if spec.required and case.labels.get(field) is not None
            )
            continue
        cells.extend(evaluate_case(case, extraction, specs))

    return build_scorecard(
        extractor=name,
        corpus=corpus_name,
        cells=cells,
        generated_at=generated_at or datetime.now(UTC).isoformat(timespec="seconds"),
    )


def corpus_grounding_summary(scorecard: Scorecard) -> dict[str, float | int]:
    """A compact summary suitable for a CI log line or a report header."""
    return {
        "claims": scorecard.claims,
        "grounded": scorecard.grounded,
        "unsupported": scorecard.claims - scorecard.grounded,
        "citation_rate": round(scorecard.citation_rate, 4),
        "grounding_rate": round(scorecard.grounding_rate, 4),
        "recall": round(scorecard.micro_recall, 4),
        "grounded_recall": round(scorecard.grounded_recall, 4),
        "recall_gap": round(scorecard.micro_recall - scorecard.grounded_recall, 4),
    }


class CaseReconciliation:
    """Arithmetic check results for a single document."""

    __slots__ = ("doc_id", "results")

    def __init__(self, doc_id: str, results: Iterable[ReconciliationResult]) -> None:
        self.doc_id = doc_id
        self.results = tuple(results)

    @property
    def failures(self) -> tuple[ReconciliationResult, ...]:
        return tuple(r for r in self.results if r.status == "fail")

    @property
    def not_checkable(self) -> tuple[ReconciliationResult, ...]:
        return tuple(r for r in self.results if r.status == "not_checkable")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CaseReconciliation(doc_id={self.doc_id!r}, results={self.results!r})"


def reconciliation_report(
    cases: Sequence[GoldenCase],
    extractions: Mapping[str, Extraction],
    specs: Mapping[str, FieldSpec],
) -> tuple[CaseReconciliation, ...]:
    """Run the arithmetic checks over every case that was attempted."""
    out: list[CaseReconciliation] = []
    for case in cases:
        extraction = extractions.get(case.doc_id)
        if extraction is None:
            continue
        out.append(CaseReconciliation(case.doc_id, reconcile(extraction.claims, specs)))
    return tuple(out)
