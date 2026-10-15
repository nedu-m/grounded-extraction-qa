"""Evidence checks: the part of the harness that catches fabricated values.

A value is only *grounded* when the text the extractor cited as its source really
does contain that value. This module answers that question for one claim, and
then answers a second, larger question: do the extracted numbers add up?

Three citation failure modes are separated on purpose, because they need
different fixes:

* ``UNCITED`` - nothing was cited at all. Fix the prompt or the extractor
  contract. Cannot be audited by a human reviewer.
* ``CONTRADICTED`` - real text was cited and it does not contain the claimed
  value. This is the fabricated-fact case, and the one that matters most.
* ``NOT_A_CITATION`` - the span is so wide it covers the whole document. It is
  technically "supported" by text that happens to contain the number, and it
  proves nothing to a reviewer.

Everything here is pure: no extractor, no I/O, no LLM. It takes a document and a
claim and returns a verdict that a test can assert on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from .models import Claim, EvidenceCheck, EvidenceStatus, FieldKind, FieldSpec
from .normalize import (
    apply_scale,
    canonical,
    canonical_date,
    canonical_text,
    money_tokens_in,
    signed_decimal,
)

__all__ = [
    "MAX_EVIDENCE_CHARS",
    "ReconciliationResult",
    "check_balance_sheet",
    "check_evidence",
    "check_totals_footing",
    "reconcile",
    "span_text",
]

MAX_EVIDENCE_CHARS = 280
"""Anything wider than this is a search result, not a citation."""


def span_text(document: str, span: tuple[int, int] | None) -> str | None:
    """Slice ``document`` by ``span``, or ``None`` when the span is unusable."""
    if span is None:
        return None
    start, end = span
    if start < 0 or end <= start or end > len(document):
        return None
    return document[start:end]


def _span_is_valid(document: str, span: tuple[int, int] | None) -> bool:
    return span_text(document, span) is not None


def _evidence_matches(cited: str, canonical_claim: str, kind: FieldKind) -> str | None:
    """Return the matched canonical token, or ``None`` when nothing matches."""
    match kind:
        case FieldKind.MONEY | FieldKind.PERCENT:
            tokens = money_tokens_in(cited)
            return canonical_claim if canonical_claim in tokens else None
        case FieldKind.DATE:
            cited_date = canonical_date(cited)
            return cited_date if cited_date == canonical_claim else None
        case FieldKind.TEXT:
            haystack = canonical_text(cited) or ""
            return canonical_claim if canonical_claim in haystack else None
        case _:  # pragma: no cover - FieldKind is exhaustive above
            raise ValueError(f"unsupported field kind: {kind!r}")


def check_evidence(document: str, claim: Claim, spec: FieldSpec) -> EvidenceCheck:
    """Decide whether ``claim`` can be traced back into ``document``.

    Note that the claim's ``scale`` is deliberately **not** used here. Grounding
    asks only "is the literal you quoted actually there?". Whether the literal
    was interpreted at the right magnitude is a value-accuracy question, scored
    in :mod:`geq.scoring`, and conflating the two hides which one broke.
    """
    canonical_claim = canonical(claim.raw_value, spec.kind)

    if claim.span is None:
        return EvidenceCheck(
            field=claim.field,
            status=EvidenceStatus.UNCITED,
            detail="value asserted with no citation",
            normalized_claim=canonical_claim,
        )

    if not _span_is_valid(document, claim.span):
        return EvidenceCheck(
            field=claim.field,
            status=EvidenceStatus.OUT_OF_BOUNDS,
            detail=f"span {claim.span} does not address the document",
            normalized_claim=canonical_claim,
        )

    cited = span_text(document, claim.span) or ""

    if len(cited) > MAX_EVIDENCE_CHARS:
        return EvidenceCheck(
            field=claim.field,
            status=EvidenceStatus.NOT_A_CITATION,
            detail=(
                f"span covers {len(cited)} characters, which is wider than a citation "
                f"({MAX_EVIDENCE_CHARS} max) and cannot be reviewed"
            ),
            cited_text=cited[:120] + "...",
            normalized_claim=canonical_claim,
        )

    if canonical_claim is None:
        return EvidenceCheck(
            field=claim.field,
            status=EvidenceStatus.UNPARSEABLE,
            detail=f"claimed value {claim.raw_value!r} cannot be canonicalised",
            cited_text=cited,
        )

    matched = _evidence_matches(cited, canonical_claim, spec.kind)
    if matched is None:
        return EvidenceCheck(
            field=claim.field,
            status=EvidenceStatus.CONTRADICTED,
            detail=(
                f"cited text does not contain the claimed value "
                f"({canonical_claim!r} not found in {cited!r})"
            ),
            cited_text=cited,
            normalized_claim=canonical_claim,
        )

    return EvidenceCheck(
        field=claim.field,
        status=EvidenceStatus.SUPPORTED,
        detail="cited text contains the claimed value",
        cited_text=cited,
        normalized_claim=canonical_claim,
        normalized_cited=matched,
    )


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """The outcome of one arithmetic check over a set of extracted values."""

    name: str
    status: Literal["pass", "fail", "not_checkable"]
    detail: str
    expected: str | None = None
    actual: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    @property
    def checkable(self) -> bool:
        return self.status != "not_checkable"


def _scaled(
    claims: Mapping[str, Claim], field: str, specs: Mapping[str, FieldSpec]
) -> Decimal | None:
    """Canonical, document-scale-adjusted value for a claimed field."""
    claim = claims.get(field)
    spec = specs.get(field)
    if claim is None or spec is None:
        return None
    base = canonical(claim.raw_value, spec.kind)
    if base is None:
        return None
    value = signed_decimal(apply_scale(base, claim.scale, spec.kind))
    return value


def check_balance_sheet(
    claims: Mapping[str, Claim],
    specs: Mapping[str, FieldSpec],
    *,
    tolerance: Decimal = Decimal(0),
) -> ReconciliationResult:
    """``assets == liabilities + equity``.

    This is the check that catches a statement that *looks* complete: every
    section labelled, every subtotal present, and the two sides disagreeing.
    """
    name = "balance_sheet_identity"
    assets = _scaled(claims, "total_assets", specs)
    liabilities = _scaled(claims, "total_liabilities", specs)
    equity = _scaled(claims, "total_equity", specs)

    missing = [
        label
        for label, value in (
            ("total_assets", assets),
            ("total_liabilities", liabilities),
            ("total_equity", equity),
        )
        if value is None
    ]
    if missing:
        return ReconciliationResult(
            name=name,
            status="not_checkable",
            detail=f"cannot reconcile: {', '.join(missing)} not extracted",
        )

    rhs = liabilities + equity
    delta = abs(assets - rhs)
    if delta <= tolerance:
        return ReconciliationResult(
            name=name,
            status="pass",
            detail=f"assets {assets} == liabilities {liabilities} + equity {equity}",
            expected=str(assets),
            actual=str(rhs),
        )
    return ReconciliationResult(
        name=name,
        status="fail",
        detail=(
            f"balance sheet does not foot: assets {assets} vs "
            f"liabilities + equity {rhs} (out by {delta})"
        ),
        expected=str(assets),
        actual=str(rhs),
    )


def check_totals_footing(
    claims: Mapping[str, Claim],
    specs: Mapping[str, FieldSpec],
    *,
    tolerance: Decimal = Decimal(0),
) -> ReconciliationResult:
    """``total liabilities and equity == total assets``, when both are present."""
    name = "totals_footing"
    assets = _scaled(claims, "total_assets", specs)
    combined = _scaled(claims, "total_liabilities_and_equity", specs)

    if assets is None or combined is None:
        return ReconciliationResult(
            name=name,
            status="not_checkable",
            detail="cannot check: total_assets and total_liabilities_and_equity both required",
        )

    if abs(assets - combined) <= tolerance:
        return ReconciliationResult(
            name=name,
            status="pass",
            detail=f"totals foot at {assets}",
            expected=str(assets),
            actual=str(combined),
        )
    return ReconciliationResult(
        name=name,
        status="fail",
        detail=f"reported totals disagree: assets {assets} vs combined total {combined}",
        expected=str(assets),
        actual=str(combined),
    )


def reconcile(
    claims: Mapping[str, Claim],
    specs: Mapping[str, FieldSpec],
    *,
    tolerance: Decimal = Decimal(0),
) -> tuple[ReconciliationResult, ...]:
    """Every arithmetic check, in a stable order."""
    return (
        check_balance_sheet(claims, specs, tolerance=tolerance),
        check_totals_footing(claims, specs, tolerance=tolerance),
    )


def failing(results: Sequence[ReconciliationResult]) -> tuple[ReconciliationResult, ...]:
    return tuple(r for r in results if r.status == "fail")
