"""Core value objects.

The harness keeps the *extractor* and the *judge* strictly separate. An extractor
reports only what it believes it read and where it read it; it never decides
whether it is correct. Every grounding and scoring decision lives in this
package, so a new extractor (deterministic rules, an LLM, a vendor API) is
measured by the same yardstick.

Two dimensions are tracked independently and deliberately not collapsed:

* **Value accuracy** - did the extractor produce the right value?
* **Grounding** - does the cited evidence actually support that value?

A system can score well on the first and zero on the second, which is the exact
failure mode this harness exists to catch: numbers that look right and are not
traceable to any document.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

# A character span into the source document, as ``(start, end)``.
Span = tuple[int, int]


class FieldKind(StrEnum):
    """How a field's value is canonicalised before it is compared."""

    MONEY = "money"
    PERCENT = "percent"
    DATE = "date"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """A single field the golden set expects to be pulled out of a document."""

    name: str
    kind: FieldKind
    label: str
    """Canonical line label as it appears in the source documents."""

    required: bool = True
    aliases: tuple[str, ...] = ()
    """Alternative line labels accepted in the source documents."""

    def all_labels(self) -> tuple[str, ...]:
        return (self.label, *self.aliases)


@dataclass(frozen=True, slots=True)
class Claim:
    """One field an extractor says it found, plus the evidence it cites."""

    field: str
    raw_value: str
    span: Span | None = None
    """Character offsets of the supporting text, or ``None`` if uncited."""

    scale: Decimal = Decimal(1)
    """Document-level multiplier read from a header such as "Amounts in thousands".

    Kept separate from ``raw_value`` on purpose. The citation must quote the
    document literally (``13,504.5``); the magnitude is a second, independently
    checkable claim, and an extractor that gets one right and the other wrong
    should be told exactly that.
    """


@dataclass(frozen=True, slots=True)
class Extraction:
    """Everything one extractor produced for one document."""

    doc_id: str
    extractor: str
    claims: Mapping[str, Claim] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def claim(self, field_name: str) -> Claim | None:
        return self.claims.get(field_name)


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """A document with hand-verified labels: the fixed set the JD asks for."""

    doc_id: str
    text: str
    labels: Mapping[str, str]
    notes: str = ""

    def label(self, field_name: str) -> str | None:
        return self.labels.get(field_name)


class EvidenceStatus(StrEnum):
    """Why a claim's citation does or does not hold up."""

    SUPPORTED = "supported"
    UNCITED = "uncited"
    """No span at all: the value is asserted with nothing behind it."""

    OUT_OF_BOUNDS = "out_of_bounds"
    """Span is negative, reversed, or past the end of the document."""

    NOT_A_CITATION = "not_a_citation"
    """Span is so wide it covers most of the document and proves nothing."""

    CONTRADICTED = "contradicted"
    """The cited text is real but does not contain the claimed value."""

    UNPARSEABLE = "unparseable"
    """The claimed value cannot be canonicalised, so it cannot be checked."""


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    """The result of attempting to trace a claim back to its source."""

    field: str
    status: EvidenceStatus
    detail: str
    cited_text: str | None = None
    normalized_claim: str | None = None
    normalized_cited: str | None = None

    @property
    def supported(self) -> bool:
        return self.status is EvidenceStatus.SUPPORTED


class Verdict(StrEnum):
    """Value accuracy for one (document, field) pair, independent of grounding."""

    RIGHT = "right"
    WRONG = "wrong"
    MISSING = "missing"
    EXTRA = "extra"
    """A field the golden set does not define: an invented field."""


@dataclass(frozen=True, slots=True)
class CellResult:
    """The outcome for one document and one field. The unit of everything below."""

    doc_id: str
    field: str
    verdict: Verdict
    gold: str | None = None
    claimed: str | None = None
    check: EvidenceCheck | None = None

    @property
    def right(self) -> bool:
        return self.verdict is Verdict.RIGHT

    @property
    def claimed_anything(self) -> bool:
        return self.verdict in (Verdict.RIGHT, Verdict.WRONG, Verdict.EXTRA)

    @property
    def grounded(self) -> bool:
        return self.check is not None and self.check.supported

    @property
    def right_and_grounded(self) -> bool:
        """The only outcome that is safe to put in front of a deal team."""
        return self.right and self.grounded

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "field": self.field,
            "verdict": self.verdict.value,
            "gold": self.gold,
            "claimed": self.claimed,
            "evidence_status": self.check.status.value if self.check else None,
            "evidence_detail": self.check.detail if self.check else None,
        }


@dataclass(frozen=True, slots=True)
class FieldScore:
    """Per-field tallies, most of them derived from :class:`CellResult` rows."""

    field: str
    gold: int = 0
    right: int = 0
    wrong: int = 0
    missing: int = 0
    extra: int = 0
    claims: int = 0
    grounded: int = 0
    right_and_grounded: int = 0

    @property
    def precision(self) -> float:
        """Of everything the extractor asserted, how much was correct."""
        return self.right / self.claims if self.claims else 0.0

    @property
    def recall(self) -> float:
        """Of the values that exist in the documents, how many were returned."""
        return self.right / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def grounding_rate(self) -> float:
        """Share of asserted values whose cited text actually supports them."""
        return self.grounded / self.claims if self.claims else 0.0

    @property
    def grounded_recall(self) -> float:
        """Share of true values returned *and* backed by real evidence."""
        if not self.gold:
            return 0.0
        return self.right_and_grounded / self.gold

    def to_dict(self) -> dict[str, float | int]:
        return {
            "gold": self.gold,
            "right": self.right,
            "wrong": self.wrong,
            "missing": self.missing,
            "extra": self.extra,
            "claims": self.claims,
            "grounded": self.grounded,
            "right_and_grounded": self.right_and_grounded,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "grounding_rate": round(self.grounding_rate, 4),
            "grounded_recall": round(self.grounded_recall, 4),
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True, slots=True)
class Scorecard:
    """The full result of scoring one extractor against one corpus."""

    extractor: str
    corpus: str
    cells: tuple[CellResult, ...]
    generated_at: str = ""
    per_field: Mapping[str, FieldScore] = field(default_factory=dict)

    # -- derivation -----------------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self.per_field)

    @property
    def _claims(self) -> tuple[CellResult, ...]:
        return tuple(c for c in self.cells if c.claimed_anything)

    def _total(self, name: str) -> int:
        return sum(getattr(s, name) for s in self.per_field.values())

    @property
    def gold(self) -> int:
        return self._total("gold")

    @property
    def right(self) -> int:
        return self._total("right")

    @property
    def wrong(self) -> int:
        return self._total("wrong")

    @property
    def missing(self) -> int:
        return self._total("missing")

    @property
    def extra(self) -> int:
        return self._total("extra")

    @property
    def claims(self) -> int:
        return self._total("claims")

    @property
    def grounded(self) -> int:
        return self._total("grounded")

    @property
    def right_and_grounded(self) -> int:
        return sum(c.right_and_grounded for c in self.cells)

    @property
    def micro_precision(self) -> float:
        return self.right / self.claims if self.claims else 0.0

    @property
    def micro_recall(self) -> float:
        return self.right / self.gold if self.gold else 0.0

    @property
    def micro_f1(self) -> float:
        p, r = self.micro_precision, self.micro_recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def macro_precision(self) -> float:
        return _mean([s.precision for s in self.per_field.values()])

    @property
    def macro_recall(self) -> float:
        return _mean([s.recall for s in self.per_field.values()])

    @property
    def macro_f1(self) -> float:
        return _mean([s.f1 for s in self.per_field.values()])

    @property
    def grounding_rate(self) -> float:
        """Headline anti-fabrication metric."""
        return self.grounded / self.claims if self.claims else 0.0

    @property
    def citation_rate(self) -> float:
        """Share of claims that carry any citation, valid or not."""
        cited = [
            c for c in self._claims if c.check and c.check.status is not EvidenceStatus.UNCITED
        ]
        return len(cited) / self.claims if self.claims else 0.0

    @property
    def grounded_recall(self) -> float:
        """The number that matters: true *and* traceable."""
        return self.right_and_grounded / self.gold if self.gold else 0.0

    @property
    def unsupported_claims(self) -> tuple[CellResult, ...]:
        return tuple(c for c in self._claims if not c.grounded)

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "extractor": self.extractor,
            "corpus": self.corpus,
            "generated_at": self.generated_at,
            "metrics": {
                "gold": self.gold,
                "right": self.right,
                "wrong": self.wrong,
                "missing": self.missing,
                "extra": self.extra,
                "claims": self.claims,
                "grounded": self.grounded,
                "right_and_grounded": self.right_and_grounded,
                "micro_precision": round(self.micro_precision, 4),
                "micro_recall": round(self.micro_recall, 4),
                "micro_f1": round(self.micro_f1, 4),
                "macro_precision": round(self.macro_precision, 4),
                "macro_recall": round(self.macro_recall, 4),
                "macro_f1": round(self.macro_f1, 4),
                "citation_rate": round(self.citation_rate, 4),
                "grounding_rate": round(self.grounding_rate, 4),
                "grounded_recall": round(self.grounded_recall, 4),
            },
            "per_field": {
                name: {
                    "gold": s.gold,
                    "right": s.right,
                    "wrong": s.wrong,
                    "missing": s.missing,
                    "extra": s.extra,
                    "claims": s.claims,
                    "grounded": s.grounded,
                    "precision": round(s.precision, 4),
                    "recall": round(s.recall, 4),
                    "f1": round(s.f1, 4),
                    "grounding_rate": round(s.grounding_rate, 4),
                    "grounded_recall": round(s.grounded_recall, 4),
                }
                for name, s in self.per_field.items()
            },
            "cells": [c.to_dict() for c in self.cells],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Scorecard:
        cells = tuple(
            CellResult(
                doc_id=row["doc_id"],
                field=row["field"],
                verdict=Verdict(row["verdict"]),
                gold=row.get("gold"),
                claimed=row.get("claimed"),
                check=(
                    EvidenceCheck(
                        field=row["field"],
                        status=EvidenceStatus(row["evidence_status"]),
                        detail=row.get("evidence_detail") or "",
                    )
                    if row.get("evidence_status")
                    else None
                ),
            )
            for row in payload["cells"]
        )
        return build_scorecard(
            extractor=payload["extractor"],
            corpus=payload["corpus"],
            cells=cells,
            generated_at=payload.get("generated_at", ""),
        )

    def failures(self, *, limit: int | None = None) -> tuple[CellResult, ...]:
        bad = [c for c in self.cells if not c.right_and_grounded]
        return tuple(bad[:limit] if limit is not None else bad)


def build_scorecard(
    *,
    extractor: str,
    corpus: str,
    cells: Iterable[CellResult],
    generated_at: str = "",
) -> Scorecard:
    """Aggregate cell-level results into a :class:`Scorecard`."""
    cells = tuple(cells)
    grouped: dict[str, list[CellResult]] = {}
    for cell in cells:
        grouped.setdefault(cell.field, []).append(cell)

    per_field: dict[str, FieldScore] = {}
    for name, rows in grouped.items():
        score = FieldScore(
            field=name,
            gold=sum(1 for r in rows if r.verdict is not Verdict.EXTRA),
            right=sum(1 for r in rows if r.verdict is Verdict.RIGHT),
            wrong=sum(1 for r in rows if r.verdict is Verdict.WRONG),
            missing=sum(1 for r in rows if r.verdict is Verdict.MISSING),
            extra=sum(1 for r in rows if r.verdict is Verdict.EXTRA),
            claims=sum(1 for r in rows if r.claimed_anything),
            grounded=sum(1 for r in rows if r.claimed_anything and r.grounded),
            right_and_grounded=sum(1 for r in rows if r.right_and_grounded),
        )
        per_field[name] = score

    return Scorecard(
        extractor=extractor,
        corpus=corpus,
        cells=cells,
        generated_at=generated_at,
        per_field=per_field,
    )
