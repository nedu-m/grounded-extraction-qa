"""Adversarial controls: extractors that are supposed to fail.

These ship in the package rather than living in the test suite for one reason: a
grounding check is only worth having if it demonstrably fires. Running these two
on every CI run turns "we check for fabricated citations" into a fact that is
re-verified continuously, and turns a regression in the checker itself into a
failing build rather than a quiet downgrade.

They are labelled ``adversarial:`` so no one mistakes them for candidate
extractors in a report.

``adversarial:fabricating``
    Produces plausible figures and cites the *label* as evidence. The citation is
    real text from the real document, which is what makes this the dangerous
    case: a reviewer skimming sees a quote and assumes it was checked.

``adversarial:uncited``
    Produces the correct value for every field and cites nothing. Recall is
    perfect, precision is perfect, and not one number can be audited. This is the
    state most LLM extraction prompts ship in, because nobody asked for a quote.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from ..models import Claim, Extraction, FieldSpec, GoldenCase

__all__ = ["CONTROL_NAMES", "FabricatingExtractor", "UncitedExtractor"]

CONTROL_NAMES = ("adversarial:fabricating", "adversarial:uncited")

_MONEY_FIELDS = (
    "total_assets",
    "total_liabilities",
    "total_equity",
    "total_liabilities_and_equity",
    "revenue",
    "net_income",
)

# Figures that look entirely reasonable in a healthcare balance sheet and appear
# in none of the documents.
_INVENTED = {
    "total_assets": "13,504,900",
    "total_liabilities": "5,320,900",
    "total_equity": "8,184,900",
    "total_liabilities_and_equity": "13,504,900",
    "revenue": "24,881,000",
    "net_income": "2,416,000",
}

_DEFAULT_LABELS = {
    "total_assets": "Total assets",
    "total_liabilities": "Total liabilities",
    "total_equity": "Total equity",
    "total_liabilities_and_equity": "Total liabilities and equity",
    "revenue": "Revenue",
    "net_income": "Net income",
}


class FabricatingExtractor:
    """Invents values and points the citation at the label instead of the figure."""

    name = "adversarial:fabricating"

    def __init__(self, specs: Mapping[str, FieldSpec] | None = None) -> None:
        self._specs: Mapping[str, FieldSpec] = specs or {}

    def extract(self, case: GoldenCase) -> Extraction:
        claims: dict[str, Claim] = {}
        for field in _MONEY_FIELDS:
            label = self._label_for(field)
            if label is None:
                continue
            start = case.text.find(label)
            if start < 0:
                continue
            claims[field] = Claim(
                field=field,
                raw_value=_INVENTED[field],
                span=(start, start + len(label)),
            )
        return Extraction(
            doc_id=case.doc_id,
            extractor=self.name,
            claims=claims,
            notes=("adversarial control: values invented, citations point at the label",),
        )

    def _label_for(self, field: str) -> str | None:
        spec = self._specs.get(field)
        return spec.label if spec is not None else _DEFAULT_LABELS.get(field)


class UncitedExtractor:
    """Returns the right answer for everything and cites nothing."""

    name = "adversarial:uncited"

    def __init__(self, specs: Mapping[str, FieldSpec] | None = None) -> None:
        # Field kinds are irrelevant: the control returns the golden text as-is.
        self._specs: Mapping[str, FieldSpec] = specs or {}

    def extract(self, case: GoldenCase) -> Extraction:
        claims = {
            name: Claim(field=name, raw_value=value, span=None)
            for name, value in case.labels.items()
            if re.match(r"^[a-z_]+$", name)
        }
        return Extraction(
            doc_id=case.doc_id,
            extractor=self.name,
            claims=claims,
            notes=("adversarial control: correct values, no citations",),
        )
