"""Three deterministic extractor revisions, plus the defects each one has.

These are not stubs standing in for "the real thing". They are the regression
fixture: each revision is a real, runnable extractor whose failure modes are
known, so the harness itself can be tested. If the grounding check ever stops
catching a fabricated citation, ``rules:v2`` stops failing and CI says so.

Revision history, and what each one actually changes:

``v1``  Naive keyword lookup. Finds the label with a plain substring search and
        takes the next figure. Cites nothing, because it was not asked to. Only
        reads the balance sheet. The common first attempt.

``v2``  Line-anchored with citations, and honours a document stated "in
        thousands". Two defects remain, both deliberate:

        * labels are matched by prefix and the *last* match wins, so
          ``Total liabilities`` resolves to the ``Total liabilities and equity``
          line. The value is wrong and the citation is perfectly faithful, which
          is exactly why value accuracy and grounding are scored separately.
        * two fields fall back to a fixed-width span guess that lands on labels
          and padding rather than on the figure, producing citations that do not
          contain the value they claim.

``v3``  Current. Exact label matching, document scale read from the header
        (thousands and millions), cover-page disclaimers skipped, the date field
        added, and spans narrowed to the figure itself.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from ..models import Claim, Extraction, FieldSpec, GoldenCase
from ..normalize import find_dates, find_money

__all__ = ["REVISIONS", "RulesExtractor", "parse_value_lines"]

REVISIONS = ("v1", "v2", "v3")

# A line read as "label, four-or-more separator characters, figure at the end".
_SEPARATOR_RE = re.compile(r"[\s.]{4,}$")
_MAX_LABEL_CHARS = 80
_MAX_LABEL_CHARS_GUARD = 200

_DISCLAIMER_RE = re.compile(
    r"\b(?:draft|confidential|preliminary|not for distribution|for internal use|"
    r"do not distribute|unaudited)\b",
    re.IGNORECASE,
)

_SCALE_WORDS = {
    "thousand": Decimal(1_000),
    "thousands": Decimal(1_000),
    "million": Decimal(1_000_000),
    "millions": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
    "billions": Decimal(1_000_000_000),
}

_THOUSANDS_RE = re.compile(r"\bin\s+thousands?\b", re.IGNORECASE)
_ANY_SCALE_RE = re.compile(r"\b(?:stated\s+)?in\s+(thousands|millions|billions)\b", re.IGNORECASE)

DEFAULT_LABELS: Mapping[str, str] = {
    "total_assets": "Total assets",
    "total_liabilities": "Total liabilities",
    "total_equity": "Total equity",
    "total_liabilities_and_equity": "Total liabilities and equity",
    "revenue": "Revenue",
    "net_income": "Net income",
}

# v1 reads only what it was pointed at first.
_V1_FIELDS = ("total_assets", "total_liabilities", "total_equity")

# v2 adds the other money fields but still cannot place a date.
_V2_FIELDS = (
    "total_assets",
    "total_liabilities",
    "total_equity",
    "total_liabilities_and_equity",
    "revenue",
    "net_income",
)

_V3_FIELDS = _V2_FIELDS

_V2_FIXED_WIDTH_FIELDS = ("total_liabilities_and_equity", "net_income")
_V2_FIXED_WIDTH = 32

_HEADER_LINES = 8
"""How far into a document to look for the entity name, date and scale."""


@dataclass(frozen=True, slots=True)
class ValueLine:
    """A parsed ``label ....... figure`` line, with absolute offsets."""

    label: str
    line_start: int
    line_end: int
    value: str
    value_start: int
    value_end: int

    @property
    def value_span(self) -> tuple[int, int]:
        return (self.value_start, self.value_end)

    @property
    def line_span(self) -> tuple[int, int]:
        return (self.line_start, self.line_end)


def _iter_lines(text: str) -> list[tuple[int, str]]:
    """Lines with their absolute start offsets. Handles LF and CRLF."""
    out: list[tuple[int, str]] = []
    pos = 0
    for raw in text.splitlines(keepends=True):
        out.append((pos, raw.rstrip("\r\n")))
        pos += len(raw)
    if not out and text:
        out.append((0, text))
    return out


def parse_value_lines(text: str) -> list[ValueLine]:
    """Parse every ``label <separator> figure`` line in ``text``.

    A line qualifies only when a figure sits at the end of it, so ordinary prose
    and section headers are never mistaken for data. All three revisions share
    this parser; what differs is how they *select* from the result.
    """
    parsed: list[ValueLine] = []
    for line_start, line_text in _iter_lines(text):
        if len(line_text) > _MAX_LABEL_CHARS_GUARD:
            continue
        money = find_money(line_text)
        if not money:
            continue
        value, value_start, value_end = money[-1]
        if line_text[value_end:].strip():
            # Something follows the figure, so this is prose, not a data row.
            continue
        label_region = line_text[:value_start]
        separator = _SEPARATOR_RE.search(label_region)
        if separator is None:
            continue
        label = label_region[: separator.start()].strip()
        if not label or len(label) > _MAX_LABEL_CHARS:
            continue
        parsed.append(
            ValueLine(
                label=label,
                line_start=line_start,
                line_end=line_start + len(line_text),
                value=value,
                value_start=line_start + value_start,
                value_end=line_start + value_end,
            )
        )
    return parsed


def _first_content_line(text: str, *, skip_disclaimers: bool) -> tuple[int, str] | None:
    for line_start, line_text in _iter_lines(text):
        stripped = line_text.strip()
        if not stripped:
            continue
        if skip_disclaimers and _DISCLAIMER_RE.search(stripped):
            continue
        return line_start, stripped
    return None


def _header(text: str) -> str:
    return "\n".join(line for _, line in _iter_lines(text)[:_HEADER_LINES])


class RulesExtractor:
    """Deterministic, offline extractor with a selectable revision.

    Labels come from the corpus spec passed in at construction, so a spec change
    is always a visible call site rather than hidden global state.
    """

    def __init__(self, revision: str = "v3", specs: Mapping[str, FieldSpec] | None = None) -> None:
        if revision not in REVISIONS:
            raise ValueError(f"unknown revision {revision!r}; expected one of {REVISIONS}")
        self.revision = revision
        self.name = f"rules:{revision}"
        self._specs: Mapping[str, FieldSpec] = specs or {}

    def extract(self, case: GoldenCase) -> Extraction:
        builder = {
            "v1": self._extract_v1,
            "v2": self._extract_v2,
            "v3": self._extract_v3,
        }[self.revision]
        return builder(case)

    def label_for(self, field: str) -> str | None:
        spec = self._specs.get(field)
        if spec is not None:
            return spec.label
        return DEFAULT_LABELS.get(field)

    # -- revisions ------------------------------------------------------

    def _extract_v1(self, case: GoldenCase) -> Extraction:
        claims: dict[str, Claim] = {}
        found = _first_content_line(case.text, skip_disclaimers=False)
        if found is not None:
            line_start, line_text = found
            claims["entity_name"] = Claim(
                field="entity_name",
                raw_value=line_text,
                span=(line_start, line_start + len(line_text)),
            )

        for field in _V1_FIELDS:
            label = self.label_for(field)
            if label is None:
                continue
            match = re.search(re.escape(label), case.text, re.IGNORECASE)
            if match is None:
                continue
            money = find_money(case.text[match.end() :])
            if not money:
                continue
            claims[field] = Claim(field=field, raw_value=money[0][0], span=None)

        return Extraction(
            doc_id=case.doc_id,
            extractor=self.name,
            claims=claims,
            notes=("v1 asserts values without citations",),
        )

    def _extract_v2(self, case: GoldenCase) -> Extraction:
        claims: dict[str, Claim] = {}
        lines = parse_value_lines(case.text)

        found = _first_content_line(case.text, skip_disclaimers=False)
        if found is not None:
            line_start, line_text = found
            claims["entity_name"] = Claim(
                field="entity_name",
                raw_value=line_text,
                span=(line_start, line_start + len(line_text)),
            )

        scale = Decimal(1_000) if _THOUSANDS_RE.search(_header(case.text)) else Decimal(1)

        for field in _V2_FIELDS:
            label = self.label_for(field)
            if label is None:
                continue
            matches = [line for line in lines if line.label.startswith(label)]
            if not matches:
                continue
            line = matches[-1]
            span = (
                (line.line_start, line.line_start + _V2_FIXED_WIDTH)
                if field in _V2_FIXED_WIDTH_FIELDS
                else line.line_span
            )
            claims[field] = Claim(field=field, raw_value=line.value, span=span, scale=scale)

        return Extraction(
            doc_id=case.doc_id,
            extractor=self.name,
            claims=claims,
            notes=("v2 matches labels by prefix and prefers the last match",),
        )

    def _extract_v3(self, case: GoldenCase) -> Extraction:
        claims: dict[str, Claim] = {}
        lines = parse_value_lines(case.text)

        found = _first_content_line(case.text, skip_disclaimers=True)
        if found is not None:
            line_start, line_text = found
            claims["entity_name"] = Claim(
                field="entity_name",
                raw_value=line_text,
                span=(line_start, line_start + len(line_text)),
            )

        dates = find_dates(case.text)
        if dates:
            literal, start, end = dates[0]
            claims["fiscal_year_end"] = Claim(
                field="fiscal_year_end", raw_value=literal, span=(start, end)
            )

        scale = self._document_scale(case.text)

        for field in _V3_FIELDS:
            label = self.label_for(field)
            if label is None:
                continue
            matches = [line for line in lines if line.label == label]
            if not matches:
                continue
            claims[field] = Claim(
                field=field,
                raw_value=matches[-1].value,
                span=matches[-1].value_span,
                scale=scale,
            )

        return Extraction(
            doc_id=case.doc_id,
            extractor=self.name,
            claims=claims,
            notes=(f"v3 applied document scale 1:{scale}",),
        )

    @staticmethod
    def _document_scale(text: str) -> Decimal:
        match = _ANY_SCALE_RE.search(_header(text))
        if match is None:
            return Decimal(1)
        return _SCALE_WORDS[match.group(1).lower()]
