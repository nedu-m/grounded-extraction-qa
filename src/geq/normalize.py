"""Canonicalisation of extracted values.

Every comparison in the harness happens between canonical strings, never between
raw text. That is what lets ``1,204,500``, ``$1,204,500.00`` and ``1204500`` be
treated as the same answer, and what stops a trailing currency symbol from being
reported as a QA failure.

Two rules worth stating up front, because getting them wrong produces QA numbers
that are confidently meaningless:

1. ``canonical`` understands a scale *suffix* attached to the literal itself
   (``1.2M`` -> ``1200000``). A *document-level* scale stated in a header
   ("Amounts in thousands of US dollars") is **not** applied here. The extractor
   reports it separately as ``Claim.scale``, so the harness can tell apart
   "read the number correctly but missed the units" from "cited the wrong line".
2. Ambiguous numeric dates are resolved by a stated rule, not by locale
   guessing. See :func:`canonical_date`.

Known limitations are encoded as failing-expectation tests in
``tests/test_normalize.py`` rather than being left undocumented.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from .models import FieldKind

__all__ = [
    "canonical",
    "canonical_date",
    "canonical_money",
    "canonical_percent",
    "canonical_text",
    "find_dates",
    "find_money",
    "money_tokens_in",
    "signed_decimal",
]

# Currency prefixes we strip. Kept as a regex fragment so ``money_tokens_in`` can
# reuse the exact same definition the single-value parser uses.
_CURRENCY = r"(?:US\$|\$|USD|NGN|EUR|GBP|EUR|\u20a6|\u20ac|\u00a3)"
_SCALES: dict[str, Decimal] = {
    "k": Decimal(1_000),
    "thousand": Decimal(1_000),
    "thousands": Decimal(1_000),
    "m": Decimal(1_000_000),
    "mn": Decimal(1_000_000),
    "million": Decimal(1_000_000),
    "millions": Decimal(1_000_000),
    "bn": Decimal(1_000_000_000),
    "b": Decimal(1_000_000_000),
    "billion": Decimal(1_000_000_000),
    "billions": Decimal(1_000_000_000),
    "tn": Decimal(1_000_000_000_000),
    "trillion": Decimal(1_000_000_000_000),
}

# Unicode minus and en/em dashes are common in documents exported from Word.
_MINUS = "-\u2212\u2013\u2014"

_NUMBER_BODY = r"\d{1,3}(?:[,\u00a0 ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"

_MONEY_RE = re.compile(
    rf"""
    (?P<paren_open>\()?
    \s*
    (?P<sign>[{re.escape(_MINUS)}+])?
    \s*
    (?:{_CURRENCY})?
    \s*
    (?P<num>{_NUMBER_BODY})
    \s*
    (?P<suffix>k|mn|bn|tn|m|b|thousand|thousands|million|millions|billion|billions|trillion)?
    \s*
    (?P<paren_close>\))?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ISO_DATE_RE = re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})")
_SLASH_DATE_RE = re.compile(r"(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{4})")
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
# "31 December 2024", "31st of December 2024", "December 31, 2024"
_LONG_DATE_RE = re.compile(
    r"(?:(?P<d1>\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:of\s+)?(?P<m1>[A-Za-z]{3,9})"
    r"|(?P<m2>[A-Za-z]{3,9})\s*(?P<d2>\d{1,2})\s*(?:st|nd|rd|th)?)"
    r"\s*,?\s*(?P<y>\d{4})"
)

_TEXT_SUFFIXES = (
    "incorporated",
    "inc",
    "limited",
    "ltd",
    "llc",
    "l.l.c",
    "plc",
    "corp",
    "corporation",
    "company",
    "co",
    "pte",
    "gmbh",
    "sa",
    "nv",
    "bv",
)


def _decimal_str(value: Decimal) -> str:
    """Render a Decimal without exponent notation and without trailing zeros."""
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _parse_number(
    num: str,
    *,
    sign: str | None,
    paren_open: bool,
    paren_close: bool,
    suffix: str | None,
) -> Decimal | None:
    """Parse a number, a sign, an accounting negative, and a scale suffix."""
    cleaned = re.sub(r"[,\u00a0\u202f ]", "", num)
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if suffix:
        value *= _SCALES[suffix.lower()]
    negative = bool(sign) and sign in _MINUS  # type: ignore[operator]
    if paren_open and paren_close:
        negative = True
    if sign == "+":
        negative = False
    return -value if negative else value


def canonical_money(raw: str) -> str | None:
    """Canonicalise a monetary literal, honouring any suffix on the literal."""
    match = _MONEY_RE.search(raw or "")
    if not match:
        return None
    value = _parse_number(
        match.group("num"),
        sign=match.group("sign"),
        paren_open=bool(match.group("paren_open")),
        paren_close=bool(match.group("paren_close")),
        suffix=match.group("suffix"),
    )
    return _decimal_str(value) if value is not None else None


def canonical_percent(raw: str) -> str | None:
    """Canonicalise a percentage as the number it names, e.g. ``12.5%`` -> ``12.5``."""
    text = re.sub(r"(?:\s*(?:%|percent|per\s+cent|pct))?", "", raw or "", flags=re.IGNORECASE)
    match = _MONEY_RE.search(text.strip())
    if not match:
        return None
    value = _parse_number(
        match.group("num"),
        sign=match.group("sign"),
        paren_open=bool(match.group("paren_open")),
        paren_close=bool(match.group("paren_close")),
        suffix=None,
    )
    return _decimal_str(value) if value is not None else None


def canonical_date(raw: str) -> str | None:
    """Canonicalise a date to ISO ``YYYY-MM-DD``.

    Ambiguity rule, applied in order:

    * ISO input wins outright.
    * If the first component of a ``a/b/yyyy`` pair is greater than 12, the input
      is day-first (``31/12/2024``).
    * Else if the second is greater than 12, it is month-first (``12/31/2024``).
    * Else the pair is genuinely ambiguous and is **rejected** rather than
      guessed. A QA harness that silently picks a convention will pass a wrong
      date with a straight face.
    """
    if not raw:
        return None
    text = raw.strip()

    if iso := _ISO_DATE_RE.search(text):
        try:
            return date(int(iso.group("y")), int(iso.group("m")), int(iso.group("d"))).isoformat()
        except ValueError:
            return None

    if long := _LONG_DATE_RE.search(text):
        day = long.group("d1") or long.group("d2")
        month = long.group("m1") or long.group("m2")
        month_num = _MONTHS.get((month or "").lower()[:9]) or _MONTHS.get((month or "").lower()[:3])
        if day and month_num:
            try:
                return date(int(long.group("y")), month_num, int(day)).isoformat()
            except ValueError:
                return None

    if slash := _SLASH_DATE_RE.search(text):
        a, b, year = int(slash.group("a")), int(slash.group("b")), int(slash.group("y"))
        if a > 12 and b <= 12:
            day, month = a, b
        elif b > 12 and a <= 12:
            month, day = a, b
        else:
            return None
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    return None


def canonical_text(raw: str) -> str | None:
    """Casefold, collapse whitespace, and drop trailing corporate suffixes."""
    if not raw:
        return None
    text = re.sub(r"\s+", " ", raw).strip()
    text = text.strip(" \t\"'\u201c\u201d,;:.")
    lowered = text.casefold()
    for suffix in sorted(_TEXT_SUFFIXES, key=len, reverse=True):
        pattern = rf"[,\s]+{re.escape(suffix)}\.?\s*$"
        if re.search(pattern, lowered):
            lowered = re.sub(pattern, "", lowered).strip(" ,.")
            break
    return lowered or None


def canonical(raw: str, kind: FieldKind) -> str | None:
    """Canonicalise ``raw`` according to ``kind``. ``None`` means unparseable."""
    match kind:
        case FieldKind.MONEY:
            return canonical_money(raw)
        case FieldKind.PERCENT:
            return canonical_percent(raw)
        case FieldKind.DATE:
            return canonical_date(raw)
        case FieldKind.TEXT:
            return canonical_text(raw)
        case _:  # pragma: no cover - FieldKind is exhaustive above
            raise ValueError(f"unsupported field kind: {kind!r}")


def signed_decimal(canonical_value: str | None) -> Decimal | None:
    """Re-read a canonical string as a Decimal, for arithmetic checks."""
    if canonical_value is None:
        return None
    try:
        return Decimal(canonical_value)
    except InvalidOperation:  # pragma: no cover - canonical output is always numeric
        return None


def find_money(text: str) -> list[tuple[str, int, int]]:
    """Every monetary literal in ``text`` with its offsets, in document order.

    This is the primitive extractors use to build a citable span: the literal it
    returns is exactly a substring of the source, so the span it reports can be
    sliced back out of the document and re-checked.
    """
    if not text:
        return []
    out: list[tuple[str, int, int]] = []
    for match in _MONEY_RE.finditer(text):
        literal = match.group(0).strip()
        if not literal:
            continue
        offset = match.group(0).index(literal)
        start = match.start() + offset
        out.append((literal, start, start + len(literal)))
    return out


def find_dates(text: str) -> list[tuple[str, int, int]]:
    """Every *valid* date literal in ``text`` with its offsets, in document order.

    Matches that do not survive :func:`canonical_date` are dropped, so an
    ordinary word near an ordinary number is never mistaken for a date.
    """
    if not text:
        return []
    candidates: list[tuple[int, int, str]] = []
    for pattern in (_ISO_DATE_RE, _LONG_DATE_RE, _SLASH_DATE_RE):
        for match in pattern.finditer(text):
            literal = match.group(0).strip()
            if literal and canonical_date(literal):
                candidates.append((match.start(), match.end(), literal))

    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    out: list[tuple[str, int, int]] = []
    last_end = -1
    for start, end, literal in candidates:
        if start < last_end:
            continue
        out.append((literal, start, end))
        last_end = end
    return out


def money_tokens_in(text: str) -> set[str]:
    """Every monetary literal in ``text``, canonicalised.

    Used to answer one question: *does this cited text actually contain the value
    that was claimed?* Trailing footnote markers and surrounding prose are
    tolerated; a different number is not.
    """
    if not text:
        return set()
    tokens: set[str] = set()
    for match in _MONEY_RE.finditer(text):
        value = _parse_number(
            match.group("num"),
            sign=match.group("sign"),
            paren_open=bool(match.group("paren_open")),
            paren_close=bool(match.group("paren_close")),
            suffix=match.group("suffix"),
        )
        if value is not None:
            tokens.add(_decimal_str(value))
    return tokens


def apply_scale(canonical_value: str | None, scale: Decimal, kind: FieldKind) -> str | None:
    """Apply a document-level scale (from an "Amounts in thousands" header)."""
    if canonical_value is None:
        return None
    if kind not in (FieldKind.MONEY, FieldKind.PERCENT) or scale == 1:
        return canonical_value
    value = signed_decimal(canonical_value)
    if value is None:  # pragma: no cover - canonical output is always numeric
        return None
    return _decimal_str(value * scale)
