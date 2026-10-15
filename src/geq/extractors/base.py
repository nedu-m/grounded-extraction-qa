"""The extractor contract.

An extractor does exactly one thing: given a document, report the values it
believes it read and the character span it read them from. It does not know
whether it is right, and it must not be able to influence how it is scored.

That constraint is what makes the harness honest. Any extractor - deterministic
rules, an LLM, a vendor API, a human - can be dropped in behind this protocol and
measured on the same golden set with the same yardstick.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Extraction, GoldenCase, Span

__all__ = ["Extractor", "locate_quote"]


@runtime_checkable
class Extractor(Protocol):
    """Anything that can turn a document into cited claims."""

    name: str

    def extract(self, case: GoldenCase) -> Extraction:  # pragma: no cover - protocol
        ...


def locate_quote(document: str, quote: str) -> tuple[Span, int] | None:
    """Map a verbatim quote back to a span in ``document``.

    LLM extractors are asked to return the text they read rather than an offset,
    because models are unreliable at counting characters and reliable at copying
    them. This converts the quote back into a span the harness can audit.

    Returns the span and the number of times the quote occurs. Repeated quotes
    are reported rather than hidden: two identical occurrences mean the citation
    does not identify a unique line, which a reviewer would want to know.
    """
    if not quote:
        return None
    start = document.find(quote)
    if start < 0:
        return None
    occurrences = document.count(quote)
    return (start, start + len(quote)), occurrences
