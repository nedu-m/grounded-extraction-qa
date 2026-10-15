"""Extractor registry.

Names are explicit so a report always says which system produced it:
``rules:v1``, ``rules:v2``, ``rules:v3``, ``llm``, or one of the
``adversarial:`` controls that are expected to fail.

``adversarial:uncited`` does depend on the golden labels to produce its perfect
values. That is deliberate and worth being precise about: it is not an extractor
that could ever pass, it is a *control* that simulates the state of an extractor
that returns correct values with no evidence, and it can only do that by being
told the answers. Nothing in the scoring path knows the difference.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..models import FieldSpec
from .adversarial import CONTROL_NAMES, FabricatingExtractor, UncitedExtractor
from .base import Extractor, locate_quote
from .rules import REVISIONS, RulesExtractor

__all__ = [
    "CONTROL_NAMES",
    "LLM_EXTRACTOR_NAME",
    "REVISIONS",
    "Extractor",
    "FabricatingExtractor",
    "RulesExtractor",
    "UncitedExtractor",
    "available_extractors",
    "build_extractor",
    "locate_quote",
]

LLM_EXTRACTOR_NAME = "llm"


def available_extractors() -> tuple[str, ...]:
    return (
        *(f"rules:{revision}" for revision in REVISIONS),
        LLM_EXTRACTOR_NAME,
        *CONTROL_NAMES,
    )


def build_extractor(name: str, specs: Mapping[str, FieldSpec] | None = None) -> Extractor:
    """Resolve an extractor name to an instance.

    ``llm`` is imported lazily so the default test run never needs network
    access, credentials, or the optional dependency.
    """
    if name.startswith("rules:"):
        return RulesExtractor(revision=name.split(":", 1)[1], specs=specs)
    if name.startswith("adversarial:"):
        control = name.split(":", 1)[1]
        if control == "fabricating":
            return FabricatingExtractor(specs=specs)
        if control == "uncited":
            return UncitedExtractor(specs=specs)
        raise ValueError(f"unknown control {name!r}")
    if name == LLM_EXTRACTOR_NAME:
        from .llm import LLMExtractor

        return LLMExtractor(specs=specs)
    raise ValueError(f"unknown extractor {name!r}; available: {', '.join(available_extractors())}")
