"""Loading and validating the golden set.

The golden set is the fixed set of examples the whole harness depends on, so it
gets validated the same way production data would. ``validate_corpus`` fails
loudly on the mistakes that quietly corrupt an evaluation:

* a label for a field that is not in the spec (typo in the label file),
* a field in the spec that no case in the corpus labels (a permanently
  unreachable recall ceiling),
* a duplicate ``doc_id`` (one document silently overwriting another).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import FieldKind, FieldSpec, GoldenCase

__all__ = ["Corpus", "CorpusError", "load_corpus", "load_specs", "validate_corpus"]

SPEC_FILENAME = "spec.json"
CASES_DIRNAME = "cases"


class CorpusError(ValueError):
    """The golden set is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class Corpus:
    """A spec plus its labelled documents."""

    name: str
    specs: dict[str, FieldSpec]
    cases: tuple[GoldenCase, ...]
    root: Path

    @property
    def required_fields(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self.specs.items() if spec.required)

    def spec(self, field_name: str) -> FieldSpec:
        return self.specs[field_name]

    def case(self, doc_id: str) -> GoldenCase:
        for candidate in self.cases:
            if candidate.doc_id == doc_id:
                return candidate
        raise KeyError(doc_id)


def load_specs(path: Path) -> dict[str, FieldSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs: dict[str, FieldSpec] = {}
    for row in payload["fields"]:
        spec = FieldSpec(
            name=row["name"],
            kind=FieldKind(row["kind"]),
            label=row["label"],
            required=bool(row.get("required", True)),
            aliases=tuple(row.get("aliases", ())),
        )
        if spec.name in specs:
            raise CorpusError(f"duplicate field spec: {spec.name!r}")
        specs[spec.name] = spec
    if not specs:
        raise CorpusError(f"{path} defines no fields")
    return specs


def load_corpus(root: str | Path) -> Corpus:
    """Load ``spec.json`` and ``cases/*.json`` from ``root``, then validate."""
    root = Path(root)
    specs = load_specs(root / SPEC_FILENAME)
    cases_dir = root / CASES_DIRNAME
    if not cases_dir.is_dir():
        raise CorpusError(f"no cases directory at {cases_dir}")

    cases: list[GoldenCase] = []
    for case_path in sorted(cases_dir.glob("*.json")):
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        cases.append(
            GoldenCase(
                doc_id=payload["doc_id"],
                text=payload["text"],
                labels=payload["labels"],
                notes=payload.get("notes", ""),
            )
        )

    corpus = Corpus(name=root.name, specs=specs, cases=tuple(cases), root=root)
    validate_corpus(corpus)
    return corpus


def validate_corpus(corpus: Corpus) -> None:
    """Raise :class:`CorpusError` if the golden set cannot support an evaluation."""
    if not corpus.cases:
        raise CorpusError(f"corpus {corpus.name!r} contains no cases")

    seen: set[str] = set()
    for case in corpus.cases:
        if case.doc_id in seen:
            raise CorpusError(f"duplicate doc_id in corpus: {case.doc_id!r}")
        seen.add(case.doc_id)

        unknown = sorted(set(case.labels) - set(corpus.specs))
        if unknown:
            raise CorpusError(
                f"{case.doc_id}: labels reference fields absent from spec.json: {unknown}"
            )

        for field_name, value in case.labels.items():
            if value is None or not str(value).strip():
                raise CorpusError(f"{case.doc_id}: empty label for {field_name!r}")

    unlabelled = sorted(
        name
        for name, spec in corpus.specs.items()
        if spec.required and not any(case.label(name) is not None for case in corpus.cases)
    )
    if unlabelled:
        raise CorpusError(
            f"required fields are never labelled by any case, so recall can never "
            f"reach 1.0: {unlabelled}"
        )
