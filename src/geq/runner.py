"""Running an extractor over a corpus, and persisting the result.

A run is stored in full - every cell, not just the totals - so two runs can be
compared cell by cell later. Aggregate metrics alone cannot answer "which
document did this regress on", and that is the only question worth asking when a
gate fails.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .corpus import Corpus
from .models import Extraction, Scorecard
from .scoring import CaseReconciliation, evaluate_corpus, reconciliation_report
from .verify import ReconciliationResult

__all__ = ["RunResult", "load_run", "run_extractor", "slugify"]


@dataclass(frozen=True, slots=True)
class RunResult:
    """A scored run plus the arithmetic checks and the raw extractions."""

    scorecard: Scorecard
    reconciliations: tuple[CaseReconciliation, ...] = ()
    extractions: Mapping[str, Extraction] | None = None

    @property
    def extractor(self) -> str:
        return self.scorecard.extractor

    @property
    def reconciliation_failures(self) -> tuple[tuple[str, ReconciliationResult], ...]:
        return tuple(
            (case.doc_id, result) for case in self.reconciliations for result in case.failures
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.scorecard.to_dict()
        payload["reconciliation"] = [
            {
                "doc_id": case.doc_id,
                "results": [
                    {
                        "name": result.name,
                        "status": result.status,
                        "detail": result.detail,
                        "expected": result.expected,
                        "actual": result.actual,
                    }
                    for result in case.results
                ],
            }
            for case in self.reconciliations
        ]
        payload["extraction_notes"] = {
            doc_id: list(extraction.notes)
            for doc_id, extraction in (self.extractions or {}).items()
            if extraction.notes
        }
        return payload


def run_extractor(corpus: Corpus, extractor_name: str) -> RunResult:
    """Extract every document with ``extractor_name`` and score the result."""
    from .extractors import build_extractor

    extractor = build_extractor(extractor_name, corpus.specs)
    extractions = {case.doc_id: extractor.extract(case) for case in corpus.cases}
    scorecard = evaluate_corpus(
        corpus.cases,
        extractions,
        corpus.specs,
        corpus_name=corpus.name,
        extractor=extractor.name,
    )
    return RunResult(
        scorecard=scorecard,
        reconciliations=reconciliation_report(corpus.cases, extractions, corpus.specs),
        extractions=extractions,
    )


def load_run(path: str | Path) -> RunResult:
    """Read a run back from the JSON written by :meth:`RunResult.to_dict`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scorecard = Scorecard.from_dict(payload)

    reconciliations: list[CaseReconciliation] = []
    for entry in payload.get("reconciliation", []):
        reconciliations.append(
            CaseReconciliation(
                entry["doc_id"],
                tuple(
                    ReconciliationResult(
                        name=row["name"],
                        status=row["status"],
                        detail=row["detail"],
                        expected=row.get("expected"),
                        actual=row.get("actual"),
                    )
                    for row in entry["results"]
                ),
            )
        )

    return RunResult(scorecard=scorecard, reconciliations=tuple(reconciliations), extractions=None)


def slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)
