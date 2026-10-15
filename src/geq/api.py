"""HTTP surface.

The harness as a service, because the tools this role supports are services. It
exists so the QA checks can run from CI, from a crate task, or from a colleague's
terminal without installing anything, and so the API itself can be tested -
status codes, payload shape, and behaviour on a repeated identical request, which
is one of the things the JD asks for explicitly.

Failures are not hidden behind a 200. A QA run whose gate fails returns 409, and
an unknown extractor returns 404, because a caller that has to parse a body to
discover the difference will eventually get it wrong.

``/`` is the human view of the same numbers the JSON endpoints return. It is
rendered from the same functions the CI gate uses, so the page cannot show a
verdict the gate does not agree with.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from .corpus import Corpus, load_corpus
from .extractors import available_extractors, build_extractor
from .extractors.llm import LLMConfig
from .models import Extraction, FieldKind, GoldenCase
from .regression import RegressionPolicy, compare_runs
from .runner import run_extractor
from .verify import check_evidence

__all__ = ["app", "create_app"]

DEFAULT_CORPUS = "data/golden"


class ExtractRequest(BaseModel):
    extractor: str = Field(default="rules:v3", examples=["rules:v3"])
    doc_id: str | None = Field(default=None, description="A document from the golden corpus")
    text: str | None = Field(default=None, description="Ad-hoc document text")
    include_span_text: bool = Field(default=True)


class ClaimOut(BaseModel):
    field: str
    kind: FieldKind
    value: str | None
    span: tuple[int, int] | None
    span_text: str | None
    evidence_status: str
    evidence_detail: str


class ExtractResponse(BaseModel):
    extractor: str
    doc_id: str
    claims: list[ClaimOut]
    missing: list[str]
    notes: list[str]


class RunRequest(BaseModel):
    extractor: str = Field(default="rules:v3")


class MetricOut(BaseModel):
    gold: int
    right: int
    claims: int
    grounded: int
    micro_recall: float
    micro_precision: float
    grounding_rate: float
    grounded_recall: float


class RunResponse(BaseModel):
    extractor: str
    corpus: str
    metrics: MetricOut
    per_field: dict[str, dict[str, float | int]]
    failures: list[dict[str, Any]]
    reconciliation: list[dict[str, Any]]


class CompareRequest(BaseModel):
    baseline: str = Field(default="rules:v2")
    candidate: str = Field(default="rules:v3")
    max_regressions: int = Field(default=0, ge=0)


@lru_cache(maxsize=4)
def _corpus(path: str) -> Corpus:
    """Load and cache a corpus. Cached by path, so tests can point elsewhere."""
    return load_corpus(path)


def _corpus_path() -> str:
    return os.environ.get("GEQ_CORPUS", DEFAULT_CORPUS)


def _resolve_case(corpus: Corpus, request: ExtractRequest) -> GoldenCase:
    if request.doc_id:
        try:
            return corpus.case(request.doc_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"unknown doc_id {request.doc_id!r}",
            ) from exc
    if request.text:
        return GoldenCase(doc_id="adhoc", text=request.text, labels={})
    raise HTTPException(
        status_code=422,
        detail="provide either doc_id (from the corpus) or text (an ad-hoc document)",
    )


def _claim_rows(
    case: GoldenCase, extraction: Extraction, corpus: Corpus, include_span_text: bool
) -> tuple[list[ClaimOut], list[str]]:
    rows: list[ClaimOut] = []
    missing: list[str] = []
    for name, spec in corpus.specs.items():
        claim = extraction.claim(name)
        if claim is None:
            if spec.required:
                missing.append(name)
            continue
        check = check_evidence(case.text, claim, spec)
        span_text = (
            case.text[claim.span[0] : claim.span[1]]
            if include_span_text and claim.span is not None
            else None
        )
        rows.append(
            ClaimOut(
                field=name,
                kind=spec.kind,
                value=claim.raw_value,
                span=claim.span,
                span_text=span_text,
                evidence_status=check.status.value,
                evidence_detail=check.detail,
            )
        )
    return rows, missing


def create_app() -> FastAPI:
    app = FastAPI(
        title="grounded-extraction-qa",
        version=__version__,
        summary="Golden-set QA for document extraction: accuracy, grounding and regression gates.",
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        from .web import dashboard_html

        return HTMLResponse(dashboard_html())

    @app.get("/assets/paper.js", include_in_schema=False)
    def paper_js() -> Response:
        from .web import PAPER_JS

        return Response(
            PAPER_JS,
            media_type="text/javascript",
            headers={"Cache-Control": "public, max-age=300, stale-while-revalidate=86400"},
        )

    @app.get("/fonts/{name}", include_in_schema=False)
    def font(name: str) -> FileResponse:
        from .web import FONT_DIR, FONT_FILES

        # An allow-list, not a path join: nothing outside the shipped faces is servable.
        if name not in FONT_FILES:
            raise HTTPException(status_code=404, detail="no such font")
        return FileResponse(
            FONT_DIR / name,
            media_type="font/woff2",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        corpus = _corpus(_corpus_path())
        return {
            "status": "ok",
            "version": __version__,
            "corpus": corpus.name,
            "documents": len(corpus.cases),
            "fields": len(corpus.specs),
            "extractors": list(available_extractors()),
            "llm": LLMConfig.from_env().public_view(),
        }

    @app.get("/fields")
    def fields() -> dict[str, Any]:
        corpus = _corpus(_corpus_path())
        return {
            "corpus": corpus.name,
            "fields": [
                {
                    "name": spec.name,
                    "kind": spec.kind.value,
                    "label": spec.label,
                    "required": spec.required,
                    "aliases": list(spec.aliases),
                }
                for spec in corpus.specs.values()
            ],
        }

    @app.post("/extract", response_model=ExtractResponse)
    def extract(request: ExtractRequest) -> ExtractResponse:
        corpus = _corpus(_corpus_path())
        try:
            extractor = build_extractor(request.extractor, corpus.specs)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        case = _resolve_case(corpus, request)
        extraction = extractor.extract(case)
        rows, missing = _claim_rows(case, extraction, corpus, request.include_span_text)
        return ExtractResponse(
            extractor=extractor.name,
            doc_id=case.doc_id,
            claims=rows,
            missing=missing,
            notes=list(extraction.notes),
        )

    @app.post("/qa/run", response_model=RunResponse)
    def qa_run(request: RunRequest) -> RunResponse:
        corpus = _corpus(_corpus_path())
        try:
            result = run_extractor(corpus, request.extractor)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        card = result.scorecard
        return RunResponse(
            extractor=card.extractor,
            corpus=card.corpus,
            metrics=MetricOut(
                gold=card.gold,
                right=card.right,
                claims=card.claims,
                grounded=card.grounded,
                micro_recall=round(card.micro_recall, 4),
                micro_precision=round(card.micro_precision, 4),
                grounding_rate=round(card.grounding_rate, 4),
                grounded_recall=round(card.grounded_recall, 4),
            ),
            per_field={name: score.to_dict() for name, score in card.per_field.items()},
            failures=[cell.to_dict() for cell in card.failures(limit=25)],
            reconciliation=[
                {
                    "doc_id": case.doc_id,
                    "results": [
                        {"name": r.name, "status": r.status, "detail": r.detail}
                        for r in case.results
                    ],
                }
                for case in result.reconciliations
            ],
        )

    @app.post("/qa/compare", responses={409: {"description": "Regression gate failed"}})
    def qa_compare(request: CompareRequest) -> dict[str, Any]:
        corpus = _corpus(_corpus_path())
        try:
            baseline = run_extractor(corpus, request.baseline)
            candidate = run_extractor(corpus, request.candidate)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        report = compare_runs(
            baseline, candidate, policy=RegressionPolicy(max_regressions=request.max_regressions)
        )
        payload = report.to_dict()
        if not report.passed:
            raise HTTPException(status_code=409, detail=payload)
        return payload

    return app


app = create_app()
