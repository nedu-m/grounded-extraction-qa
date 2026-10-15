"""The web view.

A scorecard in Markdown is the right artefact for CI and the wrong one for a
person opening a link. This renders the same numbers as a single page that makes
its argument in the first screen: an extractor can answer every field correctly
and still be unusable, and the difference is measurable.

No framework, no JavaScript, no external assets. The whole page is derived from
the same functions the CI gate uses, so the page cannot drift from the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

from .corpus import Corpus, load_corpus
from .models import EvidenceStatus, GoldenCase
from .regression import RegressionPolicy, RegressionReport, compare_runs
from .runner import RunResult, run_extractor
from .sqlqa import DataQualityReport, load_checks, load_database, run_checks
from .verify import check_evidence

__all__ = ["Dashboard", "build_dashboard", "render_dashboard"]

REVISIONS = ("rules:v1", "rules:v2", "rules:v3")
CONTROLS = ("adversarial:uncited", "adversarial:fabricating")
INSPECT_DOC = "case_02_northgate_2024_thousands"
CONTRAST_DOC = "case_01_northgate_2024"

REPO_URL = "https://github.com/nedu-m/grounded-extraction-qa"
SITE_AUTHOR = "Chukwunedum Davidson"
PORTFOLIO_URL = "https://nedum.xyz"

STATUS_CLASS = {
    EvidenceStatus.SUPPORTED: "ok",
    EvidenceStatus.UNCITED: "warn",
    EvidenceStatus.CONTRADICTED: "bad",
    EvidenceStatus.NOT_A_CITATION: "bad",
    EvidenceStatus.OUT_OF_BOUNDS: "bad",
    EvidenceStatus.UNPARSEABLE: "bad",
}


@dataclass(frozen=True, slots=True)
class Dashboard:
    """Everything the page needs, already computed."""

    corpus: Corpus
    runs: dict[str, RunResult]
    gate: RegressionReport
    sql: DataQualityReport
    llm: dict[str, Any]


def build_dashboard(
    corpus_root: str | Path = "data/golden",
    sql_schema: str | Path = "sql/schema.sql",
    sql_seed: str | Path = "sql/seed.sql",
    checks_dir: str | Path = "sql/checks",
) -> Dashboard:
    from .extractors.llm import LLMConfig

    corpus = load_corpus(corpus_root)
    names = (*REVISIONS, *CONTROLS)
    runs = {name: run_extractor(corpus, name) for name in names}

    connection = load_database(sql_schema, sql_seed)
    sql_report = run_checks(connection, load_checks(checks_dir))

    return Dashboard(
        corpus=corpus,
        runs=runs,
        gate=compare_runs(runs["rules:v2"], runs["rules:v3"], policy=RegressionPolicy()),
        sql=sql_report,
        llm=LLMConfig.from_env().public_view(),
    )


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _e(value: Any) -> str:
    return escape(str(value), quote=True)


def _chip(text: str, kind: str) -> str:
    return f'<span class="chip {kind}">{_e(text)}</span>'


def _bar(value: float, kind: str = "") -> str:
    """A one-cell bar so a table column reads as a distribution at a glance."""
    width = max(0.0, min(1.0, value)) * 100
    return (
        f'<span class="bar"><span class="fill {kind}" style="width:{width:.1f}%"></span></span>'
        f'<span class="num">{_pct(value)}</span>'
    )


def _citation_rows(run: RunResult, case: GoldenCase, specs) -> list[dict[str, Any]]:
    extraction = (run.extractions or {}).get(case.doc_id)
    rows: list[dict[str, Any]] = []
    for name, spec in specs.items():
        claim = extraction.claim(name) if extraction else None
        gold = case.label(name)
        if claim is None:
            rows.append(
                {"field": name, "value": None, "quote": None, "status": None, "right": False}
            )
            continue
        check = check_evidence(case.text, claim, spec)
        quote = case.text[claim.span[0] : claim.span[1]] if claim.span else None
        rows.append(
            {
                "field": name,
                "value": claim.raw_value,
                "quote": quote,
                "status": check.status,
                "detail": check.detail,
                "right": bool(gold) and claim.raw_value == gold,
            }
        )
    return rows


def _revision_table(runs: dict[str, RunResult]) -> str:
    cols = (
        "<colgroup><col class='c-extractor'><col class='c-metric'><col class='c-metric'>"
        "<col class='c-metric'><col class='c-hero'></colgroup>"
    )
    head = (
        "<tr><th>Extractor and what it does</th><th>Recall</th><th>Citations</th>"
        "<th>Grounding</th><th class='hero-col'>Grounded recall</th></tr>"
    )
    notes = {
        "rules:v1": "Number after the label. Cites nothing.",
        "rules:v2": "Cites sources, matches labels by prefix.",
        "rules:v3": "Exact labels, document scale, tight citation.",
        "adversarial:uncited": "Every value right, no citations at all.",
        "adversarial:fabricating": "Invented values. Citation on the label.",
    }
    rows: list[str] = []
    for name, run in runs.items():
        card = run.scorecard
        control = name.startswith("adversarial:")
        cls = " control" if control else ""
        hero = "hero-col" if name == "rules:v3" else ""
        rows.append(
            f"<tr class='{cls.strip()}'>"
            f"<td class='extractor'><code>{_e(name)}</code>"
            f"<span class='sub'>{_e(notes.get(name, ''))}</span></td>"
            f"<td>{_bar(card.micro_recall)}</td>"
            f"<td>{_bar(card.citation_rate)}</td>"
            f"<td>{_bar(card.grounding_rate, 'ground')}</td>"
            f"<td class='{hero}'>{_bar(card.grounded_recall, 'hero')}</td>"
            f"</tr>"
        )
    return f"<table class='grid metrics'>{cols}{head}{''.join(rows)}</table>"


def _inspector(run: RunResult, case: GoldenCase, specs, caption: str) -> str:
    rows = _citation_rows(run, case, specs)
    body: list[str] = []
    for row in rows:
        if row["status"] is None:
            body.append(
                f"<tr><td><code>{_e(row['field'])}</code></td>"
                f"<td colspan='2' class='note'>not extracted</td>"
                f"<td>{_chip('missing', 'warn')}</td></tr>"
            )
            continue
        cls = STATUS_CLASS.get(row["status"], "warn")
        quote = (
            f"<span class='quote'>{_e(row['quote'])}</span>"
            if row["quote"]
            else "<span class='note'>no citation</span>"
        )
        body.append(
            f"<tr><td><code>{_e(row['field'])}</code></td>"
            f"<td class='num-cell'>{_e(row['value'])}</td>"
            f"<td>{quote}</td>"
            f"<td>{_chip(row['status'].value, cls)}</td></tr>"
        )
    return (
        f"<figure class='panel'>"
        f"<figcaption><code>{_e(run.extractor)}</code> reading <code>{_e(case.doc_id)}</code>"
        f"<span class='note'>{_e(caption)}</span></figcaption>"
        f"<table class='grid'><colgroup><col style='width:24%'><col style='width:15%'>"
        f"<col><col style='width:118px'></colgroup>"
        f"<tr><th>Field</th><th>Returned</th><th class='note-col'>Cited source text</th>"
        f"<th>Evidence</th></tr>{''.join(body)}</table></figure>"
    )


def _gate_panel(gate: RegressionReport) -> str:
    verdict = "PASS" if gate.passed else "FAIL"
    cls = "ok" if gate.passed else "bad"
    deltas = "".join(
        f"<tr><td><code>{_e(d.name)}</code></td><td class='num-cell'>{_pct(d.baseline)}</td>"
        f"<td class='num-cell'>{_pct(d.candidate)}</td>"
        f"<td class='num-cell {'up' if d.delta > 0 else 'flat'}'>{'+' if d.delta > 0 else ''}{_pct(d.delta)}</td></tr>"
        for d in gate.deltas
    )
    return (
        f"<section class='panel'><h2>Release gate: {_chip(verdict, cls)}</h2>"
        f"<p class='note'><code>{_e(gate.baseline_extractor)}</code> to "
        f"<code>{_e(gate.candidate_extractor)}</code>. "
        f"{len(gate.changes)} cells compared, {len(gate.fixes)} fixed, "
        f"{len(gate.regressions)} regressed. A cell may only stay level or improve, and a change to "
        f"the golden set fails the gate outright rather than moving the scores.</p>"
        f"<table class='grid'><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr>"
        f"{deltas}</table></section>"
    )


def _sql_panel(sql: DataQualityReport) -> str:
    rows = "".join(
        f"<tr><td><code>{_e(f.name)}</code></td>"
        f"<td class='num-cell'>{f.count}</td>"
        f"<td>{_chip('clean', 'ok') if f.passed else _chip('defect', 'bad')}</td>"
        f"<td class='note note-col'>{_e(f.description)}</td></tr>"
        for f in sql.findings
    )
    return (
        f"<section class='panel'><h2>The same idea applied to SQL</h2>"
        f"<p class='note'>Six checks over a seeded data room. The fixture has one planted defect per "
        f"check, and the test suite proves each check returns nothing when the defect is removed.</p>"
        f"<table class='grid'><tr><th>Check</th><th>Rows</th><th></th>"
        f"<th class='note-col'>What it catches</th></tr>"
        f"{rows}</table></section>"
    )


def _hero(dashboard: Dashboard) -> str:
    best = dashboard.runs["rules:v3"].scorecard
    worst = dashboard.runs["adversarial:uncited"].scorecard
    card = dashboard.corpus.cases[0]
    return (
        "<section class='hero'>"
        "<div class='hero-text'>"
        "<p class='kicker'>Golden-set QA for document extraction</p>"
        "<h1>A model can answer every field correctly<br>and still be unusable.</h1>"
        "<p class='lede'>The <code>adversarial:uncited</code> control below returns the true value "
        "for every field and cites nothing. Recall is 100%. Nothing it produces can be checked "
        "against a source, which is the only reason anyone asked for it. This harness scores "
        "accuracy and traceability separately, then refuses to ship a change that loses either.</p>"
        "</div>"
        "<div class='hero-figures'>"
        f"<div class='big'><span class='big-num'>{_pct(best.grounded_recall)}</span>"
        "<span class='big-label'>grounded recall, current revision</span></div>"
        f"<div class='big muted'><span class='big-num'>{_pct(worst.grounded_recall)}</span>"
        "<span class='big-label'>for an extractor that is never wrong and never citable</span></div>"
        "<div class='big muted'><span class='big-num'>"
        f"{len(dashboard.corpus.cases) * len(dashboard.corpus.specs)}</span>"
        "<span class='big-label'>hand-labelled cells, from "
        f"{len(dashboard.corpus.cases)} synthetic statements and "
        f"{len(dashboard.corpus.specs)} fields</span></div>"
        "</div>"
        "</section>"
        f"<p class='note corpus-note'>Documents are synthetic and the figures are fictional: "
        f"<code>{_e(card.doc_id)}</code> is a hospital group's balance sheet generated by "
        f"<code>tools/make_golden_corpus.py</code>. The corpus is generated, not typed, because six "
        f"hand-built statements drift and a drifted golden set fails silently.</p>"
    )


def render_dashboard(dashboard: Dashboard) -> str:
    corpus = dashboard.corpus
    inspect_case = corpus.case(INSPECT_DOC)
    contrast_case = corpus.case(CONTRAST_DOC)
    gate = dashboard.gate
    llm = dashboard.llm

    llm_line = (
        f"Live model: <code>{_e(llm['model'])}</code> via <code>{_e(llm['base_url'])}</code>."
        if llm["configured"]
        else "No model credentials are set on this instance, so the deterministic revisions run. "
        "The live path is the same interface, and the harness measures it identically."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grounded extraction QA — {_e(SITE_AUTHOR)}</title>
<meta name="description" content="Evidence-graded QA for LLM document extraction: recall, citation grounding, fabricated-fact detection, and regression gating.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<style>
  :root {{
    --bg: #0b0d10; --panel: #12151a; --line: #232830; --ink: #e8eaed; --dim: #9aa3ae;
    --accent: #7ee0b0; --accent-dim: #2f6b52; --bad: #ff8a80; --warn: #ffd166;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.6 var(--sans); }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 40px 24px 96px; }}
  code {{ font-family: var(--mono); font-size: .88em; color: #cfe3d8; }}
  a {{ color: var(--accent); }}
  h1 {{ font-size: clamp(28px, 4.4vw, 46px); line-height: 1.1; margin: 8px 0 16px; letter-spacing: -.02em; }}
  h2 {{ font-size: 20px; margin: 0 0 6px; letter-spacing: -.01em; }}
  .kicker {{ text-transform: uppercase; letter-spacing: .14em; font-size: 12px; color: var(--accent); margin: 0; }}
  .lede {{ color: #c4cbd4; max-width: 62ch; }}
  .note {{ color: var(--dim); font-size: 14px; }}
  .corpus-note {{ margin-top: 28px; max-width: 78ch; }}
  .hero {{ display: grid; grid-template-columns: 1.5fr 1fr; gap: 40px; align-items: end; }}
  @media (max-width: 860px) {{ .hero {{ grid-template-columns: 1fr; gap: 24px; }} }}
  .hero-figures {{ display: grid; gap: 14px; }}
  .big {{ border-left: 2px solid var(--accent); padding-left: 14px; }}
  .big.muted {{ border-color: var(--line); }}
  .big-num {{ display: block; font-family: var(--mono); font-size: 30px; font-variant-numeric: tabular-nums; }}
  .big.muted .big-num {{ color: var(--dim); font-size: 22px; }}
  .big-label {{ display: block; color: var(--dim); font-size: 13px; max-width: 34ch; }}
  .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 20px; margin: 28px 0; }}
  figure.panel {{ padding: 0; overflow: hidden; }}
  figcaption {{ padding: 14px 20px; border-bottom: 1px solid var(--line); display: flex; gap: 12px; flex-wrap: wrap; align-items: baseline; }}
  table.grid {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  table.grid.metrics {{ table-layout: fixed; }}
  table.grid.metrics th, table.grid.metrics td {{ white-space: nowrap; }}
  table.grid td.extractor {{ white-space: normal; word-break: break-word; }}
  table.grid.metrics td {{ padding: 8px 10px; }}
  table.grid.metrics col.c-extractor {{ width: 30%; }}
  table.grid.metrics col.c-metric {{ width: 132px; }}
  table.grid.metrics col.c-hero {{ width: 148px; }}
  table.grid.metrics .sub {{ display: block; color: var(--dim); font-size: 12px; margin-top: 2px; white-space: normal; font-family: var(--sans); }}
  table.grid th.note-col, table.grid td.note-col {{ white-space: normal; }}
  table.grid col.c-source {{ width: 42%; }}
  table.grid th {{ text-align: left; font-weight: 600; color: var(--dim); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; padding: 10px 12px; border-bottom: 1px solid var(--line); }}
  table.grid td {{ padding: 10px 12px; border-bottom: 1px solid #1a1e24; vertical-align: top; }}
  table.grid tr:last-child td {{ border-bottom: 0; }}
  tr.control td {{ background: #14171c; }}
  .hero-col {{ background: #101a15; }}
  .num-cell {{ font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .up {{ color: var(--accent); }}
  .flat {{ color: var(--dim); }}
  .num {{ font-family: var(--mono); font-size: 12px; margin-left: 8px; color: var(--dim); }}
  .bar {{ display: inline-block; width: 60px; height: 6px; background: #1c2128; border-radius: 3px; overflow: hidden; vertical-align: middle; }}
  .bar .fill {{ display: block; height: 100%; background: #4a5560; }}
  .bar .fill.ground {{ background: var(--warn); }}
  .bar .fill.hero {{ background: var(--accent); }}
  .chip {{ display: inline-block; font-family: var(--mono); font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); color: var(--dim); text-transform: lowercase; }}
  .chip.ok {{ color: var(--accent); border-color: var(--accent-dim); }}
  .chip.bad {{ color: var(--bad); border-color: #6b2f2c; }}
  .chip.warn {{ color: var(--warn); border-color: #6b5a2c; }}
  .quote {{ font-family: var(--mono); font-size: 12px; background: #0e1116; border: 1px solid var(--line); border-radius: 4px; padding: 2px 6px; white-space: pre-wrap; overflow-wrap: anywhere; }}
  footer {{ margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--dim); font-size: 14px; display: flex; gap: 18px; flex-wrap: wrap; }}
</style>
</head>
<body>
<div class="wrap">
{_hero(dashboard)}

<section class="panel">
  <h2>Accuracy and traceability, scored separately</h2>
  <p class="note">Grounding is the share of returned values whose own citation actually contains
  that value. It is not a vibe check: the cited span is sliced out of the document and re-read.</p>
  {_revision_table(dashboard.runs)}
</section>

{_inspector(dashboard.runs["rules:v3"], inspect_case, corpus.specs, "stated in thousands: the value must be scaled and the citation still has to quote the document literally")}
{_inspector(dashboard.runs["adversarial:fabricating"], contrast_case, corpus.specs, "the control: a plausible number, and a citation pointing at the label instead of the figure")}

{_gate_panel(gate)}
{_sql_panel(dashboard.sql)}

<section class="panel">
  <h2>Reproduce it</h2>
  <p class="note">{llm_line}</p>
  <p class="note"><code>make install &amp;&amp; make ci</code> runs the linter, 288 tests, all five
  extractors, the regression gate and the SQL checks. <code>make report</code> writes the scorecards
  and <code>make compare</code> fails on any cell that got worse.</p>
  <p class="note">The deterministic revisions are trended rather than asserted: <code>rules:v1</code>
  to <code>rules:v2</code> to <code>rules:v3</code> is the same extractor being fixed prompt by
  prompt, and the gate that proves each fix holds.</p>
</section>

<footer>
  <span>{_e(SITE_AUTHOR)}</span>
  <a href="{_e(PORTFOLIO_URL)}">nedum.xyz</a>
  <a href="{_e(REPO_URL)}">github.com/nedu-m/grounded-extraction-qa</a>
  <span>MIT</span>
</footer>
</div>
</body>
</html>
"""


@lru_cache(maxsize=1)
def _cached_html(version: str) -> str:  # pragma: no cover - thin cache wrapper
    return render_dashboard(build_dashboard())


def dashboard_html() -> str:
    """Render the page, rebuilding only when the process starts."""
    from . import __version__

    return _cached_html(__version__)
