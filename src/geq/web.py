"""The web view.

A scorecard in Markdown is the right artefact for CI and the wrong one for the
person the link is sent to: a deal lead or a recruiter who has never heard the
word "recall". So the page is written for them first, in the form they already
trust for this job: an auditor's working paper, where every figure is traced to
its source and ticked. The reader can pick who prepared the schedule, click any
figure and see the exact line it came from; the engineering metrics sit in a
closed section at the bottom for whoever wants them.

No framework and no third-party hosts: the fonts are served by the app itself.
The page is derived from the same functions the CI gate uses, so it cannot drift
from the gate. A small inline script drives the working paper; everything else
renders without it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

from .corpus import Corpus, load_corpus
from .models import CellResult, EvidenceStatus, FieldKind, FieldSpec, GoldenCase, Verdict
from .regression import RegressionPolicy, RegressionReport, compare_runs
from .runner import RunResult, load_run, run_extractor
from .sqlqa import DataQualityReport, load_checks, load_database, run_checks

__all__ = ["Dashboard", "Outcome", "build_dashboard", "classify", "render_dashboard"]

REVISIONS = ("rules:v1", "rules:v2", "rules:v3")
CONTROLS = ("adversarial:uncited", "adversarial:fabricating")
LLM_ROW = "llm"
"""Shown only when a stored run exists. The model is measured once and the result
is read back, because re-running it on every page render would be slow and would
spend the operator's money to display a number that has not changed."""

BEST = "rules:v3"
UNBALANCED_DOC = "case_04_northgate_2024_unbalanced"

REPO_URL = "https://github.com/nedu-m/grounded-extraction-qa"
SITE_AUTHOR = "Chukwunedum Davidson"
PORTFOLIO_URL = "https://nedum.xyz"


# ---------------------------------------------------------------------------
# Plain-language vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reader:
    """How an extractor is introduced to someone who does not write code."""

    name: str
    blurb: str
    control: bool = False


READERS: dict[str, Reader] = {
    "rules:v3": Reader(
        "Current version",
        "Finds each figure, applies the statement's units, and points to the exact line.",
    ),
    LLM_ROW: Reader(
        "Live AI model",
        "A language model reading the same statements, checked by the same rules.",
    ),
    "rules:v2": Reader(
        "Previous version",
        "Points to its sources, but confuses lines with similar names.",
    ),
    "rules:v1": Reader(
        "First draft",
        "Grabs the number after a label and never says where it came from.",
    ),
    "adversarial:uncited": Reader(
        "Test: no sources",
        "Built to fail. Every figure correct, and not one source given.",
        control=True,
    ),
    "adversarial:fabricating": Reader(
        "Test: fabricator",
        "Built to fail. Invents believable figures and cites the row label, so each one "
        "looks sourced.",
        control=True,
    ),
}
READER_ORDER = tuple(READERS)


@dataclass(frozen=True, slots=True)
class Outcome:
    """What happened to one number, in words a deal team would use."""

    key: str
    group: str
    """One of ``proven``, ``unproven``, ``failed``, ``missed``: the four bar colours."""
    label: str
    reason: str


GROUPS = (
    ("proven", "Agreed to source", "Right, and the cited line shows it."),
    ("unproven", "No source given", "Right, but a reviewer must find it again by hand."),
    ("failed", "Exception", "Wrong, or the cited line does not back it up."),
    ("missed", "Not reported", "Left blank."),
)

SQL_PLAIN = {
    "orphaned_extracted_fields": "Figures with no document behind them",
    "foreign_key_violations": "Records that point at things that do not exist",
    "duplicate_documents": "The same file uploaded twice for one practice",
    "missing_data_room_items": "Checklist items with nothing filed against them",
    "balance_sheet_reconciliation": "Totals from two sources that disagree",
    "brand_asset_mismatch": "A marketing asset carrying another brand's name",
}

DOC_NOTES = {
    "case_01_northgate_2024": "A clean statement. Everything adds up.",
    "case_02_northgate_2024_thousands": (
        "The same figures, printed in thousands. 13,504.5 here means $13,504,500."
    ),
    "case_03_cascade_2024_millions": "Printed in millions, to one decimal place.",
    "case_04_northgate_2024_unbalanced": (
        "Looks complete, but does not balance: equity is $100,000 short."
    ),
    "case_05_riverbend_2023": (
        "A draft disclaimer sits above the company name, and the date is written month first."
    ),
    "case_06_harbour_point_2025h1": (
        "A half-year statement exported with dotted leaders, and a note that repeats a figure."
    ),
}


def classify(cell: CellResult) -> Outcome:
    """Collapse a scored cell into one of a handful of outcomes a person can act on."""
    status = cell.check.status if cell.check else None
    if cell.verdict is Verdict.MISSING:
        return Outcome("missed", "missed", "Not reported", "Left blank.")
    if cell.verdict is Verdict.RIGHT:
        if status is EvidenceStatus.SUPPORTED:
            return Outcome(
                "proven",
                "proven",
                "Agreed to source",
                "The cited line shows this exact figure.",
            )
        if status is EvidenceStatus.UNCITED:
            return Outcome(
                "unproven",
                "unproven",
                "No source given",
                "The figure happens to be right, but nothing says where it came from, so a "
                "reviewer would have to find it again by hand.",
            )
        return Outcome(
            "bad_proof",
            "failed",
            "Exception: source does not show it",
            "The figure is right, but the text it cites does not contain it.",
        )
    if status is EvidenceStatus.SUPPORTED:
        return Outcome(
            "misread",
            "failed",
            "Exception: wrong figure",
            "It cites a real line, but the wrong one, or it ignored the statement's units.",
        )
    if status is EvidenceStatus.UNCITED:
        return Outcome("wrong", "failed", "Exception: wrong figure", "Wrong, and no source given.")
    return Outcome(
        "fabricated",
        "failed",
        "Exception: not on the cited line",
        "The cited line does not contain this figure. Invented, and caught.",
    )


def _money(raw: str) -> str:
    try:
        amount = Decimal(raw)
    except (InvalidOperation, ValueError):
        return raw
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount == amount.to_integral_value():
        return f"{sign}${amount:,.0f}"
    return f"{sign}${amount:,.2f}"


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _date(raw: str) -> str:
    parts = raw.split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        year, month, day = (int(p) for p in parts)
        if 1 <= month <= 12:
            return f"{day} {_MONTHS[month - 1]} {year}"
    return raw


def display_value(raw: str | None, spec: FieldSpec) -> str:
    """A golden value as a person would write it."""
    if raw is None:
        return ""
    if spec.kind is FieldKind.MONEY:
        return _money(raw)
    if spec.kind is FieldKind.DATE:
        return _date(raw)
    return raw


def field_label(spec: FieldSpec) -> str:
    if spec.name == "entity_name":
        return "Company name"
    if spec.name == "fiscal_year_end":
        return "Period end"
    return spec.label


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dashboard:
    """Everything the page needs, already computed."""

    corpus: Corpus
    runs: dict[str, RunResult]
    gate: RegressionReport
    reverse_gate: RegressionReport
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

    stored = _stored_llm_run()
    if stored is not None:
        runs[LLM_ROW] = stored

    connection = load_database(sql_schema, sql_seed)
    sql_report = run_checks(connection, load_checks(checks_dir))
    policy = RegressionPolicy()

    return Dashboard(
        corpus=corpus,
        runs=runs,
        gate=compare_runs(runs["rules:v2"], runs["rules:v3"], policy=policy),
        reverse_gate=compare_runs(runs["rules:v3"], runs["rules:v2"], policy=policy),
        sql=sql_report,
        llm=LLMConfig.from_env().public_view(),
    )


def _utf16(text: str, index: int) -> int:
    """Python indexes code points; the browser indexes UTF-16 units."""
    return len(text[:index].encode("utf-16-le")) // 2


def _span_for(run: RunResult, case: GoldenCase, cell: CellResult) -> list[int] | None:
    extraction = (run.extractions or {}).get(case.doc_id)
    claim = extraction.claim(cell.field) if extraction else None
    span = claim.span if claim else None
    if span is None and cell.check and cell.check.cited_text:
        # A run read back from disk carries the cited text but not the offsets.
        start = case.text.find(cell.check.cited_text)
        span = (start, start + len(cell.check.cited_text)) if start >= 0 else None
    if span is None:
        return None
    start, end = span
    if not 0 <= start < end <= len(case.text):
        return None
    return [_utf16(case.text, start), _utf16(case.text, end)]


def outcome_counts(run: RunResult) -> dict[str, int]:
    counts = {group: 0 for group, _, _ in GROUPS}
    for cell in run.scorecard.cells:
        if cell.verdict is Verdict.EXTRA:
            continue
        counts[classify(cell).group] += 1
    return counts


def explorer_payload(dashboard: Dashboard) -> dict[str, Any]:
    """The data the document explorer needs, keyed for the browser."""
    corpus = dashboard.corpus
    docs = []
    for case in corpus.cases:
        entity = case.label("entity_name") or case.doc_id
        period = case.label("fiscal_year_end")
        docs.append(
            {
                "id": case.doc_id,
                "title": f"{entity}, {_date(period)}" if period else entity,
                "entity": entity,
                "period": _date(period) if period else "",
                "note": DOC_NOTES.get(case.doc_id, ""),
                "text": case.text,
            }
        )

    readers = []
    for key in READER_ORDER:
        run = dashboard.runs.get(key)
        if run is None:
            continue
        cells: dict[str, dict[str, Any]] = {}
        by_doc = {case.doc_id: case for case in corpus.cases}
        for cell in run.scorecard.cells:
            spec = corpus.specs.get(cell.field)
            golden = by_doc.get(cell.doc_id)
            if spec is None or golden is None or cell.verdict is Verdict.EXTRA:
                continue
            outcome = classify(cell)
            cells.setdefault(cell.doc_id, {})[cell.field] = {
                "reported": cell.claimed or "",
                "expected": display_value(cell.gold, spec),
                "outcome": outcome.key,
                "group": outcome.group,
                "label": outcome.label,
                "reason": outcome.reason,
                "span": _span_for(run, golden, cell),
                "cited": " ".join((cell.check.cited_text or "").split()) if cell.check else "",
            }
        reader = READERS[key]
        readers.append(
            {
                "key": key,
                "name": reader.name,
                "blurb": reader.blurb,
                "control": reader.control,
                "cells": cells,
            }
        )

    return {
        "fields": [{"name": n, "label": field_label(s)} for n, s in corpus.specs.items()],
        "docs": docs,
        "readers": readers,
        "groups": [{"key": g, "label": label} for g, label, _ in GROUPS],
    }


def _stored_llm_run() -> RunResult | None:
    """Read a previously measured live-model run, if one was shipped or produced."""
    path = Path(os.environ.get("GEQ_LLM_REPORT", "reports/llm.json"))
    if not path.is_file():
        return None
    try:
        return load_run(path)
    except (KeyError, ValueError):
        # A malformed report degrades to "not shown" rather than taking the page
        # down; the deterministic rows are the part that must always render.
        return None


# ---------------------------------------------------------------------------
# HTML
#
# The page is an auditor's working paper. Auditors trace every figure in a
# schedule back to its source and tick it; this tool does the same to an AI's
# extraction, so the page borrows that paper's grammar instead of a dashboard's:
# ledger-ruled schedules, typed figures, pencil tick marks with a legend, totals
# with single and double rules, and a sign-off stamp.
# ---------------------------------------------------------------------------

FONT_FILES = (
    "libre-franklin-var.woff2",
    "libre-franklin-italic-var.woff2",
    "courier-prime-400.woff2",
    "courier-prime-700.woff2",
)
FONT_DIR = Path(__file__).parent / "static" / "fonts"

DEFAULT_READER = "rules:v2"
"""The schedule opens on a reader with both ticks and exceptions, so the first
screen shows the check doing its job rather than a page of green."""


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _e(value: Any) -> str:
    return escape(str(value), quote=True)


def _json_for_script(payload: Any) -> str:
    """JSON that cannot close the script tag it lives in, whatever a document says."""
    return (
        json.dumps(payload, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _figure(raw: str) -> str:
    """A money string as typed on a working paper: no symbol, negatives in brackets."""
    try:
        amount = Decimal(raw)
    except (InvalidOperation, ValueError):
        return raw
    text = f"{abs(amount):,.0f}" if amount == amount.to_integral_value() else f"{abs(amount):,.2f}"
    return f"({text})" if amount < 0 else text


def _tick(group: str, label: str = "") -> str:
    sr = f'<span class="sr">{_e(label)}</span>' if label else ""
    return (
        f'<svg class="tick g-{group}" aria-hidden="true" viewBox="0 0 20 20">'
        f'<use href="#t-{group}"/></svg>{sr}'
    )


_SPRITE = """<svg width="0" height="0" style="position:absolute" aria-hidden="true">
<symbol id="t-proven" viewBox="0 0 20 20"><path d="M3.5 10.8l4.1 4.3L16.6 3.8" fill="none"
 stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/></symbol>
<symbol id="t-unproven" viewBox="0 0 20 20"><circle cx="10" cy="10" r="6" fill="none"
 stroke="currentColor" stroke-width="1.9"/></symbol>
<symbol id="t-failed" viewBox="0 0 20 20"><circle cx="10" cy="10" r="7.6" fill="none"
 stroke="currentColor" stroke-width="1.8"/><path d="M6.9 6.9l6.2 6.2M13.1 6.9l-6.2 6.2"
 stroke="currentColor" stroke-width="2.1" stroke-linecap="round"/></symbol>
<symbol id="t-missed" viewBox="0 0 20 20"><path d="M5 10h10" stroke="currentColor"
 stroke-width="2.2" stroke-linecap="round"/></symbol>
</svg>"""


def _legend() -> str:
    return "".join(
        f"<li>{_tick(group)}<span><strong>{_e(label)}.</strong> {_e(desc)}</span></li>"
        for group, label, desc in GROUPS
    )


def _masthead() -> str:
    return f"""
<header class="masthead">
  <span class="mark">Grounded extraction QA</span>
  <nav aria-label="About this project">
    <a href="{_e(PORTFOLIO_URL)}">{_e(SITE_AUTHOR)}</a>
    <a href="{_e(REPO_URL)}">Source on GitHub</a>
  </nav>
</header>"""


def _opening() -> str:
    return """
<section class="opening">
  <h1>Every figure the AI reports, ticked back to the line that proves it.</h1>
  <p class="lede">Auditors trace each number in a working paper to its source and tick it. This
  tool does the same for AI that reads a practice's financial statements during an acquisition.
  A figure is accepted only when the line it cites really shows it. Anything else becomes an
  exception, even when the number happens to be right.</p>
</section>"""


def _working_paper(dashboard: Dashboard) -> str:
    payload = explorer_payload(dashboard)
    default_reader = (
        DEFAULT_READER if DEFAULT_READER in dashboard.runs else payload["readers"][0]["key"]
    )
    reader_buttons = "".join(
        f'<button type="button" class="preparer{" test" if r["control"] else ""}" '
        f'data-reader="{_e(r["key"])}" '
        f'aria-pressed="{"true" if r["key"] == default_reader else "false"}">'
        f"{_e(r['name'])}</button>"
        for r in payload["readers"]
    )
    doc_options = "".join(
        f'<option value="{_e(d["id"])}">{_e(d["entity"])}, {_e(d["period"])}</option>'
        for d in payload["docs"]
    )
    first = payload["docs"][0]
    return f"""
<section class="paper" id="working-paper" aria-labelledby="wp-title"
  data-default-reader="{_e(default_reader)}">
  <div class="paper-head">
    <h2 id="wp-title">Working paper <span class="ref">A-1</span></h2>
    <label class="head-cell entity">Statement
      <select id="doc">{doc_options}</select></label>
    <div class="head-cell preparers">
      <span class="head-label" id="prep-label">Prepared by</span>
      <div class="preparer-row" role="group" aria-labelledby="prep-label">{reader_buttons}</div>
    </div>
    <div class="head-cell reviewer"><span class="head-label">Checked by</span>
      <span class="hand">this tool</span></div>
  </div>
  <p class="prep-note" id="prep-note"></p>
  <div class="paper-body">
    <div class="schedule-wrap">
      <table class="schedule">
        <caption class="sr">Figures reported, with the result of tracing each to its source.
        Select a row to see its source line.</caption>
        <thead><tr><th scope="col">Item</th><th scope="col" class="num">Reported</th>
        <th scope="col" class="tk"><span class="sr">Tick</span></th>
        <th scope="col" class="wp">Source</th></tr></thead>
        <tbody id="schedule"><tr><td colspan="4">Turn on JavaScript to work through the
        schedule.</td></tr></tbody>
      </table>
    </div>
    <figure class="source">
      <figcaption><span>Source document</span>
        <span class="doc-note" id="doc-note">{_e(first["note"])}</span></figcaption>
      <div class="source-text" id="source" tabindex="0"
        aria-label="Source document text"><pre>{_e(first["text"])}</pre></div>
    </figure>
  </div>
  <ul class="legend" aria-label="Tick mark legend">{_legend()}</ul>
  <script id="paper-data" type="application/json">{_json_for_script(payload)}</script>
</section>"""


def _summary(dashboard: Dashboard) -> str:
    rows = []
    total_cells = 0
    for key in READER_ORDER:
        run = dashboard.runs.get(key)
        if run is None:
            continue
        reader = READERS[key]
        counts = outcome_counts(run)
        total = sum(counts.values()) or 1
        total_cells = total
        bar = "".join(
            f'<span class="seg g-{group}" style="width:{counts[group] / total * 100:.2f}%"></span>'
            for group, _, _ in GROUPS
            if counts[group]
        )
        cls = " class='test'" if reader.control else ""
        rows.append(
            f"<tr{cls}><th scope='row'>{_e(reader.name)}"
            f"<span class='blurb'>{_e(reader.blurb)}</span></th>"
            + "".join(f"<td class='num c-{g}'>{counts[g]}</td>" for g, _, _ in GROUPS)
            + f"<td class='bar-cell'><span class='bar' aria-hidden='true'>{bar}</span></td>"
            f"<td class='num usable'><strong>{counts['proven']}</strong>"
            f"<span class='of'> of {total}</span></td></tr>"
        )
    heads = "".join(
        f"<th scope='col' class='num c-{g}'>{_tick(g)}<span class='th-label'>{_e(label)}</span></th>"
        for g, label, _ in GROUPS
    )
    docs = len(dashboard.corpus.cases)
    fields = len(dashboard.corpus.specs)
    return f"""
<section class="sheet" aria-labelledby="summary-title">
  <h2 id="summary-title">Review summary</h2>
  <p class="intro">{docs} statements, {fields} figures each: {total_cells} figures per preparer.
  Only figures agreed to source are counted as usable, because they are the only ones a deal
  team can rely on without checking them again.</p>
  <div class="table-wrap"><table class="summary">
    <thead><tr><th scope="col">Prepared by</th>{heads}<th scope="col" class="bar-cell">
    <span class="sr">Distribution</span></th><th scope="col" class="num">Usable</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
  <p class="pencil">The "no sources" test gets every figure right and is still usable on none
  of them. Scoring correctness alone would have rated it perfect.</p>
</section>"""


def _footing(dashboard: Dashboard) -> str:
    run = dashboard.runs[BEST]
    fail = next(
        (
            result
            for case in run.reconciliations
            if case.doc_id == UNBALANCED_DOC
            for result in case.failures
            if result.name == "balance_sheet_identity"
        ),
        None,
    )
    if fail is None or fail.expected is None or fail.actual is None:
        return ""
    case = dashboard.corpus.case(UNBALANCED_DOC)
    liabilities = case.label("total_liabilities") or "0"
    equity = case.label("total_equity") or "0"
    entity = case.label("entity_name") or UNBALANCED_DOC
    gap = Decimal(fail.actual) - Decimal(fail.expected)
    return f"""
<section class="sheet footing-sheet" aria-labelledby="footing-title">
  <div class="footing-copy">
    <h2 id="footing-title">It also catches a statement that does not add up</h2>
    <p>Every figure on {_e(entity)}'s statement was read and agreed to source. The statement
    still does not foot: liabilities plus equity should equal total assets, and here they fall
    short. A tool that only copied numbers would carry the gap straight into the deal summary.</p>
  </div>
  <table class="footing" aria-label="Footing of {_e(entity)}'s balance sheet">
    <tr><th scope="row">Total liabilities</th><td>{_figure(liabilities)}</td></tr>
    <tr><th scope="row">Total equity</th><td class="rule">{_figure(equity)}</td></tr>
    <tr><th scope="row">Liabilities and equity</th><td>{_figure(fail.actual)}</td></tr>
    <tr><th scope="row">Total assets</th><td class="double">{_figure(fail.expected)}</td></tr>
    <tr class="diff"><th scope="row">Difference</th>
      <td>{_tick("failed", "Exception")}{_figure(str(gap))}</td></tr>
  </table>
  <p class="pencil footing-note">Does not foot by {_figure(str(abs(gap)))}. Flagged before it
  reaches the deal team.</p>
</section>"""


def _signoff(dashboard: Dashboard) -> str:
    def block(report: RegressionReport, title: str) -> str:
        ok = report.passed
        return (
            f'<div class="signoff-block {"ok" if ok else "no"}">'
            f"<h3>{_e(title)}</h3>"
            f"<dl><div><dt>Figures fixed</dt><dd>{len(report.fixes)}</dd></div>"
            f"<div><dt>Figures made worse</dt><dd>{len(report.regressions)}</dd></div></dl>"
            f'<p class="stamp {"approved" if ok else "refused"}">'
            f"{'Approved to ship' if ok else 'Refused'}</p></div>"
        )

    return f"""
<section class="sheet" aria-labelledby="signoff-title">
  <h2 id="signoff-title">No update ships if it makes a single figure worse</h2>
  <p class="intro">Every change to the reader, including a change to an AI prompt, is worked
  through the same statements before it ships. If any one figure gets worse, sign-off is refused,
  even when the overall score goes up.</p>
  <div class="signoffs">
    {block(dashboard.gate, "Previous version to current version")}
    {block(dashboard.reverse_gate, "The same change, rolled backwards")}
  </div>
</section>"""


def _checklist(sql: DataQualityReport) -> str:
    rows = "".join(
        f"<tr><th scope='row'>{_e(SQL_PLAIN.get(f.name, f.name))}</th>"
        f"<td>{_tick('proven' if not f.passed else 'unproven')}"
        f"{'Found ' + str(f.count) + (' records' if f.count != 1 else ' record') if not f.passed else 'Nothing found'}"
        f"</td></tr>"
        for f in sql.findings
    )
    return f"""
<section class="sheet" aria-labelledby="checklist-title">
  <h2 id="checklist-title">The same discipline for the data room</h2>
  <p class="intro">A sample diligence database, seeded on purpose with one problem of each kind,
  so every check should find something. The test suite also proves each check finds nothing once
  its problem is removed, so a check cannot pass by always raising the alarm.</p>
  <table class="checklist"><thead><tr><th scope="col">Planted problem</th>
  <th scope="col">Result</th></tr></thead><tbody>{rows}</tbody></table>
</section>"""


def _revision_table(runs: dict[str, RunResult]) -> str:
    head = (
        "<tr><th scope='col'>Extractor</th><th scope='col'>Recall</th><th scope='col'>Precision"
        "</th><th scope='col'>Citing</th><th scope='col'>Grounding</th>"
        "<th scope='col'>Grounded recall</th></tr>"
    )
    rows = "".join(
        f"<tr><th scope='row'><code>{_e(name)}</code></th>"
        f"<td>{_pct(run.scorecard.micro_recall)}</td>"
        f"<td>{_pct(run.scorecard.micro_precision)}</td>"
        f"<td>{_pct(run.scorecard.citation_rate)}</td>"
        f"<td>{_pct(run.scorecard.grounding_rate)}</td>"
        f"<td><strong>{_pct(run.scorecard.grounded_recall)}</strong></td></tr>"
        for name, run in runs.items()
    )
    return f"<div class='table-wrap'><table class='metrics'>{head}{rows}</table></div>"


def _supporting(dashboard: Dashboard) -> str:
    llm = dashboard.llm
    llm_line = (
        f"Live model on this instance: <code>{_e(llm['model'])}</code> via "
        f"<code>{_e(llm['base_url'])}</code>."
        if llm["configured"]
        else "No model credentials are set on this instance, so the deterministic revisions run. "
        "The live model path uses the same interface and is measured identically."
    )
    gate = dashboard.gate
    deltas = "".join(
        f"<tr><th scope='row'><code>{_e(d.name)}</code></th><td>{_pct(d.baseline)}</td>"
        f"<td>{_pct(d.candidate)}</td><td>{'+' if d.delta > 0 else ''}{_pct(d.delta)}</td></tr>"
        for d in gate.deltas
    )
    return f"""
<details class="sheet supporting">
  <summary>Supporting schedules for engineers</summary>
  <p><strong>Grounding</strong> means the cited span is sliced out of the document and re-read:
  a citation that does not contain its own value fails, and the offending text is recorded.
  <strong>Grounded recall</strong> is the share of true values returned correct <em>and</em>
  grounded, and it is the headline metric.</p>
  {_revision_table(dashboard.runs)}
  <h3>Release gate: <code>{_e(gate.baseline_extractor)}</code> to
  <code>{_e(gate.candidate_extractor)}</code>, {"PASS" if gate.passed else "FAIL"}</h3>
  <div class="table-wrap"><table class="metrics"><tr><th scope="col">Metric</th>
  <th scope="col">Baseline</th><th scope="col">Candidate</th><th scope="col">Delta</th></tr>
  {deltas}</table></div>
  <p>The gate fails on any regressed cell, any drop in a headline metric, any change to the
  golden set itself, and any arithmetic check that stops reconciling.</p>
  <p>{llm_line}</p>
  <p><code>make install &amp;&amp; make ci</code> runs the linter, the test suite, every
  extractor, the regression gate and the SQL checks. <code>make compare</code> exits non-zero on
  any cell that got worse. The documents are synthetic and every figure is fictional; the golden
  set is generated by <code>tools/make_golden_corpus.py</code> so it cannot drift.</p>
</details>"""


_CSS = """
@font-face { font-family: "Libre Franklin"; font-style: normal; font-weight: 100 900;
  font-display: swap; src: url(/fonts/libre-franklin-var.woff2) format("woff2"); }
@font-face { font-family: "Libre Franklin"; font-style: italic; font-weight: 100 900;
  font-display: swap; src: url(/fonts/libre-franklin-italic-var.woff2) format("woff2"); }
@font-face { font-family: "Courier Prime"; font-style: normal; font-weight: 400;
  font-display: swap; src: url(/fonts/courier-prime-400.woff2) format("woff2"); }
@font-face { font-family: "Courier Prime"; font-style: normal; font-weight: 700;
  font-display: swap; src: url(/fonts/courier-prime-700.woff2) format("woff2"); }

/* A ledger pad on a desk in daylight: green-tinted paper, blue-black typed ink,
   and three pencils (green for agreed, red for exceptions, blue for the reviewer). */
:root {
  --desk: #d9e1cc; --paper: #f5f8ee; --paper-head: #e9efdc;
  --rule: #c3d3ae; --rule-strong: #87a26f; --ink: #17222e; --ink-2: #47534a;
  --pencil-green: #2b6a35; --pencil-red: #b5302a; --pencil-blue: #2c4f9e;
  --graphite: #6a6f68; --highlight: #fbe98a; --highlight-soft: #fdf5c9;
  --typed: "Courier Prime", "Courier New", monospace;
  --sans: "Libre Franklin", "Franklin Gothic Medium", "Arial Narrow", sans-serif;
  --shadow: 0 1px 2px rgba(40, 56, 30, .10), 0 10px 28px -12px rgba(40, 56, 30, .28);
  color-scheme: light;
}
* { box-sizing: border-box; }
html { scrollbar-color: var(--rule-strong) var(--paper-head); }
body { margin: 0; background: var(--desk); color: var(--ink);
  font: 400 17px/1.6 var(--sans); font-variant-numeric: tabular-nums; caret-color: var(--pencil-blue); }
::selection { background: var(--highlight); color: var(--ink); }
a { color: var(--pencil-blue); text-underline-offset: .2em; text-decoration-thickness: 1px; }
a:hover { text-decoration-thickness: 2px; }
:focus-visible { outline: 2px solid var(--pencil-blue); outline-offset: 3px; border-radius: 2px; }
code { font-family: var(--typed); font-size: .95em; }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0);
  white-space: nowrap; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 24px 72px; }

.masthead { display: flex; justify-content: space-between; align-items: baseline; gap: 16px;
  flex-wrap: wrap; padding: 22px 0 18px; border-bottom: 1px solid var(--rule-strong); }
.mark { font-weight: 800; letter-spacing: -.01em; }
.masthead nav { display: flex; gap: 22px; font-size: 15px; }

.opening { padding: 44px 0 30px; display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
  gap: 48px; align-items: end; }
h1 { font-size: clamp(32px, 4.2vw, 54px); line-height: 1.04; font-weight: 800;
  letter-spacing: -.03em; margin: 0; text-wrap: balance; }
.lede { margin: 0; font-size: 18px; max-width: 54ch; color: var(--ink-2); }
h2 { font-size: 26px; line-height: 1.2; font-weight: 800; letter-spacing: -.02em; margin: 0 0 10px;
  text-wrap: balance; }
h3 { font-size: 17px; font-weight: 700; margin: 0 0 8px; }

/* The working paper */
.paper { background: var(--paper); box-shadow: var(--shadow); border-radius: 3px;
  border: 1px solid var(--rule-strong); }
.paper-head { display: grid; grid-template-columns: auto minmax(250px, 1.2fr) minmax(0, 2fr) auto;
  border-bottom: 2px solid var(--rule-strong); background: var(--paper-head); }
.paper-head > * { padding: 12px 16px; border-right: 1px solid var(--rule); min-width: 0; }
.paper-head > :last-child { border-right: 0; }
.paper-head h2 { font-size: 18px; margin: 0; align-self: center; white-space: nowrap; }
.ref { font-family: var(--typed); font-weight: 700; color: var(--pencil-blue); }
.head-cell { display: flex; flex-direction: column; gap: 6px; }
.head-label, .entity { font-size: 12px; font-weight: 600; letter-spacing: .04em; color: var(--ink-2); }
select { font: 500 15px/1.3 var(--sans); color: var(--ink); background: var(--paper);
  border: 1px solid var(--rule-strong); border-radius: 3px; padding: 7px 8px; width: 100%; }
.preparer-row { display: flex; flex-wrap: wrap; gap: 6px; }
button.preparer { font: 500 14px/1.2 var(--sans); color: var(--ink); background: transparent;
  border: 1px solid var(--rule-strong); border-radius: 3px; padding: 7px 10px; cursor: pointer;
  transition: background-color .15s ease-out, color .15s ease-out; }
button.preparer:hover { background: var(--paper); }
button.preparer.test { border-style: dashed; }
button.preparer[aria-pressed="true"] { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.hand { font-family: var(--typed); font-weight: 700; color: var(--pencil-blue); font-size: 16px; }
.prep-note { margin: 0; padding: 10px 16px; font-size: 15px; color: var(--ink-2);
  border-bottom: 1px solid var(--rule); }
.paper-body { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
.schedule-wrap { border-right: 2px solid var(--rule-strong); }
table { border-collapse: collapse; width: 100%; }
.schedule th, .schedule td { border-bottom: 1px solid var(--rule); padding: 9px 12px; text-align: left;
  vertical-align: middle; }
.schedule thead th { font-size: 12px; font-weight: 600; color: var(--ink-2); letter-spacing: .04em;
  border-bottom: 1px solid var(--rule-strong); }
.schedule td + td, .schedule th + th { border-left: 1px solid var(--rule); }
.num { text-align: right !important; }
.tk { width: 40px; text-align: center !important; }
.wp { width: 74px; }
.schedule tbody tr.row { cursor: pointer; transition: background-color .15s ease-out; }
.schedule tbody tr.row:hover { background: var(--highlight-soft); }
.schedule tbody tr.row.active { background: var(--highlight-soft); }
.schedule tbody tr.row.active .item { text-decoration: underline 2px var(--highlight); text-underline-offset: .25em; }
.item { font-weight: 600; }
.fig { font-family: var(--typed); font-size: 16px; white-space: nowrap; }
.fig.blank { color: var(--graphite); font-family: var(--sans); font-size: 14px; font-style: italic; }
.wpref { font-family: var(--typed); font-weight: 700; color: var(--pencil-blue); font-size: 15px; }
.wpref.none { color: var(--graphite); font-weight: 400; }
.tick { width: 20px; height: 20px; display: inline-block; vertical-align: -4px; flex: none; }
.g-proven { color: var(--pencil-green); }
.g-unproven { color: var(--graphite); }
.g-failed { color: var(--pencil-red); }
.g-missed { color: var(--graphite); }
tr.note td { padding: 4px 12px 14px 15px; background: var(--highlight-soft); }
.note-body { font-size: 15px; line-height: 1.5; }
.note-label { font-weight: 700; }
.note-label.g-proven { color: var(--pencil-green); }
.note-label.g-failed { color: var(--pencil-red); }
.note-label.g-unproven, .note-label.g-missed { color: var(--ink-2); }
.note-cite { display: block; margin-top: 6px; font-family: var(--typed); font-size: 14px;
  overflow-wrap: anywhere; }
.note-cite mark { background: var(--highlight); padding: 0 3px; }

.source { margin: 0; display: flex; flex-direction: column; min-width: 0; }
.source figcaption { display: flex; gap: 10px 16px; flex-wrap: wrap; align-items: baseline;
  padding: 10px 14px; border-bottom: 1px solid var(--rule-strong); font-size: 12px; font-weight: 600;
  color: var(--ink-2); letter-spacing: .04em; }
.doc-note { font-weight: 400; letter-spacing: 0; font-size: 14px; font-style: italic; }
.source-text { overflow: auto; max-height: 560px; padding: 4px 0 12px; }
.source-text pre { margin: 0; padding: 0 14px; font-family: var(--typed); font-size: 14px; line-height: 24px; }
.ln { display: grid; grid-template-columns: 34px minmax(52ch, max-content) 24px; gap: 8px;
  font-family: var(--typed); font-size: 14px; line-height: 23px; padding-right: 12px;
  border-bottom: 1px solid var(--rule); }
.lines { width: max-content; min-width: 100%; }
.ln .no { text-align: right; color: var(--graphite); user-select: none; padding-left: 4px; }
.ln .tx { white-space: pre; }
.ln .mk { display: flex; align-items: center; }
.ln mark { background: transparent; color: inherit; border-radius: 2px;
  box-shadow: inset 0 -.45em 0 var(--highlight-soft); }
.ln mark.active { background: var(--highlight); box-shadow: none; }
.ln.on { background: rgba(251, 233, 138, .18); }

.legend { list-style: none; margin: 0; padding: 12px 16px; display: flex; flex-wrap: wrap;
  gap: 8px 26px; border-top: 2px solid var(--rule-strong); font-size: 14px; color: var(--ink-2); }
.legend li { display: flex; gap: 8px; align-items: flex-start; }
.legend strong { color: var(--ink); }

/* Sheets: the rest of the file */
.sheet { background: var(--paper); box-shadow: var(--shadow); border: 1px solid var(--rule-strong);
  border-radius: 3px; padding: 32px 36px; margin-top: 40px; }
.intro { margin: 0 0 22px; max-width: 70ch; color: var(--ink-2); }
.pencil { font-style: italic; color: var(--pencil-blue); max-width: 70ch; margin: 18px 0 0; }
.table-wrap { overflow-x: auto; }
.summary th, .summary td { padding: 12px 12px; border-bottom: 1px solid var(--rule); text-align: left;
  vertical-align: middle; }
.summary thead th { border-bottom: 2px solid var(--rule-strong); font-size: 13px; font-weight: 600;
  color: var(--ink-2); vertical-align: bottom; }
.summary thead th .tick { display: block; margin-left: auto; margin-bottom: 4px; }
.th-label { display: block; white-space: nowrap; }
.summary tbody th { font-weight: 700; min-width: 210px; }
.summary .blurb { display: block; font-weight: 400; font-size: 14px; color: var(--ink-2); line-height: 1.4; }
.summary tr.test th::after { content: "Test preparer, built to fail"; display: block; font-weight: 400;
  font-size: 13px; font-style: italic; color: var(--pencil-blue); margin-top: 2px; }
.summary td.num { font-family: var(--typed); font-size: 16px; }
.summary td.c-proven { color: var(--pencil-green); font-weight: 700; }
.summary td.c-failed { color: var(--pencil-red); }
.bar-cell { width: 26%; min-width: 160px; }
.bar { display: flex; height: 12px; background: var(--paper-head); border: 1px solid var(--rule); }
.seg.g-proven { background: var(--pencil-green); }
.seg.g-unproven { background: repeating-linear-gradient(135deg, var(--graphite) 0 2px, transparent 2px 5px); }
.seg.g-failed { background: var(--pencil-red); }
.seg.g-missed { background: var(--rule); }
.usable { white-space: nowrap; }
.usable strong { font-size: 22px; }
.of { color: var(--ink-2); font-family: var(--sans); font-size: 14px; }

.footing-sheet { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 12px 56px;
  align-items: start; }
.footing-copy p { margin: 0; max-width: 58ch; color: var(--ink-2); }
.footing th { text-align: left; font-weight: 500; padding: 7px 0; }
.footing td { text-align: right; font-family: var(--typed); font-size: 18px; padding: 7px 0;
  white-space: nowrap; }
.footing td.rule { border-bottom: 1px solid var(--ink); }
.footing td.double { border-bottom: 4px double var(--ink); }
.footing tr.diff th, .footing tr.diff td { color: var(--pencil-red); font-weight: 700; padding-top: 14px; }
.footing tr.diff .tick { margin-right: 8px; }
.footing-note { grid-column: 2; margin-top: 4px; }

.signoffs { display: grid; grid-template-columns: 1fr 1fr; gap: 0; border: 1px solid var(--rule-strong); }
.signoff-block { padding: 22px 24px 26px; position: relative; min-height: 190px; }
.signoff-block + .signoff-block { border-left: 1px solid var(--rule-strong); }
.signoff-block dl { display: flex; gap: 36px; margin: 10px 0 0; }
.signoff-block dt { font-size: 14px; color: var(--ink-2); }
.signoff-block dd { margin: 0; font-family: var(--typed); font-size: 30px; font-weight: 700; }
.stamp { position: absolute; right: 24px; bottom: 22px; margin: 0; padding: 6px 14px;
  font-weight: 800; font-size: 18px; letter-spacing: .08em; text-transform: uppercase;
  border: 3px solid currentColor; border-radius: 4px; transform: rotate(-6deg); opacity: .9;
  mix-blend-mode: multiply; }
.stamp.approved { color: var(--pencil-green); }
.stamp.refused { color: var(--pencil-red); }
.stamp.pre { opacity: 0; transform: rotate(-6deg) scale(1.6); }
.stamp.pressed { transition: opacity .32s cubic-bezier(.16, 1, .3, 1), transform .32s cubic-bezier(.16, 1, .3, 1); }

.checklist th, .checklist td { padding: 11px 0; border-bottom: 1px solid var(--rule); text-align: left; }
.checklist thead th { font-size: 13px; font-weight: 600; color: var(--ink-2); border-bottom: 2px solid var(--rule-strong); }
.checklist tbody th { font-weight: 500; padding-right: 24px; }
.checklist td { white-space: nowrap; font-family: var(--typed); }
.checklist td .tick { margin-right: 8px; }

.supporting summary { cursor: pointer; font-weight: 700; font-size: 18px; list-style: none; }
.supporting summary::-webkit-details-marker { display: none; }
.supporting summary::before { content: "+"; display: inline-block; width: 1.2em; font-family: var(--typed);
  color: var(--pencil-blue); }
.supporting[open] summary::before { content: "\\2212"; }
.supporting p { max-width: 75ch; color: var(--ink-2); }
.metrics { margin: 14px 0; font-size: 15px; }
.metrics th, .metrics td { padding: 7px 12px 7px 0; border-bottom: 1px solid var(--rule); text-align: left;
  white-space: nowrap; }
.metrics td { font-family: var(--typed); }

footer { margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--rule-strong);
  display: flex; gap: 10px 24px; flex-wrap: wrap; font-size: 15px; color: var(--ink-2); }

@media (max-width: 980px) {
  .paper-head { grid-template-columns: 1fr 1fr; }
  .paper-head .preparers { grid-column: 1 / -1; order: 3; border-right: 0; border-top: 1px solid var(--rule); }
  .paper-head .reviewer { display: none; }
  .paper-body { grid-template-columns: 1fr; }
  .schedule-wrap { border-right: 0; border-bottom: 2px solid var(--rule-strong); }
  .opening { grid-template-columns: 1fr; gap: 18px; }
  .footing-sheet { grid-template-columns: 1fr; }
  .footing-note { grid-column: 1; }
}
@media (max-width: 640px) {
  .wrap { padding: 0 12px 56px; }
  .opening { padding: 30px 4px 22px; }
  .paper-head { grid-template-columns: 1fr; }
  .paper-head > * { border-right: 0; border-bottom: 1px solid var(--rule); }
  .sheet { padding: 22px 16px; }
  .schedule { table-layout: fixed; }
  .schedule thead th:first-child { width: 36%; }
  .tk { width: 34px; }
  .wp { width: 50px; }
  .schedule th, .schedule td { padding: 9px 7px; }
  .fig { font-size: 14px; white-space: normal; overflow-wrap: anywhere; }
  .item { font-size: 15px; line-height: 1.3; }
  .ln { grid-template-columns: 22px minmax(0, max-content); gap: 6px; font-size: 11.5px;
    line-height: 21px; padding-right: 8px; }
  .ln .mk { display: none; }
  .source-text pre { font-size: 11.5px; }
  .source-text { max-height: 380px; }
  .signoffs { grid-template-columns: 1fr; }
  .signoff-block + .signoff-block { border-left: 0; border-top: 1px solid var(--rule-strong); }
  .summary thead { display: none; }
  .summary tbody tr { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px 14px;
    padding: 16px 0; border-bottom: 1px solid var(--rule); }
  .summary tbody th, .summary tbody td { border: 0; padding: 0; min-width: 0; }
  .summary tbody td.num:not(.usable) { display: none; }
  .summary .bar-cell { grid-column: 1 / -1; width: auto; min-width: 0; }
  .summary tbody th { grid-column: 1; grid-row: 1; }
  .summary td.usable { grid-column: 2; grid-row: 1; align-self: start; }
  .checklist td { white-space: normal; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""

_JS = """
(function () {
  var paper = document.getElementById('working-paper');
  var data = JSON.parse(document.getElementById('paper-data').textContent);
  var docs = {}, readers = {}, groups = {};
  data.docs.forEach(function (d) { docs[d.id] = d; });
  data.readers.forEach(function (r) { readers[r.key] = r; });
  data.groups.forEach(function (g) { groups[g.key] = g.label; });
  var state = { reader: paper.dataset.defaultReader, doc: data.docs[0].id, field: null };
  var docSel = document.getElementById('doc');
  var tbody = document.getElementById('schedule');
  var source = document.getElementById('source');
  var docNote = document.getElementById('doc-note');
  var prepNote = document.getElementById('prep-note');
  var buttons = paper.querySelectorAll('button.preparer');

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function tick(group) {
    return '<svg class="tick g-' + group + '" aria-hidden="true" viewBox="0 0 20 20">' +
      '<use href="#t-' + group + '"/></svg>';
  }
  function cells() { return readers[state.reader].cells[state.doc] || {}; }
  function lineOf(text, index) { return text.slice(0, index).split('\\n').length; }

  function firstException() {
    var c = cells(), pick = null;
    data.fields.some(function (f) {
      if (c[f.name] && c[f.name].group === 'failed') { pick = f.name; return true; }
      return false;
    });
    return pick;
  }

  function renderSchedule() {
    var c = cells(), text = docs[state.doc].text, html = '';
    data.fields.forEach(function (f) {
      var cell = c[f.name];
      if (!cell) return;
      var on = f.name === state.field;
      var fig = cell.reported ? '<span class="fig">' + esc(cell.reported) + '</span>' :
        '<span class="fig blank">blank</span>';
      var ref = cell.span ? '<span class="wpref">L' + lineOf(text, cell.span[0]) + '</span>' :
        '<span class="wpref none">none</span>';
      html += '<tr class="row' + (on ? ' active' : '') + '" tabindex="0" data-field="' + f.name +
        '" aria-expanded="' + on + '"><th scope="row" class="item">' + esc(f.label) + '</th>' +
        '<td class="num">' + fig + '</td><td class="tk">' + tick(cell.group) +
        '<span class="sr">' + esc(cell.label) + '</span></td><td class="wp">' + ref + '</td></tr>';
      if (on) {
        var should = (cell.group === 'failed' || cell.group === 'missed') && cell.expected ?
          ' Should be <strong>' + esc(cell.expected) + '</strong>.' : '';
        var cite = cell.cited ? '<span class="note-cite">Cited: <mark>' + esc(cell.cited) +
          '</mark></span>' : '';
        html += '<tr class="note"><td colspan="4"><div class="note-body">' +
          '<span class="note-label g-' + cell.group + '">' + esc(cell.label) + '.</span> ' +
          esc(cell.reason) + should + cite + '</div></td></tr>';
      }
    });
    tbody.innerHTML = html;
  }

  function renderSource() {
    var doc = docs[state.doc], text = doc.text, c = cells(), marks = [];
    data.fields.forEach(function (f) {
      var cell = c[f.name];
      if (cell && cell.span) marks.push({ s: cell.span[0], e: cell.span[1], f: f.name, g: cell.group });
    });
    marks.sort(function (a, b) { return a.s - b.s || b.e - a.e; });
    var lines = text.replace(/\\n$/, '').split('\\n'), offset = 0, html = '';
    lines.forEach(function (line, i) {
      var start = offset, end = offset + line.length, pos = start, tx = '', mk = '', on = false;
      marks.forEach(function (m) {
        if (m.s < start || m.s >= end || m.s < pos) return;
        var stop = Math.min(m.e, end), active = m.f === state.field;
        on = on || active;
        tx += esc(text.slice(pos, m.s)) + '<mark data-field="' + m.f + '"' +
          (active ? ' class="active"' : '') + '>' + esc(text.slice(m.s, stop)) + '</mark>';
        if (!mk || active) mk = tick(m.g);
        pos = stop;
      });
      tx += esc(text.slice(pos, end));
      html += '<div class="ln' + (on ? ' on' : '') + '"><span class="no">' + (i + 1) +
        '</span><span class="tx">' + (tx || ' ') + '</span><span class="mk">' + mk + '</span></div>';
      offset = end + 1;
    });
    source.innerHTML = '<div class="lines">' + html + '</div>';
    docNote.textContent = doc.note;
  }

  function render(scroll) {
    buttons.forEach(function (b) {
      b.setAttribute('aria-pressed', b.dataset.reader === state.reader ? 'true' : 'false');
    });
    prepNote.textContent = readers[state.reader].blurb;
    renderSchedule();
    renderSource();
    var active = source.querySelector('mark.active');
    if (active && scroll) {
      var behave = scroll === 'instant' ? 'auto' : 'smooth';
      var top = active.getBoundingClientRect().top - source.getBoundingClientRect().top +
        source.scrollTop - source.clientHeight / 3;
      var r = active.getBoundingClientRect(), box = source.getBoundingClientRect();
      var left = r.right - box.left + source.scrollLeft - source.clientWidth + 24;
      source.scrollTo({ top: Math.max(0, top), left: Math.max(0, left), behavior: behave });
    }
  }

  function choose(reader, doc) {
    state.reader = reader; state.doc = doc; state.field = firstException(); render(true);
  }
  buttons.forEach(function (b) {
    b.addEventListener('click', function () { choose(b.dataset.reader, state.doc); });
  });
  docSel.addEventListener('change', function () { choose(state.reader, docSel.value); });
  function pick(row) {
    if (!row) return;
    state.field = state.field === row.dataset.field ? null : row.dataset.field;
    render(true);
    var again = tbody.querySelector('tr.row[data-field="' + row.dataset.field + '"]');
    if (again) again.focus({ preventScroll: true });
  }
  tbody.addEventListener('click', function (e) { pick(e.target.closest('tr.row')); });
  tbody.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(e.target.closest('tr.row')); }
  });
  source.addEventListener('click', function (e) {
    var m = e.target.closest('mark');
    if (m) { state.field = m.dataset.field; render(false); }
  });
  state.field = firstException();
  render('instant');

  var stamps = document.querySelectorAll('.stamp');
  var still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!still && 'IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        en.target.classList.add('pressed');
        requestAnimationFrame(function () { en.target.classList.remove('pre'); });
        io.unobserve(en.target);
      });
    }, { threshold: 0.9 });
    stamps.forEach(function (s) {
      if (s.getBoundingClientRect().top > window.innerHeight) { s.classList.add('pre'); io.observe(s); }
    });
  }
})();
"""


PAPER_JS = _JS
"""Served from /assets/paper.js, never inlined: the host's Content-Security-Policy
allows scripts from this origin only, which keeps an injected inline script dead
even if document text ever escaped the renderer."""


def render_dashboard(dashboard: Dashboard) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grounded extraction QA | {_e(SITE_AUTHOR)}</title>
<meta name="description" content="Every figure an AI pulls from a financial statement, traced to the line that proves it. Anything else is flagged as an exception.">
<meta name="theme-color" content="#d9e1cc">
<link rel="preload" href="/fonts/libre-franklin-var.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="/fonts/courier-prime-400.woff2" as="font" type="font/woff2" crossorigin>
<style>{_CSS}</style>
</head>
<body>
{_SPRITE}
<div class="wrap">
{_masthead()}
<main>
{_opening()}
{_working_paper(dashboard)}
{_summary(dashboard)}
{_footing(dashboard)}
{_signoff(dashboard)}
{_checklist(dashboard.sql)}
{_supporting(dashboard)}
</main>
<footer>
  <span>Prepared by {_e(SITE_AUTHOR)}</span>
  <a href="{_e(PORTFOLIO_URL)}">nedum.xyz</a>
  <a href="{_e(REPO_URL)}">Source on GitHub</a>
  <span>All companies and figures are fictional.</span>
</footer>
</div>
<script src="/assets/paper.js" defer></script>
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
