"""Dashboard tests.

The page is the artefact a reviewer opens, so the things that break it are worth
asserting: a metric that does not appear, a control that is missing from the
argument, and unescaped document text reaching the markup.
"""

from __future__ import annotations

import json
import re

import pytest

from geq.models import CellResult, EvidenceCheck, EvidenceStatus, Verdict
from geq.web import (
    CONTROLS,
    REVISIONS,
    Dashboard,
    _json_for_script,
    build_dashboard,
    classify,
    explorer_payload,
    outcome_counts,
    render_dashboard,
)


@pytest.fixture(scope="module")
def dashboard(corpus) -> Dashboard:
    """Built from the shipped paths, the same way the service builds it."""
    return build_dashboard()


@pytest.fixture(scope="module")
def html(dashboard: Dashboard) -> str:
    return render_dashboard(dashboard)


class TestDashboardData:
    def test_every_revision_and_control_is_scored(self, dashboard):
        assert set(dashboard.runs) == {*REVISIONS, *CONTROLS}

    def test_the_page_is_not_missing_an_extractor(self, dashboard):
        for name, run in dashboard.runs.items():
            assert run.scorecard.extractor == name

    def test_the_gate_on_the_page_is_the_gate_ci_runs(self, dashboard):
        assert dashboard.gate.passed
        assert dashboard.gate.baseline_extractor == "rules:v2"
        assert dashboard.gate.candidate_extractor == "rules:v3"

    def test_the_sql_report_carries_every_check(self, dashboard):
        assert len(dashboard.sql.findings) == 6
        assert all(f.description for f in dashboard.sql.findings)

    def test_the_reverse_gate_blocks_the_same_change_rolled_back(self, dashboard):
        assert not dashboard.reverse_gate.passed
        assert len(dashboard.reverse_gate.regressions) == len(dashboard.gate.fixes)

    def test_provider_configuration_is_reported_without_the_key(self, dashboard):
        assert set(dashboard.llm) == {"configured", "model", "base_url", "api_key_env"}
        assert dashboard.llm["model"]


class TestOutcomes:
    """The plain-language buckets are what a stakeholder reads, so they must be exact."""

    @pytest.mark.parametrize(
        ("verdict", "status", "key", "group"),
        [
            (Verdict.RIGHT, EvidenceStatus.SUPPORTED, "proven", "proven"),
            (Verdict.RIGHT, EvidenceStatus.UNCITED, "unproven", "unproven"),
            (Verdict.RIGHT, EvidenceStatus.CONTRADICTED, "bad_proof", "failed"),
            (Verdict.WRONG, EvidenceStatus.SUPPORTED, "misread", "failed"),
            (Verdict.WRONG, EvidenceStatus.UNCITED, "wrong", "failed"),
            (Verdict.WRONG, EvidenceStatus.CONTRADICTED, "fabricated", "failed"),
            (Verdict.MISSING, None, "missed", "missed"),
        ],
    )
    def test_every_combination_lands_in_one_bucket(self, verdict, status, key, group):
        check = EvidenceCheck(field="f", status=status, detail="") if status else None
        outcome = classify(CellResult(doc_id="d", field="f", verdict=verdict, check=check))
        assert (outcome.key, outcome.group) == (key, group)

    def test_only_right_and_grounded_counts_as_proven(self, dashboard):
        for run in dashboard.runs.values():
            assert outcome_counts(run)["proven"] == run.scorecard.right_and_grounded

    def test_counts_cover_every_golden_cell(self, dashboard):
        for run in dashboard.runs.values():
            assert sum(outcome_counts(run).values()) == run.scorecard.gold

    def test_the_controls_score_zero_proven(self, dashboard):
        for name in CONTROLS:
            assert outcome_counts(dashboard.runs[name])["proven"] == 0


class TestExplorerPayload:
    def test_every_reader_document_and_field_is_present(self, dashboard):
        payload = explorer_payload(dashboard)
        assert {r["key"] for r in payload["readers"]} == set(dashboard.runs)
        assert len(payload["docs"]) == len(dashboard.corpus.cases)
        for reader in payload["readers"]:
            for doc in payload["docs"]:
                assert set(reader["cells"][doc["id"]]) == set(dashboard.corpus.specs)

    def test_proven_spans_point_at_the_reported_figure(self, dashboard):
        payload = explorer_payload(dashboard)
        texts = {d["id"]: d["text"] for d in payload["docs"]}
        best = next(r for r in payload["readers"] if r["key"] == "rules:v3")
        for doc_id, cells in best["cells"].items():
            for cell in cells.values():
                start, end = cell["span"]
                assert cell["reported"] in texts[doc_id][start:end]

    def test_a_fabricated_figure_shows_what_its_source_really_says(self, dashboard):
        payload = explorer_payload(dashboard)
        fab = next(r for r in payload["readers"] if r["key"] == "adversarial:fabricating")
        cell = fab["cells"]["case_01_northgate_2024"]["total_assets"]
        assert cell["outcome"] == "fabricated"
        assert cell["cited"] == "Total assets"
        assert cell["expected"] == "$13,504,500"

    def test_the_embedded_json_cannot_close_its_script_tag(self):
        hostile = {"text": "</script><script>alert(1)</script>"}
        encoded = _json_for_script(hostile)
        assert "</" not in encoded
        assert json.loads(encoded) == hostile


class TestDashboardRendering:
    def test_it_leads_with_what_the_tool_does(self, html):
        assert "Every figure the AI reports, ticked back to the line that proves it." in html

    def test_it_has_no_eyebrow_label_above_the_heading(self, html):
        assert 'class="kicker"' not in html

    def test_the_working_paper_opens_on_a_preparer_with_exceptions(self, html):
        assert 'data-default-reader="rules:v2"' in html
        assert "Working paper" in html

    def test_every_preparer_is_named_in_plain_language(self, html):
        for name in (
            "Current version",
            "Previous version",
            "First draft",
            "Test: no sources",
            "Test: fabricator",
        ):
            assert name in html

    def test_the_tick_legend_explains_every_mark(self, html):
        for label in ("Agreed to source", "No source given", "Exception", "Not reported"):
            assert label in html
        for group in ("proven", "unproven", "failed", "missed"):
            assert f'id="t-{group}"' in html

    def test_the_summary_counts_only_agreed_figures_as_usable(self, html):
        assert "Review summary" in html
        assert "<strong>48</strong><span class='of'> of 48</span>" in html

    def test_the_unbalanced_statement_is_footed_with_its_difference(self, html):
        assert "does not add up" in html
        assert "13,404,500" in html
        assert "(100,000)" in html

    def test_both_sign_off_outcomes_are_shown(self, html):
        assert "Approved to ship" in html
        assert "Refused" in html

    def test_the_data_room_checks_are_described_in_plain_terms(self, html):
        assert "The same file uploaded twice for one practice" in html
        assert "A marketing asset carrying another brand&#x27;s name" in html

    def test_engineering_metrics_are_kept_but_collapsed(self, html):
        assert '<details class="sheet supporting">' in html
        assert "Grounded recall" in html
        for name in (*REVISIONS, *CONTROLS):
            assert name in html

    def test_the_page_says_the_figures_are_fictional(self, html):
        assert "synthetic" in html
        assert "fictional" in html

    def test_no_em_dashes_in_the_copy(self, html):
        assert "\u2014" not in html

    def test_it_loads_nothing_from_another_host(self, html):
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")
        assert not re.search(
            r"""(src|href)=["']?(https?:)?//""",
            html.replace('href="https://nedum.xyz"', "").replace(
                'href="https://github.com/nedu-m/grounded-extraction-qa"', ""
            ),
        )
        assert not re.search(r"url\((https?:)?//", html)

    def test_no_inline_script_can_run_under_the_hosts_policy(self, html):
        """The host sends script-src 'self': an inline script would be blocked."""
        for attrs, body in re.findall(r"<script\b([^>]*)>(.*?)</script>", html, flags=re.S):
            assert "src=" in attrs or 'type="application/json"' in attrs
            assert "src=" not in attrs or not body.strip()

    def test_every_font_it_asks_for_is_shipped(self, html):
        from geq.web import FONT_DIR, FONT_FILES

        for name in re.findall(r"/fonts/([\w.-]+\.woff2)", html):
            assert name in FONT_FILES
        for name in FONT_FILES:
            assert (FONT_DIR / name).is_file()

    def test_no_template_braces_leaked_into_the_output(self, html):
        css_and_js = re.sub(r"<(style|script)\b[^>]*>.*?</\1>", "", html, flags=re.S)
        assert "{{" not in css_and_js
        assert "}}" not in css_and_js
