"""Dashboard tests.

The page is the artefact a reviewer opens, so the things that break it are worth
asserting: a metric that does not appear, a control that is missing from the
argument, and unescaped document text reaching the markup.
"""

from __future__ import annotations

import pytest

from geq.models import FieldKind, FieldSpec
from geq.web import (
    CONTROLS,
    REVISIONS,
    Dashboard,
    _citation_rows,
    build_dashboard,
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

    def test_provider_configuration_is_reported_without_the_key(self, dashboard):
        assert set(dashboard.llm) == {"configured", "model", "base_url", "api_key_env"}
        assert dashboard.llm["model"]


class TestDashboardRendering:
    def test_it_leads_with_the_argument(self, html):
        assert "answer every field correctly" in html
        assert "and still be unusable" in html

    def test_the_headline_metric_is_present(self, html):
        assert "grounded recall" in html
        assert "100.0%" in html

    def test_both_controls_appear_in_the_table(self, html):
        for control in CONTROLS:
            assert control in html

    def test_the_uncited_control_shows_perfect_recall_and_no_evidence(self, html, dashboard):
        card = dashboard.runs["adversarial:uncited"].scorecard
        assert card.micro_recall == pytest.approx(1.0)
        assert card.grounded_recall == pytest.approx(0.0)
        assert "Every value right, no citations at all." in html

    def test_the_fabricated_citation_is_shown_being_caught(self, html):
        assert "contradicted" in html
        assert "Total assets" in html, "the citation text itself should be visible"

    def test_the_gate_is_shown_as_passing(self, html):
        assert "Release gate" in html
        assert "PASS" in html

    def test_the_sql_checks_are_shown(self, html):
        assert "orphaned_extracted_fields" in html
        assert "brand_asset_mismatch" in html

    def test_the_page_says_the_documents_are_synthetic(self, html):
        assert "synthetic" in html
        assert "fictional" in html

    def test_it_is_a_complete_standalone_document(self, html):
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")
        assert "<style>" in html
        assert "http://" not in html.replace("http://127.0.0.1", ""), "no external assets"

    def test_no_template_braces_leaked_into_the_output(self, html):
        assert "{{" not in html
        assert "}}" not in html


class TestEscaping:
    def test_document_text_cannot_inject_markup(self):
        """Ad-hoc documents are reviewer-supplied text that reaches this renderer."""
        spec = FieldSpec(name="total_assets", kind=FieldKind.MONEY, label="Total assets")
        case = _case_with_label("<script>alert(1)</script>")
        run = _run_with_claim(case, spec, "<script>alert(1)</script>")
        rows = _citation_rows(run, case, {spec.name: spec})
        assert rows[0]["value"] == "<script>alert(1)</script>"
        # The value is returned raw by design; escaping is the renderer's job, and
        # _e() is what the template uses. Prove that path directly.
        from geq.web import _e

        assert "&lt;script&gt;" in _e(rows[0]["value"])


def _case_with_label(text: str):
    from geq.models import GoldenCase

    return GoldenCase(doc_id="doc", text=text, labels={"total_assets": text})


def _run_with_claim(case, spec, claim_text: str):
    from geq.models import Claim, Extraction
    from geq.runner import RunResult
    from geq.scoring import evaluate_corpus

    claim = Claim(field="total_assets", raw_value=claim_text, span=(0, len(claim_text)))
    extraction = Extraction(doc_id=case.doc_id, extractor="test", claims={"total_assets": claim})
    card = evaluate_corpus([case], {case.doc_id: extraction}, {spec.name: spec}, extractor="test")
    return RunResult(scorecard=card, extractions={case.doc_id: extraction})
