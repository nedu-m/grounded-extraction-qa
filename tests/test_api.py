"""API tests.

The JD asks for testing APIs directly and for checking "behaviour on repeated
identical requests", so the last test in this file does exactly that. The gate is
also checked to return 409 rather than a 200 with a sad body, because a caller
that has to parse JSON to find out whether QA passed will eventually get it
wrong.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from geq.api import create_app


@pytest.fixture
def client(corpus, monkeypatch) -> TestClient:
    monkeypatch.setenv("GEQ_CORPUS", str(corpus.root))
    return TestClient(create_app())


class TestHealth:
    def test_reports_the_corpus_and_the_available_extractors(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["documents"] == 6
        assert body["fields"] == 8
        assert "rules:v3" in body["extractors"]

    def test_fields_endpoint_describes_every_field(self, client):
        body = client.get("/fields").json()
        names = [f["name"] for f in body["fields"]]
        assert "total_assets" in names
        assert len(names) == 8
        kinds = {f["name"]: f["kind"] for f in body["fields"]}
        assert kinds["fiscal_year_end"] == "date"
        assert kinds["entity_name"] == "text"


class TestExtract:
    def test_extracts_a_corpus_document_with_traceable_claims(self, client):
        response = client.post(
            "/extract", json={"doc_id": "case_01_northgate_2024", "extractor": "rules:v3"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["missing"] == []
        assert len(body["claims"]) == 8
        for claim in body["claims"]:
            assert claim["evidence_status"] == "supported"
            assert claim["span_text"] == claim["value"]

    def test_reports_uncited_claims_rather_than_hiding_them(self, client):
        body = client.post(
            "/extract", json={"doc_id": "case_01_northgate_2024", "extractor": "rules:v1"}
        ).json()
        statuses = {c["evidence_status"] for c in body["claims"]}
        assert "uncited" in statuses
        assert "fiscal_year_end" in body["missing"]
        assert body["notes"]

    def test_an_ad_hoc_document_can_be_submitted(self, client):
        text = "ACME HOLDINGS\nAmounts in US dollars\nTotal assets      1,000\n"
        body = client.post("/extract", json={"text": text}).json()
        fields = {c["field"]: c["value"] for c in body["claims"]}
        assert fields["total_assets"] == "1,000"
        assert fields["entity_name"] == "ACME HOLDINGS"

    def test_span_text_can_be_suppressed(self, client):
        body = client.post(
            "/extract", json={"doc_id": "case_01_northgate_2024", "include_span_text": False}
        ).json()
        assert all(c["span_text"] is None for c in body["claims"])
        assert all(c["span"] is not None for c in body["claims"])

    def test_an_unknown_document_is_a_404(self, client):
        response = client.post("/extract", json={"doc_id": "nope"})
        assert response.status_code == 404
        assert "unknown doc_id" in response.json()["detail"]

    def test_an_unknown_extractor_is_a_404(self, client):
        response = client.post(
            "/extract", json={"doc_id": "case_01_northgate_2024", "extractor": "magic"}
        )
        assert response.status_code == 404

    def test_neither_document_nor_text_is_a_422(self, client):
        response = client.post("/extract", json={})
        assert response.status_code == 422
        assert "doc_id" in response.json()["detail"]


class TestQARun:
    def test_a_perfect_revision_scores_perfectly(self, client):
        body = client.post("/qa/run", json={"extractor": "rules:v3"}).json()
        assert body["metrics"]["grounded_recall"] == pytest.approx(1.0)
        assert body["metrics"]["grounding_rate"] == pytest.approx(1.0)
        assert body["metrics"]["gold"] == 48
        assert body["failures"] == []

    def test_a_weak_revision_shows_its_work(self, client):
        body = client.post("/qa/run", json={"extractor": "rules:v1"}).json()
        assert body["metrics"]["grounded_recall"] < body["metrics"]["micro_recall"]
        assert body["failures"]
        assert body["per_field"]["total_assets"]["recall"] < 1.0

    def test_per_field_scores_are_returned_for_every_field(self, client):
        body = client.post("/qa/run", json={"extractor": "rules:v3"}).json()
        assert set(body["per_field"]) == {
            "entity_name",
            "fiscal_year_end",
            "total_assets",
            "total_liabilities",
            "total_equity",
            "total_liabilities_and_equity",
            "revenue",
            "net_income",
        }

    def test_arithmetic_failures_surface_through_the_api(self, client):
        body = client.post("/qa/run", json={"extractor": "rules:v3"}).json()
        failed = [
            row
            for case in body["reconciliation"]
            for row in case["results"]
            if row["status"] == "fail"
        ]
        assert len(failed) == 1
        assert failed[0]["name"] == "balance_sheet_identity"

    def test_an_unknown_extractor_is_a_404(self, client):
        assert client.post("/qa/run", json={"extractor": "magic"}).status_code == 404


class TestQACompare:
    def test_an_improvement_returns_200(self, client):
        response = client.post(
            "/qa/compare", json={"baseline": "rules:v2", "candidate": "rules:v3"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["passed"] is True
        assert body["counts"]["regressed"] == 0

    def test_a_regression_returns_409_with_the_reasons(self, client):
        response = client.post(
            "/qa/compare", json={"baseline": "rules:v3", "candidate": "rules:v2"}
        )
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["passed"] is False
        assert detail["counts"]["regressed"] == 28
        assert any("regressed" in reason for reason in detail["failures"])

    def test_a_tolerant_policy_can_let_a_known_regression_through(self, client):
        response = client.post(
            "/qa/compare",
            json={
                "baseline": "rules:v3",
                "candidate": "rules:v2",
                "max_regressions": 100,
            },
        )
        assert response.status_code == 409, "the metric drop alone must still block it"

    def test_an_unknown_extractor_is_a_404(self, client):
        assert (
            client.post(
                "/qa/compare", json={"baseline": "nope", "candidate": "rules:v3"}
            ).status_code
            == 404
        )


class TestRepeatedIdenticalRequests:
    def test_the_same_extract_request_returns_the_same_bytes_twice(self, client):
        payload = {"doc_id": "case_02_northgate_2024_thousands", "extractor": "rules:v3"}
        first = client.post("/extract", json=payload)
        second = client.post("/extract", json=payload)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()

    def test_the_same_compare_request_returns_the_same_verdict_twice(self, client):
        payload = {"baseline": "rules:v1", "candidate": "rules:v3"}
        first = client.post("/qa/compare", json=payload)
        second = client.post("/qa/compare", json=payload)
        assert first.status_code == second.status_code == 200
        assert first.json()["counts"] == second.json()["counts"]

    def test_a_repeated_failing_request_keeps_failing_the_same_way(self, client):
        payload = {"baseline": "rules:v3", "candidate": "rules:v1"}
        first = client.post("/qa/compare", json=payload)
        second = client.post("/qa/compare", json=payload)
        assert first.status_code == second.status_code == 409
        assert first.json() == second.json()


class TestFonts:
    def test_a_shipped_face_is_served_with_a_long_cache(self, client):
        response = client.get("/fonts/libre-franklin-var.woff2")
        assert response.status_code == 200
        assert response.headers["content-type"] == "font/woff2"
        assert "immutable" in response.headers["cache-control"]

    @pytest.mark.parametrize("name", ["../api.py", "..%2Fapi.py", "LICENSE.md", "nope.woff2"])
    def test_anything_else_is_not_found(self, client, name):
        assert client.get(f"/fonts/{name}").status_code == 404


class TestPaperScript:
    def test_the_working_paper_script_is_served_from_this_origin(self, client):
        response = client.get("/assets/paper.js")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/javascript")
        assert "paper-data" in response.text

    def test_the_page_points_at_it(self, client):
        assert '<script src="/assets/paper.js" defer></script>' in client.get("/").text
