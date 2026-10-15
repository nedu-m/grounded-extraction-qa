"""The real LLM path, tested without a network.

The extractor's orchestration is the part that can be wrong; the HTTP call is not
interesting. ``_complete`` is monkeypatched so the contract handling - a value not
present in its own quote, a quote not present in the document, a missing key - is
exercised for real.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from geq.extractors import LLM_EXTRACTOR_NAME, available_extractors, build_extractor
from geq.extractors.base import locate_quote
from geq.extractors.llm import (
    API_KEY_ENV_VARS,
    DEFAULT_MODEL,
    LLMConfig,
    LLMExtractor,
    quote_must_contain_value,
)
from geq.models import EvidenceStatus, FieldKind, FieldSpec, GoldenCase
from geq.verify import check_evidence

DOC = "NORTHGATE CLINICAL GROUP\nTotal assets      13,504,500\nRevenue      24,880,000\n"

CASE = GoldenCase(doc_id="doc", text=DOC, labels={})

SPECS = {
    "entity_name": FieldSpec(name="entity_name", kind=FieldKind.TEXT, label="ENTITY_NAME"),
    "total_assets": FieldSpec(name="total_assets", kind=FieldKind.MONEY, label="Total assets"),
    "revenue": FieldSpec(name="revenue", kind=FieldKind.MONEY, label="Revenue"),
}


class TestQuoteContract:
    def test_a_quote_containing_its_value_is_accepted(self):
        assert quote_must_contain_value("Total assets 13,504,500", "13,504,500")

    @pytest.mark.parametrize(
        ("quote", "value"),
        [("", "1"), ("Total assets", ""), ("Total assets 13,504,500", "24,880,000")],
    )
    def test_a_quote_that_does_not_contain_its_value_is_rejected(self, quote, value):
        assert not quote_must_contain_value(quote, value)


class TestQuoteLocation:
    def test_locates_a_verbatim_quote(self):
        found = locate_quote(DOC, "Total assets")
        assert found is not None
        span, occurrences = found
        assert DOC[span[0] : span[1]] == "Total assets"
        assert occurrences == 1

    def test_reports_a_repeated_quote_rather_than_hiding_the_ambiguity(self):
        text = "Total assets 1\nTotal assets 2\n"
        found = locate_quote(text, "Total assets")
        assert found is not None
        _, occurrences = found
        assert occurrences == 2

    def test_absent_text_is_not_located(self):
        assert locate_quote(DOC, "Total equity") is None

    def test_an_empty_quote_is_not_located(self):
        assert locate_quote(DOC, "") is None


class TestWithoutCredentials:
    def test_a_missing_key_degrades_to_an_explained_no_op(self, monkeypatch):
        for name in API_KEY_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        extraction = LLMExtractor(SPECS).extract(CASE)
        assert extraction.claims == {}
        assert extraction.notes
        assert "DEEPSEEK_API_KEY" in extraction.notes[0]
        assert "rules:v3" in extraction.notes[0], "the message should point at the offline path"


class TestOrchestration:
    @staticmethod
    def _with_answers(answers: dict[str, dict[str, Any]], specs=SPECS) -> LLMExtractor:
        extractor = LLMExtractor(specs, api_key="test-key")
        remaining = dict(answers)

        def fake_complete(document: str, spec: FieldSpec) -> dict[str, Any]:
            return remaining.pop(spec.name, {"value": None, "quote": None})

        extractor._complete = fake_complete  # type: ignore[method-assign]
        return extractor

    def test_a_compliant_answer_becomes_a_grounded_claim(self):
        extractor = self._with_answers(
            {"total_assets": {"value": "13,504,500", "quote": "Total assets      13,504,500"}}
        )
        claims = extractor.extract(CASE).claims
        assert "total_assets" in claims
        check = check_evidence(DOC, claims["total_assets"], SPECS["total_assets"])
        assert check.status is EvidenceStatus.SUPPORTED

    def test_a_value_absent_from_its_own_quote_is_downgraded_to_uncited(self):
        extractor = self._with_answers(
            {"total_assets": {"value": "999,999,999", "quote": "Total assets      13,504,500"}}
        )
        extraction = extractor.extract(CASE)
        assert extraction.claims["total_assets"].span is None
        assert any("not present in its own quote" in note for note in extraction.notes)

    def test_a_quote_absent_from_the_document_is_downgraded_to_uncited(self):
        extractor = self._with_answers(
            {"revenue": {"value": "24,880,000", "quote": "Revenue (restated) 24,880,000"}}
        )
        extraction = extractor.extract(CASE)
        assert extraction.claims["revenue"].span is None
        assert any("not found in the document" in note for note in extraction.notes)

    def test_a_null_answer_is_simply_absent(self):
        extractor = self._with_answers({})
        assert extractor.extract(CASE).claims == {}

    def test_a_repeated_quote_is_flagged_in_the_notes(self):
        text = "Total assets 13,504,500\nTotal assets 13,504,500\n"
        case = GoldenCase(doc_id="doc", text=text, labels={})
        extractor = self._with_answers(
            {"total_assets": {"value": "13,504,500", "quote": "Total assets 13,504,500"}}
        )
        extraction = extractor.extract(case)
        assert any("occurs 2 times" in note for note in extraction.notes)

    def test_the_entity_name_is_not_asked_of_the_model(self):
        extractor = self._with_answers({})
        extraction = extractor.extract(CASE)
        assert "entity_name" not in extraction.claims
        assert extraction.notes == () or all("entity_name" not in n for n in extraction.notes)


class TestRegistry:
    def test_the_llm_extractor_is_listed(self):
        assert LLM_EXTRACTOR_NAME in available_extractors()

    def test_building_it_does_not_require_credentials(self):
        assert isinstance(build_extractor("llm", SPECS), LLMExtractor)

    def test_the_default_model_is_configurable_through_the_environment(self, monkeypatch):
        monkeypatch.setenv("GEQ_LLM_MODEL", "some-model")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        assert LLMExtractor(SPECS).model == "some-model"
        assert DEFAULT_MODEL, "a default must exist for operators who do not set one"

    def test_an_unknown_extractor_name_lists_the_valid_options(self):
        with pytest.raises(ValueError, match="available"):
            build_extractor("nope")


class TestProviderConfig:
    """The provider is a chat-completions endpoint, and DeepSeek is only the default."""

    def test_deepseek_is_the_default(self):
        config = LLMConfig.from_env({})
        assert config.model == DEFAULT_MODEL
        assert config.base_url == "https://api.deepseek.com"
        assert config.completions_url == "https://api.deepseek.com/chat/completions"
        assert not config.configured

    def test_the_deepseek_key_is_picked_up_first(self):
        config = LLMConfig.from_env({"DEEPSEEK_API_KEY": "d", "OPENAI_API_KEY": "o"})
        assert config.api_key == "d"

    def test_a_generic_key_works_as_a_fallback(self):
        assert LLMConfig.from_env({"GEQ_LLM_API_KEY": "g"}).api_key == "g"
        assert LLMConfig.from_env({"OPENAI_API_KEY": "o"}).api_key == "o"

    def test_the_endpoint_and_model_can_be_redirected(self):
        config = LLMConfig.from_env(
            {
                "DEEPSEEK_API_KEY": "d",
                "GEQ_LLM_BASE_URL": "https://example.invalid/v1/",
                "GEQ_LLM_MODEL": "other",
            }
        )
        assert config.base_url == "https://example.invalid/v1"
        assert config.completions_url == "https://example.invalid/v1/chat/completions"
        assert config.model == "other"

    def test_the_public_view_never_contains_the_key(self):
        config = LLMConfig.from_env({"DEEPSEEK_API_KEY": "super-secret-value"})
        view = config.public_view()
        assert view["configured"] is True
        assert "super-secret-value" not in json.dumps(view)
        assert view["api_key_env"] == "DEEPSEEK_API_KEY"

    def test_an_unknown_provider_name_still_lists_the_valid_options(self):
        with pytest.raises(ValueError, match="adversarial:fabricating"):
            build_extractor("nonsense")

    def test_listing_models_without_a_key_refuses_clearly(self):
        from geq.extractors.llm import list_models

        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
            list_models(LLMConfig.from_env({}))

    def test_the_cli_reports_missing_credentials_rather_than_crashing(self, monkeypatch):
        from geq.cli import main

        for name in API_KEY_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        assert main(["models"]) == 2
