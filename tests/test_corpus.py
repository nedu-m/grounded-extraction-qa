"""Golden-set loading and validation.

A golden set that is silently broken is worse than no golden set, because it
produces confident numbers. These tests check that each way it can break is
caught at load time rather than at scorecard-reading time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geq.corpus import CorpusError, load_corpus, load_specs, validate_corpus
from geq.models import FieldKind

SPEC = {
    "fields": [{"name": "total_assets", "kind": "money", "label": "Total assets", "required": True}]
}


def write_case(
    directory: Path, doc_id: str, labels: dict[str, str], text: str = "Total assets 1"
) -> None:
    cases = directory / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    (cases / f"{doc_id}.json").write_text(
        json.dumps({"doc_id": doc_id, "text": text, "labels": labels}), encoding="utf-8"
    )


def write_spec(directory: Path, spec: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "spec.json").write_text(json.dumps(spec or SPEC), encoding="utf-8")


class TestLoading:
    def test_the_shipped_corpus_loads(self, corpus):
        assert corpus.name == "golden"
        assert len(corpus.cases) == 6
        assert len(corpus.specs) == 8
        assert corpus.required_fields == tuple(corpus.specs)

    def test_specs_carry_kind_label_and_required(self, corpus):
        assets = corpus.specs["total_assets"]
        assert assets.kind is FieldKind.MONEY
        assert assets.label == "Total assets"
        assert assets.required
        assert corpus.specs["fiscal_year_end"].kind is FieldKind.DATE
        assert corpus.specs["entity_name"].kind is FieldKind.TEXT

    def test_cases_are_ordered_deterministically(self, corpus):
        ids = [case.doc_id for case in corpus.cases]
        assert ids == sorted(ids)

    def test_lookup_by_id(self, corpus):
        case = corpus.case("case_01_northgate_2024")
        assert case.label("total_assets") == "13504500"
        with pytest.raises(KeyError):
            corpus.case("nope")

    def test_aliases_default_to_empty(self, corpus):
        assert corpus.specs["revenue"].all_labels() == ("Revenue",)


class TestValidation:
    def test_a_case_referencing_an_unknown_field_fails_loudly(self, tmp_path):
        write_spec(tmp_path)
        write_case(tmp_path, "a", {"total_assets": "1", "typoo_field": "2"})
        with pytest.raises(CorpusError, match=r"absent from spec\.json"):
            load_corpus(tmp_path)

    def test_duplicate_doc_ids_are_rejected(self, tmp_path):
        write_spec(tmp_path)
        write_case(tmp_path, "a", {"total_assets": "1"})
        # Same doc_id in a second file.
        (tmp_path / "cases" / "second.json").write_text(
            json.dumps({"doc_id": "a", "text": "Total assets 1", "labels": {"total_assets": "1"}}),
            encoding="utf-8",
        )
        with pytest.raises(CorpusError, match="duplicate doc_id"):
            load_corpus(tmp_path)

    def test_a_required_field_no_case_labels_is_rejected(self, tmp_path):
        write_spec(
            tmp_path,
            {
                "fields": [
                    {"name": "total_assets", "kind": "money", "label": "Total assets"},
                    {"name": "never_labelled", "kind": "money", "label": "Never"},
                ]
            },
        )
        write_case(tmp_path, "a", {"total_assets": "1"})
        with pytest.raises(CorpusError, match=r"recall can never reach 1\.0"):
            load_corpus(tmp_path)

    def test_an_empty_label_is_rejected(self, tmp_path):
        write_spec(tmp_path)
        write_case(tmp_path, "a", {"total_assets": "   "})
        with pytest.raises(CorpusError, match="empty label"):
            load_corpus(tmp_path)

    def test_a_corpus_with_no_cases_is_rejected(self, tmp_path):
        write_spec(tmp_path)
        (tmp_path / "cases").mkdir()
        with pytest.raises(CorpusError, match="contains no cases"):
            load_corpus(tmp_path)

    def test_a_spec_with_no_fields_is_rejected(self, tmp_path):
        write_spec(tmp_path, {"fields": []})
        with pytest.raises(CorpusError, match="defines no fields"):
            load_specs(tmp_path / "spec.json")

    def test_a_duplicate_field_spec_is_rejected(self, tmp_path):
        write_spec(
            tmp_path,
            {
                "fields": [
                    {"name": "total_assets", "kind": "money", "label": "Total assets"},
                    {"name": "total_assets", "kind": "money", "label": "Assets again"},
                ]
            },
        )
        with pytest.raises(CorpusError, match="duplicate field spec"):
            load_specs(tmp_path / "spec.json")

    def test_a_missing_cases_directory_is_reported_clearly(self, tmp_path):
        write_spec(tmp_path)
        with pytest.raises(CorpusError, match="no cases directory"):
            load_corpus(tmp_path)

    def test_an_optional_field_may_go_unlabelled(self, tmp_path):
        write_spec(
            tmp_path,
            {
                "fields": [
                    {"name": "total_assets", "kind": "money", "label": "Total assets"},
                    {"name": "nice_to_have", "kind": "money", "label": "Nice", "required": False},
                ]
            },
        )
        write_case(tmp_path, "a", {"total_assets": "1"})
        corpus = load_corpus(tmp_path)
        assert corpus.specs["nice_to_have"].required is False

    def test_validate_accepts_a_healthy_corpus(self, corpus):
        validate_corpus(corpus)
