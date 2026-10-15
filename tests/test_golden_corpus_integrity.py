"""Guards on the golden set itself.

The corpus is the measuring instrument. If it drifts, every number the harness
produces is wrong in a way that looks like success, so it gets the same scrutiny
as the code:

* the committed JSON must match what the generator produces (no hand edits),
* every label must be traceable to the document it came from,
* exactly one document must be unbalanced, and it must be the one labelled as
  unbalanced, or the arithmetic checks have stopped discriminating.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal

import pytest
from tools.make_golden_corpus import (
    SPEC,
    build_case_payload,
    build_cases,
    render_amount,
    totals,
)

from geq.models import Claim, FieldKind
from geq.normalize import apply_scale, canonical, money_tokens_in
from geq.verify import reconcile

SCALE_RE = re.compile(r"\bAmounts in (thousands|millions) of\b", re.IGNORECASE)
SCALE_WORDS = {"thousands": Decimal(1_000), "millions": Decimal(1_000_000)}

UNBALANCED_DOC_ID = "case_04_northgate_2024_unbalanced"


def document_scale(text: str) -> Decimal:
    match = SCALE_RE.search(text)
    return SCALE_WORDS[match.group(1).lower()] if match else Decimal(1)


def claims_from_labels(case, specs) -> dict[str, Claim]:
    """Rebuild claims from the labels, at document magnitude, to check the arithmetic."""
    scale = document_scale(case.text)
    claims: dict[str, Claim] = {}
    for name, value in case.labels.items():
        spec = specs.get(name)
        if spec is None or spec.kind is not FieldKind.MONEY:
            continue
        rendered = canonical(value, spec.kind)
        assert rendered is not None
        # Undo the document scale so the literal is what a reader would see.
        literal_scale = Decimal(1) / scale if scale != 1 else Decimal(1)
        claims[name] = Claim(
            field=name,
            raw_value=apply_scale(rendered, literal_scale, spec.kind) or rendered,
            span=(0, 0),
            scale=scale,
        )
    return claims


class TestGeneratorAgreesWithCommittedFiles:
    def test_spec_matches(self, repo_root):
        committed = json.loads((repo_root / "data" / "golden" / "spec.json").read_text())
        assert committed == SPEC

    def test_every_case_matches_byte_for_byte(self, repo_root):
        for config in build_cases():
            path = repo_root / "data" / "golden" / "cases" / f"{config.doc_id}.json"
            assert path.exists(), f"missing case file {path.name}"
            committed = json.loads(path.read_text())
            generated = build_case_payload(config)
            assert committed["text"] == generated["text"], f"{config.doc_id}: text drifted"
            assert committed["labels"] == generated["labels"], f"{config.doc_id}: labels drifted"

    def test_no_orphan_case_files(self, repo_root):
        expected = {f"{config.doc_id}.json" for config in build_cases()}
        actual = {p.name for p in (repo_root / "data" / "golden" / "cases").glob("*.json")}
        assert actual == expected

    def test_every_case_is_unique_and_described(self, corpus):
        ids = [case.doc_id for case in corpus.cases]
        assert len(ids) == len(set(ids))
        for case in corpus.cases:
            assert case.notes, f"{case.doc_id} has no explanatory note"


class TestDocumentsAreSelfConsistent:
    def test_every_label_appears_in_its_own_document(self, corpus):
        """A label that cannot be found in the text makes recall unachievable."""
        for case in corpus.cases:
            scale = document_scale(case.text)
            tokens = money_tokens_in(case.text)
            for name, value in case.labels.items():
                spec = corpus.specs[name]
                if spec.kind is not FieldKind.MONEY:
                    continue
                canonical_value = canonical(value, spec.kind)
                assert canonical_value is not None
                scaled_back = apply_scale(canonical_value, Decimal(1) / scale, spec.kind)
                assert scaled_back in tokens, (
                    f"{case.doc_id}/{name}: {value} at scale 1:{scale} "
                    f"({scaled_back}) does not appear in the document"
                )

    def test_only_the_named_case_is_unbalanced(self, corpus):
        statuses = {}
        for case in corpus.cases:
            results = reconcile(claims_from_labels(case, corpus.specs), corpus.specs)
            identity = next(r for r in results if r.name == "balance_sheet_identity")
            statuses[case.doc_id] = identity.status

        unbalanced = sorted(doc for doc, status in statuses.items() if status == "fail")
        assert unbalanced == [UNBALANCED_DOC_ID], (
            "the arithmetic checks must catch exactly the planted defect; "
            f"they flagged {unbalanced}"
        )

    def test_the_unbalanced_case_still_has_agreeing_printed_totals(self, corpus):
        """It must look complete, or it does not test what it claims to test."""
        case = corpus.case(UNBALANCED_DOC_ID)
        results = {
            r.name: r for r in reconcile(claims_from_labels(case, corpus.specs), corpus.specs)
        }
        assert results["balance_sheet_identity"].status == "fail"
        assert results["totals_footing"].status == "pass"

    def test_every_balanced_case_foots_exactly(self, corpus):
        for case in corpus.cases:
            if case.doc_id == UNBALANCED_DOC_ID:
                continue
            results = reconcile(claims_from_labels(case, corpus.specs), corpus.specs)
            assert all(r.status == "pass" for r in results), case.doc_id

    def test_labels_cover_every_required_field_in_every_case(self, corpus):
        for case in corpus.cases:
            for name, spec in corpus.specs.items():
                if spec.required:
                    assert case.label(name) is not None, f"{case.doc_id} is missing {name}"


class TestGeneratorInternals:
    def test_render_amount_refuses_values_it_cannot_represent(self):
        with pytest.raises(ValueError, match="not representable"):
            render_amount(1_234_567, 1_000_000)

    def test_render_amount_at_scale_is_exact_when_it_returns(self):
        assert render_amount(13_504_500, 1_000) == "13,504.5"
        assert render_amount(13_000_000, 1_000_000) == "13.0"
        assert render_amount(1_204_500, 1) == "1,204,500"

    def test_totals_derive_both_sides_of_the_identity(self):
        figures = build_cases()[0].amounts
        derived = totals(figures)
        assert derived["total_assets"] == derived["sum_of_liabilities_and_equity"]

    def test_the_equity_override_breaks_the_identity_by_a_known_amount(self):
        figures = build_cases()[0].amounts
        clean = totals(figures)
        broken = totals(figures, equity_override=figures["retained_earnings"] - 100_000)
        assert clean["total_assets"] - broken["sum_of_liabilities_and_equity"] == 100_000
