"""Scoring tests.

The one that matters most is ``test_right_value_with_no_citation_scores_zero_grounded_recall``:
an extractor can hit 100% recall and be worth nothing. If that test ever passes
for the wrong reason, the harness has stopped doing its job.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from geq.models import (
    Claim,
    Extraction,
    FieldKind,
    FieldSpec,
    GoldenCase,
    Scorecard,
    Verdict,
)
from geq.scoring import corpus_grounding_summary, evaluate_case, evaluate_corpus, scaled_claim_value

DOC = "Total assets 13,504.5\nTotal equity 8,184.1\nRevenue 24,880.0\nAs at 31 December 2024\n"

SPECS = {
    "total_assets": FieldSpec(name="total_assets", kind=FieldKind.MONEY, label="Total assets"),
    "total_equity": FieldSpec(name="total_equity", kind=FieldKind.MONEY, label="Total equity"),
    "revenue": FieldSpec(name="revenue", kind=FieldKind.MONEY, label="Revenue"),
}

CASE = GoldenCase(
    doc_id="doc",
    text=DOC,
    labels={"total_assets": "13504500", "total_equity": "8184100", "revenue": "24880000"},
)


def locate(needle: str) -> tuple[int, int]:
    start = DOC.index(needle)
    return (start, start + len(needle))


def extraction(**claims: Claim | None) -> Extraction:
    return Extraction(
        doc_id="doc",
        extractor="test",
        claims={name: claim for name, claim in claims.items() if claim is not None},
    )


class TestCellVerdicts:
    def test_right_when_value_matches_once_scaled(self):
        result = extraction(
            total_assets=Claim("total_assets", "13,504.5", locate("13,504.5"), Decimal(1_000))
        )
        cells = {c.field: c for c in evaluate_case(CASE, result, SPECS)}
        assert cells["total_assets"].verdict is Verdict.RIGHT
        assert cells["total_assets"].grounded
        assert cells["total_assets"].right_and_grounded

    def test_wrong_when_the_magnitude_is_missed(self):
        result = extraction(total_assets=Claim("total_assets", "13,504.5", locate("13,504.5")))
        cell = next(c for c in evaluate_case(CASE, result, SPECS) if c.field == "total_assets")
        assert cell.verdict is Verdict.WRONG
        assert cell.grounded, "the citation is real even though the value is not"

    def test_missing_when_nothing_is_claimed(self):
        cells = {c.field: c for c in evaluate_case(CASE, extraction(), SPECS)}
        assert all(c.verdict is Verdict.MISSING for c in cells.values())
        assert cells["revenue"].gold == "24880000"

    def test_extra_when_the_field_is_not_in_the_golden_set(self):
        result = extraction(invented=Claim("invented", "42", (0, 2)))
        cells = {c.field: c for c in evaluate_case(CASE, result, SPECS)}
        assert cells["invented"].verdict is Verdict.EXTRA
        assert cells["invented"].gold is None

    def test_an_unparseable_claim_cannot_be_right(self):
        result = extraction(total_assets=Claim("total_assets", "n/a", locate("13,504.5")))
        cell = next(c for c in evaluate_case(CASE, result, SPECS) if c.field == "total_assets")
        assert cell.verdict is Verdict.WRONG

    def test_required_fields_absent_from_the_golden_set_are_not_scored(self):
        specs = dict(SPECS)
        specs["optional"] = FieldSpec(
            name="optional", kind=FieldKind.MONEY, label="Optional", required=False
        )
        case = GoldenCase(doc_id="doc", text=DOC, labels=CASE.labels)
        assert "optional" not in {c.field for c in evaluate_case(case, extraction(), specs)}

    def test_scaled_claim_value_is_what_a_reader_would_report(self):
        claim = Claim("total_assets", "13,504.5", locate("13,504.5"), Decimal(1_000))
        assert scaled_claim_value(claim, SPECS["total_assets"]) == "13504500"


class TestScorecardMetrics:
    def test_right_value_with_no_citation_scores_zero_grounded_recall(self):
        """Confident, correct, and completely unauditable."""
        result = extraction(
            total_assets=Claim("total_assets", "13,504.5", None, Decimal(1_000)),
            total_equity=Claim("total_equity", "8,184.1", None, Decimal(1_000)),
            revenue=Claim("revenue", "24,880.0", None, Decimal(1_000)),
        )
        card = evaluate_corpus([CASE], {"doc": result}, SPECS)
        assert card.micro_recall == pytest.approx(1.0)
        assert card.micro_precision == pytest.approx(1.0)
        assert card.grounding_rate == pytest.approx(0.0)
        assert card.citation_rate == pytest.approx(0.0)
        assert card.grounded_recall == pytest.approx(0.0)
        assert len(card.unsupported_claims) == 3

    def test_fully_grounded_extraction_scores_one_everywhere(self):
        result = extraction(
            total_assets=Claim("total_assets", "13,504.5", locate("13,504.5"), Decimal(1_000)),
            total_equity=Claim("total_equity", "8,184.1", locate("8,184.1"), Decimal(1_000)),
            revenue=Claim("revenue", "24,880.0", locate("24,880.0"), Decimal(1_000)),
        )
        card = evaluate_corpus([CASE], {"doc": result}, SPECS)
        assert card.grounded_recall == pytest.approx(1.0)
        assert card.grounding_rate == pytest.approx(1.0)
        assert card.failures() == ()

    def test_mixed_extraction_has_expected_counts(self):
        result = extraction(
            total_assets=Claim("total_assets", "13,504.5", locate("13,504.5"), Decimal(1_000)),
            total_equity=Claim("total_equity", "8,184.1", locate("8,184.1")),
            # revenue omitted entirely
        )
        card = evaluate_corpus([CASE], {"doc": result}, SPECS)
        assert (card.gold, card.right, card.wrong, card.missing, card.claims) == (3, 1, 1, 1, 2)
        assert card.micro_recall == pytest.approx(1 / 3)
        assert card.micro_precision == pytest.approx(0.5)
        assert card.grounding_rate == pytest.approx(1.0)
        assert card.grounded_recall == pytest.approx(1 / 3)

    def test_a_document_never_attempted_counts_as_misses_not_as_absence(self):
        card = evaluate_corpus([CASE], {}, SPECS)
        assert card.gold == 3
        assert card.right == 0
        assert card.missing == 3
        assert card.micro_recall == 0.0

    def test_per_field_scores_sum_to_the_totals(self):
        result = extraction(
            total_assets=Claim("total_assets", "13,504.5", locate("13,504.5"), Decimal(1_000))
        )
        card = evaluate_corpus([CASE], {"doc": result}, SPECS)
        assert sum(s.gold for s in card.per_field.values()) == card.gold
        assert sum(s.claims for s in card.per_field.values()) == card.claims
        assert sum(s.grounded for s in card.per_field.values()) == card.grounded

    def test_an_empty_scorecard_does_not_divide_by_zero(self):
        card = evaluate_corpus([], {}, SPECS)
        assert card.grounded_recall == 0.0
        assert card.micro_f1 == 0.0

    def test_grounding_summary_surfaces_the_recall_gap(self):
        result = extraction(
            total_assets=Claim("total_assets", "13,504.5", None, Decimal(1_000)),
            total_equity=Claim("total_equity", "8,184.1", None, Decimal(1_000)),
            revenue=Claim("revenue", "24,880.0", None, Decimal(1_000)),
        )
        summary = corpus_grounding_summary(evaluate_corpus([CASE], {"doc": result}, SPECS))
        assert summary["recall"] == pytest.approx(1.0)
        assert summary["grounded_recall"] == pytest.approx(0.0)
        assert summary["recall_gap"] == pytest.approx(1.0)
        assert summary["unsupported"] == 3


class TestScorecardSerialisation:
    def test_round_trips_through_json(self):
        result = extraction(
            total_assets=Claim("total_assets", "13,504.5", locate("13,504.5"), Decimal(1_000)),
            total_equity=Claim("total_equity", "8,184.1", locate("8,184.1")),
        )
        card = evaluate_corpus([CASE], {"doc": result}, SPECS)
        restored = Scorecard.from_dict(card.to_dict())
        assert restored.grounded_recall == pytest.approx(card.grounded_recall)
        assert restored.micro_recall == pytest.approx(card.micro_recall)
        assert restored.grounding_rate == pytest.approx(card.grounding_rate)
        assert restored.extractor == card.extractor
        assert restored.gold == card.gold

    def test_failures_respects_the_limit(self):
        card = evaluate_corpus([CASE], {}, SPECS)
        assert len(card.failures(limit=2)) == 2
