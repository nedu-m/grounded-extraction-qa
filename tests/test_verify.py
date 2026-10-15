"""Evidence and arithmetic checks.

The grounding tests are the centre of this file. Each citation failure mode gets
its own case, because they call for different fixes: an uncited value is a prompt
problem, a contradicted value is a fabrication, and a whole-document citation is
a schema problem.
"""

from __future__ import annotations

from decimal import Decimal
from typing import ClassVar

import pytest

from geq.models import Claim, EvidenceStatus, FieldKind, FieldSpec
from geq.verify import (
    MAX_EVIDENCE_CHARS,
    check_balance_sheet,
    check_evidence,
    check_totals_footing,
    failing,
    reconcile,
    span_text,
)

DOC = (
    "NORTHGATE CLINICAL GROUP\n"
    "Amounts in thousands of US dollars\n"
    "Total assets                                13,504.5\n"
    "Total liabilities and equity                13,504.5\n"
    "As at 31 December 2024\n"
)

MONEY = FieldSpec(name="total_assets", kind=FieldKind.MONEY, label="Total assets")


def span_of(haystack: str, needle: str) -> tuple[int, int]:
    start = haystack.index(needle)
    return (start, start + len(needle))


def claim_at(needle: str, raw_value: str, **kwargs) -> Claim:
    return Claim(
        field="total_assets",
        raw_value=raw_value,
        span=span_of(DOC, needle),
        **kwargs,
    )


class TestSpanText:
    def test_slices_the_document(self):
        assert span_text(DOC, (0, 8)) == "NORTHGAT"

    @pytest.mark.parametrize(
        "span",
        [None, (-1, 5), (10, 10), (10, 5), (0, len(DOC) + 1)],
    )
    def test_unusable_spans_return_none(self, span):
        assert span_text(DOC, span) is None


class TestEvidenceStatuses:
    def test_no_citation_is_uncited(self):
        check = check_evidence(DOC, Claim(field="total_assets", raw_value="13,504.5"), MONEY)
        assert check.status is EvidenceStatus.UNCITED
        assert not check.supported
        assert check.normalized_claim == "13504.5"

    def test_span_past_the_end_is_out_of_bounds(self):
        claim = Claim(field="total_assets", raw_value="13,504.5", span=(0, len(DOC) + 50))
        assert check_evidence(DOC, claim, MONEY).status is EvidenceStatus.OUT_OF_BOUNDS

    def test_reversed_span_is_out_of_bounds(self):
        claim = Claim(field="total_assets", raw_value="13,504.5", span=(40, 10))
        assert check_evidence(DOC, claim, MONEY).status is EvidenceStatus.OUT_OF_BOUNDS

    def test_a_whole_document_span_is_not_a_citation(self):
        wide = "x" * (MAX_EVIDENCE_CHARS + 1)
        claim = Claim(field="total_assets", raw_value="13,504.5", span=(0, len(wide)))
        check = check_evidence(wide, claim, MONEY)
        assert check.status is EvidenceStatus.NOT_A_CITATION
        assert "wider than a citation" in check.detail

    def test_citing_real_text_that_lacks_the_value_is_contradicted(self):
        claim = claim_at("Total liabilities and equity", "13,504.5")
        check = check_evidence(DOC, claim, MONEY)
        assert check.status is EvidenceStatus.CONTRADICTED
        assert "does not contain the claimed value" in check.detail

    def test_pointing_at_the_value_is_supported(self):
        claim = claim_at("13,504.5", "13,504.5")
        check = check_evidence(DOC, claim, MONEY)
        assert check.status is EvidenceStatus.SUPPORTED
        assert check.supported
        assert check.normalized_cited == "13504.5"

    def test_citing_the_whole_line_is_supported(self):
        """A conservative citation is still a valid one."""
        claim = claim_at("Total assets                                13,504.5", "13,504.5")
        assert check_evidence(DOC, claim, MONEY).status is EvidenceStatus.SUPPORTED

    def test_an_unparseable_value_is_reported_as_such(self):
        claim = claim_at("13,504.5", "not stated")
        check = check_evidence(DOC, claim, MONEY)
        assert check.status is EvidenceStatus.UNPARSEABLE

    def test_offset_drift_of_one_character_breaks_grounding(self):
        start, end = span_of(DOC, "13,504.5")
        drifted = Claim(field="total_assets", raw_value="13,504.5", span=(start, end - 2))
        assert check_evidence(DOC, drifted, MONEY).status is EvidenceStatus.CONTRADICTED

    def test_a_footnote_marker_in_the_citation_is_tolerated(self):
        text = "Total assets 13,504.5 (note 4)"
        claim = Claim(field="total_assets", raw_value="13,504.5", span=(0, len(text)))
        assert check_evidence(text, claim, MONEY).status is EvidenceStatus.SUPPORTED


class TestEvidenceByKind:
    def test_date_accepts_an_equivalent_format(self):
        spec = FieldSpec(name="fiscal_year_end", kind=FieldKind.DATE, label="DATE")
        claim = Claim(
            field="fiscal_year_end",
            raw_value="31 December 2024",
            span=span_of(DOC, "As at 31 December 2024"),
        )
        assert check_evidence(DOC, claim, spec).status is EvidenceStatus.SUPPORTED

    def test_date_contradicted_by_a_different_date(self):
        spec = FieldSpec(name="fiscal_year_end", kind=FieldKind.DATE, label="DATE")
        claim = Claim(
            field="fiscal_year_end", raw_value="2024-12-31", span=span_of(DOC, "NORTHGATE")
        )
        assert check_evidence(DOC, claim, spec).status is EvidenceStatus.CONTRADICTED

    def test_text_is_matched_case_insensitively(self):
        spec = FieldSpec(name="entity_name", kind=FieldKind.TEXT, label="ENTITY_NAME")
        claim = Claim(
            field="entity_name",
            raw_value="Northgate Clinical Group",
            span=span_of(DOC, "NORTHGATE CLINICAL GROUP"),
        )
        assert check_evidence(DOC, claim, spec).status is EvidenceStatus.SUPPORTED


class TestGroundingIgnoresScale:
    """The design decision that keeps the two failure modes apart."""

    def test_a_faithful_citation_of_the_wrong_magnitude_is_grounded(self):
        claim = claim_at("13,504.5", "13,504.5", scale=Decimal(1))
        check = check_evidence(DOC, claim, MONEY)
        assert check.status is EvidenceStatus.SUPPORTED
        assert claim.scale == Decimal(1)

    def test_scale_does_not_change_the_grounding_verdict(self):
        small = claim_at("13,504.5", "13,504.5", scale=Decimal(1))
        correct = claim_at("13,504.5", "13,504.5", scale=Decimal(1_000))
        assert (
            check_evidence(DOC, small, MONEY).status
            is check_evidence(DOC, correct, MONEY).status
            is EvidenceStatus.SUPPORTED
        )


class TestReconciliation:
    SPECS: ClassVar[dict[str, FieldSpec]] = {
        "total_assets": FieldSpec(name="total_assets", kind=FieldKind.MONEY, label="Total assets"),
        "total_liabilities": FieldSpec(
            name="total_liabilities", kind=FieldKind.MONEY, label="Total liabilities"
        ),
        "total_equity": FieldSpec(name="total_equity", kind=FieldKind.MONEY, label="Total equity"),
        "total_liabilities_and_equity": FieldSpec(
            name="total_liabilities_and_equity",
            kind=FieldKind.MONEY,
            label="Total liabilities and equity",
        ),
    }

    @staticmethod
    def claims(assets, liabilities, equity, combined, scale=Decimal(1)) -> dict[str, Claim]:
        def make(field: str, value: str) -> Claim:
            return Claim(field=field, raw_value=value, span=(0, 0), scale=scale)

        return {
            "total_assets": make("total_assets", assets),
            "total_liabilities": make("total_liabilities", liabilities),
            "total_equity": make("total_equity", equity),
            "total_liabilities_and_equity": make("total_liabilities_and_equity", combined),
        }

    def test_a_balanced_sheet_passes(self):
        result = check_balance_sheet(
            self.claims("13504500", "5320400", "8184100", "13504500"), self.SPECS
        )
        assert result.status == "pass"
        assert result.ok

    def test_an_unbalanced_sheet_fails_and_reports_the_gap(self):
        result = check_balance_sheet(
            self.claims("13504500", "5320400", "8084100", "13504500"), self.SPECS
        )
        assert result.status == "fail"
        assert result.expected == "13504500"
        assert result.actual == "13404500"
        assert "out by 100000" in result.detail

    def test_the_printed_totals_can_agree_while_the_components_do_not(self):
        """The 'looks complete but is not' case: only the identity check catches it."""
        claims = self.claims("13504500", "5320400", "8084100", "13504500")
        assert check_totals_footing(claims, self.SPECS).status == "pass"
        assert check_balance_sheet(claims, self.SPECS).status == "fail"

    def test_scale_is_applied_before_reconciling(self):
        """Both sides scaled by 1000, so a mirror-image error still cancels out."""
        claims = self.claims("13504.5", "5320.4", "8184.1", "13504.5", scale=Decimal(1_000))
        assert check_balance_sheet(claims, self.SPECS).status == "pass"

    def test_a_rounding_tolerance_can_be_granted_explicitly(self):
        claims = self.claims("13504500", "5320400", "8084100", "13504500")
        assert check_balance_sheet(claims, self.SPECS, tolerance=Decimal(100_000)).status == "pass"
        assert check_balance_sheet(claims, self.SPECS, tolerance=Decimal(99_999)).status == "fail"

    def test_missing_components_are_not_checkable_rather_than_passing(self):
        partial = self.claims("13504500", "5320400", "8184100", "13504500")
        del partial["total_equity"]
        result = check_balance_sheet(partial, self.SPECS)
        assert result.status == "not_checkable"
        assert not result.checkable
        assert "total_equity" in result.detail

    def test_totals_footing_needs_both_sides(self):
        partial = self.claims("13504500", "5320400", "8184100", "13504500")
        del partial["total_liabilities_and_equity"]
        assert check_totals_footing(partial, self.SPECS).status == "not_checkable"

    def test_reconcile_returns_both_checks_in_a_stable_order(self):
        results = reconcile(self.claims("13504500", "5320400", "8084100", "13504500"), self.SPECS)
        assert [r.name for r in results] == ["balance_sheet_identity", "totals_footing"]
        assert len(failing(results)) == 1

    def test_unparseable_components_are_treated_as_missing(self):
        claims = self.claims("n/a", "5320400", "8184100", "13504500")
        assert check_balance_sheet(claims, self.SPECS).status == "not_checkable"
