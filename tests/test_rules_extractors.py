"""The three rules revisions, including the defects each one is supposed to have.

A regression fixture whose defects are not asserted is a fixture that will
quietly become a vacuous pass. Every defect here is pinned to a specific document
and field, so a change to the harness that hides one shows up as a failure.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from geq.extractors.rules import REVISIONS, RulesExtractor, parse_value_lines
from geq.models import EvidenceStatus
from geq.verify import check_evidence

LAYOUT = "\n".join(
    [
        "NORTHGATE CLINICAL GROUP",
        "Amounts in US dollars",
        "Cash and cash equivalents                   1,204,500",
        "Total assets .............................. 13,504,500",
        "Gross profit                                9,960,000",
        "Note 7: Statement unsigned as at the reporting date",
        "Net income 3,415,000 (restated)",
        "",
    ]
)


class TestLineParsing:
    def test_reads_space_padded_and_dot_leader_layouts(self):
        labels = {line.label: line.value for line in parse_value_lines(LAYOUT)}
        assert labels["Cash and cash equivalents"] == "1,204,500"
        assert labels["Total assets"] == "13,504,500"
        assert labels["Gross profit"] == "9,960,000"

    def test_a_figure_followed_by_prose_is_not_a_data_row(self):
        labels = [line.label for line in parse_value_lines(LAYOUT)]
        assert "Net income" not in labels

    def test_prose_and_headers_are_ignored(self):
        labels = [line.label for line in parse_value_lines(LAYOUT)]
        assert "NORTHGATE CLINICAL GROUP" not in labels
        assert "Amounts in US dollars" not in labels

    def test_spans_slice_back_to_the_value(self):
        for line in parse_value_lines(LAYOUT):
            assert LAYOUT[line.value_start : line.value_end] == line.value

    def test_line_span_covers_the_label_and_the_value(self):
        line = next(x for x in parse_value_lines(LAYOUT) if x.label == "Total assets")
        text = LAYOUT[line.line_start : line.line_end]
        assert text.startswith("Total assets")
        assert text.endswith("13,504,500")

    def test_offsets_are_absolute_across_multiple_lines(self):
        lines = parse_value_lines(LAYOUT)
        assert lines[1].line_start > lines[0].line_start
        assert LAYOUT[lines[1].value_start : lines[1].value_end] == lines[1].value

    def test_an_unknown_revision_is_rejected(self):
        with pytest.raises(ValueError, match="unknown revision"):
            RulesExtractor("v9")

    def test_available_revisions_are_exactly_v1_v2_v3(self):
        assert REVISIONS == ("v1", "v2", "v3")


class TestV1Defects:
    """ "Read the number after the label" and cite nothing."""

    def test_money_values_carry_no_citation(self, corpus, v1_run):
        case = corpus.case("case_01_northgate_2024")
        extraction = RulesExtractor("v1", corpus.specs).extract(case)
        for field in ("total_assets", "total_liabilities", "total_equity"):
            assert extraction.claims[field].span is None

    def test_values_are_right_but_ungrounded(self, corpus, v1_run):
        cell = next(
            c
            for c in v1_run.scorecard.cells
            if c.doc_id == "case_01_northgate_2024" and c.field == "total_assets"
        )
        assert cell.right
        assert not cell.grounded
        assert cell.check is not None and cell.check.status is EvidenceStatus.UNCITED

    def test_it_ignores_the_document_scale(self, corpus, v1_run):
        cell = next(
            c
            for c in v1_run.scorecard.cells
            if c.doc_id == "case_02_northgate_2024_thousands" and c.field == "total_assets"
        )
        assert cell.verdict.value == "wrong"

    def test_it_reads_the_income_statement_only_up_to_the_balance_sheet(self, corpus, v1_run):
        claimed = {c.field for c in v1_run.scorecard.cells if c.claimed_anything}
        assert "revenue" not in claimed
        assert "net_income" not in claimed
        assert "fiscal_year_end" not in claimed

    def test_it_is_fooled_by_a_note_line_mentioning_the_total(self, corpus, v1_run):
        cell = next(
            c
            for c in v1_run.scorecard.cells
            if c.doc_id == "case_06_harbour_point_2025h1" and c.field == "total_assets"
        )
        assert cell.verdict.value == "wrong"
        assert cell.claimed == "1,000,000", "it read the footnote's figure, not the total"


class TestV2Defects:
    """Citations added, and two defects left in on purpose."""

    def test_prefix_matching_resolves_total_liabilities_to_the_wrong_line(self, corpus):
        case = corpus.case("case_01_northgate_2024")
        extraction = RulesExtractor("v2", corpus.specs).extract(case)
        assert extraction.claims["total_liabilities"].raw_value == "13,504,500"
        assert extraction.claims["total_liabilities"].raw_value == case.labels[
            "total_liabilities_and_equity"
        ].replace("13504500", "13,504,500")

    def test_that_wrong_value_is_still_faithfully_cited(self, corpus, v2_run):
        cell = next(
            c
            for c in v2_run.scorecard.cells
            if c.doc_id == "case_01_northgate_2024" and c.field == "total_liabilities"
        )
        assert cell.verdict.value == "wrong"
        assert cell.grounded, "a faithful citation of the wrong line is still a citation"

    def test_the_fixed_width_span_misses_the_figure(self, corpus, v2_run):
        cell = next(
            c
            for c in v2_run.scorecard.cells
            if c.doc_id == "case_01_northgate_2024" and c.field == "net_income"
        )
        assert cell.verdict.value == "right", "the value was read correctly"
        assert not cell.grounded, "but the citation does not contain it"
        assert cell.check is not None and cell.check.status is EvidenceStatus.CONTRADICTED

    def test_it_handles_thousands_but_not_millions(self, corpus, v2_run):
        thousands = next(
            c
            for c in v2_run.scorecard.cells
            if c.doc_id == "case_02_northgate_2024_thousands" and c.field == "total_assets"
        )
        millions = next(
            c
            for c in v2_run.scorecard.cells
            if c.doc_id == "case_03_cascade_2024_millions" and c.field == "total_assets"
        )
        assert thousands.verdict.value == "right"
        assert millions.verdict.value == "wrong"

    def test_it_still_cannot_place_a_date(self, corpus, v2_run):
        cell = next(
            c
            for c in v2_run.scorecard.cells
            if c.doc_id == "case_01_northgate_2024" and c.field == "fiscal_year_end"
        )
        assert cell.verdict.value == "missing"

    def test_it_takes_a_cover_page_disclaimer_for_the_entity_name(self, corpus, v2_run):
        cell = next(
            c
            for c in v2_run.scorecard.cells
            if c.doc_id == "case_05_riverbend_2023" and c.field == "entity_name"
        )
        assert cell.verdict.value == "wrong"
        assert "PRELIMINARY" in (cell.claimed or "")


class TestV3Quality:
    """The revision that is allowed to ship. Fails loudly if it degrades."""

    def test_every_cell_is_right_and_grounded(self, v3_run):
        assert v3_run.scorecard.failures() == ()
        assert v3_run.scorecard.grounded_recall == pytest.approx(1.0)

    def test_grounding_and_citation_rates_are_complete(self, v3_run):
        assert v3_run.scorecard.grounding_rate == pytest.approx(1.0)
        assert v3_run.scorecard.citation_rate == pytest.approx(1.0)

    def test_no_field_is_below_perfect_recall(self, v3_run):
        for name, score in v3_run.scorecard.per_field.items():
            assert score.recall == pytest.approx(1.0), name
            assert score.precision == pytest.approx(1.0), name

    def test_spans_are_narrow_and_slice_back_to_the_value(self, corpus):
        extractor = RulesExtractor("v3", corpus.specs)
        for case in corpus.cases:
            extraction = extractor.extract(case)
            for claim in extraction.claims.values():
                assert claim.span is not None
                start, end = claim.span
                assert case.text[start:end] == claim.raw_value
                assert end - start < 60

    def test_it_reads_the_document_scale_for_thousands_and_millions(self, corpus):
        extractor = RulesExtractor("v3", corpus.specs)
        thousands = extractor.extract(corpus.case("case_02_northgate_2024_thousands"))
        millions = extractor.extract(corpus.case("case_03_cascade_2024_millions"))
        assert thousands.claims["total_assets"].scale == Decimal(1_000)
        assert millions.claims["total_assets"].scale == Decimal(1_000_000)

    def test_it_skips_the_cover_page_disclaimer(self, corpus):
        case = corpus.case("case_05_riverbend_2023")
        extraction = RulesExtractor("v3", corpus.specs).extract(case)
        assert extraction.claims["entity_name"].raw_value == "RIVERBEND ORTHOPAEDIC ASSOCIATES"
        assert "PRELIMINARY" not in extraction.claims["entity_name"].raw_value

    def test_it_reads_the_date_in_the_form_the_document_uses(self, corpus):
        extractor = RulesExtractor("v3", corpus.specs)
        month_first = extractor.extract(corpus.case("case_05_riverbend_2023"))
        mid_year = extractor.extract(corpus.case("case_06_harbour_point_2025h1"))
        assert month_first.claims["fiscal_year_end"].raw_value == "December 31, 2023"
        assert mid_year.claims["fiscal_year_end"].raw_value == "30 June 2025"

    def test_it_is_not_fooled_by_the_note_line(self, corpus):
        extraction = RulesExtractor("v3", corpus.specs).extract(
            corpus.case("case_06_harbour_point_2025h1")
        )
        assert extraction.claims["total_assets"].raw_value == "8,840,000"


class TestRevisionProgression:
    """The story the reports tell, asserted rather than described."""

    def test_each_revision_is_at_least_as_good_as_the_last(self, runs):
        recalls = [runs[rev].scorecard.micro_recall for rev in REVISIONS]
        grounded = [runs[rev].scorecard.grounded_recall for rev in REVISIONS]
        assert recalls == sorted(recalls)
        assert grounded == sorted(grounded)
        assert grounded[0] < grounded[1] < grounded[2]

    def test_grounding_only_ever_improves(self, runs):
        rates = [runs[rev].scorecard.grounding_rate for rev in REVISIONS]
        assert rates == sorted(rates)

    def test_v1_is_the_only_revision_with_uncited_values(self, runs):
        assert runs["v1"].scorecard.citation_rate < 1.0
        assert runs["v2"].scorecard.citation_rate == pytest.approx(1.0)
        assert runs["v3"].scorecard.citation_rate == pytest.approx(1.0)

    def test_the_corpus_is_always_the_same_size(self, runs):
        sizes = {len(run.scorecard.cells) for run in runs.values()}
        assert sizes == {48}


class TestClaimInvariantsAcrossEveryRevision:
    def test_spans_are_always_inside_the_document(self, corpus):
        for revision in REVISIONS:
            extractor = RulesExtractor(revision, corpus.specs)
            for case in corpus.cases:
                for claim in extractor.extract(case).claims.values():
                    if claim.span is None:
                        continue
                    start, end = claim.span
                    assert 0 <= start < end <= len(case.text)

    def test_every_claim_survives_a_grounding_check_without_crashing(self, corpus):
        for revision in REVISIONS:
            extractor = RulesExtractor(revision, corpus.specs)
            for case in corpus.cases:
                for name, claim in extractor.extract(case).claims.items():
                    spec = corpus.specs.get(name)
                    if spec is None:
                        continue
                    assert check_evidence(case.text, claim, spec) is not None

    def test_a_revision_never_claims_an_unknown_field(self, corpus):
        known = set(corpus.specs)
        for revision in REVISIONS:
            extractor = RulesExtractor(revision, corpus.specs)
            for case in corpus.cases:
                assert set(extractor.extract(case).claims) <= known

    def test_scale_is_one_or_a_power_of_ten(self, corpus):
        for revision in REVISIONS:
            extractor = RulesExtractor(revision, corpus.specs)
            for case in corpus.cases:
                for claim in extractor.extract(case).claims.values():
                    if claim.scale != 1:
                        assert claim.scale in (Decimal(1_000), Decimal(1_000_000))

    def test_uncited_claims_never_carry_a_span(self, corpus):
        for revision in REVISIONS:
            extractor = RulesExtractor(revision, corpus.specs)
            for case in corpus.cases:
                for claim in extractor.extract(case).claims.values():
                    assert (claim.span is None) or claim.span[1] > claim.span[0]

    def test_extraction_records_the_extractor_it_came_from(self, corpus):
        for revision in REVISIONS:
            extractor = RulesExtractor(revision, corpus.specs)
            extraction = extractor.extract(corpus.cases[0])
            assert extraction.extractor == f"rules:{revision}"
            assert extraction.doc_id == corpus.cases[0].doc_id

    def test_labels_come_from_the_corpus_spec(self, corpus):
        extractor = RulesExtractor("v3", corpus.specs)
        assert extractor.label_for("total_assets") == corpus.specs["total_assets"].label

    def test_falling_back_to_default_labels_without_a_spec(self):
        extractor = RulesExtractor("v3")
        assert extractor.label_for("total_assets") == "Total assets"
        assert extractor.label_for("nonexistent") is None
