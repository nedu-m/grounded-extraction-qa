"""The failure modes this harness exists to catch.

Each class here is an extractor that would look fine in a demo. They are the
reason the grounding check, and not just recall, decides whether a number is
allowed near a reviewer.

These run against the real golden corpus, not a toy document, because a grounding
check that only works on a two-line fixture is not worth having.
"""

from __future__ import annotations

import re
from decimal import Decimal

import pytest

from geq.extractors import available_extractors, build_extractor
from geq.extractors.adversarial import CONTROL_NAMES, FabricatingExtractor, UncitedExtractor
from geq.models import Claim, EvidenceStatus, Extraction, GoldenCase
from geq.scoring import evaluate_corpus
from geq.verify import check_evidence

PRETTY_NUMBER = re.compile(r"\d[\d,]*\.?\d*")


def _build_control(name: str, specs):
    """The shipped controls, built the way the registry builds them."""
    return build_extractor(name, specs)


class WholeDocumentExtractor:
    """Cites the entire document for every field.

    Technically the value is in the cited text. It proves nothing, and a reviewer
    handed that citation cannot check anything without rereading the document.
    """

    name = "whole_document"

    def extract(self, case: GoldenCase) -> Extraction:
        span = (0, len(case.text))
        return Extraction(
            doc_id=case.doc_id,
            extractor=self.name,
            claims={
                "total_assets": Claim("total_assets", "13,504,500", span),
                "revenue": Claim("revenue", "24,880,000", span),
            },
        )


class OffsetDriftExtractor:
    """Correct values, spans off by one character.

    The most common way a real integration breaks: offsets computed over a
    normalised copy of the text, then applied to the original.
    """

    name = "offset_drift"

    def extract(self, case: GoldenCase) -> Extraction:
        claims: dict[str, Claim] = {}
        for name, value in (("total_assets", "13,504,500"), ("revenue", "24,880,000")):
            start = case.text.find(value)
            if start < 0:
                continue
            claims[name] = Claim(
                field=name, raw_value=value, span=(start + 1, start + 1 + len(value))
            )
        return Extraction(doc_id=case.doc_id, extractor=self.name, claims=claims)


@pytest.fixture(scope="module")
def card_for(corpus):
    def build(extractor) -> object:
        extractions = {case.doc_id: extractor.extract(case) for case in corpus.cases}
        return evaluate_corpus(corpus.cases, extractions, corpus.specs, extractor=extractor.name)

    return build


@pytest.fixture(scope="module")
def fabricating(corpus):
    return _build_control("adversarial:fabricating", corpus.specs)


@pytest.fixture(scope="module")
def uncited(corpus):
    return _build_control("adversarial:uncited", corpus.specs)


class TestFabricatedValues:
    def test_a_fabricated_number_is_contradicted_by_its_own_citation(
        self, corpus, card_for, fabricating
    ):
        card = card_for(fabricating)
        assert card.claims > 0
        assert card.grounding_rate == pytest.approx(0.0)
        assert card.grounded_recall == pytest.approx(0.0)
        assert card.unsupported_claims != ()

    def test_every_claim_is_flagged_with_a_reason_a_reviewer_can_act_on(
        self, corpus, card_for, fabricating
    ):
        card = card_for(fabricating)
        for cell in card.unsupported_claims:
            assert cell.check is not None
            assert cell.check.status is EvidenceStatus.CONTRADICTED
            assert "does not contain the claimed value" in cell.check.detail


class TestWholeDocumentCitations:
    def test_a_document_wide_span_is_rejected(self, corpus, card_for):
        card = card_for(WholeDocumentExtractor())
        assert card.claims == 2 * len(corpus.cases)
        assert card.grounding_rate == pytest.approx(0.0)
        for cell in card.unsupported_claims:
            assert cell.check is not None
            assert cell.check.status is EvidenceStatus.NOT_A_CITATION

    def test_the_citation_is_truncated_in_the_report_rather_than_dumped(self, corpus, card_for):
        card = card_for(WholeDocumentExtractor())
        cell = card.unsupported_claims[0]
        assert cell.check is not None
        assert cell.check.cited_text is not None
        assert cell.check.cited_text.endswith("...")
        assert len(cell.check.cited_text) < 200


class TestOffsetDrift:
    def test_one_character_of_drift_breaks_grounding(self, corpus, card_for):
        card = card_for(OffsetDriftExtractor())
        assert card.claims > 0
        assert card.grounding_rate == pytest.approx(0.0)
        statuses = {cell.check.status for cell in card.unsupported_claims if cell.check}
        assert statuses == {EvidenceStatus.CONTRADICTED}


class TestControlsAreWiredIn:
    """The controls ship in the package, so CI re-proves the checker on every run."""

    def test_the_controls_are_registered(self):
        names = available_extractors()
        for name in CONTROL_NAMES:
            assert name in names

    def test_the_shipped_control_is_the_one_under_test(self, fabricating):
        assert isinstance(fabricating, FabricatingExtractor)
        assert fabricating.name == "adversarial:fabricating"

    def test_the_uncited_control_is_a_perfect_liar(self, uncited, corpus):
        extraction = uncited.extract(corpus.cases[0])
        assert extraction.claims, "the control must actually return values"
        assert all(claim.span is None for claim in extraction.claims.values())
        assert isinstance(uncited, UncitedExtractor)

    def test_an_unknown_control_is_rejected(self):
        with pytest.raises(ValueError, match="unknown control"):
            _build_control("adversarial:nonsense", {})


class TestConfidentButUnverifiable:
    def test_perfect_recall_and_zero_grounded_recall_coexist(self, corpus, card_for, uncited):
        card = card_for(uncited)
        assert card.micro_recall == pytest.approx(1.0)
        assert card.micro_precision == pytest.approx(1.0)
        assert card.citation_rate == pytest.approx(0.0)
        assert card.grounded_recall == pytest.approx(0.0)

    def test_the_gap_between_recall_and_grounded_recall_is_the_whole_argument(
        self, corpus, card_for, uncited
    ):
        card = card_for(uncited)
        assert card.micro_recall - card.grounded_recall == pytest.approx(1.0)

    def test_unsupported_claims_are_reported_as_uncited_not_merely_absent(
        self, corpus, card_for, uncited
    ):
        card = card_for(uncited)
        for cell in card.unsupported_claims:
            assert cell.check is not None
            assert cell.check.status is EvidenceStatus.UNCITED
            assert cell.claimed is not None, "the value was asserted, it just has no evidence"


class TestGroundingChecksTheRightThing:
    """Direct checks on the primitive, so a bug cannot hide behind an aggregate."""

    def test_a_value_present_but_cited_from_the_wrong_number_is_caught(self):
        text = "Total current assets 4,614,500\nTotal assets 13,504,500\n"
        start = text.index("4,614,500")
        claim = Claim(
            field="total_assets",
            raw_value="13,504,500",
            span=(start, start + len("4,614,500")),
        )
        spec = corpus_spec("total_assets")
        assert check_evidence(text, claim, spec).status is EvidenceStatus.CONTRADICTED

    def test_scale_alone_does_not_earn_grounding(self):
        text = "Total assets 13,504.5\n"
        claim = Claim(
            field="total_assets",
            raw_value="13,504.5",
            span=(text.index("13,504.5"), len(text) - 1),
            scale=Decimal(1_000),
        )
        assert (
            check_evidence(text, claim, corpus_spec("total_assets")).status
            is EvidenceStatus.SUPPORTED
        )


def corpus_spec(field: str):
    from geq.models import FieldKind, FieldSpec

    return FieldSpec(name=field, kind=FieldKind.MONEY, label="Total assets")
