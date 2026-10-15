"""Canonicalisation tests.

Two of these assert behaviour that is *wrong on purpose* and marked as a known
limitation. They are tests rather than comments so the limitation cannot quietly
become a silent pass: if the parser is ever fixed, the test fails and the fix
gets a changelog entry.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from geq.models import FieldKind
from geq.normalize import (
    apply_scale,
    canonical,
    canonical_date,
    canonical_money,
    canonical_percent,
    canonical_text,
    find_dates,
    find_money,
    money_tokens_in,
    signed_decimal,
)


class TestMoney:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1,204,500", "1204500"),
            ("$1,204,500.00", "1204500"),
            ("1,204,500.00", "1204500"),
            ("USD 1,204,500", "1204500"),
            ("1.2M", "1200000"),
            ("1.2 million", "1200000"),
            ("1.2m", "1200000"),
            ("13,504.5", "13504.5"),
            ("13,504.50", "13504.5"),
            ("(1,204,500)", "-1204500"),
            ("-1,204,500", "-1204500"),
            ("\u22121,204,500", "-1204500"),
            ("-1,204,500.00 ", "-1204500"),
            ("0", "0"),
        ],
    )
    def test_parses_the_shapes_that_appear_in_real_statements(self, raw, expected):
        assert canonical_money(raw) == expected

    def test_a_bare_integer_is_not_truncated(self):
        """Regression: the grouped-number branch used to match ``135`` out of ``13504500``.

        A greedy alternation that does not require separators matched the first
        three digits and stopped, so every unseparated value scored as wrong
        while looking perfectly correct in the report.
        """
        assert canonical_money("13504500") == "13504500"
        assert canonical_money("24880000") == "24880000"
        assert canonical_money("106663") == "106663"

    @pytest.mark.parametrize("raw", ["", "n/a", "not stated", "-", "nil"])
    def test_unparseable_returns_none_rather_than_guessing(self, raw):
        assert canonical_money(raw) is None

    def test_a_bare_number_survives_the_prose_around_it(self):
        """Deliberate: value parsing is not the place to judge relevance.

        ``see note 3`` is read as 3. That is correct for a parser and would be
        wrong for an extractor, which is exactly why a value also has to be
        cited and the citation checked separately.
        """
        assert canonical_money("see note 3") == "3"

    def test_european_decimal_comma_is_a_known_limitation(self):
        """Known limitation, deliberately tracked.

        ``13.504,50`` is read as ``13.504``. Nothing in this corpus uses comma
        decimals, so the parser has never needed them. When a real document does,
        this test is the tripwire.
        """
        assert canonical_money("13.504,50") == "13.504"


class TestPercentDateText:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("12.5%", "12.5"), ("12.5 per cent", "12.5"), ("12.5 percent", "12.5"), ("100 %", "100")],
    )
    def test_percent(self, raw, expected):
        assert canonical_percent(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("31 December 2024", "2024-12-31"),
            ("31st December 2024", "2024-12-31"),
            ("31st of December 2024", "2024-12-31"),
            ("December 31, 2023", "2023-12-31"),
            ("2024-12-31", "2024-12-31"),
            ("30 June 2025", "2025-06-30"),
            ("As at 31 December 2024", "2024-12-31"),
        ],
    )
    def test_dates(self, raw, expected):
        assert canonical_date(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("31/12/2024", "2024-12-31"), ("12/31/2024", "2024-12-31")],
    )
    def test_slash_dates_are_resolved_only_when_unambiguous(self, raw, expected):
        assert canonical_date(raw) == expected

    def test_an_ambiguous_slash_date_is_rejected_not_guessed(self):
        """03/04/2024 is either 3 April or 4 March depending on locale. Refuse."""
        assert canonical_date("03/04/2024") is None

    @pytest.mark.parametrize("raw", ["31 February 2024", "2024-13-01", "", "not a date"])
    def test_invalid_dates_return_none(self, raw):
        assert canonical_date(raw) is None

    def test_text_is_case_and_suffix_insensitive(self):
        assert canonical_text("NORTHGATE CLINICAL GROUP") == "northgate clinical group"
        assert canonical_text("Northgate Clinical Group, LLC") == "northgate clinical group"
        assert canonical_text("  Northgate   Clinical Group  ") == "northgate clinical group"

    def test_canonical_dispatches_on_kind(self):
        assert canonical("1,204,500", FieldKind.MONEY) == "1204500"
        assert canonical("12.5%", FieldKind.PERCENT) == "12.5"
        assert canonical("December 31, 2023", FieldKind.DATE) == "2023-12-31"
        assert canonical("Northgate Clinical Group", FieldKind.TEXT) == "northgate clinical group"


class TestTokenScanning:
    def test_finds_every_literal_with_a_usable_span(self):
        text = "Total assets 13,504.5 (note 4) and 1,000,000"
        tokens = money_tokens_in(text)
        assert {"13504.5", "4", "1000000"} <= tokens

    def test_returned_spans_slice_back_to_the_literal(self):
        """Invariant the whole grounding check rests on."""
        text = "Cash and cash equivalents 1,204,500 and Total assets 13,504,500"
        for literal, start, end in find_money(text):
            assert text[start:end] == literal

    def test_scanning_a_document_with_no_figures_is_empty(self):
        assert money_tokens_in("Amounts in thousands of US dollars") == set()

    def test_find_dates_locates_a_date_inside_a_sentence(self):
        text = "As at 31 December 2024"
        matches = find_dates(text)
        assert matches
        literal, start, end = matches[0]
        assert text[start:end] == literal
        assert canonical_date(literal) == "2024-12-31"

    def test_find_dates_ignores_words_near_numbers(self):
        assert find_dates("Total current assets 4,614,500") == []
        assert find_dates("Net income 2,415,000") == []


class TestScaling:
    def test_scale_multiplies_money(self):
        assert apply_scale("13504.5", Decimal(1_000), FieldKind.MONEY) == "13504500"
        assert apply_scale("13.0", Decimal(1_000_000), FieldKind.MONEY) == "13000000"

    def test_scale_of_one_is_a_no_op(self):
        assert apply_scale("13504500", Decimal(1), FieldKind.MONEY) == "13504500"

    def test_scale_does_not_touch_dates_or_text(self):
        assert apply_scale("2024-12-31", Decimal(1_000), FieldKind.DATE) == "2024-12-31"
        assert apply_scale("northgate", Decimal(1_000), FieldKind.TEXT) == "northgate"

    def test_none_passes_through(self):
        assert apply_scale(None, Decimal(1_000), FieldKind.MONEY) is None

    def test_signed_decimal_reads_canonical_output(self):
        assert signed_decimal("-1204500") == Decimal("-1204500")
        assert signed_decimal(None) is None
