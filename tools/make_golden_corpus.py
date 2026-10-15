"""Generate the synthetic golden corpus.

The corpus is generated rather than hand-typed for one reason: the documents have
to be *arithmetically consistent* everywhere except where a defect is planted on
purpose. Six aligned statements typed by hand will drift, and a drifted golden
set produces QA numbers that are wrong in a way nobody notices.

Every document here is fictional. No real organisation's figures appear in this
repository.

Run with ``python tools/make_golden_corpus.py``. Output is deterministic, and
``tests/test_golden_corpus_integrity.py`` fails if the committed JSON stops
matching what this script produces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "data" / "golden"

LABEL_COL = 44
"""Column at which every figure starts.

Deliberately wide: it means a citation guessed at a fixed 32-character width
lands on labels and padding instead of on the number, which is the defect
``rules:v2`` reproduces.
"""

SCALE_WORDS = {
    1: "US dollars",
    1_000: "thousands of US dollars",
    1_000_000: "millions of US dollars",
}


@dataclass(frozen=True)
class CaseConfig:
    """One synthetic statement."""

    doc_id: str
    entity: str
    date_line: str
    scale: int
    amounts: dict[str, int]
    prefix_lines: tuple[str, ...] = ()
    extra_lines: tuple[str, ...] = ()
    dot_leaders: bool = False
    combined_total: int | None = None
    """Printed "Total liabilities and equity". Defaults to total assets.

    Set this to make a statement print a combined total that its own components
    do not add up to: the "looks complete but is not" case.
    """

    equity_override: int | None = None
    """Retained earnings override, used to unbalance an otherwise clean case."""

    notes: str = ""


# --------------------------------------------------------------------------
# Figure sets
# --------------------------------------------------------------------------

# Set A - full dollars. Everything foots.
FIGURES_A = {
    "cash": 1_204_500,
    "receivables": 3_410_000,
    "ppe": 8_890_000,
    "payables": 1_120_400,
    "long_term_debt": 4_200_000,
    "share_capital": 1_000_000,
    "retained_earnings": 7_184_100,
    "revenue": 24_880_000,
    "cost_of_sales": 14_920_000,
    "net_income": 2_415_000,
}

# Set B - a smaller practice, different entity.
FIGURES_B = {
    "cash": 842_900,
    "receivables": 1_986_400,
    "ppe": 4_512_000,
    "payables": 764_300,
    "long_term_debt": 2_150_000,
    "share_capital": 500_000,
    "retained_earnings": 3_927_000,
    "revenue": 15_420_000,
    "cost_of_sales": 9_180_000,
    "net_income": 1_336_000,
}

# Set C - a six-month interim period.
FIGURES_C = {
    "cash": 655_200,
    "receivables": 2_204_800,
    "ppe": 5_980_000,
    "payables": 931_500,
    "long_term_debt": 2_800_000,
    "share_capital": 750_000,
    "retained_earnings": 4_358_500,
    "revenue": 12_640_000,
    "cost_of_sales": 7_584_000,
    "net_income": 1_182_000,
}

# Set D - round millions, so that a value written to one decimal place is exact
# once the document-level scale is applied. A statement rounded to the nearest
# million cannot be scored exactly, and pretending otherwise would bake a silent
# rounding tolerance into the harness.
FIGURES_D = {
    "cash": 1_200_000,
    "receivables": 3_400_000,
    "ppe": 8_400_000,
    "payables": 1_100_000,
    "long_term_debt": 5_400_000,
    "share_capital": 1_000_000,
    "retained_earnings": 5_500_000,
    "revenue": 24_900_000,
    "cost_of_sales": 14_900_000,
    "net_income": 2_400_000,
}


# --------------------------------------------------------------------------
# Derivation
# --------------------------------------------------------------------------


def totals(amounts: dict[str, int], equity_override: int | None = None) -> dict[str, int]:
    """Every subtotal a statement would print, derived from the primitives."""
    retained = equity_override if equity_override is not None else amounts["retained_earnings"]
    total_current_assets = amounts["cash"] + amounts["receivables"]
    total_assets = total_current_assets + amounts["ppe"]
    total_liabilities = amounts["payables"] + amounts["long_term_debt"]
    total_equity = amounts["share_capital"] + retained
    return {
        "total_current_assets": total_current_assets,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "total_equity": total_equity,
        "gross_profit": amounts["revenue"] - amounts["cost_of_sales"],
        "sum_of_liabilities_and_equity": total_liabilities + total_equity,
    }


def render_amount(amount: int, scale: int) -> str:
    """Render a figure at the document's stated scale, exactly.

    Raises if the value cannot be represented at that scale, because a rounded
    figure is not a checkable figure.
    """
    if amount % scale != 0 and scale != 1:
        # One decimal place is available, so the value must be a multiple of
        # scale / 10.
        if amount % (scale // 10) != 0:
            raise ValueError(f"{amount} is not representable at scale 1:{scale}")
    value = Decimal(amount) / Decimal(scale)
    text = f"{value:,.0f}" if scale == 1 else f"{value:,.1f}"
    return text


def money_line(label: str, amount: int, scale: int, *, leaders: bool = False) -> str:
    """A label at column 0 and a figure at ``LABEL_COL``, dot leaders optional."""
    rendered = f"({render_amount(-amount, scale)})" if amount < 0 else render_amount(amount, scale)
    pad = LABEL_COL - len(label)
    if leaders:
        if pad < 4:
            raise ValueError(f"label too long for dot leaders: {label!r}")
        return f"{label} {'.' * (pad - 2)} {rendered}"
    return f"{label}{' ' * pad}{rendered}"


def render_document(cfg: CaseConfig) -> str:
    a = cfg.amounts
    t = totals(a, cfg.equity_override)
    combined = cfg.combined_total if cfg.combined_total is not None else t["total_assets"]

    def line(label: str, amount: int) -> str:
        return money_line(label, amount, cfg.scale, leaders=cfg.dot_leaders)

    blocks = [
        *cfg.prefix_lines,
        cfg.entity.upper(),
        "Consolidated Balance Sheet",
        cfg.date_line,
        f"Amounts in {SCALE_WORDS[cfg.scale]}",
        *cfg.extra_lines,
        "",
        "ASSETS",
        line("Cash and cash equivalents", a["cash"]),
        line("Trade and other receivables", a["receivables"]),
        line("Total current assets", t["total_current_assets"]),
        line("Property, plant and equipment", a["ppe"]),
        line("Total assets", t["total_assets"]),
        "",
        "LIABILITIES",
        line("Trade and other payables", a["payables"]),
        line("Long-term borrowings", a["long_term_debt"]),
        line("Total liabilities", t["total_liabilities"]),
        "",
        "EQUITY",
        line("Share capital", a["share_capital"]),
        line("Retained earnings", a["retained_earnings"] if cfg.equity_override is None else cfg.equity_override),
        line("Total equity", t["total_equity"]),
        "",
        line("Total liabilities and equity", combined),
        "",
        "CONSOLIDATED STATEMENT OF OPERATIONS",
        line("Revenue", a["revenue"]),
        line("Cost of sales", -a["cost_of_sales"]),
        line("Gross profit", t["gross_profit"]),
        line("Net income", a["net_income"]),
    ]
    return "\n".join(blocks) + "\n"


def labels_for(cfg: CaseConfig) -> dict[str, str]:
    a = cfg.amounts
    t = totals(a, cfg.equity_override)
    combined = cfg.combined_total if cfg.combined_total is not None else t["total_assets"]
    return {
        "entity_name": cfg.entity,
        "fiscal_year_end": _iso_date(cfg.date_line),
        "total_assets": str(t["total_assets"]),
        "total_liabilities": str(t["total_liabilities"]),
        "total_equity": str(t["total_equity"]),
        "total_liabilities_and_equity": str(combined),
        "revenue": str(a["revenue"]),
        "net_income": str(a["net_income"]),
    }


_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04", "may": "05",
    "june": "06", "july": "07", "august": "08", "september": "09",
    "october": "10", "november": "11", "december": "12",
}


def _iso_date(date_line: str) -> str:
    """Turn a rendering date phrase into the ISO label, without guessing.

    Kept deliberately dumb and explicit: if the date line changes shape, the
    label generator should fail loudly rather than invent a date.
    """
    words = date_line.replace(",", " ").split()
    month = next((w.lower() for w in words if w.lower() in _MONTHS), None)
    year = next((w for w in words if len(w) == 4 and w.isdigit()), None)
    day = next((w for w in words if w.isdigit() and len(w) <= 2), None)
    if not (month and year and day):
        raise ValueError(f"cannot derive an ISO date from {date_line!r}")
    return f"{year}-{_MONTHS[month]}-{int(day):02d}"


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------


def build_cases() -> list[CaseConfig]:
    return [
        CaseConfig(
            doc_id="case_01_northgate_2024",
            entity="Northgate Clinical Group",
            date_line="As at 31 December 2024",
            scale=1,
            amounts=FIGURES_A,
            notes="Clean control case. Everything foots and every field is present.",
        ),
        CaseConfig(
            doc_id="case_02_northgate_2024_thousands",
            entity="Northgate Clinical Group",
            date_line="As at 31 December 2024",
            scale=1_000,
            amounts=FIGURES_A,
            notes=(
                "Same figures as case_01 but stated in thousands. An extractor that "
                "reads the number and ignores the header is out by 1000x."
            ),
        ),
        CaseConfig(
            doc_id="case_03_cascade_2024_millions",
            entity="Cascade Specialty Holdings",
            date_line="As at 31 December 2024",
            scale=1_000_000,
            amounts=FIGURES_D,
            notes="Stated in millions, rounded to one decimal but exact at that scale.",
        ),
        CaseConfig(
            doc_id="case_04_northgate_2024_unbalanced",
            entity="Northgate Clinical Group",
            date_line="As at 31 December 2024",
            scale=1,
            amounts=FIGURES_A,
            equity_override=7_084_100,
            notes=(
                "Planted defect: retained earnings are 100,000 low, so liabilities plus "
                "equity do not equal total assets. Every subtotal is present and the "
                "printed combined total still agrees with assets, so nothing looks "
                "wrong until the identity is checked."
            ),
        ),
        CaseConfig(
            doc_id="case_05_riverbend_2023",
            entity="Riverbend Orthopaedic Associates",
            date_line="For the year ended December 31, 2023",
            scale=1,
            amounts=FIGURES_B,
            prefix_lines=("PRELIMINARY DRAFT - NOT FOR DISTRIBUTION",),
            notes=(
                "Cover-page disclaimer above the entity name, and a month-first date. "
                "Catches extractors that take the first non-empty line as the entity."
            ),
        ),
        CaseConfig(
            doc_id="case_06_harbour_point_2025h1",
            entity="Harbour Point Medical Partners",
            date_line="As at 30 June 2025",
            scale=1,
            amounts=FIGURES_C,
            dot_leaders=True,
            extra_lines=(
                money_line(
                    "Note 4: Total assets held in trust", 1_000_000, 1, leaders=True
                ),
                "Note 7: Statement unsigned as at the reporting date",
            ),
            notes=(
                "Exported-with-leader-dots layout, a non-standard period end, and a note "
                "line containing the words 'Total assets' whose figure is not the total. "
                "The distractor is planted deliberately."
            ),
        ),
    ]


SPEC = {
    "fields": [
        {"name": "entity_name", "kind": "text", "label": "ENTITY_NAME", "required": True},
        {"name": "fiscal_year_end", "kind": "date", "label": "DATE", "required": True},
        {"name": "total_assets", "kind": "money", "label": "Total assets", "required": True},
        {"name": "total_liabilities", "kind": "money", "label": "Total liabilities", "required": True},
        {"name": "total_equity", "kind": "money", "label": "Total equity", "required": True},
        {
            "name": "total_liabilities_and_equity",
            "kind": "money",
            "label": "Total liabilities and equity",
            "required": True,
        },
        {"name": "revenue", "kind": "money", "label": "Revenue", "required": True},
        {"name": "net_income", "kind": "money", "label": "Net income", "required": True},
    ]
}


def build_case_payload(cfg: CaseConfig) -> dict[str, object]:
    return {
        "doc_id": cfg.doc_id,
        "text": render_document(cfg),
        "labels": labels_for(cfg),
        "notes": cfg.notes,
    }


def write_corpus(root: Path = GOLDEN_DIR) -> list[Path]:
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(SPEC, indent=2) + "\n", encoding="utf-8")
    written.append(spec_path)

    for cfg in build_cases():
        path = cases_dir / f"{cfg.doc_id}.json"
        path.write_text(
            json.dumps(build_case_payload(cfg), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def main() -> None:
    for path in write_corpus():
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
