# Golden corpus

Six synthetic consolidated statements and the field spec they are labelled
against. **Every figure is fictional.** No real organisation's accounts appear in
this repository.

## Why it is generated

`tools/make_golden_corpus.py` builds these documents and a test asserts the
committed JSON still matches the generator byte for byte.

The alternative, hand-typing six statements, fails in a way that is hard to
notice: subtotals stop adding up, one statement drifts from the others, and the
harness starts reporting extraction failures that are actually corpus failures.
Generating means every statement foots *by construction*, and the only
inconsistency in the set is the one planted on purpose.

`render_amount` refuses to emit a figure that cannot be represented exactly at
the document's stated scale. A rounded figure is not a checkable figure, and
accepting one would bake a silent rounding tolerance into every comparison.

## The documents

| Document | What it exercises |
| --- | --- |
| `case_01_northgate_2024` | Clean control. Everything foots, every field present. |
| `case_02_northgate_2024_thousands` | Same figures stated in thousands. An extractor that reads the number and ignores the header is out by 1000x. |
| `case_03_cascade_2024_millions` | Stated in millions, exact at that scale. An extractor that only understands "thousands" fails here. |
| `case_04_northgate_2024_unbalanced` | **Planted defect.** Retained earnings are 100,000 low, so liabilities plus equity do not equal total assets. Every subtotal is present and the printed combined total still agrees with assets, so nothing looks wrong until the identity is checked. |
| `case_05_riverbend_2023` | A cover-page disclaimer above the entity name, and a month-first date. Catches "the first non-empty line is the entity". |
| `case_06_harbour_point_2025h1` | Dot-leader layout, a non-standard period end, and a note line containing the words "Total assets" whose figure is not the total. |

Exactly one document is unbalanced, and
`tests/test_golden_corpus_integrity.py` fails if that stops being true: the
arithmetic checks have to keep discriminating, and a second unbalanced statement
would mean they were firing on a corpus problem rather than on extraction.

## The fields

Eight, in `spec.json`. `entity_name` and `fiscal_year_end` are read from the
document header; the rest are figures from the statements.

Labels are stored at **absolute magnitude**, so a statement "in thousands" that
prints `13,504.5` is labelled `13504500`. This is what lets the harness tell
apart "misread the number" from "misread the units", which are different bugs
with different fixes. A test asserts every label is findable in its own document
at the stated scale, so a typo in a label fails the build rather than quietly
capping recall.

## Adding a case

1. Add a `CaseConfig` to `build_cases()` in the generator, with a note explaining
   what the document is for. A case without a stated purpose is not worth
   labelling.
2. Run `python tools/make_golden_corpus.py`.
3. Run `pytest`. The integrity tests check the labels against the document, the
   arithmetic, and the uniqueness of the defect.
