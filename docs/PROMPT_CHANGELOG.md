# Prompt changelog

Three revisions of the same extractor, kept runnable so the change and its
measured effect stay in the same place. Each entry states what changed, why, and
what the scorecard said afterwards.

The numbers below are produced by `make report && make compare`, not typed by
hand. If a revision stops having the effect recorded here, `tests/test_rules_extractors.py`
fails.

## v1: read the number after the label

The first attempt, and roughly what a "just get the fields out" prompt produces.

- Find the label with a substring search over the whole document.
- Take the first figure that appears after it.
- Return no citation, because the prompt never asked for one.
- Read the balance sheet only.

**Measured:** recall 33.3%, precision 66.7%, citations 25.0%, grounding 25.0%,
**grounded recall 10.4%**.

**What went wrong, and what it teaches.** Four failures, all of them invisible if
you only count correct values:

| Symptom | Cause | Fixed in |
| --- | --- | --- |
| A footnote's figure read as the total | Substring search finds `Note 4: Total assets held in trust` before the real line | v3 |
| Values out by 1000x on a statement "in thousands" | The header was never read | v2 |
| No citation at all on the figures | The prompt never asked for one | v2 |
| The cover page's `PRELIMINARY DRAFT` read as the entity name | "First non-empty line" is not an entity name rule | v3 |

## v2: line-anchored, with citations

- Parse `label <separator> figure` rows instead of searching for substrings.
- Ask for a citation, and record the offsets.
- Read the document header for "in thousands" and apply the scale.
- Add the remaining balance-sheet and income-statement fields.

**Measured:** recall 62.5%, precision 71.4%, citations 100%, grounding 71.4%,
**grounded recall 41.7%**. 28 cells fixed against v1, 0 regressed.

**Two defects left in on purpose**, because they are the interesting ones:

1. **Prefix matching with a "totals come last" preference.** `Total liabilities`
   also matches `Total liabilities and equity`, and the later line wins. The
   value is wrong and the citation is *faithfully* citing the line it read. Every
   grounded-recall-only dashboard would call this a citation problem; it is not,
   it is a value problem, and the report says so.
2. **A fixed-width span guess.** Two fields fall back to "take 32 characters from
   the start of the line", which lands on the label and the padding rather than
   the figure. The value is right and the citation does not contain it. This is
   the mirror image of defect 1, and it is why the two dimensions are scored
   separately.

Both are reproduced in `tests/test_rules_extractors.py::TestV2Defects`, pinned to
a specific document and field, so a change to the scoring code that hides either
one shows up as a failure.

## v3: exact labels, scale from the header, narrow citations

- Match the label exactly, so `Total liabilities` can no longer claim the
  combined-total line.
- Read thousands *and* millions from the header.
- Skip cover-page disclaimers before taking the entity name.
- Add the date field, in whichever form the document uses.
- Narrow every citation to the figure itself.

**Measured:** recall 100%, precision 100%, citations 100%, grounding 100%,
**grounded recall 100%**. Against v2: 28 cells fixed, 0 regressed, and the gate
passes.

Run the same change backwards and the gate says: 28 regressions, grounded recall
down 58.3 points, grounding rate down 28.6 points, and five balance sheets that
stopped footing.

## Why the revisions live in the repository

A prompt change is a code change, and its effect should be a number CI can
compare rather than a claim in a pull request description. Keeping `v1` and `v2`
runnable costs a few hundred lines and buys three things:

1. The gate can be tested in both directions on every commit.
2. The grounding checker has known-bad input available without a network call.
3. A future regression has a shape to be recognised by: it will look like one of
   these two defects, not like a novel failure.

## Rules that hold for every revision

- An extractor returns values and spans. It never decides whether it is right.
- A citation that does not contain its own value is a defect, not a lower score.
- The document's stated scale is a separate claim from the citation.
- A value that cannot be canonicalised is never counted as correct.
