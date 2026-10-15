# grounded-extraction-qa

Golden-set QA for LLM document extraction: value accuracy, evidence grounding,
fabricated-fact detection, and regression gating across prompt revisions.

**Live:** [qa.nedum.xyz](https://qa.nedum.xyz) · **Author:** [Chukwunedum Davidson](https://nedum.xyz)

---

## The page

[qa.nedum.xyz](https://qa.nedum.xyz) is written for the person the link is sent
to, not for whoever runs the CI. It takes the form a finance reader already trusts
for this job: an auditor's working paper, where every figure is traced to its
source and ticked.

- **The working paper.** Pick who prepared the schedule (the current version, two
  older ones, or one of the two built-to-fail controls) and which statement. Every
  figure carries a tick mark and a source line reference. Selecting one shows the
  reviewer's note and highlights the cited line in the document beside it.
- **Four outcomes, in plain words.** Agreed to source, no source given,
  exception, not reported. Only agreed figures count as usable.
- **The review summary** counts usable figures per preparer, which is where the
  "no sources" control scores zero despite getting every figure right.
- **A statement that does not foot** is set out with single and double rules
  and its difference in brackets.
- **The release gate** reads as a sign-off: approved one way, refused when the
  same change is rolled backwards.
- **Supporting schedules**, collapsed, carry the recall, precision and grounding
  tables for engineers.

The page loads nothing from another host. Fonts are served from `/fonts`, the
script from `/assets/paper.js`, and the host's Content-Security-Policy allows
scripts from this origin only, so nothing injected into the markup can run.

## The problem this exists for

A document extractor can return the correct value for every field and still be
unusable. If the value arrives without a citation, nobody can check it, and a
reviewer signing off on a deal summary is taking the model's word for it. The
failure that actually hurts is not a missing number, it is a confident wrong
number with a plausible-looking reference next to it.

So this harness refuses to score accuracy alone. Every claim is scored twice:

| Metric | Question it answers |
| --- | --- |
| **Recall** | Of the values present in the documents, how many were returned? |
| **Precision** | Of the values returned, how many were correct? |
| **Citation rate** | Did the extractor say where each value came from? |
| **Grounding rate** | Does the cited text *actually contain* the value claimed? |
| **Grounded recall** | Share of true values that were returned **and** traceable. The headline. |

The gap between recall and grounded recall is the entire argument. Grounding is
not a confidence score or a self-report: the cited span is sliced out of the
document and re-read, and a citation that does not contain its own value is
recorded as a failure with the offending text attached.

## Results

48 hand-labelled cells: six synthetic statements, eight fields each. Reproduce
with `make report && make summary`.

| Extractor | Recall | Precision | Citing | Grounding | **Grounded recall** |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rules:v1` naive keyword lookup | 33.3% | 66.7% | 25.0% | 25.0% | **10.4%** |
| `rules:v2` line-anchored, cited | 62.5% | 71.4% | 100% | 71.4% | **41.7%** |
| `rules:v3` exact labels, scaled | 100% | 100% | 100% | 100% | **100%** |
| `adversarial:uncited` correct, uncited | 100% | 100% | 0% | 0% | **0%** |
| `adversarial:fabricating` invented, cited | 0% | 0% | 100% | 0% | **0%** |

The last two rows are the point. They ship inside the package as controls, and CI
runs them on every commit. `adversarial:uncited` produces the true value for
every field and cites nothing, so it posts perfect recall and zero citable
output. `adversarial:fabricating` invents plausible figures and points each
citation at the line *label*, which is real text from the real document. A
reviewer skimming sees a quote and assumes someone checked it. Both score 0%
grounded recall, and if the checker ever stops catching them the build fails
rather than the score quietly improving.

## What the gate refuses

`geq compare` fails on any cell that gets worse, on any of these grounds:

- **A cell regressed.** Value accuracy and grounding are tracked independently,
  so losing a citation is a regression even when the number stays right.
- **A headline metric dropped.** Grounded recall, grounding rate and recall each
  have a zero-drop allowance by default.
- **The golden set changed.** If the two runs scored different cells the gate
  fails outright, because a moving corpus produces meaningless improvements. This
  is the check that stops the gate being gameable.
- **An arithmetic check regressed.** A statement that reconciled and no longer
  does is a real regression regardless of extraction scores.

`rules:v2` → `rules:v3` passes with 28 cells fixed and 0 regressed. Run the same
change backwards and the gate reports 28 regressions, a 58.3 point fall in
grounded recall, and five balance sheets that stopped footing.

## Design decisions worth stating

**Accuracy and grounding are never collapsed.** A value can be right and
uncited, wrong and faithfully cited, or right with a citation pointing at the
wrong line. Each combination needs a different fix, so each is reported
separately. `cellular grounding ignores scale on purpose`: an extractor that
reads `13,504.5` correctly out of a statement "in thousands" and reports it
unscaled has a faithful citation and a wrong answer, and the report says exactly
that rather than blaming the citation.

**Document scale is a separate claim from the citation.** The citation must quote
the document literally (`13,504.5`). The magnitude is a second, independently
checkable claim (`Claim.scale`). Conflating them hides which one broke.

**A wide citation is not a citation.** A span covering more than 280 characters
is rejected as `not_a_citation` even though it technically contains the value,
because it cannot be reviewed.

**Ambiguous dates are rejected, not guessed.** `03/04/2024` is either 3 April or
4 March depending on locale. The parser refuses rather than picking a convention
and passing a wrong date with a straight face.

**The golden set is generated, not typed.** `tools/make_golden_corpus.py` derives
every subtotal from primitives so the statements foot by construction, and a test
asserts the committed JSON still matches the generator. Six hand-built statements
drift, and a drifted golden set fails silently.

**The corpus is validated like production data.** A label for a field that is not
in the spec, a required field no case labels, or a duplicate `doc_id` all fail at
load time rather than at scorecard-reading time.

## Beyond extraction: SQL data checks

The same discipline applied where the JD asks for it, over a seeded diligence
data room. Each check is a SQL file that returns the offending rows and nothing
else, so a reviewer can paste it into a database client.

| Check | Planted defect it catches |
| --- | --- |
| `orphaned_extracted_fields` | Extracted values whose parent document does not exist |
| `foreign_key_violations` | Referential integrity, checked by the database not by hand |
| `duplicate_documents` | The same file received twice against one target |
| `missing_data_room_items` | A required checklist item with nothing filed against it |
| `balance_sheet_reconciliation` | Extracted totals that do not foot, across two sources |
| `brand_asset_mismatch` | A generated asset carrying another brand's name |

Every check has a negative control in `tests/test_sqlqa.py`: the suite proves
each returns nothing when its defect is removed, so a check that fires
unconditionally cannot masquerade as a working one.

## The live model path

The default provider is DeepSeek, reached over the OpenAI chat-completions
protocol. Anything speaking the same protocol works:

```bash
export DEEPSEEK_API_KEY=...            # or GEQ_LLM_API_KEY, or OPENAI_API_KEY
export GEQ_LLM_MODEL=deepseek-v4-pro   # optional; deepseek-flash is the cheaper option
export GEQ_LLM_BASE_URL=https://api.deepseek.com   # optional
geq models                             # what the key can actually use
geq run --extractor llm
```

`geq models` exists because a hard-coded model name is a guess about someone
else's catalogue. A model that has been retired fails as an HTTP 400 that reads
like a prompt problem, and the guess is cheap to check.

Two rules make a model's output auditable rather than merely plausible:

1. The model is asked for a **verbatim quote**, never for character offsets.
   Models are unreliable at counting characters and reliable at copying text, so
   the quote is mapped back to a span in-process. Model-generated offsets are a
   fabricated citation waiting to happen.
2. Its `value` may only be what appears inside its own quote. A response that
   breaks that contract is downgraded to uncited and flagged in the notes rather
   than quietly accepted.

Nothing in the offline suite touches the network: the contract checks
(`quote_must_contain_value`, `locate_quote`) are pure functions with their own
tests.

## Layout

```
src/geq/
  models.py        value objects; the metrics live with their definitions
  normalize.py     canonicalisation: money, percent, dates, text
  verify.py        evidence checks and the arithmetic checks
  scoring.py       cells into precision, recall, grounding and grounded recall
  regression.py    the gate between two runs
  corpus.py        golden-set loading and validation
  sqlqa.py         data-quality checks loaded from sql/checks/*.sql
  api.py           HTTP surface; failures return 409, not a 200
  web.py           the working-paper page: the same numbers, for a non-technical reader
  static/fonts/    self-hosted Libre Franklin and Courier Prime (OFL)
  extractors/      rules v1/v2/v3, the DeepSeek path, the adversarial controls
data/golden/       six labelled statements and the field spec
sql/               schema, seed with planted defects, six checks
tools/             the corpus generator
docs/              production-readiness checklist, prompt changelog
```

## Running it

```bash
make install     # venv + dev deps (uv)
make ci          # lint, 343 tests, all five extractors, the gate, the SQL checks
make report      # write reports/*.json and *.md for every extractor
make compare     # regression-gate v3 against v2; non-zero exit on any regression
make serve       # http://127.0.0.1:8000
```

`make sqlqa` prints the data-quality report. The seeded fixture is *supposed* to
be dirty, so it exits 0; pass `--expect-clean` against a real database to make
findings fail the command.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | The working-paper page |
| `GET /assets/paper.js` | The page's only script, served from this origin |
| `GET /fonts/{name}` | Self-hosted faces, from an allow-list |
| `GET /health` | Corpus, available extractors, whether model credentials are present |
| `GET /fields` | The field spec |
| `POST /extract` | One document, every claim with its citation and evidence status |
| `POST /qa/run` | Score an extractor over the corpus |
| `POST /qa/compare` | Regression gate; **409** when it fails |

A repeated identical request returns a byte-identical response, and that is
asserted in the suite rather than assumed. `postman/collection.json` exercises
all six.

## Known limitations

Tracked as failing-expectation tests in `tests/test_normalize.py` rather than as
comments, so a fix has to be a deliberate decision.

- **European decimal commas are not parsed.** `13.504,50` reads as `13.504`.
  Nothing in this corpus uses comma decimals; the test is the tripwire.
- **A figure followed by trailing prose on the same line is not treated as a data
  row.** `Net income 3,415,000 (restated)` fails to parse, which produces a miss
  rather than a wrong value. Safe, but it is a miss.
- **Rounding is not absorbed.** A statement "in millions" rounded to the nearest
  million cannot be scored exactly, and the corpus uses values that are exact at
  their stated scale instead of quietly baking in a tolerance. Where a tolerance
  is wanted, `check_balance_sheet` takes one explicitly.
- **One is not zero.** A grounding rate of 100% means every citation contained
  its value. It does not mean the value is in the right *context*: a figure
  quoted from a note rather than from the statement it claims to summarise would
  still pass. Catching that needs the note-versus-statement distinction, which is
  a field-level rule this harness does not yet model.
- **The corpus is small and synthetic.** 48 cells proves the machinery, not the
  extractor. The failure modes are chosen, not sampled.

## What I would do next

1. Grow the corpus from real (redacted) statements, weighted towards the layouts
   that actually break: rotated scans, footnoted tables, restated comparatives.
2. Add a **cross-field consistency** dimension: values that are individually
   grounded but mutually impossible, which is where the note-versus-statement
   gap above gets caught.
3. Record every run and chart grounded recall per revision, so a prompt change
   that helps one document and hurts another is visible before it merges.
4. Add an ambiguity check on citations: a quote that occurs more than once in a
   document is currently flagged in the notes, and should probably be a scored
   failure.

## Licence

MIT.
