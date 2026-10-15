# Production-ready checklist

The role this was built for asks for a *definition of done* that the organisation
adopts, applied to the first tool. This is that checklist, written so that it can
be applied to something other than this repository.

Each line is a gate, not an aspiration. A gate is either satisfied with a named
artefact you can open, or it is not satisfied, and "we tested it" is not an
artefact. The last section is this repository checked against itself.

## How to use it

Copy the table, keep the two columns, and fill the evidence column with a path,
a URL, or a command. If the evidence cell needs a sentence to explain why a
failure is acceptable, the gate is not passed; it is waived, and waivers belong
in the same document with a name and a date against them.

## The checklist

### Correctness

| # | Gate | Evidence required |
| --- | --- | --- |
| 1 | A fixed set of examples exists, with expected outputs written by a human | Path to the labelled set |
| 2 | Every field the tool claims to produce has at least one labelled example | Coverage report against the field spec |
| 3 | The labelled set is validated in CI: no unknown fields, no unlabelled required field, no duplicate document | Test name that fails on each |
| 4 | Accuracy is reported per field, not only as an average | Per-field table |
| 5 | The labelled set is versioned in the repository and its changes are reviewed | Corpus change fails the gate |

### Traceability

| # | Gate | Evidence required |
| --- | --- | --- |
| 6 | Every produced value cites the source it came from | Citation rate |
| 7 | The citation is checked, not trusted: the cited text must contain the value | Grounding rate |
| 8 | A citation wider than a readable excerpt is rejected | Check with a limit and a test |
| 9 | Values that arrive without a citation are reported as defects, not as gaps | Failing-cell report |
| 10 | The tool cannot score well on accuracy while scoring zero on traceability | Grounded recall is the headline metric |

### Arithmetic and domain rules

| # | Gate | Evidence required |
| --- | --- | --- |
| 11 | Values that must reconcile are reconciled, not just extracted | Named checks |
| 12 | A document that looks complete and does not foot is caught | Planted-defect test |
| 13 | Document-level units are honoured or the output is failed | Scale test, thousands and millions |
| 14 | Tolerant comparisons are explicit and bounded, never silent | Tolerance parameter with a test |

### Regression safety

| # | Gate | Evidence required |
| --- | --- | --- |
| 15 | Two revisions can be compared cell by cell | Compare command |
| 16 | A cell that gets worse fails the build | Gate test, both directions |
| 17 | Losing a citation fails the build even when the value stays right | Grounding-regression test |
| 18 | A change to the expected outputs fails the build instead of moving the score | Corpus-change test |
| 19 | The gate's own failure modes are tested | Tests that invert the gate |

### Data quality

| # | Gate | Evidence required |
| --- | --- | --- |
| 20 | Referential integrity is checked by the database, not by hand | Integrity check |
| 21 | Duplicate records are detected, with the criterion stated | Duplicate check |
| 22 | Orphaned records are detected | Orphan check |
| 23 | Totals reconcile across two sources that disagree | Reconciliation check |
| 24 | Every check has a negative control: it returns nothing on clean data | Clean-fixture test |

### Interfaces

| # | Gate | Evidence required |
| --- | --- | --- |
| 25 | Failures are not returned as success | Status-code test |
| 26 | A repeated identical request returns an identical response | Idempotency test |
| 27 | Error responses name what was wrong | 404 and 422 tests |
| 28 | The interface is exercised by a collection a reviewer can run | Postman / OpenAPI link |

### Operations

| # | Gate | Evidence required |
| --- | --- | --- |
| 29 | No secret is committed; configuration comes from the environment | `.gitignore` plus a secret scan |
| 30 | Credentials absent means a clear, actionable message, never a stack trace | Test for the no-credential path |
| 31 | The health endpoint states which optional capabilities are configured | Health payload |
| 32 | Every check runs in CI on every change | Workflow file |
| 33 | A reviewer can reproduce the headline number from the README in one command | `make ci` |

## This repository against the checklist

| # | Status | Evidence |
| --- | --- | --- |
| 1 | pass | `data/golden/cases/*.json`, six statements, 48 labelled cells |
| 2 | pass | `tests/test_golden_corpus_integrity.py::test_labels_cover_every_required_field_in_every_case` |
| 3 | pass | `tests/test_corpus.py::TestValidation`, four distinct failure modes |
| 4 | pass | per-field table in `geq run` and in `GET /qa/run` |
| 5 | pass | `test_only_the_named_case_is_unbalanced`, `test_every_case_matches_byte_for_byte` |
| 6 | pass | citation rate; `rules:v1` reports 25.0% |
| 7 | pass | `geq.verify.check_evidence`; grounding rate |
| 8 | pass | `MAX_EVIDENCE_CHARS = 280`, `test_a_whole_document_span_is_not_a_citation` |
| 9 | pass | `EvidenceStatus.UNCITED`; `adversarial:uncited` scores 0% grounded recall |
| 10 | pass | `adversarial:uncited` reports 100% recall and 0% grounded recall in the same row |
| 11 | pass | `check_balance_sheet`, `check_totals_footing` |
| 12 | pass | `case_04_northgate_2024_unbalanced`, `test_the_unbalanced_case_still_has_agreeing_printed_totals` |
| 13 | pass | `Claim.scale`; `rules:v2` fails the millions statement and passes the thousands one |
| 14 | pass | `tolerance` parameter, `test_a_rounding_tolerance_can_be_granted_explicitly` |
| 15 | pass | `geq compare`, `compare_runs` |
| 16 | pass | `test_v3_to_v2_fails_the_gate` |
| 17 | pass | `cell_state`, independent value and grounding regression detection |
| 18 | pass | `TestCorpusChangeDetection` |
| 19 | pass | `TestRealRevisionComparison` asserts the gate in both directions |
| 20 | pass | `sql/checks/02_foreign_key_violations.sql` |
| 21 | pass | `03_duplicate_documents.sql`, criterion is `(entity_id, sha256)` |
| 22 | pass | `01_orphaned_extracted_fields.sql` |
| 23 | pass | `05_balance_sheet_reconciliation.sql`, out by 100,000 by design |
| 24 | pass | `tests/test_sqlqa.py::TestCleanDataIsClean` |
| 25 | pass | `POST /qa/compare` returns 409, `test_a_regression_returns_409_with_the_reasons` |
| 26 | pass | `TestRepeatedIdenticalRequests`, three cases |
| 27 | pass | 404 on unknown document and unknown extractor, 422 on a missing body |
| 28 | pass | `postman/collection.json`, six requests |
| 29 | pass | `.gitignore` covers `.env*`; the deploy script writes the key with `0600` at runtime |
| 30 | pass | `test_a_missing_key_degrades_to_an_explained_no_op` |
| 31 | pass | `GET /health` reports provider, model and whether a key is configured, never the key |
| 32 | pass | `.github/workflows/ci.yml` runs lint, tests, all five extractors, the gate and the SQL checks |
| 33 | pass | `make ci` |

Two gates are **waived**, and they are the honest gaps:

| # | Gate | Waiver |
| --- | --- | --- |
| 13 | Document-level units | Waived for `adversarial:fabricating` only, which is a control rather than a candidate extractor and is expected to fail. |
| 14 | Bounded tolerances | Waived for the millions statement, which is generated with values exact at its stated scale rather than scored with a rounding tolerance. Recorded in the README's limitations, not hidden here. |
