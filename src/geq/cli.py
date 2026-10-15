"""Command line interface.

Every command prints Markdown and writes it next to the JSON, so a CI log, a
report file and a pull request comment all show the same thing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .corpus import load_corpus
from .normalize import canonical
from .regression import RegressionPolicy, compare_runs
from .report import render_comparison, render_scorecard, render_summary
from .runner import load_run, run_extractor
from .scoring import scaled_claim_value
from .sqlqa import load_checks, load_database, run_checks
from .verify import check_evidence

__all__ = ["build_parser", "main"]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _cmd_run(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    result = run_extractor(corpus, args.extractor)

    if args.out:
        base = Path(args.out)
        _write(base.with_suffix(".json"), json.dumps(result.to_dict(), indent=2) + "\n")
        _write(base.with_suffix(".md"), render_scorecard(result) + "\n")

    print(render_scorecard(result, failure_limit=args.failure_limit))
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    directory = Path(args.reports)
    runs = [load_run(path) for path in sorted(directory.glob("*.json"))]
    if not runs:
        print(f"no run JSON found in {directory}", file=sys.stderr)
        return 2
    markdown = render_summary(runs)
    if args.out:
        _write(Path(args.out), markdown + "\n")
    print(markdown)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    baseline = load_run(args.baseline)
    candidate = load_run(args.candidate)
    report = compare_runs(
        baseline,
        candidate,
        policy=RegressionPolicy(max_regressions=args.max_regressions),
    )
    markdown = render_comparison(report)
    if args.out:
        _write(Path(args.out), markdown + "\n")
    print(markdown)
    return 0 if report.passed else 1


def _cmd_sqlqa(args: argparse.Namespace) -> int:
    connection = load_database(args.schema, args.seed or None)
    report = run_checks(connection, load_checks(args.checks))
    if args.out:
        _write(Path(args.out), report.render_markdown())
    print(report.render_markdown())
    if args.expect_clean and not report.passed:
        return 1
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    case = corpus.case(args.doc)
    from .extractors import build_extractor

    extractor = build_extractor(args.extractor, corpus.specs)
    extraction = extractor.extract(case)

    print(f"# {extractor.name} on `{case.doc_id}`")
    print()
    print("| field | claimed | span | evidence | gold | right |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, spec in corpus.specs.items():
        claim = extraction.claim(name)
        gold = case.label(name)
        if claim is None:
            print(f"| `{name}` | _missing_ | | | `{gold}` | no |")
            continue
        check = check_evidence(case.text, claim, spec)
        span = f"{claim.span[0]}:{claim.span[1]}" if claim.span else "-"
        detail = check.detail if len(check.detail) < 90 else check.detail[:87] + "..."
        # The same comparison the scorer makes. Comparing raw strings here said
        # "no" for a correctly formatted value, which put the CLI's verdict in
        # conflict with the scorecard it was reporting on.
        right = (
            "yes" if scaled_claim_value(claim, spec) == canonical(gold or "", spec.kind) else "no"
        )
        print(
            f"| `{name}` | `{claim.raw_value}` | `{span}` | {check.status.value}: {detail} | "
            f"`{gold}` | {right} |"
        )
    for note in extraction.notes:
        print()
        print(f"> {note}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from .extractors.llm import LLMConfig, list_models

    config = LLMConfig.from_env()
    if config.model and args.model:
        config = LLMConfig(
            model=args.model,
            base_url=config.base_url,
            api_key=config.api_key,
            api_key_env=config.api_key_env,
        )
    if not config.configured:
        print(
            "no LLM API key is set. Export DEEPSEEK_API_KEY, then run this again.",
            file=sys.stderr,
        )
        return 2
    try:
        models = list_models(config)
    except RuntimeError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 1

    print(f"# Models available to the configured key ({config.base_url})")
    print()
    for name in models:
        marker = " <- configured" if name == config.model else ""
        print(f"- `{name}`{marker}")
    if config.model not in models:
        print()
        print(
            f"!! {config.model!r} is not in that list. Set GEQ_LLM_MODEL to one of "
            f"the names above; a request for an unavailable model fails as an HTTP "
            f"error that looks like a prompt problem."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geq",
        description="Golden-set QA harness for document extraction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="score an extractor against the golden corpus")
    run.add_argument("--corpus", default="data/golden")
    run.add_argument("--extractor", default="rules:v3")
    run.add_argument("--out", help="write <out>.json and <out>.md")
    run.add_argument("--failure-limit", type=int, default=12)
    run.set_defaults(func=_cmd_run)

    summary = sub.add_parser("summary", help="compare every run JSON in a directory")
    summary.add_argument("--reports", default="reports")
    summary.add_argument("--out")
    summary.set_defaults(func=_cmd_summary)

    compare = sub.add_parser("compare", help="regression-gate one run against another")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--out")
    compare.add_argument("--max-regressions", type=int, default=0)
    compare.set_defaults(func=_cmd_compare)

    sqlqa = sub.add_parser("sqlqa", help="run the SQL data-quality checks")
    sqlqa.add_argument("--schema", default="sql/schema.sql")
    sqlqa.add_argument("--seed", default="sql/seed.sql")
    sqlqa.add_argument("--checks", default="sql/checks")
    sqlqa.add_argument("--out")
    sqlqa.add_argument(
        "--expect-clean",
        action="store_true",
        help="exit non-zero if any check returns rows (for real data, not the fixture)",
    )
    sqlqa.set_defaults(func=_cmd_sqlqa)

    models = sub.add_parser("models", help="list the models the configured LLM key can use")
    models.add_argument("--model", help="check a specific name against the catalogue")
    models.set_defaults(func=_cmd_models)

    extract = sub.add_parser("extract", help="show one document's claims and citations")
    extract.add_argument("--corpus", default="data/golden")
    extract.add_argument("--doc", required=True)
    extract.add_argument("--extractor", default="rules:v3")
    extract.set_defaults(func=_cmd_extract)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
