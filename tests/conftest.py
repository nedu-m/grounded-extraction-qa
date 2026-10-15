"""Shared fixtures.

Expensive things (parsing the corpus, running all three revisions) are built once
per session. Everything else is a pure function and gets no fixture at all.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from geq.corpus import Corpus, load_corpus
from geq.extractors.rules import REVISIONS
from geq.runner import RunResult, run_extractor
from geq.sqlqa import load_checks, load_database, run_checks

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "golden"
SQL_DIR = REPO_ROOT / "sql"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return load_corpus(CORPUS_DIR)


@pytest.fixture(scope="session")
def runs(corpus: Corpus) -> dict[str, RunResult]:
    return {revision: run_extractor(corpus, f"rules:{revision}") for revision in REVISIONS}


@pytest.fixture(scope="session")
def v1_run(runs: dict[str, RunResult]) -> RunResult:
    return runs["v1"]


@pytest.fixture(scope="session")
def v2_run(runs: dict[str, RunResult]) -> RunResult:
    return runs["v2"]


@pytest.fixture(scope="session")
def v3_run(runs: dict[str, RunResult]) -> RunResult:
    return runs["v3"]


@pytest.fixture(scope="session")
def checks():
    return load_checks(SQL_DIR / "checks")


@pytest.fixture
def dirty_database() -> sqlite3.Connection:
    """The fixture database, with every planted defect present."""
    return load_database(SQL_DIR / "schema.sql", SQL_DIR / "seed.sql")


@pytest.fixture
def clean_database() -> sqlite3.Connection:
    """The negative control: the same checks against data with nothing wrong.

    A check that passes on both is not a check, so this fixture exists to make
    sure every planted defect is the *only* reason its check fires.
    """
    connection = load_database(SQL_DIR / "schema.sql")
    connection.executescript(
        """
        INSERT INTO entities (id, name, brand_name) VALUES
            ('e1', 'Northgate Clinical Group', 'Northgate Clinical Group'),
            ('e2', 'Cascade Specialty Holdings', 'Cascade Specialty');

        INSERT INTO data_room_requirements (entity_id, category) VALUES
            ('e1', 'financial_statements'),
            ('e2', 'financial_statements');

        INSERT INTO documents (id, entity_id, category, sha256, received_at) VALUES
            ('d1', 'e1', 'financial_statements', 'aa11', '2026-02-11T09:00:00Z'),
            ('d2', 'e2', 'financial_statements', 'bb22', '2026-03-02T09:00:00Z');

        INSERT INTO extracted_fields (document_id, field, value, page) VALUES
            ('d1', 'total_assets',      '13504500', 2),
            ('d1', 'total_liabilities', '5320400',  2),
            ('d1', 'total_equity',      '8184100',  2),
            ('d2', 'total_assets',      '13000000', 2),
            ('d2', 'total_liabilities', '6500000',  2),
            ('d2', 'total_equity',      '6500000',  2);

        INSERT INTO brand_assets (entity_id, brand_name, asset_type, status) VALUES
            ('e1', 'Northgate Clinical Group', 'logo', 'approved'),
            ('e2', 'Cascade Specialty',        'logo', 'approved');
        """
    )
    return connection


@pytest.fixture
def dirty_report(dirty_database: sqlite3.Connection, checks):
    return run_checks(dirty_database, checks)
