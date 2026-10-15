"""SQL data-quality checks.

The second half of the JD: "validate data at scale through row counts, duplicate
detection, orphaned records, referential integrity, and reconciliation of totals".

Each check is a SQL file under ``sql/checks/`` that returns the offending rows
and nothing else. Keeping them as files rather than as strings has one practical
benefit: a reviewer can paste a check straight into a database client and look at
the rows it is complaining about, without reading any Python.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Check",
    "DataQualityReport",
    "Finding",
    "load_checks",
    "load_database",
    "run_checks",
]

_HEADER_RE = re.compile(r"^--\s*(check|description):\s*(?P<value>.+)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Check:
    """One named query and the reason it exists."""

    name: str
    description: str
    sql: str
    path: Path

    def run(self, connection: sqlite3.Connection) -> tuple[dict[str, Any], ...]:
        cursor = connection.execute(self.sql)
        columns = [column[0] for column in cursor.description or ()]
        return tuple(dict(zip(columns, row, strict=True)) for row in cursor.fetchall())


@dataclass(frozen=True, slots=True)
class Finding:
    """The rows one check returned."""

    name: str
    description: str
    rows: tuple[dict[str, Any], ...]

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def passed(self) -> bool:
        return not self.rows

    def summary(self) -> str:
        if self.passed:
            return "ok"
        return (
            f"{self.count} row(s): {self.rows[0]}"
            if self.count == 1
            else f"{self.count} rows, first: {self.rows[0]}"
        )


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    findings: tuple[Finding, ...]

    @property
    def failed(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if not f.passed)

    @property
    def passed(self) -> bool:
        return not self.failed

    @property
    def total_rows(self) -> int:
        return sum(f.count for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks_run": len(self.findings),
            "checks_failed": len(self.failed),
            "findings": [
                {
                    "name": f.name,
                    "description": f.description,
                    "count": f.count,
                    "rows": list(f.rows),
                }
                for f in self.findings
            ],
        }

    def render_markdown(self) -> str:
        lines = [
            "# SQL data-quality checks",
            "",
            f"{len(self.findings) - len(self.failed)}/{len(self.findings)} checks clean, "
            f"{self.total_rows} offending row(s).",
            "",
            "| check | rows | description |",
            "| --- | ---: | --- |",
        ]
        for finding in self.findings:
            mark = "ok" if finding.passed else f"**{finding.count}**"
            lines.append(f"| `{finding.name}` | {mark} | {finding.description} |")
        for finding in self.failed:
            lines.append("")
            lines.append(f"## `{finding.name}`")
            lines.append("")
            lines.append(finding.description)
            lines.append("")
            headers = list(finding.rows[0])
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in finding.rows:
                lines.append("| " + " | ".join(str(row.get(h)) for h in headers) + " |")
        return "\n".join(lines) + "\n"


def load_checks(directory: str | Path) -> tuple[Check, ...]:
    """Load every ``*.sql`` check in ``directory``, ordered by filename."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"no checks directory at {directory}")

    checks: list[Check] = []
    for path in sorted(directory.glob("*.sql")):
        text = path.read_text(encoding="utf-8")
        headers = {m.group(1): m.group("value").strip() for m in _HEADER_RE.finditer(text)}
        if "check" not in headers:
            raise ValueError(f"{path} is missing a '-- check: <name>' header")
        checks.append(
            Check(
                name=headers["check"],
                description=headers.get("description", ""),
                sql=text,
                path=path,
            )
        )
    if not checks:
        raise ValueError(f"no checks found in {directory}")
    names = [c.name for c in checks]
    if len(set(names)) != len(names):
        raise ValueError(
            f"duplicate check names: {sorted({n for n in names if names.count(n) > 1})}"
        )
    return tuple(checks)


def load_database(
    schema_path: str | Path,
    seed_path: str | Path | None = None,
) -> sqlite3.Connection:
    """Build an in-memory database from the schema, then the seed.

    An empty ``seed_path`` means "schema only", which is what a caller wants
    when checking a real database where the schema already exists.

    Foreign key enforcement is switched **off** while the seed loads, which is a
    deliberate reproduction of how the legacy import ran. The point of the
    ``foreign_key_violations`` check is to find rows that should not exist and do.
    """
    connection = sqlite3.connect(":memory:")
    connection.isolation_level = None  # autocommit: PRAGMA foreign_keys takes effect
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.executescript(Path(schema_path).read_text(encoding="utf-8"))
    if seed_path:
        connection.executescript(Path(seed_path).read_text(encoding="utf-8"))
    return connection


def run_checks(
    connection: sqlite3.Connection,
    checks: Iterable[Check] | None = None,
    *,
    checks_dir: str | Path | None = None,
) -> DataQualityReport:
    """Run every check and collect its findings."""
    resolved = tuple(checks) if checks is not None else load_checks(checks_dir or _default_dir())
    findings = tuple(
        Finding(name=check.name, description=check.description, rows=check.run(connection))
        for check in resolved
    )
    return DataQualityReport(findings=findings)


def _default_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "sql" / "checks"
