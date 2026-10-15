"""SQL data-quality checks.

Each planted defect gets its own test asserting the *count*, not just that the
check fired. A check that returns every row is as useless as one that returns
none, and only a count tells them apart.

The clean-database tests are the negative control. Without them, a check that
fires unconditionally would look like a working check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geq.sqlqa import (
    Check,
    DataQualityReport,
    Finding,
    load_checks,
    load_database,
    run_checks,
)

EXPECTED_CHECKS = {
    "orphaned_extracted_fields",
    "foreign_key_violations",
    "duplicate_documents",
    "missing_data_room_items",
    "balance_sheet_reconciliation",
    "brand_asset_mismatch",
}


class TestCheckLoading:
    def test_all_six_checks_load(self, checks):
        assert {c.name for c in checks} == EXPECTED_CHECKS
        assert len(checks) == 6

    def test_checks_are_ordered_by_filename(self, checks):
        assert [c.path.name for c in checks] == sorted(c.path.name for c in checks)

    def test_every_check_documents_itself(self, checks):
        for check in checks:
            assert len(check.description) > 30, check.name

    def test_a_check_without_a_header_is_rejected(self, tmp_path):
        (tmp_path / "bad.sql").write_text("SELECT 1;", encoding="utf-8")
        with pytest.raises(ValueError, match="missing a '-- check: <name>' header"):
            load_checks(tmp_path)

    def test_duplicate_check_names_are_rejected(self, tmp_path):
        body = "-- check: same\n-- description: duplicated\nSELECT 1;"
        (tmp_path / "a.sql").write_text(body, encoding="utf-8")
        (tmp_path / "b.sql").write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate check names"):
            load_checks(tmp_path)

    def test_an_empty_directory_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no checks found"):
            load_checks(tmp_path)

    def test_a_missing_directory_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checks(tmp_path / "nope")


class TestPlantedDefectsAreFound:
    def test_orphaned_extracted_fields(self, dirty_report):
        finding = _finding(dirty_report, "orphaned_extracted_fields")
        assert finding.count == 1
        assert finding.rows[0]["missing_document_id"] == "d999"

    def test_referential_integrity_by_the_database(self, dirty_report):
        finding = _finding(dirty_report, "foreign_key_violations")
        assert finding.count == 2
        parents = {row["parent"] for row in finding.rows}
        assert parents == {"documents", "entities"}

    def test_duplicate_documents(self, dirty_report):
        finding = _finding(dirty_report, "duplicate_documents")
        assert finding.count == 1
        row = finding.rows[0]
        assert row["entity_id"] == "e1"
        assert row["copies"] == 2
        assert row["document_ids"] == "d2, d3"

    def test_missing_data_room_items(self, dirty_report):
        finding = _finding(dirty_report, "missing_data_room_items")
        assert finding.count == 1
        assert finding.rows[0]["missing_category"] == "tax_returns"

    def test_balance_sheet_reconciliation(self, dirty_report):
        finding = _finding(dirty_report, "balance_sheet_reconciliation")
        assert finding.count == 1
        row = finding.rows[0]
        assert row["entity_id"] == "e1"
        assert row["out_by"] == 100_000

    def test_brand_asset_mismatch(self, dirty_report):
        finding = _finding(dirty_report, "brand_asset_mismatch")
        assert finding.count == 1
        row = finding.rows[0]
        assert row["asset_brand"] == "Cascade Specialty"
        assert row["owning_brand"] == "Northgate Clinical Group"

    def test_the_report_says_how_bad_it_is(self, dirty_report):
        assert dirty_report.passed is False
        assert len(dirty_report.failed) == 6
        assert dirty_report.total_rows == 7


class TestCleanDataIsClean:
    def test_every_check_passes_on_healthy_data(self, clean_database, checks):
        report = run_checks(clean_database, checks)
        assert report.passed, [f.name for f in report.failed]
        assert report.total_rows == 0
        assert len(report.findings) == 6

    def test_healthy_data_fails_only_when_the_defect_is_reintroduced(self, clean_database, checks):
        clean_database.execute(
            "INSERT INTO brand_assets (entity_id, brand_name, asset_type, status) "
            "VALUES ('e1', 'Cascade Specialty', 'ad_creative', 'draft')"
        )
        report = run_checks(clean_database, checks)
        assert not report.passed
        assert [f.name for f in report.failed] == ["brand_asset_mismatch"]


class TestDatabaseLoading:
    def test_the_seed_loads_with_enforcement_off(self, dirty_database):
        """A deliberate reproduction of the legacy import, not an oversight."""
        rows = dirty_database.execute("PRAGMA foreign_keys").fetchall()
        assert rows == [(0,)]

    def test_the_schema_alone_loads_with_no_rows(self, tmp_path):
        connection = load_database(Path("sql/schema.sql"))
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == {
            "entities",
            "documents",
            "data_room_requirements",
            "extracted_fields",
            "brand_assets",
        }
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone() == (0,)


class TestReportRendering:
    def test_markdown_lists_every_check(self, dirty_report):
        markdown = dirty_report.render_markdown()
        for name in EXPECTED_CHECKS:
            assert f"`{name}`" in markdown

    def test_markdown_shows_the_rows_it_is_complaining_about(self, dirty_report):
        markdown = dirty_report.render_markdown()
        assert "d999" in markdown
        assert "tax_returns" in markdown

    def test_a_clean_report_says_so(self, clean_database, checks):
        markdown = run_checks(clean_database, checks).render_markdown()
        assert "6/6 checks clean" in markdown
        assert "0 offending row(s)" in markdown

    def test_to_dict_is_serialisable(self, dirty_report):
        import json

        payload = dirty_report.to_dict()
        assert payload["passed"] is False
        assert payload["checks_run"] == 6
        assert payload["checks_failed"] == 6
        json.dumps(payload)

    def test_finding_summary_is_readable_for_one_and_many_rows(self):
        one = Finding(name="a", description="d", rows=({"x": 1},))
        many = Finding(name="a", description="d", rows=({"x": 1}, {"x": 2}))
        none = Finding(name="a", description="d", rows=())
        assert one.summary().startswith("1 row(s)")
        assert many.summary().startswith("2 rows")
        assert none.summary() == "ok"
        assert none.passed


def _finding(report: DataQualityReport, name: str) -> Finding:
    return next(f for f in report.findings if f.name == name)


def test_check_run_returns_named_columns(dirty_database):
    check = Check(
        name="probe", description="", sql="SELECT 1 AS one, 'two' AS two;", path=Path("x.sql")
    )
    rows = check.run(dirty_database)
    assert rows == ({"one": 1, "two": "two"},)
