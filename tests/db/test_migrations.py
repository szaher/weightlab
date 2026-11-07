"""Phase 1 Task 1: migration bootstrap, fail-closed versions, backups and integrity (doc 10 s3/s7, doc 18 s7)."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

import weightlab.db.schema as mig
from weightlab.db import Database, integrity
from weightlab.db.schema import DatabaseVersionError, Migration, connect, migrate, migrations

DOC10_TABLES = {"schema_migrations", "projects", "models", "model_files", "tensors", "adapters", "comparisons",
                "bookmarks", "jobs", "experiments", "exports"}  # fmt: skip
ADDED_TABLES = {"comparison_entries", "app_settings"}


def tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def file_digest(path: Path) -> str:
    return hashlib.sha256(b"".join(p.read_bytes() for p in sorted(path.parent.glob(path.name + "*")))).hexdigest()


def test_fresh_database_gets_latest_schema(tmp_path: Path) -> None:
    db = Database(tmp_path / "w.sqlite3")
    assert db.migration.from_version == 0
    assert db.migration.applied == tuple(range(1, mig.latest_version() + 1))
    assert db.migration.backup is None  # nothing to back up
    with db.session() as c:
        assert tables(c) == DOC10_TABLES | ADDED_TABLES
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert integrity(c).ok
        row = c.execute("SELECT version, checksum FROM schema_migrations").fetchone()
        assert row["checksum"] == migrations()[0].checksum


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    Database(tmp_path / "w.sqlite3")
    again = Database(tmp_path / "w.sqlite3")
    assert again.migration.applied == ()
    assert not (tmp_path / "backups").exists()  # no backup without pending migrations


def test_doc10_columns_are_all_present(tmp_path: Path) -> None:
    """Every column of the suggested schema exists (additions are allowed, removals are not)."""
    doc10 = {
        "projects": {"id", "name", "created_at", "updated_at", "settings_json"},
        "models": {"id", "project_id", "display_name", "selected_root", "entry_path", "format", "architecture_hint",
                   "lineage_json", "quick_fingerprint_json", "sha256", "status", "imported_at"},
        "model_files": {"id", "model_id", "relpath", "size_bytes", "mtime_ns", "file_id", "sha256",
                        "fingerprint_method", "verified_at"},
        "tensors": {"id", "model_id", "raw_key", "rank", "shape_json", "dtype", "shard_relpath", "byte_length", "role",
                    "layer_index"},
        "adapters": {"id", "project_id", "base_model_id", "config_json", "adapter_path", "mapping_json", "status"},
        "comparisons": {"id", "project_id", "model_a_id", "model_b_id", "compatibility_json", "created_at"},
        "bookmarks": {"id", "project_id", "view_json", "note", "created_at"},
        "jobs": {"id", "project_id", "kind", "spec_json", "state", "progress", "result_json", "error_json",
                 "created_at", "started_at", "finished_at"},
        "experiments": {"id", "project_id", "spec_json", "status", "manifest_path", "created_at"},
        "exports": {"id", "project_id", "format", "export_path", "manifest_json", "created_at"},
    }  # fmt: skip
    db = Database(tmp_path / "w.sqlite3")
    with db.session() as c:
        for table, cols in doc10.items():
            have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
            assert cols <= have, (table, cols - have)
        idx = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert {"idx_tensors_model_layer", "idx_jobs_project_state"} <= idx


def test_newer_database_is_refused_unmodified(tmp_path: Path) -> None:
    path = tmp_path / "w.sqlite3"
    Database(path)
    c = connect(path)
    c.execute("INSERT INTO schema_migrations VALUES (99, '2030-01-01T00:00:00Z', 'x')")
    c.close()
    before = file_digest(path)
    with pytest.raises(DatabaseVersionError, match="newer"):
        migrate(path)
    assert file_digest(path) == before


def test_edited_applied_migration_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "w.sqlite3"
    Database(path)
    c = connect(path)
    c.execute("UPDATE schema_migrations SET checksum = 'tampered'")
    c.close()
    before = file_digest(path)
    with pytest.raises(DatabaseVersionError, match="checksum"):
        migrate(path)
    assert file_digest(path) == before


@pytest.mark.parametrize("journal", ["delete", "truncate", "persist", "wal"])
def test_foreign_database_is_refused_byte_for_byte_unmodified(tmp_path: Path, journal: str) -> None:
    """Regression (PR #3 review): connect() set journal_mode=WAL before the ownership check, which rewrites
    the header of a rollback-journal database (bytes 18-19), so a refused foreign file was still modified."""
    path = tmp_path / "other.sqlite3"
    c = sqlite3.connect(path)
    c.execute(f"PRAGMA journal_mode = {journal}")
    c.execute("CREATE TABLE notes (x TEXT)")
    c.execute("INSERT INTO notes VALUES ('keep me')")
    c.commit()
    c.close()
    before, header, files = file_digest(path), path.read_bytes()[:100], sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(DatabaseVersionError, match="not a WeightLab database"):
        migrate(path)
    with pytest.raises(DatabaseVersionError):
        Database(path)
    assert path.read_bytes()[:100] == header
    assert file_digest(path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == files  # no -wal/-shm created


def test_non_sqlite_file_is_refused_unmodified(tmp_path: Path) -> None:
    path = tmp_path / "weightlab.sqlite3"
    path.write_bytes(b"not a database, just someone's notes\n" * 200)
    before = file_digest(path)
    with pytest.raises(DatabaseVersionError, match="not a readable SQLite database"):
        migrate(path)
    assert file_digest(path) == before


def _with_extra(monkeypatch: pytest.MonkeyPatch, sql: str) -> None:
    extra = Migration(2, "0002_test.sql", sql)
    base = migrations()
    monkeypatch.setattr(mig, "migrations", lambda: [*base, extra])


def test_upgrade_takes_a_backup_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "w.sqlite3"
    db = Database(path)
    with db.session() as c:
        c.execute("INSERT INTO projects VALUES ('prj_1', 'keep me', 't', 't', '{}')")
    _with_extra(monkeypatch, "CREATE TABLE added_later (id TEXT PRIMARY KEY);")
    result = migrate(path)
    assert result.applied == (2,)
    assert result.backup is not None
    assert result.backup.exists()
    old = sqlite3.connect(result.backup)
    assert old.execute("SELECT name FROM projects").fetchall() == [("keep me",)]
    assert old.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 1
    old.close()
    c = connect(path)
    assert "added_later" in tables(c)
    c.close()


def test_failed_migration_rolls_back_completely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "w.sqlite3"
    Database(path)
    _with_extra(monkeypatch, "CREATE TABLE half_done (id TEXT);\nINSERT INTO no_such_table VALUES (1);")
    with pytest.raises(sqlite3.OperationalError):
        migrate(path)
    c = connect(path)
    assert "half_done" not in tables(c)  # the first statement was rolled back with the failing one
    assert c.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 1
    c.close()


def test_statement_splitter_respects_strings_and_comments() -> None:
    sql = "-- header; not a statement\nCREATE TABLE a (x TEXT DEFAULT ';');\n\n-- c\nINSERT INTO a VALUES ('x;y');\n"
    assert mig._statements(sql) == ["CREATE TABLE a (x TEXT DEFAULT ';');", "INSERT INTO a VALUES ('x;y');"]
    with pytest.raises(RuntimeError, match="incomplete"):
        mig._statements("CREATE TABLE b (x")


def test_integrity_reports_foreign_key_violations(tmp_path: Path) -> None:
    path = tmp_path / "w.sqlite3"
    Database(path)
    c = connect(path)
    c.execute("PRAGMA foreign_keys = OFF")
    c.execute("INSERT INTO bookmarks VALUES ('bmk_1', 'prj_missing', '{}', NULL, 't')")
    c.execute("PRAGMA foreign_keys = ON")
    report = integrity(c)
    c.close()
    assert report.foreign_key_violations == 1
    assert report.ok is False
