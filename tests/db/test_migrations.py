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
ADDED_TABLES = {
    "comparison_entries",
    "app_settings",
    "comparison_reports",
    "comparison_report_seq",
    "sqlite_sequence",
}  # 0003: ADR-020 (AUTOINCREMENT keeps sqlite_sequence)


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
    base = migrations()
    extra = Migration(len(base) + 1, f"{len(base) + 1:04d}_test.sql", sql)
    monkeypatch.setattr(mig, "migrations", lambda: [*base, extra])


def test_upgrade_takes_a_backup_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "w.sqlite3"
    db = Database(path)
    with db.session() as c:
        c.execute("INSERT INTO projects VALUES ('prj_1', 'keep me', 't', 't', '{}')")
    _with_extra(monkeypatch, "CREATE TABLE added_later (id TEXT PRIMARY KEY);")
    result = migrate(path)
    assert result.applied == (mig.latest_version(),)  # the extra one
    assert result.backup is not None
    assert result.backup.exists()
    old = sqlite3.connect(result.backup)
    assert old.execute("SELECT name FROM projects").fetchall() == [("keep me",)]
    assert (
        old.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == mig.latest_version() - 1
    )  # before the extra one
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
    assert (
        c.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == mig.latest_version() - 1
    )  # before the extra one
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


def _hot_wal_database(path: Path, statements: list[str]) -> None:
    """A WAL database whose committed transactions are still only in a non-empty -wal (as after a crash): no
    checkpoint on close, and the -shm removed."""
    c = sqlite3.connect(path, isolation_level=None)
    c.execute("PRAGMA journal_mode = WAL")
    c.execute("PRAGMA wal_autocheckpoint = 0")
    c.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)
    for s in statements:
        c.execute(s)
    c.close()
    Path(f"{path}-shm").unlink(missing_ok=True)
    assert Path(f"{path}-wal").stat().st_size > 0


def _snapshot(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


def test_foreign_database_with_hot_wal_is_refused_without_sidecar_files(tmp_path: Path) -> None:
    """PR #3 review, round 2: the hot-WAL branch opened the original with mode=ro, and SQLite created a -shm
    next to it. The inspection must leave the directory exactly as it was: same files, same bytes."""
    path = tmp_path / "other.sqlite3"
    _hot_wal_database(path, ["CREATE TABLE notes (x TEXT)", "INSERT INTO notes VALUES ('only in the WAL')"])
    before = _snapshot(tmp_path)
    assert set(before) == {"other.sqlite3", "other.sqlite3-wal"}
    with pytest.raises(DatabaseVersionError, match="not a WeightLab database"):
        migrate(path)
    with pytest.raises(DatabaseVersionError):
        Database(path)
    assert _snapshot(tmp_path) == before


def test_hot_wal_contents_are_inspected_not_ignored(tmp_path: Path) -> None:
    """A WeightLab database whose newer-version row exists only in the WAL: an inspection that ignored the WAL
    (immutable) would see version 1 and let migrate() open the original. It must be refused, untouched."""
    path = tmp_path / "w.sqlite3"
    Database(path)
    _hot_wal_database(path, ["INSERT INTO schema_migrations VALUES (99, '2030-01-01T00:00:00Z', 'x')"])
    before = _snapshot(tmp_path)
    with pytest.raises(DatabaseVersionError, match="newer"):
        migrate(path)
    assert _snapshot(tmp_path) == before


def test_preflight_refuses_a_database_that_keeps_changing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The snapshot copy is retried when the files change while being copied, and refused if they never settle."""
    path = tmp_path / "w.sqlite3"
    _hot_wal_database(path, ["CREATE TABLE notes (x TEXT)"])
    real = mig._copy_snapshot
    calls = 0

    def busy(src: Path, wal: Path, dest: Path) -> None:
        nonlocal calls
        calls += 1
        real(src, wal, dest)
        with wal.open("ab") as f:  # a concurrent writer appends to the WAL during every copy
            f.write(b"\0")

    monkeypatch.setattr(mig, "_copy_snapshot", busy)
    with pytest.raises(DatabaseVersionError, match="changing"):
        migrate(path)
    assert calls == mig.SNAPSHOT_ATTEMPTS


def test_alpha1_database_upgrades_to_comparison_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 0.1.0a1 database (schema version 2) with a comparison gains migration 0003 without losing data; the
    legacy comparison is kept as is (record_version absent), so report jobs refuse it (ADR-020 item 8)."""
    path = tmp_path / "alpha1.sqlite3"
    alpha1 = [m for m in migrations() if m.version <= 2]
    with monkeypatch.context() as m:
        m.setattr(mig, "migrations", lambda: alpha1)
        db = Database(path)
        with db.session() as c:
            c.execute("INSERT INTO projects VALUES ('prj_1', 'keep me', 't', 't', '{}')")
            for mid in ("mod_a", "mod_b"):
                c.execute("INSERT INTO models (id, project_id, display_name, selected_root, entry_path, format, "
                          "lineage_json, quick_fingerprint_json, status, imported_at) VALUES (?, 'prj_1', 'm', "
                          "'/r', '/r/m', 'safetensors', '{}', '{}', 'ready', 't')", (mid,))  # fmt: skip
            c.execute("INSERT INTO comparisons VALUES ('cmp_1', 'prj_1', 'mod_a', 'mod_b', "
                      "'{\"lineage\": {}, \"counts\": {}, \"policy\": \"strict\"}', 't')")  # fmt: skip
        assert "comparison_reports" not in tables(connect(path))
    result = migrate(path)
    assert result.from_version == 2
    assert result.applied == (3,)
    assert result.backup is not None
    assert result.backup.exists()
    c = connect(path)
    assert {"comparison_reports", "comparison_report_seq"} <= tables(c)
    assert c.execute("SELECT compatibility_json FROM comparisons").fetchone()[0].startswith('{"lineage"')
    assert integrity(c).ok
    c.close()
