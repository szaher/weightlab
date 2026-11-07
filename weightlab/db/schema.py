"""SQLite connection policy and the migration bootstrap (doc 10 sections 3 and 7, doc 18 section 7).

* Every connection: `foreign_keys=ON`, `busy_timeout`, `synchronous=FULL`; file databases use WAL.
* Migrations are numbered SQL files in `weightlab/db/migrations/`, applied in order, each in one
  transaction together with its `schema_migrations` row (version, time, SHA-256 of the file).
* Fail closed: a database written by a newer WeightLab (version above the latest known), one whose
  applied migration no longer matches the shipped file, or a foreign SQLite file is refused without being
  modified. The check runs on a read-only connection before the connection policy (WAL) is applied.
* Before upgrading a database that already holds a schema, a backup is taken with the SQLite backup
  API (never a raw copy of a live WAL database).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from weightlab.core.errors import ErrorCode, WeightLabError

_FILE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


class DatabaseVersionError(WeightLabError):
    """The database cannot be used by this version of WeightLab; it was not modified."""

    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.CONFLICT, detail)


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MigrationResult:
    from_version: int
    to_version: int
    applied: tuple[int, ...]
    backup: Path | None


def migrations() -> list[Migration]:
    found = []
    for entry in resources.files("weightlab.db.migrations").iterdir():
        m = _FILE.match(entry.name)
        if m:
            found.append(Migration(int(m.group(1)), entry.name, entry.read_text(encoding="utf-8")))
    found.sort(key=lambda x: x.version)
    if [x.version for x in found] != list(range(1, len(found) + 1)):
        raise RuntimeError("migration files must be numbered 0001, 0002, ... without gaps")
    return found


def latest_version() -> int:
    return len(migrations())


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def connect(path: str | os.PathLike[str]) -> sqlite3.Connection:
    """A connection with the WeightLab policy. Autocommit mode: callers open transactions explicitly
    (`with transaction(conn):`), so nothing is ever committed implicitly."""
    conn = sqlite3.connect(os.fspath(path), isolation_level=None, check_same_thread=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = FULL")
    if os.fspath(path) != ":memory:":
        mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            conn.close()
            raise WeightLabError(ErrorCode.CONFLICT, f"database refused WAL journal mode ({mode})")
    return conn


def current_version(conn: sqlite3.Connection) -> int:
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'").fetchone()
    if not exists:
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return int(row[0] or 0)


def _has_user_tables(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").fetchone()
        is not None
    )


def _verify_applied(conn: sqlite3.Connection, known: list[Migration]) -> None:
    by_version = {m.version: m for m in known}
    for row in conn.execute("SELECT version, checksum FROM schema_migrations ORDER BY version"):
        m = by_version.get(int(row["version"]))
        if m is None:
            raise DatabaseVersionError(
                f"database has migration {row['version']}, unknown to this WeightLab (latest {len(known)}); "
                "it was written by a newer version and was not modified"
            )
        if row["checksum"] != m.checksum:
            raise DatabaseVersionError(
                f"applied migration {m.name} does not match the shipped file (checksum differs); refusing to continue"
            )


def backup(conn: sqlite3.Connection, target: Path) -> Path:
    """Consistent copy of a live database through the SQLite backup API."""
    target.parent.mkdir(parents=True, exist_ok=True)
    dest = sqlite3.connect(os.fspath(target))
    try:
        conn.backup(dest)
    finally:
        dest.close()
    return target


def _check_schema(conn: sqlite3.Connection, known: list[Migration]) -> int:
    """The database's schema version, or DatabaseVersionError when WeightLab must not touch it."""
    start = current_version(conn)
    if start == 0 and _has_user_tables(conn):
        raise DatabaseVersionError("database has tables but no schema_migrations; not a WeightLab database")
    if start:
        _verify_applied(conn, known)
    if start > len(known):  # guarded by _verify_applied too; kept explicit
        raise DatabaseVersionError(f"database schema {start} is newer than this WeightLab ({len(known)})")
    return start


def _preflight(path: str | os.PathLike[str], known: list[Migration]) -> None:
    """Check an existing database through a read-only connection *before* the WeightLab connection policy
    runs: `PRAGMA journal_mode = WAL` rewrites the file header, so a foreign, newer or edited database
    must be refused before connect() ever sees it."""
    raw = os.fspath(path)
    p = Path(raw)
    if raw == ":memory:" or not p.exists() or p.stat().st_size == 0:
        return  # nothing to protect: connect() creates it
    # immutable=1: no locks, no -wal/-shm created (a plain mode=ro open of a WAL database leaves both behind).
    # It ignores a WAL, so when a non-empty -wal exists (a crashed writer) the ro open must read it instead.
    wal = Path(raw + "-wal")
    hot = wal.exists() and wal.stat().st_size > 0
    query = "?mode=ro" if hot else "?mode=ro&immutable=1"
    conn = sqlite3.connect(p.resolve().as_uri() + query, uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        _check_schema(conn, known)
    except sqlite3.DatabaseError as e:
        raise DatabaseVersionError(f"not a readable SQLite database ({e}); it was not modified") from None
    finally:
        conn.close()


def migrate(path: str | os.PathLike[str], backup_dir: Path | None = None) -> MigrationResult:
    """Bring the database at `path` to the latest schema. Creates it when absent. An existing database is
    verified read-only first and refused, byte for byte unmodified, unless it is a WeightLab database
    whose applied migrations all match the shipped files."""
    known = migrations()
    _preflight(path, known)
    conn = connect(path)
    try:
        start = _check_schema(conn, known)  # again under the policy connection (the file may have been replaced)
        pending = [m for m in known if m.version > start]
        saved: Path | None = None
        if pending and start > 0:
            stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
            where = backup_dir or Path(os.fspath(path)).parent / "backups"
            saved = backup(conn, where / f"weightlab-schema{start}-{stamp}.sqlite3")
        for m in pending:
            try:
                conn.execute("BEGIN IMMEDIATE")
                for statement in _statements(m.sql):
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at, checksum) VALUES (?, ?, ?)",
                    (m.version, now(), m.checksum),
                )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return MigrationResult(start, len(known), tuple(m.version for m in pending), saved)
    finally:
        conn.close()


def _statements(sql: str) -> list[str]:
    """Split a migration into complete statements (sqlite3.complete_statement), so each runs inside
    the migration's single transaction; `executescript` would commit implicitly."""
    out: list[str] = []
    buf = ""
    for line in sql.splitlines(keepends=True):
        stripped = line.strip()
        if not buf and (not stripped or stripped.startswith("--")):
            continue
        buf += line
        if sqlite3.complete_statement(buf):
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        raise RuntimeError("migration ends with an incomplete SQL statement")
    return out


@dataclass(frozen=True)
class IntegrityReport:
    ok: bool
    schema_version: int
    latest_version: int
    integrity: str
    foreign_key_violations: int

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def integrity(conn: sqlite3.Connection) -> IntegrityReport:
    """`weightlab doctor` checks: integrity_check, foreign_key_check and schema version (doc 10 section 7)."""
    result = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    fk = len(conn.execute("PRAGMA foreign_key_check").fetchall())
    version = current_version(conn)
    latest = latest_version()
    return IntegrityReport(result == "ok" and fk == 0 and version == latest, version, latest, result, fk)


def quick_health(conn: sqlite3.Connection) -> bool:
    """Constant-cost liveness probe for the unauthenticated /health route: the database answers and is at the
    latest schema. The full integrity_check / foreign_key_check scan grows with the database and stays in
    integrity() (`weightlab doctor`), not on a route anyone on the machine can call (PR #3 review)."""
    try:
        return current_version(conn) == latest_version()
    except sqlite3.Error:
        return False
