"""Local project database (Phase 1 Task 1): SQLite schema, migrations and repositories (doc 10)."""

from __future__ import annotations

import contextlib
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from weightlab.db.schema import MigrationResult, connect, integrity, latest_version, migrate, quick_health
from weightlab.db.store import install_functions

DB_FILE_NAME = "weightlab.sqlite3"

__all__ = [
    "DB_FILE_NAME",
    "Database",
    "MigrationResult",
    "connect",
    "integrity",
    "latest_version",
    "migrate",
    "quick_health",
]


class Database:
    """The project database at `path`, migrated on open. One short-lived connection per unit of
    work (`with db.session() as conn:`), so request threads never share a connection."""

    def __init__(self, path: str | os.PathLike[str], backup_dir: Path | None = None) -> None:
        self.path = Path(os.fspath(path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migration = migrate(self.path, backup_dir)

    @contextlib.contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.path)
        install_functions(conn)
        try:
            yield conn
        finally:
            conn.close()
