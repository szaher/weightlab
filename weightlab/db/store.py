"""Repository functions over the project database (explicit SQL, no ORM).

Conventions: rows come back as plain dicts with JSON columns decoded; every write runs inside
`transaction(conn)`; JSON is stored canonically (sorted keys, no NaN). Look-ups that take a
`project_id` return None for an object of another project, so the API can answer 404 without
revealing that the object exists (doc 09 "Ownership").
"""

from __future__ import annotations

import contextlib
import json
import secrets
import sqlite3
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from weightlab.db.schema import now

TENSOR_BATCH = 5000  # rows per executemany batch (doc 10 section 3: batched transactions)


def new_id(prefix: str) -> str:
    """Opaque server-issued ID matching ^[a-z]{3}_[A-Za-z0-9]{1,64}$."""
    if len(prefix) != 3 or not prefix.isalpha() or not prefix.islower():
        raise ValueError("ID prefix must be three lowercase letters")
    return f"{prefix}_{secrets.token_hex(12)}"


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _row(r: sqlite3.Row | None, json_cols: Sequence[str] = ()) -> dict[str, Any] | None:
    if r is None:
        return None
    d = dict(r)
    for c in json_cols:
        if c in d and d[c] is not None:
            d[c] = json.loads(d[c])
    return d


def _fold(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold()


def install_functions(conn: sqlite3.Connection) -> None:
    """Unicode-aware case-insensitive substring match (SQLite LIKE/lower fold ASCII only)."""

    def contains(haystack: str | None, needle: str | None) -> int:
        if haystack is None or needle is None:
            return 0
        return int(_fold(needle) in _fold(haystack))

    conn.create_function("wl_contains", 2, contains, deterministic=True)


# -- settings ----------------------------------------------------------------------------------
def get_settings(conn: sqlite3.Connection, defaults: dict[str, Any]) -> dict[str, Any]:
    row = conn.execute("SELECT settings_json FROM app_settings WHERE id = 1").fetchone()
    stored = json.loads(row[0]) if row else {}
    return {**defaults, **{k: v for k, v in stored.items() if k in defaults}}


def put_settings(conn: sqlite3.Connection, values: dict[str, Any]) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT INTO app_settings (id, settings_json, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET settings_json = excluded.settings_json, updated_at = excluded.updated_at",
            (dumps(values), now()),
        )


# -- projects ----------------------------------------------------------------------------------
def create_project(conn: sqlite3.Connection, name: str) -> dict[str, Any]:
    pid, t = new_id("prj"), now()
    with transaction(conn):
        conn.execute(
            "INSERT INTO projects (id, name, created_at, updated_at, settings_json) VALUES (?, ?, ?, ?, '{}')",
            (pid, name, t, t),
        )
    return {"id": pid, "name": name, "created_at": t}


def list_projects(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT id, name, created_at FROM projects ORDER BY created_at, id")]


def get_project(conn: sqlite3.Connection, project_id: str) -> dict[str, Any] | None:
    return _row(conn.execute("SELECT id, name, created_at FROM projects WHERE id = ?", (project_id,)).fetchone())


def delete_project(conn: sqlite3.Connection, project_id: str) -> bool:
    """Removes the project's rows (cascade). Checkpoint files are never touched: nothing here opens them."""
    with transaction(conn):
        return conn.execute("DELETE FROM projects WHERE id = ?", (project_id,)).rowcount == 1


# -- models, files, tensors --------------------------------------------------------------------
@dataclass(frozen=True)
class FileRow:
    relpath: str
    size_bytes: int
    mtime_ns: int
    file_id: str | None
    sha256: str | None
    fingerprint_method: str
    ctime_ns: int | None = None  # migration 0002; None for side files whose content is checked instead


@dataclass(frozen=True)
class TensorRow:
    id: str
    raw_key: str
    shape: tuple[int, ...]
    dtype: str
    shard_relpath: str
    byte_length: int
    numel: int
    file_offset_begin: int
    file_offset_end: int
    role: str | None
    layer_index: int | None


def insert_model(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    display_name: str,
    selected_root: str,
    entry_path: str,
    fmt: str,
    lineage: dict[str, Any],
    files: Sequence[FileRow],
    tensors: Iterable[TensorRow],
    warnings: Sequence[str] = (),
    model_id: str | None = None,
) -> str:
    mid, t = model_id or new_id("mod"), now()
    quick = {"files": [[f.relpath, f.size_bytes, f.mtime_ns, f.file_id] for f in files]}
    with transaction(conn):
        conn.execute(
            "INSERT INTO models (id, project_id, display_name, selected_root, entry_path, format, architecture_hint, "
            "lineage_json, quick_fingerprint_json, sha256, status, imported_at, warnings_json) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, 'ready', ?, ?)",
            (mid, project_id, display_name, selected_root, entry_path, fmt, dumps(lineage), dumps(quick), t,
             dumps(list(warnings))),
        )  # fmt: skip
        conn.executemany(
            "INSERT INTO model_files (id, model_id, relpath, size_bytes, mtime_ns, file_id, sha256, fingerprint_method,"
            " verified_at, ctime_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
            [(new_id("mfl"), mid, f.relpath, f.size_bytes, f.mtime_ns, f.file_id, f.sha256, f.fingerprint_method,
              f.ctime_ns) for f in files],
        )  # fmt: skip
        batch: list[tuple[Any, ...]] = []
        sql = (
            "INSERT INTO tensors (id, model_id, raw_key, rank, shape_json, dtype, shard_relpath, byte_length, role, "
            "layer_index, numel, file_offset_begin, file_offset_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        for tr in tensors:
            batch.append(
                (tr.id, mid, tr.raw_key, len(tr.shape), dumps(list(tr.shape)), tr.dtype, tr.shard_relpath,
                 tr.byte_length, tr.role, tr.layer_index, tr.numel, tr.file_offset_begin, tr.file_offset_end)
            )  # fmt: skip
            if len(batch) >= TENSOR_BATCH:
                conn.executemany(sql, batch)
                batch.clear()
        if batch:
            conn.executemany(sql, batch)
    return mid


_MODEL_COLS = (
    "m.id, m.project_id, m.display_name, m.selected_root, m.entry_path, m.format, m.status, m.lineage_json, "
    "m.warnings_json, m.imported_at, (SELECT COUNT(*) FROM tensors t WHERE t.model_id = m.id) AS tensor_count"
)


def list_models(conn: sqlite3.Connection, project_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"SELECT {_MODEL_COLS} FROM models m WHERE m.project_id = ? ORDER BY m.imported_at, m.id", (project_id,)
    )
    return [_row(r, ("lineage_json", "warnings_json")) for r in rows]  # type: ignore[misc]


def get_model(conn: sqlite3.Connection, model_id: str, project_id: str | None = None) -> dict[str, Any] | None:
    q = f"SELECT {_MODEL_COLS} FROM models m WHERE m.id = ?" + (" AND m.project_id = ?" if project_id else "")
    args = (model_id, project_id) if project_id else (model_id,)
    return _row(conn.execute(q, args).fetchone(), ("lineage_json", "warnings_json"))


def model_files(conn: sqlite3.Connection, model_id: str) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM model_files WHERE model_id = ? ORDER BY relpath", (model_id,))]


def relink_model(
    conn: sqlite3.Connection,
    model_id: str,
    entry_path: str,
    selected_root: str,
    file_ids: dict[str, str | None],
    ctimes: dict[str, int | None] | None = None,
) -> None:
    """Point a model at its new place and record the new file identities (file ID and ctime_ns, which a move
    changes), atomically (one transaction)."""
    with transaction(conn):
        conn.execute(
            "UPDATE models SET entry_path = ?, selected_root = ?, status = 'ready' WHERE id = ?",
            (entry_path, selected_root, model_id),
        )
        conn.executemany(
            "UPDATE model_files SET file_id = ?, ctime_ns = ? WHERE model_id = ? AND relpath = ?",
            [(fid, (ctimes or {}).get(rel), model_id, rel) for rel, fid in file_ids.items()],
        )


_TENSOR_SORT = {"raw_key": "raw_key", "numel": "numel", "byte_length": "byte_length", "layer_index": "layer_index"}


def query_tensors(
    conn: sqlite3.Connection,
    model_id: str,
    *,
    q: str | None = None,
    dtypes: Sequence[str] = (),
    roles: Sequence[str] = (),
    layer: int | None = None,
    rank: int | None = None,
    min_numel: int | None = None,
    sort: str = "raw_key",
    order: str = "asc",
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    where, args = ["model_id = ?"], [model_id]
    if q:
        where.append("wl_contains(raw_key, ?)")
        args.append(q)
    if dtypes:
        where.append(f"dtype IN ({','.join('?' * len(dtypes))})")
        args += list(dtypes)
    if roles:
        where.append(f"role IN ({','.join('?' * len(roles))})")
        args += list(roles)
    if layer is not None:
        where.append("layer_index = ?")
        args.append(layer)  # type: ignore[arg-type]
    if rank is not None:
        where.append("rank = ?")
        args.append(rank)  # type: ignore[arg-type]
    if min_numel is not None:
        where.append("numel >= ?")
        args.append(min_numel)  # type: ignore[arg-type]
    col = _TENSOR_SORT[sort]
    direction = "DESC" if order == "desc" else "ASC"
    # NULL layer indices sort last in either direction; raw_key then id make the order total.
    nulls = f"{col} IS NULL, " if col == "layer_index" else ""
    clause = " AND ".join(where)
    total = int(conn.execute(f"SELECT COUNT(*) FROM tensors WHERE {clause}", args).fetchone()[0])
    rows = conn.execute(
        f"SELECT * FROM tensors WHERE {clause} ORDER BY {nulls}{col} {direction}, raw_key {direction}, id "
        "LIMIT ? OFFSET ?",
        [*args, limit, offset],
    )
    return total, [_row(r, ("shape_json",)) for r in rows]  # type: ignore[misc]


def get_tensor(conn: sqlite3.Connection, tensor_id: str, model_id: str | None = None) -> dict[str, Any] | None:
    q = "SELECT * FROM tensors WHERE id = ?" + (" AND model_id = ?" if model_id else "")
    return _row(conn.execute(q, (tensor_id, model_id) if model_id else (tensor_id,)).fetchone(), ("shape_json",))


def tensor_by_key(conn: sqlite3.Connection, model_id: str, raw_key: str) -> dict[str, Any] | None:
    return _row(
        conn.execute("SELECT * FROM tensors WHERE model_id = ? AND raw_key = ?", (model_id, raw_key)).fetchone(),
        ("shape_json",),
    )


# -- comparisons -------------------------------------------------------------------------------
def insert_comparison(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    model_a_id: str,
    model_b_id: str,
    compatibility: dict[str, Any],
    entries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    cid, t = new_id("cmp"), now()
    with transaction(conn):
        conn.execute(
            "INSERT INTO comparisons (id, project_id, model_a_id, model_b_id, compatibility_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, project_id, model_a_id, model_b_id, dumps(compatibility), t),
        )
        conn.executemany(
            "INSERT INTO comparison_entries (id, comparison_id, ordinal, raw_key, status, tensor_id_a, tensor_id_b, "
            "detail_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(e["entry_id"], cid, i, e["raw_key"], e["status"], e.get("tensor_id_a"), e.get("tensor_id_b"),
              dumps({k: v for k, v in e.items() if k not in ("entry_id", "raw_key", "status", "tensor_id_a",
                                                             "tensor_id_b")}))
             for i, e in enumerate(entries)],
        )  # fmt: skip
    return get_comparison(conn, cid)  # type: ignore[return-value]


def get_comparison(
    conn: sqlite3.Connection, comparison_id: str, project_id: str | None = None
) -> dict[str, Any] | None:
    q = "SELECT * FROM comparisons WHERE id = ?" + (" AND project_id = ?" if project_id else "")
    args = (comparison_id, project_id) if project_id else (comparison_id,)
    return _row(conn.execute(q, args).fetchone(), ("compatibility_json",))


def list_entries(
    conn: sqlite3.Connection, comparison_id: str, statuses: Sequence[str] = (), limit: int = 100, offset: int = 0
) -> tuple[int, list[dict[str, Any]]]:
    where, args = ["comparison_id = ?"], [comparison_id]
    if statuses:
        where.append(f"status IN ({','.join('?' * len(statuses))})")
        args += list(statuses)
    clause = " AND ".join(where)
    total = int(conn.execute(f"SELECT COUNT(*) FROM comparison_entries WHERE {clause}", args).fetchone()[0])
    rows = conn.execute(
        f"SELECT * FROM comparison_entries WHERE {clause} ORDER BY ordinal LIMIT ? OFFSET ?", [*args, limit, offset]
    )
    return total, [_row(r, ("detail_json",)) for r in rows]  # type: ignore[misc]


def get_entry(conn: sqlite3.Connection, comparison_id: str, entry_id: str) -> dict[str, Any] | None:
    return _row(
        conn.execute(
            "SELECT * FROM comparison_entries WHERE id = ? AND comparison_id = ?", (entry_id, comparison_id)
        ).fetchone(),
        ("detail_json",),
    )


def next_report_seq(conn: sqlite3.Connection) -> int:
    """A strictly increasing submission number for comparison_report jobs (AUTOINCREMENT never reuses one)."""
    with transaction(conn):
        seq = int(conn.execute("INSERT INTO comparison_report_seq DEFAULT VALUES RETURNING seq").fetchone()[0])
        conn.execute("DELETE FROM comparison_report_seq WHERE seq < ?", (seq,))
    return seq


def publish_comparison_report(
    conn: sqlite3.Connection,
    comparison_id: str,
    job_id: str,
    job_seq: int,
    consistency: str,
    complete: bool,
    report_json: str,
) -> bool:
    """Inside the caller's transaction (the job's terminal-state transaction): store the report unless a report of a
    later-submitted job is already stored. True when this job's report is the stored one afterwards."""
    conn.execute(
        "INSERT INTO comparison_reports (comparison_id, job_id, job_seq, consistency, complete, report_json, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(comparison_id) DO UPDATE SET job_id = excluded.job_id, "
        "job_seq = excluded.job_seq, consistency = excluded.consistency, complete = excluded.complete, "
        "report_json = excluded.report_json, created_at = excluded.created_at "
        "WHERE excluded.job_seq > comparison_reports.job_seq",
        (comparison_id, job_id, job_seq, consistency, int(complete), report_json, now()),
    )
    row = conn.execute("SELECT job_id FROM comparison_reports WHERE comparison_id = ?", (comparison_id,)).fetchone()
    return row is not None and row["job_id"] == job_id


def get_comparison_report(conn: sqlite3.Connection, comparison_id: str) -> dict[str, Any] | None:
    """The stored report row; `report_json` stays the stored JSON text (served without re-parsing)."""
    r = conn.execute("SELECT * FROM comparison_reports WHERE comparison_id = ?", (comparison_id,)).fetchone()
    return None if r is None else dict(r)


def list_comparisons(conn: sqlite3.Connection, project_id: str) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM comparisons WHERE project_id = ? ORDER BY created_at DESC, rowid DESC",
                        (project_id,))  # fmt: skip
    return [_row(r, ("compatibility_json",)) for r in rows]  # type: ignore[misc]


def comparison_report_summary(conn: sqlite3.Connection, comparison_id: str) -> dict[str, Any] | None:
    """The stored report's metadata, without its JSON body."""
    r = conn.execute("SELECT job_id, complete, consistency, created_at FROM comparison_reports WHERE comparison_id = ?",
                     (comparison_id,)).fetchone()  # fmt: skip
    return None if r is None else {"job_id": r["job_id"], "complete": bool(r["complete"]),
                                   "consistency": r["consistency"], "created_at": r["created_at"]}  # fmt: skip


def latest_report_job_id(conn: sqlite3.Connection, comparison_id: str) -> str | None:
    """The most recently submitted comparison_report job of the comparison, in any state."""
    r = conn.execute("SELECT id FROM jobs WHERE kind = 'comparison_report' AND "
                     "json_extract(spec_json, '$.comparison_id') = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                     (comparison_id,)).fetchone()  # fmt: skip
    return None if r is None else str(r["id"])


def comparison_entry_stats(conn: sqlite3.Connection, comparison_id: str, model_a_id: str, model_b_id: str
                           ) -> tuple[int, int]:  # fmt: skip
    """(number of entries, tensor bytes of both sides of every analyzable entry), computed in SQL so no entry list
    is materialized."""
    n = int(conn.execute("SELECT COUNT(*) FROM comparison_entries WHERE comparison_id = ?",
                         (comparison_id,)).fetchone()[0])  # fmt: skip
    nbytes = conn.execute(
        "SELECT COALESCE(SUM(COALESCE(ta.byte_length, 0) + COALESCE(tb.byte_length, 0)), 0) "
        "FROM comparison_entries e "
        "LEFT JOIN tensors ta ON ta.model_id = ? AND ta.raw_key = e.raw_key "
        "LEFT JOIN tensors tb ON tb.model_id = ? AND tb.raw_key = e.raw_key "
        "WHERE e.comparison_id = ? AND json_extract(e.detail_json, '$.analyzable') = 1",
        (model_a_id, model_b_id, comparison_id),
    ).fetchone()[0]
    return n, int(nbytes)


# -- adapters ----------------------------------------------------------------------------------
def insert_adapter(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    base_model_id: str,
    adapter_path: str,
    config: dict[str, Any],
    modules: Sequence[dict[str, Any]],
    status: str,
    lineage: dict[str, Any],
    files: dict[str, Any],
    warnings: Sequence[str],
    adapter_id: str | None = None,
) -> str:
    aid = adapter_id or new_id("adp")
    with transaction(conn):
        conn.execute(
            "INSERT INTO adapters (id, project_id, base_model_id, config_json, adapter_path, mapping_json, status, "
            "lineage_json, files_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (aid, project_id, base_model_id, dumps(config), adapter_path,
             dumps({"modules": list(modules), "warnings": list(warnings)}), status, dumps(lineage), dumps(files),
             now()),
        )  # fmt: skip
    return aid


_ADAPTER_JSON = ("config_json", "mapping_json", "lineage_json", "files_json")


def list_adapters(conn: sqlite3.Connection, project_id: str) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM adapters WHERE project_id = ? ORDER BY created_at, id", (project_id,))
    return [_row(r, _ADAPTER_JSON) for r in rows]  # type: ignore[misc]


def get_adapter(conn: sqlite3.Connection, adapter_id: str, project_id: str | None = None) -> dict[str, Any] | None:
    q = "SELECT * FROM adapters WHERE id = ?" + (" AND project_id = ?" if project_id else "")
    args = (adapter_id, project_id) if project_id else (adapter_id,)
    return _row(conn.execute(q, args).fetchone(), _ADAPTER_JSON)


# -- bookmarks ---------------------------------------------------------------------------------
def insert_bookmark(
    conn: sqlite3.Connection, project_id: str, view: dict[str, Any], note: str | None
) -> dict[str, Any]:
    bid, t = new_id("bmk"), now()
    with transaction(conn):
        conn.execute(
            "INSERT INTO bookmarks (id, project_id, view_json, note, created_at) VALUES (?, ?, ?, ?, ?)",
            (bid, project_id, dumps(view), note, t),
        )
    return {"id": bid, "view": view, "note": note, "created_at": t}


def list_bookmarks(conn: sqlite3.Connection, project_id: str) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM bookmarks WHERE project_id = ? ORDER BY created_at, id", (project_id,))
    return [{"id": r["id"], "view": json.loads(r["view_json"]), "note": r["note"], "created_at": r["created_at"]}
            for r in rows]  # fmt: skip


def delete_bookmark(conn: sqlite3.Connection, project_id: str, bookmark_id: str) -> bool:
    with transaction(conn):
        cur = conn.execute("DELETE FROM bookmarks WHERE id = ? AND project_id = ?", (bookmark_id, project_id))
        return cur.rowcount == 1
