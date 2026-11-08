"""Model import in place, fingerprints, relink and catalog rows (doc 09 s1/1.1, doc 10 s3-4, doc 12 s3).

Import reads only headers through the Phase 0 reader (strict SafeTensors, selected-root containment,
regular files only). Paths are stored in the local database and never returned by the API.
Every later use of a model reopens it and compares each shard's quick fingerprint (relpath, size,
mtime_ns, ctime_ns, device:inode) with the stored rows; any difference is FILE_CHANGED, never a stale result.
"""

from __future__ import annotations

import hashlib
import ntpath
import os
import sqlite3
import stat
import sys
from pathlib import Path
from typing import Any

from weightlab.analysis.roles import layer_index, role
from weightlab.api.errors import ApiError
from weightlab.api.schemas import ModelImport, ModelRelink
from weightlab.core.errors import ErrorCode, WeightLabError
from weightlab.db import store
from weightlab.db.store import FileRow, TensorRow
from weightlab.formats.checkpoint import Checkpoint, fingerprint_side_file, open_checkpoint
from weightlab.formats.fileio import FileIdentity, change_time_ns

CONTRACT_ROLES = frozenset(
    {
        "embedding",
        "attention.q",
        "attention.k",
        "attention.v",
        "attention.o",
        "mlp.up",
        "mlp.down",
        "mlp.gate",
        "norm",
        "lm_head",
        "other",
        "unknown",
    }
)


def tensor_id(model_id: str, raw_key: str) -> str:
    """Deterministic, URL-safe opaque ID for (model, raw key); the key itself never appears in a URL."""
    return "tns_" + hashlib.sha256(f"{model_id}\0{raw_key}".encode()).hexdigest()[:40]


def contract_role(key: str) -> str:
    r = role(key)
    return r if r in CONTRACT_ROLES else "other"  # e.g. fused attention.qkv has no contract value


def _is_absolute(p: str) -> bool:
    """Absolute on the host OS. Windows needs a drive or UNC prefix (C:\\..., \\\\?\\C:\\..., \\\\server\\share);
    a rooted path without a drive ("\\models") is relative to the current drive and refused."""
    if sys.platform == "win32":
        return ntpath.isabs(p) and bool(ntpath.splitdrive(p)[0])
    return os.path.isabs(p)


def validate_paths(entry_path: str, selected_root: str) -> tuple[Path, Path, list[str]]:
    """Typed absolute paths only (doc 09 s1.1). Returns (entry, root, warnings)."""
    if "\0" in entry_path or "\0" in selected_root:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "paths must not contain NUL characters")
    if not (_is_absolute(entry_path) and _is_absolute(selected_root)):
        raise ApiError(ErrorCode.VALIDATION_ERROR, "entry_path and selected_root must be absolute paths")
    warnings = []
    if selected_root.startswith("\\\\") and not selected_root.startswith("\\\\?\\"):
        warnings.append("network location (UNC path): reads may be slow")
    root = Path(os.path.abspath(selected_root))
    entry = Path(os.path.abspath(entry_path))
    try:
        st_root = os.stat(root)
    except FileNotFoundError:
        raise ApiError(ErrorCode.NOT_FOUND, "selected root does not exist") from None
    except PermissionError as e:
        raise ApiError(
            ErrorCode.VALIDATION_ERROR, f"permission denied: selected root ({e.strerror})", status=403
        ) from None
    if not stat.S_ISDIR(st_root.st_mode):
        raise ApiError(ErrorCode.VALIDATION_ERROR, "selected root is not a directory")
    try:
        st_entry = os.stat(entry)
    except FileNotFoundError:
        raise ApiError(ErrorCode.NOT_FOUND, "entry path does not exist") from None
    except PermissionError as e:
        raise ApiError(
            ErrorCode.VALIDATION_ERROR, f"permission denied: entry path ({e.strerror})", status=403
        ) from None
    if not (stat.S_ISDIR(st_entry.st_mode) or stat.S_ISREG(st_entry.st_mode)):
        raise ApiError(ErrorCode.INVALID_FORMAT, "entry path is not a regular file or directory")
    return entry, root, warnings


def _relpath(ck: Checkpoint, shard_id: str, root: Path) -> str:
    """Shard path relative to the selected root, POSIX form, lexical (keeps HF-cache snapshot names)."""
    p = os.path.abspath(ck.shard_path(shard_id))
    return Path(os.path.relpath(p, os.path.abspath(root))).as_posix()


def _file_id(ident: FileIdentity) -> str:
    return f"{ident.dev}:{ident.ino}"


def _file_rows(ck: Checkpoint, root: Path) -> list[FileRow]:
    """Every file the catalog depends on: the shards (quick fingerprint) and, for a sharded checkpoint, the
    index that maps keys to shards (quick fingerprint plus SHA-256 of its bytes, which the reader reads in
    full anyway). Without the index row, an index rewritten to map keys differently while naming the same
    shards went undetected (PR #3 review)."""
    idents = ck.file_identities()
    rows = [
        FileRow(
            _relpath(ck, sid, root),
            ident.size,
            ident.mtime_ns,
            _file_id(ident),
            None,
            "size_mtime",
            change_time_ns(ident),
        )
        for sid, ident in idents.items()
    ]
    if ck.index is not None:
        rel = Path(os.path.relpath(os.path.abspath(ck.index.path), os.path.abspath(root))).as_posix()
        i = ck.index.identity
        rows.append(FileRow(rel, i.size, i.mtime_ns, _file_id(i), ck.index.sha256, "sha256"))
    return rows


def import_model(conn: sqlite3.Connection, project_id: str, req: ModelImport) -> dict[str, Any]:
    entry, root, warnings = validate_paths(req.entry_path, req.selected_root)
    with open_checkpoint(entry, root) as ck:
        model_id = store.new_id("mod")
        files = _file_rows(ck, root)
        rel = {sid: _relpath(ck, sid, root) for sid in ck.shard_ids}
        rows = (
            TensorRow(
                tensor_id(model_id, m.key),
                m.key,
                m.shape,
                m.dtype,
                rel[m.shard_id],
                m.byte_length,
                m.numel,
                m.data_offset,
                m.data_offset + m.byte_length,
                contract_role(m.key),
                layer_index(m.key),
            )
            for m in ck.metadata()
        )
        fmt = "sharded_safetensors" if ck.entry.name.endswith(".index.json") else "safetensors"
        name = req.display_name or (entry.name if entry.is_dir() else ck.entry.name)
        store.insert_model(
            conn,
            project_id=project_id,
            display_name=name[:120],
            selected_root=str(root),
            entry_path=str(ck.entry),
            fmt=fmt,
            lineage={},
            files=files,
            tensors=rows,
            warnings=[*warnings, *ck.warnings],
            model_id=model_id,
        )
    return model_view(conn, model_id)


def live_status(conn: sqlite3.Connection, model: dict[str, Any]) -> str:
    """Stored status, downgraded by a cheap stat of every recorded file: missing -> disconnected,
    size/mtime/identity differs -> changed (doc 18 s7: a moved model is disconnected, not deleted)."""
    root = Path(model["selected_root"])
    for f in store.model_files(conn, model["id"]):
        try:
            st = os.stat(root / f["relpath"])
        except OSError:
            return "disconnected"
        if (st.st_size, st.st_mtime_ns) != (f["size_bytes"], f["mtime_ns"]) or f"{st.st_dev}:{st.st_ino}" != f[
            "file_id"
        ]:
            return "changed"
        if f["ctime_ns"] is not None and st.st_ctime_ns != f["ctime_ns"]:
            return "changed"
    return str(model["status"])


def model_view(conn: sqlite3.Connection, model_id: str, project_id: str | None = None) -> dict[str, Any]:
    m = store.get_model(conn, model_id, project_id)
    if m is None:
        raise ApiError(ErrorCode.NOT_FOUND, "model not found")
    methods = {f["fingerprint_method"] for f in store.model_files(conn, model_id)}
    out: dict[str, Any] = {
        "id": m["id"],
        "project_id": m["project_id"],
        "display_name": m["display_name"],
        "format": m["format"],
        "status": live_status(conn, m),
        "tensor_count": m["tensor_count"],
        "fingerprint_method": methods.pop() if len(methods) == 1 else "size_mtime",
    }
    if m["warnings_json"]:
        out["warnings"] = [w[:500] for w in m["warnings_json"][:100]]
    return out


def open_model(conn: sqlite3.Connection, model_id: str, project_id: str | None = None) -> Checkpoint:
    """Reopen an imported model and prove it is the imported bytes' files (quick fingerprint)."""
    m = store.get_model(conn, model_id, project_id)
    if m is None:
        raise ApiError(ErrorCode.NOT_FOUND, "model not found")
    root = Path(m["selected_root"])
    _assert_same_index(conn, model_id, root)  # before parsing: an edited index must be FILE_CHANGED, not a parse error
    try:
        ck = open_checkpoint(m["entry_path"], root)
    except WeightLabError as e:
        if e.code in (ErrorCode.NOT_FOUND, ErrorCode.MISSING_SHARD):
            raise ApiError(
                ErrorCode.FILE_CHANGED, "model files are missing (moved or deleted); relink or re-import"
            ) from None
        # The files parsed at import. If they no longer do and their stat fingerprint differs, the checkpoint
        # changed on disk: that is FILE_CHANGED (re-import), not a format error of the imported model.
        if live_status(conn, m) in ("changed", "disconnected"):
            raise ApiError(
                ErrorCode.FILE_CHANGED, f"model files changed since import and no longer parse ({e.detail})"
            ) from None
        raise
    try:
        _assert_same_files(conn, model_id, ck, root, compare_file_id=True)
    except BaseException:
        ck.close()
        raise
    return ck


def _assert_same_index(conn: sqlite3.Connection, model_id: str, root: Path) -> None:
    """Content check of the recorded index file (rows with a SHA-256), by its own read."""
    for f in store.model_files(conn, model_id):
        if f["sha256"] is None:
            continue
        try:
            now = fingerprint_side_file(root / f["relpath"], root, "index file")
        except WeightLabError:
            raise ApiError(
                ErrorCode.FILE_CHANGED, f"{Path(f['relpath']).name} is missing or unreadable; relink or re-import"
            ) from None
        if now.sha256 != f["sha256"]:
            raise ApiError(ErrorCode.FILE_CHANGED, f"{Path(f['relpath']).name} changed since import (content)")


def _assert_same_files(
    conn: sqlite3.Connection, model_id: str, ck: Checkpoint, root: Path, *, compare_file_id: bool
) -> None:
    stored = {f["relpath"]: f for f in store.model_files(conn, model_id)}
    now = {r.relpath: r for r in _file_rows(ck, root)}
    if stored.keys() != now.keys():
        raise ApiError(ErrorCode.FILE_CHANGED, "the model's shard set differs from the imported one")
    for rel, f in stored.items():
        r = now[rel]
        # compare_file_id=False (relink): a move changes the inode and the ctime, so neither is compared.
        if (r.size_bytes, r.mtime_ns) != (f["size_bytes"], f["mtime_ns"]) or (
            compare_file_id
            and (r.file_id != f["file_id"] or (f["ctime_ns"] is not None and r.ctime_ns != f["ctime_ns"]))
        ):
            raise ApiError(ErrorCode.FILE_CHANGED, f"{Path(rel).name} changed since import (size, time or identity)")
        if f["sha256"] is not None and r.sha256 != f["sha256"]:  # the index: bytes re-read on every open
            raise ApiError(ErrorCode.FILE_CHANGED, f"{Path(rel).name} changed since import (content)")


def relink(conn: sqlite3.Connection, model_id: str, req: ModelRelink) -> dict[str, Any]:
    """Point a moved model at its new place: every file's (relpath, size, mtime_ns) and the whole tensor
    catalog must match the import, else 409 FILE_CHANGED (the user must re-import)."""
    m = store.get_model(conn, model_id)
    if m is None:
        raise ApiError(ErrorCode.NOT_FOUND, "model not found")
    entry, root, _ = validate_paths(req.entry_path, req.selected_root)
    # Quick fingerprints first, by stat only: a changed file must be FILE_CHANGED, not whatever error
    # the reader would raise on its new content.
    for f in store.model_files(conn, model_id):
        try:
            st = os.stat(root / f["relpath"])
        except OSError:
            raise ApiError(ErrorCode.FILE_CHANGED, f"{Path(f['relpath']).name} is not under the new root") from None
        if (st.st_size, st.st_mtime_ns) != (f["size_bytes"], f["mtime_ns"]):
            raise ApiError(ErrorCode.FILE_CHANGED, f"{Path(f['relpath']).name} differs from the imported file")
    _assert_same_index(conn, model_id, root)
    with open_checkpoint(entry, root) as ck:
        _assert_same_files(conn, model_id, ck, root, compare_file_id=False)  # a move changes the inode
        total, rows = store.query_tensors(conn, model_id, limit=max(1, len(ck.metadata())) + 1)
        catalog = {r["raw_key"]: (tuple(r["shape_json"]), r["dtype"], r["file_offset_begin"]) for r in rows}
        now = {m_.key: (m_.shape, m_.dtype, m_.data_offset) for m_ in ck.metadata()}
        if total != len(now) or catalog != now:
            raise ApiError(ErrorCode.FILE_CHANGED, "tensor catalog differs from the imported model; re-import it")
        # One transaction: a failure must not leave the new path with the old file identities.
        rows_now = _file_rows(ck, root)
        store.relink_model(
            conn,
            model_id,
            str(ck.entry),
            str(root),
            {r.relpath: r.file_id for r in rows_now},
            {r.relpath: r.ctime_ns for r in rows_now},
        )
    return model_view(conn, model_id)


def tensor_view(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": t["id"],
        "model_id": t["model_id"],
        "raw_key": t["raw_key"],
        "shape": t["shape_json"],
        "dtype": t["dtype"],
        "rank": t["rank"],
        "numel": t["numel"],
        "byte_length": t["byte_length"],
        "shard_relpath": t["shard_relpath"],
        "file_offset_begin": t["file_offset_begin"],
        "file_offset_end": t["file_offset_end"],
        "role": t["role"],
        "role_is_heuristic": True,
        "layer_index": t["layer_index"],
    }
