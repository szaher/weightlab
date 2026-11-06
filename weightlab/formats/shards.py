"""Hugging Face sharded index validation (doc 04 section 3, SEC-02).

Lexical violations inside the index (absolute, drive-qualified, UNC, backslash, `..`, non-string)
are format errors of the index file: INVALID_FORMAT, as listed in `malformed_expected`. A shard
path that is lexically clean but resolves outside the selected root through a symlink or reparse
point is PATH_OUTSIDE_ROOT (doc 12 section 4).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from weightlab.core.errors import ErrorCode, fail
from weightlab.formats.base import strict_json

INDEX_SUFFIX = ".safetensors.index.json"
_DRIVE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class IndexDoc:
    weight_map: dict[str, str]  # tensor key -> normalized POSIX relative shard path
    total_size: int | None


def check_relative(rel: object) -> str:
    """Return the normalized POSIX form of a shard path, or raise INVALID_FORMAT."""
    if not isinstance(rel, str):
        raise fail(ErrorCode.INVALID_FORMAT, "shard path is not a string")
    if not rel or "\x00" in rel:
        raise fail(ErrorCode.INVALID_FORMAT, "shard path is empty or holds NUL")
    if "\\" in rel:
        raise fail(ErrorCode.INVALID_FORMAT, "backslash in shard path")
    if _DRIVE.match(rel) or ":" in rel:
        raise fail(ErrorCode.INVALID_FORMAT, "drive-qualified or colon-bearing shard path")
    if rel.startswith("/"):
        raise fail(ErrorCode.INVALID_FORMAT, "absolute shard path")
    parts = PurePosixPath(rel).parts
    if ".." in parts:
        raise fail(ErrorCode.INVALID_FORMAT, "path outside selected root ('..' component)")
    return PurePosixPath(*parts).as_posix()


def parse_index(data: bytes) -> IndexDoc:
    doc = strict_json(data, "index")
    if not isinstance(doc, dict):
        raise fail(ErrorCode.INVALID_FORMAT, "index is not a JSON object")
    wm = doc.get("weight_map")
    if not isinstance(wm, dict):
        raise fail(ErrorCode.INVALID_FORMAT, "index has no weight_map object")
    meta = doc.get("metadata", {})
    if not isinstance(meta, dict):
        raise fail(ErrorCode.INVALID_FORMAT, "index metadata is not an object")
    total = meta.get("total_size")
    if total is not None and type(total) is not int:
        raise fail(ErrorCode.INVALID_FORMAT, "metadata.total_size is not an integer")
    return IndexDoc({k: check_relative(v) for k, v in wm.items()}, total)


def is_within(path: str, root: str) -> bool:
    """Both arguments must already be realpaths."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def resolve_shard(index_dir: Path, rel: str, real_root: str) -> Path:
    """Resolve a validated relative shard path; containment is checked on the real path."""
    candidate = index_dir / Path(*PurePosixPath(rel).parts)
    if not is_within(os.path.realpath(candidate), real_root):
        raise fail(ErrorCode.PATH_OUTSIDE_ROOT, f"shard {rel!r} resolves outside the selected root")
    if not candidate.is_file():
        raise fail(ErrorCode.MISSING_SHARD, f"shard {rel!r} not found")
    return candidate
