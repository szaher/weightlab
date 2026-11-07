"""Tile payloads (doc 09 section 3, contract TileMetadata / TileJson / TileStatus) and tile identity (doc 03
section 5, doc 10 section 6).

Binary body (`application/vnd.weightlab.tile.v2`): rows*cols float32 little-endian display values in row-major
order, then rows*cols uint8 status bitfields; exactly rows*cols*5 bytes. JSON (tests and debugging, <= 16,384
cells): `values[i][j]` is the float32 display value when the cell is drawable and null otherwise, `status` the same
bitfield. Both forms are validated on encode and decode with the same rules:

* reserved bits 32/64/128 are never set;
* drawable ((status & 17) == 0) <=> a finite float32 value is present;
* NO_FINITE needs at least one non-finite kind bit (a block always holds >= 1 element) and excludes
  NOT_REPRESENTABLE (which requires a finite aggregate);
* at level 0 one cell is one element: status is 0 (finite), 3 (NaN), 5 (+Inf), 9 (-Inf) or 16 (finite but beyond
  float32);
* metadata: rows/cols match the body; row/col ranges match rows/cols at the level; finite + non-finite counts equal
  the covered element count; quality `exact` carries no sampling fields.

Cache key: SHA-256 of canonical (sorted keys, compact, ASCII-safe UTF-8, no NaN) JSON of the identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from weightlab.analysis.tile import (
    DRAWABLE_MASK,
    HAS_NAN,
    HAS_NEG_INF,
    HAS_POS_INF,
    NO_FINITE,
    NOT_REPRESENTABLE,
    PYRAMID_RECORD_FORMAT,
    RESERVED_BITS,
    TILE_FORMAT,
    TILE_NUMERICS_VERSION,
    TILE_SCHEMA_VERSION,
    Tile,
)
from weightlab.core.errors import ErrorCode, fail
from weightlab.formats.selection import View

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]

JSON_MAX_CELLS = 16_384
LEVEL0_STATUSES = frozenset(
    {0, NO_FINITE | HAS_NAN, NO_FINITE | HAS_POS_INF, NO_FINITE | HAS_NEG_INF, NOT_REPRESENTABLE}
)


def _bad(why: str) -> Exception:
    return fail(ErrorCode.VALIDATION_ERROR, f"invalid tile: {why}")


# ----------------------------------------------------------------------------------------------------- validation
def validate_cells(values: F32, status: U8, level: int) -> None:
    if values.shape != status.shape or values.ndim != 2:
        raise _bad("values and status must be 2D arrays of the same shape")
    if values.dtype != np.float32 or status.dtype != np.uint8:
        raise _bad("values must be float32 and status uint8")
    if np.any(status & RESERVED_BITS):
        raise _bad("reserved status bits are set")
    drawable = (status & DRAWABLE_MASK) == 0
    finite = np.isfinite(values)
    if np.any(drawable & ~finite):
        raise _bad("a drawable cell has no finite value")
    if np.any(~drawable & finite):
        raise _bad("a non-drawable cell carries a finite value (placeholders must be NaN)")
    no_finite = (status & NO_FINITE) != 0
    if np.any(no_finite & ((status & (HAS_NAN | HAS_POS_INF | HAS_NEG_INF)) == 0)):
        raise _bad("NO_FINITE without any non-finite kind")
    if np.any(no_finite & ((status & NOT_REPRESENTABLE) != 0)):
        raise _bad("NO_FINITE and NOT_REPRESENTABLE are exclusive")
    if level == 0 and not set(np.unique(status).tolist()) <= LEVEL0_STATUSES:
        raise _bad("a level-0 cell status must be 0, 3, 5, 9 or 16")


def validate_metadata(meta: Mapping[str, Any], rows: int, cols: int) -> None:
    if meta.get("schema_version") != TILE_SCHEMA_VERSION:
        raise _bad("schema_version")
    if meta["rows"] != rows or meta["cols"] != cols:
        raise _bad("metadata rows/cols differ from the body")
    block = 1 << int(meta["level"])
    h = meta["row_end"] - meta["row_start"]
    w = meta["col_end"] - meta["col_start"]
    if h <= 0 or w <= 0 or -(-h // block) != rows or -(-w // block) != cols:
        raise _bad("row/col ranges do not match rows/cols at this level")
    if meta["row_start"] % block or meta["col_start"] % block:
        raise _bad("ranges are not anchored at a block boundary")
    if meta["finite_count"] + meta["nonfinite_count"] != h * w:
        raise _bad("finite + non-finite counts differ from the covered element count")
    if meta["quality"] == "exact" and any(meta.get(k) is not None for k in ("sampled_rows", "sampling_policy", "seed")):
        raise _bad("an exact tile carries sampling fields")
    if meta["value_precision"] != "float32_visualization":
        raise _bad("value_precision")


# ---------------------------------------------------------------------------------------------------------- binary
def encode_binary(values: F32, status: U8, level: int) -> bytes:
    validate_cells(values, status, level)
    return values.astype("<f4").tobytes() + status.tobytes()


def decode_binary(body: bytes, rows: int, cols: int, level: int) -> tuple[F32, U8]:
    n = rows * cols
    if len(body) != n * 5:
        raise _bad(f"body is {len(body)} bytes, expected rows*cols*5 = {n * 5}")
    values = np.frombuffer(body, "<f4", count=n).astype(np.float32).reshape(rows, cols)
    status = np.frombuffer(body, np.uint8, count=n, offset=n * 4).copy().reshape(rows, cols)
    validate_cells(values, status, level)
    return values, status


# -------------------------------------------------------------------------------------------------- metadata, JSON
def view_json(view: View | None) -> dict[str, Any]:
    if view is None:
        return {"row_axis": None, "col_axis": None, "fixed_indices": {}, "transpose": False}
    return {
        "row_axis": view.row_axis,
        "col_axis": view.col_axis,
        "fixed_indices": {str(k): int(v) for k, v in sorted(view.fixed_indices.items())},
        "transpose": bool(view.transpose),
    }


def metadata(tile: Tile, *, cache_key: str, source: Mapping[str, Any], view: View | None) -> dict[str, Any]:
    g = tile.geometry
    return {
        "schema_version": TILE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "rows": g.rows,
        "cols": g.cols,
        "row_start": g.row_start,
        "row_end": g.row_end,
        "col_start": g.col_start,
        "col_end": g.col_end,
        "level": g.level,
        "quality": tile.quality,
        "value_precision": tile.value_precision,
        "sampled_rows": None,
        "sampling_policy": None,
        "seed": None,
        "aggregation": tile.aggregation,
        "finite_count": tile.finite_count,
        "nonfinite_count": tile.nonfinite_count,
        "source": dict(source),
        "view": view_json(view),
        "warnings": [w[:500] for w in tile.warnings[:50]],
    }


def headers(meta: Mapping[str, Any]) -> dict[str, str]:
    """Binary-response metadata headers (doc 09 section 3). Sampled-Rows only for row-sampled tiles."""
    h = {
        "Content-Type": TILE_FORMAT,
        "X-WeightLab-Rows": str(meta["rows"]),
        "X-WeightLab-Cols": str(meta["cols"]),
        "X-WeightLab-Row-Start": str(meta["row_start"]),
        "X-WeightLab-Row-End": str(meta["row_end"]),
        "X-WeightLab-Col-Start": str(meta["col_start"]),
        "X-WeightLab-Col-End": str(meta["col_end"]),
        "X-WeightLab-Level": str(meta["level"]),
        "X-WeightLab-Quality": meta["quality"],
        "X-WeightLab-Value-Precision": meta["value_precision"],
        "X-WeightLab-Aggregation": meta["aggregation"],
        "X-WeightLab-Cache-Key": meta["cache_key"],
    }
    if meta.get("sampled_rows") is not None:
        h["X-WeightLab-Sampled-Rows"] = str(meta["sampled_rows"])
    return h


def to_json(meta: Mapping[str, Any], values: F32, status: U8) -> dict[str, Any]:
    rows, cols = values.shape
    if rows * cols > JSON_MAX_CELLS:
        raise fail(ErrorCode.NOT_ACCEPTABLE, f"JSON tiles are limited to {JSON_MAX_CELLS} cells; use {TILE_FORMAT}")
    validate_metadata(meta, rows, cols)
    validate_cells(values, status, int(meta["level"]))
    drawable = (status & DRAWABLE_MASK) == 0
    vals = [
        [float(v) if d else None for v, d in zip(vr, dr, strict=True)]
        for vr, dr in zip(values.tolist(), drawable.tolist(), strict=True)
    ]
    return {"metadata": dict(meta), "values": vals, "status": status.astype(int).tolist()}


def from_json(doc: Mapping[str, Any]) -> tuple[dict[str, Any], F32, U8]:
    meta = dict(doc["metadata"])
    raw_vals: Sequence[Sequence[float | None]] = doc["values"]
    raw_status: Sequence[Sequence[int]] = doc["status"]
    rows, cols = len(raw_status), len(raw_status[0]) if raw_status else 0
    if len(raw_vals) != rows or any(len(r) != cols for r in raw_vals) or any(len(r) != cols for r in raw_status):
        raise _bad("values and status must be rectangular and of the same shape")
    status = np.array(raw_status, dtype=np.int64)
    if np.any((status < 0) | (status > 255)):
        raise _bad("status out of byte range")
    st = status.astype(np.uint8)
    values = np.full((rows, cols), np.nan, np.float32)
    for i, row in enumerate(raw_vals):
        for j, v in enumerate(row):
            drawable = (int(st[i, j]) & DRAWABLE_MASK) == 0
            if (v is None) == drawable:
                raise _bad(f"cell {i},{j}: null exactly for non-drawable cells")
            if v is not None:
                f = np.float32(v)
                if float(f) != v:
                    raise _bad(f"cell {i},{j}: value is not a float32")
                values[i, j] = f
    validate_metadata(meta, rows, cols)
    validate_cells(values, st, int(meta["level"]))
    return meta, values, st


# --------------------------------------------------------------------------------------------------------- identity
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def tile_identity(
    *,
    source_kind: str,
    files: Sequence[Mapping[str, Any]],
    tensor_keys: Sequence[str],
    view: View | None,
    level: int,
    x: int,
    y: int,
    size: int,
    aggregation: str,
    quality_mode: str = "exact",
    seed: int | None = None,
    operation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything that determines a tile's bytes (doc 10 section 6). Opaque database IDs are deliberately absent:
    the same files, tensor and view give the same key; any input fingerprint change gives a new one."""
    return {
        "kind": "tile",
        "source_kind": source_kind,
        "files": sorted((dict(f) for f in files), key=lambda f: str(f["relpath"])),
        "tensor_keys": list(tensor_keys),
        "view": view_json(view),
        "operation": dict(operation or {}),
        "numerics_version": TILE_NUMERICS_VERSION,
        "pyramid_record_format": PYRAMID_RECORD_FORMAT,
        "tile_format": TILE_FORMAT,
        "level": level,
        "x": x,
        "y": y,
        "size": size,
        "aggregation": aggregation,
        "quality_mode": quality_mode,
        "seed": seed,
    }


def cache_key(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
