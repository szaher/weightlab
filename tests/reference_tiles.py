"""Independent reference for heatmap tiles (doc 03 section 5, doc 05 section 6, doc 09 section 3).

Shares no code with weightlab: tensors are decoded straight from the file bytes (header JSON + np.frombuffer, BF16 by
bit shift, F16 by NumPy), views are plain NumPy indexing/transposition, and every block is aggregated on its own with
exactly rounded sums (math.fsum) or Python Fractions. Status bits follow the fixture generator's rule
(scripts/generate_demo_fixtures.py `status`)."""

from __future__ import annotations

import json
import math
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F32_MAX = float(np.finfo(np.float32).max)
_NP = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "U64": "<u8", "I32": "<i4", "I16": "<i2", "I8": "i1",
       "U32": "<u4", "U16": "<u2", "U8": "u1", "BOOL": "u1"}  # fmt: skip


def decode_tensor(path: Path, key: str) -> npt.NDArray[Any]:
    """The stored tensor: floats widened to float64 (exact), integers as Python ints (object array)."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    entry = json.loads(raw[8 : 8 + n])[key]
    a, b = entry["data_offsets"]
    data = raw[8 + n + a : 8 + n + b]
    dt, shape = entry["dtype"], entry["shape"]
    if dt == "BF16":
        bits = np.frombuffer(data, "<u2").astype(np.uint32) << 16
        return bits.view(np.float32).astype(np.float64).reshape(shape)
    arr = np.frombuffer(data, _NP[dt]).reshape(shape)
    if dt.startswith(("I", "U")):
        return np.array([int(v) for v in arr.ravel()], dtype=object).reshape(shape)
    return arr.astype(np.float64)


def view2d(
    t: npt.NDArray[Any], row_axis: int | None, col_axis: int | None, fixed: dict[int, int], transpose: bool
) -> npt.NDArray[Any]:
    if t.ndim == 0:
        return t.reshape(1, 1)
    if t.ndim == 1:
        return t.reshape(1, -1)
    assert row_axis is not None
    assert col_axis is not None
    index = tuple(slice(None) if a in (row_axis, col_axis) else fixed[a] for a in range(t.ndim))
    m = t[index]  # remaining axes in increasing order
    if row_axis > col_axis:
        m = m.T
    return m.T if transpose else m


def _status(block: list[float], display: float | None) -> int:
    finite = [v for v in block if math.isfinite(v)]
    bits = 0 if finite else 1
    bits |= 2 if any(math.isnan(v) for v in block) else 0
    bits |= 4 if any(v == math.inf for v in block) else 0
    bits |= 8 if any(v == -math.inf for v in block) else 0
    if finite and (display is None or not abs(display) <= F32_MAX):
        bits |= 16
    return bits


def aggregate(block: list[float], aggregation: str) -> float | None:
    """The mathematically exact aggregate over finite elements, rounded once to float64 (None: no finite value or
    beyond the float64 range)."""
    finite = [v for v in block if math.isfinite(v)]
    if not finite:
        return None
    if aggregation == "min":
        return min(finite)
    if aggregation == "max":
        return max(finite)
    n = len(finite)
    if aggregation == "mean":
        q = sum((Fraction(v) for v in finite), Fraction(0)) / n
        if q == 0 and all(math.copysign(1.0, v) < 0 for v in finite):
            return -0.0  # IEEE: a sum of negative zeros is -0.0
    elif aggregation == "mean_abs":
        q = sum((Fraction(abs(v)) for v in finite), Fraction(0)) / n
    else:  # rms
        ms = sum((Fraction(v) ** 2 for v in finite), Fraction(0)) / n
        return math.sqrt(ms) if ms < Fraction(F32_MAX) ** 2 else None  # beyond float32 anyway
    return float(q) if abs(q) <= Fraction(1.7976931348623157e308) else None


def ref_tile(view: npt.NDArray[Any], level: int, x: int, y: int, size: int, aggregation: str) -> dict[str, Any]:
    """Reference tile of a 2D view: values (float32, NaN where not drawable), status, geometry, counts."""
    b = 1 << level
    span = size * b
    r0, c0 = y * span, x * span
    r1, c1 = min(view.shape[0], r0 + span), min(view.shape[1], c0 + span)
    sub = view[r0:r1, c0:c1]
    rows, cols = -(-(r1 - r0) // b), -(-(c1 - c0) // b)
    values = np.full((rows, cols), np.nan, np.float32)
    status = np.zeros((rows, cols), np.uint8)
    exact = np.full((rows, cols), np.nan)
    for i in range(rows):
        for j in range(cols):
            blk = [float(v) for v in np.asarray(sub[i * b : (i + 1) * b, j * b : (j + 1) * b]).ravel()]
            v = aggregate(blk, aggregation)
            st = _status(blk, v if v is not None else (None if not any(map(math.isfinite, blk)) else math.inf))
            status[i, j] = st
            if st & 17 == 0 and v is not None:
                values[i, j] = np.float32(v)
                exact[i, j] = v
    flat = [float(v) for v in np.asarray(sub).ravel()]
    finite = sum(1 for v in flat if math.isfinite(v))
    return {"values": values, "status": status, "exact": exact, "rows": rows, "cols": cols,
            "range": (r0, r1, c0, c1), "finite": finite, "nonfinite": len(flat) - finite}  # fmt: skip
