"""Exact heatmap tiles by direct bounded reads: the numerical foundation of Phase 1 Task 4B (doc 03 section 5,
doc 05 section 6, doc 09 section 3).

Geometry. A view of a tensor is a logical 2D array (R, C) (`formats.selection.resolve_view`: axes, fixed indices,
transpose). At level L one output cell aggregates a block of 2^L x 2^L raw cells; a tile is `size` (<= 256) output
cells per side. Tile (x, y) covers raw rows [y*size*2^L, (y+1)*size*2^L) and raw columns [x*size*2^L, ...), both
clipped to the view: blocks are anchored at the origin, so edge blocks hold only their real cells and every
aggregate is count-weighted automatically.

Aggregation. Per block: counts of finite / NaN / +Inf / -Inf elements, sums of x, |x| and x^2 over finite elements
in float64, and min / max over finite elements (the record of doc 03 section 5). Display values are derived from
these only: mean = sum/n, mean_abs = sum_abs/n, rms = sqrt(sum_sq/n), min, max; partial records merge by addition
and min/max, so RMS is never an average of RMS values. A float64 accumulator that leaves the finite range although
every input was finite sets an overflow flag; an aggregate that depends on an overflowed field, or whose value
exceeds the float32 range, is NOT_REPRESENTABLE and is never drawn. A block without finite elements is NO_FINITE.

Reading. Only positioned reads of bounded rectangles (`Checkpoint.read_roi_raw`), each at most `band_cells` cells,
oriented along the storage-contiguous axis of the view; their total read bytes are planned and checked against the
read budget before the first byte is read. Covered source bytes above `direct_tile_max_mib` are refused with
RESOURCE_LIMIT: such tiles need the exact pyramid (Task 4B, PR #7). Nothing is ever sampled here.

`coverage` (quality) is always `exact`; `value_precision` is always `float32_visualization`; `numeric_precision`
says whether the float64 aggregate before display is a source value (`lossless`), float64 arithmetic or a rounded
64-bit integer (`rounded`), or withheld somewhere in the tile (`overflowed`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from weightlab.analysis.diff import NUMERICS_VERSION
from weightlab.core.errors import ErrorCode, fail
from weightlab.core.limits import MAX_SAFE_INTEGER
from weightlab.formats.base import Consistency
from weightlab.formats.checkpoint import Checkpoint
from weightlab.formats.dtypes import DType, lookup
from weightlab.formats.selection import ResolvedView, View, resolve_view

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]

TILE_SIZE = 256
MAX_LEVEL = 52  # contract: TileRequest.tile.level maximum
TILE_FORMAT = "application/vnd.weightlab.tile.v2"
TILE_SCHEMA_VERSION = 2
PYRAMID_RECORD_FORMAT = 1  # part of every identity (doc 03 section 5), also for direct tiles
TILE_NUMERICS_VERSION = NUMERICS_VERSION
F32_MAX = float(np.finfo(np.float32).max)
MIB = 1024 * 1024

Aggregation = Literal["mean", "mean_abs", "rms", "min", "max"]
AGGREGATIONS: tuple[Aggregation, ...] = ("mean", "mean_abs", "rms", "min", "max")

# Tile status bitfield (contract TileStatus).
NO_FINITE = 1
HAS_NAN = 2
HAS_POS_INF = 4
HAS_NEG_INF = 8
NOT_REPRESENTABLE = 16
RESERVED_BITS = 32 | 64 | 128
DRAWABLE_MASK = NO_FINITE | NOT_REPRESENTABLE  # a cell is drawn iff (status & 17) == 0

# Aggregate record flags (doc 03 section 5, record format v1).
SUM_OVERFLOW = 1
SUM_ABS_OVERFLOW = 2
SUM_SQ_OVERFLOW = 4
_DEPENDS: dict[str, int] = {
    "mean": SUM_OVERFLOW,
    "mean_abs": SUM_ABS_OVERFLOW,
    "rms": SUM_SQ_OVERFLOW,
    "min": 0,
    "max": 0,
}


# ------------------------------------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class TileGeometry:
    level: int
    x: int
    y: int
    size: int
    block: int  # 2^level raw cells per output cell side
    row_start: int  # raw view rows [row_start, row_end), clipped
    row_end: int
    col_start: int
    col_end: int
    rows: int  # output cells
    cols: int

    @property
    def covered_cells(self) -> int:
        return (self.row_end - self.row_start) * (self.col_end - self.col_start)


def _check_int(name: str, v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool):
        raise fail(ErrorCode.VALIDATION_ERROR, f"{name} must be an integer")
    return v


def tile_geometry(view_rows: int, view_cols: int, level: int, x: int, y: int, size: int = TILE_SIZE) -> TileGeometry:
    """Raw coordinates of tile (x, y) at `level` in an R x C view; INVALID_ROI if the tile starts outside it."""
    level, x, y, size = (_check_int(n, v) for n, v in (("level", level), ("x", x), ("y", y), ("size", size)))
    if not 0 <= level <= MAX_LEVEL:
        raise fail(ErrorCode.VALIDATION_ERROR, f"level must be within 0..{MAX_LEVEL}")
    if not 1 <= size <= TILE_SIZE:
        raise fail(ErrorCode.VALIDATION_ERROR, f"size must be within 1..{TILE_SIZE}")
    if not (0 <= x <= MAX_SAFE_INTEGER and 0 <= y <= MAX_SAFE_INTEGER):
        raise fail(ErrorCode.INVALID_ROI, "tile coordinates must be within [0, 2^53-1]")
    block = 1 << level
    span = size * block
    r0, c0 = y * span, x * span
    if r0 >= view_rows or c0 >= view_cols:
        raise fail(
            ErrorCode.INVALID_ROI, f"tile ({x}, {y}) at level {level} is outside the {view_rows}x{view_cols} view"
        )
    r1, c1 = min(view_rows, r0 + span), min(view_cols, c0 + span)
    return TileGeometry(level, x, y, size, block, r0, r1, c0, c1, -(-(r1 - r0) // block), -(-(c1 - c0) // block))


# --------------------------------------------------------------------------------------------- block aggregates
@dataclass
class BlockAggregates:
    """Record-format-v1 fields for a grid of blocks (struct of arrays). Merges by addition and min/max."""

    finite: I64
    nan: I64
    pos_inf: I64
    neg_inf: I64
    flags: U8
    sum: F64
    sum_abs: F64
    sum_sq: F64
    min: F64  # +inf where no finite element (never read then: finite == 0 is NO_FINITE)
    max: F64

    @classmethod
    def empty(cls, rows: int, cols: int) -> BlockAggregates:
        def z() -> I64:
            return np.zeros((rows, cols), np.int64)

        def neg0() -> F64:  # -0.0 is the additive identity that keeps the sign of a lone -0.0
            return np.full((rows, cols), -0.0)

        return cls(z(), z(), z(), z(), np.zeros((rows, cols), np.uint8), neg0(), neg0(), neg0(),
                   np.full((rows, cols), np.inf), np.full((rows, cols), -np.inf))  # fmt: skip

    @classmethod
    def of_values(cls, v: F64, row_phase: int, col_phase: int, block: int) -> BlockAggregates:
        """Aggregates of a 2D float64 array whose first row / column sit at offset `row_phase` / `col_phase`
        (mod `block`) from a block boundary. Returns one record per (partial) block the array touches.

        The array is padded to whole blocks with each reduction's identity (-0.0 for sums, which also keeps the sign
        of a lone -0.0; 0 for counts; +/-inf for min/max) and reduced over a (rows, b, cols, b) reshape."""
        fin = np.isfinite(v)
        with np.errstate(over="ignore", invalid="ignore"):
            fv = np.where(fin, v, -0.0)
            out = cls(
                _blocks(fin, row_phase, col_phase, block, False).sum(axis=(1, 3), dtype=np.int64),
                _blocks(np.isnan(v), row_phase, col_phase, block, False).sum(axis=(1, 3), dtype=np.int64),
                _blocks(np.isposinf(v), row_phase, col_phase, block, False).sum(axis=(1, 3), dtype=np.int64),
                _blocks(np.isneginf(v), row_phase, col_phase, block, False).sum(axis=(1, 3), dtype=np.int64),
                np.zeros(_grid(v.shape, row_phase, col_phase, block), np.uint8),
                _blocks(fv, row_phase, col_phase, block, -0.0).sum(axis=(1, 3), initial=-0.0),
                _blocks(np.abs(fv), row_phase, col_phase, block, -0.0).sum(axis=(1, 3), initial=-0.0),
                _blocks(fv * fv, row_phase, col_phase, block, -0.0).sum(axis=(1, 3), initial=-0.0),
                _blocks(np.where(fin, v, np.inf), row_phase, col_phase, block, np.inf).min(axis=(1, 3)),
                _blocks(np.where(fin, v, -np.inf), row_phase, col_phase, block, -np.inf).max(axis=(1, 3)),
            )
        out._flag_overflow()
        return out

    def merge_into(self, other: BlockAggregates, r: int, c: int) -> None:
        """Adds `other` into this grid at block offset (r, c)."""
        h, w = other.finite.shape
        sl = (slice(r, r + h), slice(c, c + w))
        for name in ("finite", "nan", "pos_inf", "neg_inf"):
            dst = getattr(self, name)
            dst[sl] += getattr(other, name)  # int64: a tile covers < 2^63 cells by construction
        with np.errstate(over="ignore", invalid="ignore"):
            self.sum[sl] += other.sum
            self.sum_abs[sl] += other.sum_abs
            self.sum_sq[sl] += other.sum_sq
        np.minimum(self.min[sl], other.min, out=self.min[sl])
        np.maximum(self.max[sl], other.max, out=self.max[sl])
        self.flags[sl] |= other.flags
        sub = BlockAggregates(*(getattr(self, f)[sl] for f in _FIELDS))
        sub._flag_overflow()
        self.flags[sl] = sub.flags

    def _flag_overflow(self) -> None:
        # Every summand is finite, so a non-finite sum (Inf, or NaN from Inf - Inf) is an overflow.
        self.flags |= np.where(~np.isfinite(self.sum), SUM_OVERFLOW, 0).astype(np.uint8)
        self.flags |= np.where(~np.isfinite(self.sum_abs), SUM_ABS_OVERFLOW, 0).astype(np.uint8)
        self.flags |= np.where(~np.isfinite(self.sum_sq), SUM_SQ_OVERFLOW, 0).astype(np.uint8)

    def display(self, aggregation: Aggregation) -> tuple[F32, U8]:
        """float32 display values and status bitfields. Non-drawable cells hold a NaN placeholder."""
        if aggregation not in AGGREGATIONS:
            raise fail(ErrorCode.VALIDATION_ERROR, f"unknown aggregation {aggregation!r}")
        n = self.finite
        status = np.zeros(n.shape, np.uint8)
        status |= np.where(n == 0, NO_FINITE, 0).astype(np.uint8)
        status |= np.where(self.nan > 0, HAS_NAN, 0).astype(np.uint8)
        status |= np.where(self.pos_inf > 0, HAS_POS_INF, 0).astype(np.uint8)
        status |= np.where(self.neg_inf > 0, HAS_NEG_INF, 0).astype(np.uint8)
        has = n > 0
        safe_n = np.where(has, n, 1).astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if aggregation == "mean":
                val = self.sum / safe_n
            elif aggregation == "mean_abs":
                val = self.sum_abs / safe_n
            elif aggregation == "rms":
                val = np.sqrt(self.sum_sq / safe_n)
            elif aggregation == "min":
                val = self.min.copy()
            else:
                val = self.max.copy()
            dep = _DEPENDS[aggregation]
            bad = has & (((self.flags & dep) != 0) | ~(np.abs(val) <= F32_MAX))
        status |= np.where(bad, NOT_REPRESENTABLE, 0).astype(np.uint8)
        drawable = has & ~bad
        values = np.full(n.shape, np.nan, np.float32)
        values[drawable] = val[drawable].astype(np.float32)
        return values, status


_FIELDS = ("finite", "nan", "pos_inf", "neg_inf", "flags", "sum", "sum_abs", "sum_sq", "min", "max")


def _grid(shape: tuple[int, ...], row_phase: int, col_phase: int, block: int) -> tuple[int, int]:
    return -(-(row_phase + shape[0]) // block), -(-(col_phase + shape[1]) // block)


def _blocks(a: npt.NDArray[Any], row_phase: int, col_phase: int, block: int, fill: Any) -> npt.NDArray[Any]:
    """`a` placed at (row_phase, col_phase) in a fill-padded array of whole blocks, viewed as (rows, b, cols, b)."""
    gr, gc = _grid(a.shape, row_phase, col_phase, block)
    h, w = a.shape
    if (row_phase, col_phase, gr * block, gc * block) != (0, 0, h, w):
        padded = np.full((gr * block, gc * block), fill, dtype=a.dtype)
        padded[row_phase : row_phase + h, col_phase : col_phase + w] = a
        a = padded
    return a.reshape(gr, block, gc, block)


# ------------------------------------------------------------------------------------------------------- budget
@dataclass(frozen=True)
class TileBudget:
    """The tile engine's own limits, separate from the ROI-table budget (doc 03 section 5, doc 11)."""

    max_tile_cells: int = 65_536  # output cells (settings max_tile_cells, read-only 256 x 256)
    direct_tile_max_bytes: int = 64 * MIB  # covered source bytes readable directly (settings direct_tile_max_mib)
    max_read_bytes: int = 256 * MIB  # bytes actually read, incl. stripe overhead of non-contiguous views
    band_cells: int = 1 << 18  # cells per positioned-read rectangle
    memory_bytes: int = 512 * MIB  # working set of one tile computation

    def __post_init__(self) -> None:
        for name in ("max_tile_cells", "direct_tile_max_bytes", "max_read_bytes", "band_cells", "memory_bytes"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> TileBudget:
        """Budget from the server settings: direct_tile_max_mib bounds covered bytes, reads may exceed it by at
        most READ_AMPLIFICATION (stripes of sliced or transposed views), and a tile may use the memory budget
        divided by the number of concurrent tile computations."""
        direct = int(settings["direct_tile_max_mib"]) * MIB
        return cls(
            max_tile_cells=int(settings["max_tile_cells"]),
            direct_tile_max_bytes=direct,
            max_read_bytes=READ_AMPLIFICATION * direct,
            memory_bytes=int(settings["memory_budget_mib"]) * MIB // max(1, int(settings["max_active_tiles"])),
        )


READ_AMPLIFICATION = 4
# Working-set estimate: per band cell: index arrays (3 x 8 B), raw element (<= 8 B), float64 value, masks and
# float64 temporaries of the reduction (~5 x 8 B); per output cell: one record (4 x 8 B counts, 5 x 8 B, flags)
# plus display buffers.
BYTES_PER_BAND_CELL = 96
BYTES_PER_OUTPUT_CELL = 96


def working_set_bytes(budget: TileBudget, output_cells: int, read_chunk_bytes: int) -> int:
    return budget.band_cells * BYTES_PER_BAND_CELL + output_cells * BYTES_PER_OUTPUT_CELL + read_chunk_bytes


# --------------------------------------------------------------------------------------------------------- tile
@dataclass(frozen=True)
class Tile:
    geometry: TileGeometry
    aggregation: Aggregation
    source_dtype: str
    values: F32  # (rows, cols) float32 display values; NaN placeholder where not drawable
    status: U8  # (rows, cols) TileStatus bitfields
    finite_count: int  # source elements under the tile
    nonfinite_count: int
    numeric_precision: Literal["lossless", "rounded", "overflowed"]
    covered_bytes: int
    bytes_read: int
    quality: Literal["exact"] = "exact"
    value_precision: Literal["float32_visualization"] = "float32_visualization"
    warnings: tuple[str, ...] = field(default=())


def _as_float64(vals: npt.NDArray[Any], dt: DType) -> tuple[F64, bool]:
    """Values for aggregation and whether any was rounded on the way (64-bit integers beyond 2^53)."""
    if dt.kind == "float":
        return vals.astype(np.float64, copy=False), False
    if dt.kind == "bool":
        return vals.astype(np.float64), False
    rounded = False
    if dt.size == 8 and vals.size:
        big = vals > MAX_SAFE_INTEGER if dt.kind == "uint" else (vals > MAX_SAFE_INTEGER) | (vals < -MAX_SAFE_INTEGER)
        rounded = bool(big.any())
    return vals.astype(np.float64), rounded


def _rectangles(
    rv: ResolvedView, shape: tuple[int, ...], g: TileGeometry, band_cells: int
) -> list[tuple[int, int, int, int]]:
    """Read rectangles covering the tile's raw region, each <= band_cells, long along the view axis that is
    contiguous in storage (smaller stride), so each rectangle reads few, dense stripes."""
    strides = [1] * len(shape)
    for k in range(len(shape) - 2, -1, -1):
        strides[k] = strides[k + 1] * shape[k + 1]
    rs = strides[rv.row_axis] if rv.row_axis is not None else math.inf
    cs = strides[rv.col_axis] if rv.col_axis is not None else 1
    height, width = g.row_end - g.row_start, g.col_end - g.col_start
    out: list[tuple[int, int, int, int]] = []
    if cs <= rs:  # rows are the slow axis: wide rectangles, row bands outside
        w = min(width, band_cells)
        h = max(1, min(height, band_cells // w))
        for r in range(g.row_start, g.row_end, h):
            for c in range(g.col_start, g.col_end, w):
                out.append((r, min(g.row_end, r + h), c, min(g.col_end, c + w)))
    else:  # columns are the slow axis (e.g. a transposed view): tall rectangles, column bands outside
        h = min(height, band_cells)
        w = max(1, min(width, band_cells // h))
        for c in range(g.col_start, g.col_end, w):
            for r in range(g.row_start, g.row_end, h):
                out.append((r, min(g.row_end, r + h), c, min(g.col_end, c + w)))
    return out


def compute_tile(
    ck: Checkpoint,
    key: str,
    view: View | None,
    *,
    level: int,
    x: int,
    y: int,
    size: int = TILE_SIZE,
    aggregation: Aggregation,
    budget: TileBudget | None = None,
    consistency: Consistency = Consistency.BEST_EFFORT,
) -> Tile:
    """One exact tile by direct bounded reads. Raises RESOURCE_LIMIT before reading when the tile needs more than
    the direct-read, read or memory budget (such tiles need the exact pyramid), and FILE_CHANGED if the source
    changes at any point; no partial tile is ever returned."""
    budget = budget or TileBudget()
    if aggregation not in AGGREGATIONS:
        raise fail(ErrorCode.VALIDATION_ERROR, f"aggregation must be one of {', '.join(AGGREGATIONS)}")
    meta = ck.tensor(key)
    rv = resolve_view(meta.shape, view)
    g = tile_geometry(rv.rows, rv.cols, level, x, y, size)
    if g.rows * g.cols > budget.max_tile_cells:
        raise fail(
            ErrorCode.RESOURCE_LIMIT, f"tile has {g.rows * g.cols} cells, above max_tile_cells {budget.max_tile_cells}"
        )
    dt = lookup(meta.dtype)
    covered = g.covered_cells * dt.size
    if covered > budget.direct_tile_max_bytes:
        raise fail(
            ErrorCode.RESOURCE_LIMIT,
            f"tile covers {covered} source bytes, above direct_tile_max_mib "
            f"({budget.direct_tile_max_bytes // MIB} MiB); "
            "it needs the exact pyramid, which is not available yet",
        )
    need = working_set_bytes(budget, g.rows * g.cols, ck.limits.read_chunk_bytes)
    if need > budget.memory_bytes:
        raise fail(
            ErrorCode.RESOURCE_LIMIT,
            f"tile needs ~{need} bytes of working memory, above its budget {budget.memory_bytes}",
        )
    band = min(budget.band_cells, ck.limits.roi_hard_max_cells)
    rects = _rectangles(rv, meta.shape, g, band)
    planned = sum(ck.roi_read_bytes(key, r0, r1, c0, c1, view, max_cells=band) for r0, r1, c0, c1 in rects)
    if planned > budget.max_read_bytes:
        raise fail(
            ErrorCode.RESOURCE_LIMIT,
            f"tile needs {planned} bytes of reads, above its read budget {budget.max_read_bytes}",
        )

    acc = BlockAggregates.empty(g.rows, g.cols)
    rounded = False
    bytes_read = 0
    for r0, r1, c0, c1 in rects:
        raw = ck.read_roi_raw(key, r0, r1, c0, c1, view, max_cells=band, consistency=consistency)
        bytes_read += raw.stripe_bytes
        vals, rnd = _as_float64(raw.values(), dt)
        rounded |= rnd
        rel_r, rel_c = r0 - g.row_start, c0 - g.col_start
        part = BlockAggregates.of_values(vals, rel_r % g.block, rel_c % g.block, g.block)
        acc.merge_into(part, rel_r // g.block, rel_c // g.block)
    ck.check()  # a change after the last read also invalidates the tile

    values, status = acc.display(aggregation)
    finite = int(acc.finite.sum())
    not_repr = int(np.count_nonzero(status & NOT_REPRESENTABLE))
    exact_agg = aggregation in ("min", "max") or (g.level == 0 and aggregation in ("mean", "mean_abs"))
    precision: Literal["lossless", "rounded", "overflowed"] = (
        "overflowed" if not_repr else "rounded" if rounded or not exact_agg else "lossless"
    )
    warnings: list[str] = []
    if rounded:
        warnings.append(
            "64-bit integer values beyond +/-(2^53-1) were rounded to float64 before aggregation; "
            "exact values come from /tensors/roi"
        )
    if not_repr:
        warnings.append(
            f"{not_repr} cell(s) overflowed float64 or exceed the float32 display range; "
            "they are withheld (NOT_REPRESENTABLE)"
        )
    return Tile(
        geometry=g,
        aggregation=aggregation,
        source_dtype=dt.name,
        values=values,
        status=status,
        finite_count=finite,
        nonfinite_count=g.covered_cells - finite,
        numeric_precision=precision,
        covered_bytes=covered,
        bytes_read=bytes_read,
        warnings=tuple(warnings),
    )
