"""Exact aggregate pyramids (doc 03 section 5, ADR-012): one streaming pass over a source view builds the level-4
base records and every coarser level by merging children, in the frozen record format v1.

Records (little-endian, `pyramid_record_format = 1`):

* base (L4, <= 256 cells): finite, nan, pos_inf, neg_inf counts as uint16; flags uint8 + 7 B padding; sum, sum_abs,
  sum_sq, min, max as float64 -> 56 B;
* coarse (L >= 5): the same with uint64 counts -> 80 B.

Merging is the PR #6 record algebra (`tile.BlockAggregates`): counts and sums add (count-weighted means, RMS from sums
of squares), min/max combine, overflow flags propagate (a float64 sum that leaves the finite range although every
input was finite). Counter additions are checked; an overflow aborts the build with RESOURCE_LIMIT. A record without
finite elements stores sums -0.0, min +Inf, max -Inf and no flags (its display is NO_FINITE, never a number).

Orientation. The pass follows storage order: when the view's columns are the slow storage axis (a transposed matrix),
the builder streams the transposed view, so reads stay sequential; such a pyramid stores its grids transposed
(`orientation = "cols"`), which the reader undoes. Contiguous views (any rank-0/1/2 tensor, contiguous slices of
higher ranks) are read by plain positioned reads of whole row bands; other views by bounded `read_roi_raw`
rectangles. Nothing larger than one band (at most `band_cells` cells) and one row of records per level is in memory.

File (`.wlpyr`, file_version 1): a 16-byte preamble (magic, file version, record format); each level's records,
row-major in stored orientation; a CRC32 table (one uint32 per run of 256 records of a stored row); a canonical JSON
trailer (identity, dimensions, level offsets, per-level SHA-256, CRC table SHA-256); a 56-byte footer (magic, trailer
offset and length, trailer SHA-256). Loading validates the footer, the trailer checksum and its every field against
the identity being asked for and a freshly computed layout (offsets, record sizes, file size); every tile read
verifies the CRC32 of the record runs it touches and checks each record (counts sum to the block's real cell count,
flags and padding, finite-less records, overflow flags <-> non-finite sums). Any mismatch raises PyramidCorrupt.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from weightlab.analysis.tile import (
    PYRAMID_RECORD_FORMAT,
    TILE_NUMERICS_VERSION,
    Aggregation,
    BlockAggregates,
    Tile,
    TileGeometry,
    _as_float64,
    columns_are_slow,
    finish_tile,
    storage_strides,
    tile_geometry,
)
from weightlab.core.errors import ErrorCode, fail
from weightlab.formats.checkpoint import Checkpoint
from weightlab.formats.dtypes import lookup
from weightlab.formats.selection import ResolvedView, View, resolve_view

BASE_LEVEL = 4
CHUNK_RECORDS = 256  # records per CRC32 run of a stored row
FILE_VERSION = 1
MAGIC = b"WLPYRAMD"
FOOTER_MAGIC = b"WLPYREND"
PREAMBLE = struct.Struct("<8sII")  # magic, file_version, record format
FOOTER = struct.Struct("<8sQQ32s")  # magic, trailer offset, trailer length, sha256(trailer)
TRAILER_MAX = 64 * 1024  # identity (a few fingerprints, one key <= 4 KiB) and <= 53 levels fit easily
FORMAT_NAME = "weightlab.pyramid"
UINT16_MAX = 2**16 - 1
INT64_MAX = 2**63 - 1

_FIELDS = [("finite", "nan", "pos_inf", "neg_inf"), ("flags",), ("sum", "sum_abs", "sum_sq", "min", "max")]
BASE_DTYPE = np.dtype(
    [("finite", "<u2"), ("nan", "<u2"), ("pos_inf", "<u2"), ("neg_inf", "<u2"), ("flags", "u1"), ("pad", "V7"),
     ("sum", "<f8"), ("sum_abs", "<f8"), ("sum_sq", "<f8"), ("min", "<f8"), ("max", "<f8")]
)  # fmt: skip
COARSE_DTYPE = np.dtype(
    [("finite", "<u8"), ("nan", "<u8"), ("pos_inf", "<u8"), ("neg_inf", "<u8"), ("flags", "u1"), ("pad", "V7"),
     ("sum", "<f8"), ("sum_abs", "<f8"), ("sum_sq", "<f8"), ("min", "<f8"), ("max", "<f8")]
)  # fmt: skip
assert BASE_DTYPE.itemsize == 56 and COARSE_DTYPE.itemsize == 80  # noqa: PT018  (frozen record sizes)

COUNTS = ("finite", "nan", "pos_inf", "neg_inf")
FLOATS = ("sum", "sum_abs", "sum_sq", "min", "max")
FLAG_BITS = 1 | 2 | 4


class PyramidCorrupt(Exception):
    """A stored pyramid failed validation; it must be discarded and rebuilt, never served."""


class BuildControl(Protocol):
    """Called after every read with the bytes just read: reports progress, enforces cancellation and the wall-time
    budget (by raising), between bounded chunks."""

    def tick(self, nbytes: int) -> None: ...


class _NoControl:
    def tick(self, nbytes: int) -> None:
        return None


# --------------------------------------------------------------------------------------------------------- layout
def record_dtype(level: int) -> np.dtype[Any]:
    return BASE_DTYPE if level == BASE_LEVEL else COARSE_DTYPE


@dataclass(frozen=True)
class LevelInfo:
    level: int
    stored_rows: int  # grid in stored orientation
    stored_cols: int
    record_size: int
    offset: int
    length: int
    crc_offset: int  # index of this level's first entry in the CRC table
    crc_count: int

    @property
    def chunks_per_row(self) -> int:
        return -(-self.stored_cols // CHUNK_RECORDS)


@dataclass(frozen=True)
class Layout:
    rows: int  # view rows / cols
    cols: int
    orientation: str  # "rows": stored grid = view grid; "cols": stored grid = transposed view grid
    levels: tuple[LevelInfo, ...]
    crc_offset: int
    crc_length: int

    @property
    def stored_shape(self) -> tuple[int, int]:
        return (self.cols, self.rows) if self.orientation == "cols" else (self.rows, self.cols)

    @property
    def top_level(self) -> int:
        return self.levels[-1].level

    @property
    def data_bytes(self) -> int:
        """Bytes before the trailer: preamble, records and CRC table."""
        return self.crc_offset + self.crc_length

    def level(self, level: int) -> LevelInfo:
        return self.levels[min(level, self.top_level) - BASE_LEVEL]

    @classmethod
    def plan(cls, rows: int, cols: int, orientation: str) -> Layout:
        if rows < 1 or cols < 1:
            raise fail(ErrorCode.VALIDATION_ERROR, "a pyramid needs a non-empty view")
        sr, sc = (cols, rows) if orientation == "cols" else (rows, cols)
        levels: list[LevelInfo] = []
        off = PREAMBLE.size
        crc = 0
        lv = BASE_LEVEL
        while True:
            gr, gc = -(-sr // (1 << lv)), -(-sc // (1 << lv))
            size = record_dtype(lv).itemsize
            n_crc = gr * -(-gc // CHUNK_RECORDS)
            levels.append(LevelInfo(lv, gr, gc, size, off, gr * gc * size, crc, n_crc))
            off += gr * gc * size
            crc += n_crc
            if gr == 1 and gc == 1:
                break
            lv += 1
        return cls(rows, cols, orientation, tuple(levels), off, crc * 4)

    def as_json(self) -> list[dict[str, int]]:
        return [
            {"level": li.level, "stored_rows": li.stored_rows, "stored_cols": li.stored_cols,
             "record_size": li.record_size, "offset": li.offset, "length": li.length}
            for li in self.levels
        ]  # fmt: skip


def estimate_bytes(rows: int, cols: int) -> int:
    """Upper bound of a pyramid file's size (records + CRC table + trailer + footer)."""
    lay = Layout.plan(rows, cols, "rows")
    return lay.data_bytes + TRAILER_MAX + FOOTER.size


# ---------------------------------------------------------------------------------------------- record conversion
def to_records(agg: BlockAggregates, level: int) -> npt.NDArray[np.void]:
    """Records of one grid (stored orientation) in the frozen layout. Counts are checked against the field width."""
    dt = record_dtype(level)
    out = np.zeros(agg.finite.shape, dtype=dt)
    for f in COUNTS:
        v = getattr(agg, f)
        if v.size and (int(v.min()) < 0 or (level == BASE_LEVEL and int(v.max()) > UINT16_MAX)):
            raise fail(ErrorCode.RESOURCE_LIMIT, f"pyramid counter {f} outside its record field")
        out[f] = v
    out["flags"] = agg.flags
    for f in FLOATS:
        out[f] = getattr(agg, f)
    none = agg.finite == 0
    out["sum"][none] = out["sum_abs"][none] = out["sum_sq"][none] = -0.0
    out["min"][none] = np.inf
    out["max"][none] = -np.inf
    out["flags"][none] = 0
    return out


def from_records(rec: npt.NDArray[np.void]) -> BlockAggregates:
    def counts(f: str) -> npt.NDArray[np.int64]:
        v = rec[f]
        if v.dtype == np.dtype("<u8") and v.size and int(v.max()) > INT64_MAX:
            raise PyramidCorrupt(f"counter {f} beyond int64")
        return v.astype(np.int64)

    return BlockAggregates(
        counts("finite"), counts("nan"), counts("pos_inf"), counts("neg_inf"),
        rec["flags"].astype(np.uint8),
        *(rec[f].astype(np.float64) for f in FLOATS),
    )  # fmt: skip


def base_records(v: npt.NDArray[np.float64]) -> BlockAggregates:
    """Level-4 records of an array whose first row and column sit on 16-cell block boundaries.

    Each block's (up to) 256 values are laid out contiguously in one fixed order (row-major within the block, the
    reduction identity in the cells beyond a clipped edge) and reduced along that axis, so a block's float64 sums
    are bitwise the same however the stream is cut into bands and column chunks."""
    h, w = v.shape
    gr, gc = -(-h // 16), -(-w // 16)
    fin = np.isfinite(v)

    def blocks(a: npt.NDArray[Any], fill: Any) -> npt.NDArray[Any]:
        if (gr * 16, gc * 16) != (h, w):
            p = np.full((gr * 16, gc * 16), fill, dtype=a.dtype)
            p[:h, :w] = a
            a = p
        return np.ascontiguousarray(a.reshape(gr, 16, gc, 16).transpose(0, 2, 1, 3)).reshape(gr, gc, 256)

    with np.errstate(over="ignore", invalid="ignore"):
        fv = np.where(fin, v, -0.0)
        out = BlockAggregates(
            blocks(fin, False).sum(axis=2, dtype=np.int64),
            blocks(np.isnan(v), False).sum(axis=2, dtype=np.int64),
            blocks(np.isposinf(v), False).sum(axis=2, dtype=np.int64),
            blocks(np.isneginf(v), False).sum(axis=2, dtype=np.int64),
            np.zeros((gr, gc), np.uint8),
            blocks(fv, -0.0).sum(axis=2, initial=-0.0),
            blocks(np.abs(fv), -0.0).sum(axis=2, initial=-0.0),
            blocks(fv * fv, -0.0).sum(axis=2, initial=-0.0),
            blocks(np.where(fin, v, np.inf), np.inf).min(axis=2),
            blocks(np.where(fin, v, -np.inf), -np.inf).max(axis=2),
        )
    out._flag_overflow()
    return out


def checked_add(a: npt.NDArray[np.int64], b: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Counter addition that refuses to wrap (doc 03 section 5): RESOURCE_LIMIT instead."""
    if a.size and bool(np.any(a > INT64_MAX - b)):
        raise fail(ErrorCode.RESOURCE_LIMIT, "pyramid counter overflow")
    return a + b


def merge_pairs(agg: BlockAggregates, axis: int) -> BlockAggregates:
    """Parent records of consecutive pairs along `axis` (a trailing odd record merges alone)."""
    n = agg.finite.shape[axis]
    even = n - n % 2

    def take(a: npt.NDArray[Any], start: int, stop: int, step: int) -> npt.NDArray[Any]:
        sl = [slice(None), slice(None)]
        sl[axis] = slice(start, stop, step)
        return a[tuple(sl)]

    def cat(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.concatenate((a, b), axis=axis) if n % 2 else a

    def pair(f: str, op: Callable[[Any, Any], Any]) -> npt.NDArray[Any]:
        a = getattr(agg, f)
        return cat(op(take(a, 0, even, 2), take(a, 1, even, 2)), take(a, even, n, 1))

    with np.errstate(over="ignore", invalid="ignore"):
        out = BlockAggregates(
            pair("finite", checked_add), pair("nan", checked_add),
            pair("pos_inf", checked_add), pair("neg_inf", checked_add),
            pair("flags", np.bitwise_or),
            pair("sum", np.add), pair("sum_abs", np.add), pair("sum_sq", np.add),
            pair("min", np.minimum), pair("max", np.maximum),
        )  # fmt: skip
    out._flag_overflow()
    return out


def concat_rows(a: BlockAggregates | None, b: BlockAggregates) -> BlockAggregates:
    if a is None:
        return b
    return BlockAggregates(*(np.concatenate((getattr(a, f), getattr(b, f)), axis=0) for f in _AGG_FIELDS))


_AGG_FIELDS = ("finite", "nan", "pos_inf", "neg_inf", "flags", "sum", "sum_abs", "sum_sq", "min", "max")


def _rows(agg: BlockAggregates, start: int, stop: int) -> BlockAggregates:
    return BlockAggregates(*(getattr(agg, f)[start:stop] for f in _AGG_FIELDS))


# ------------------------------------------------------------------------------------------------------- sinks
class Sink(Protocol):
    def write_at(self, offset: int, data: bytes) -> None: ...


class FileSink:
    """Writes the pyramid into a (temporary) file; the caller publishes it atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._f = open(path, "wb")  # noqa: SIM115  (closed by close())

    def write_at(self, offset: int, data: bytes) -> None:
        self._f.seek(offset)
        self._f.write(data)

    def close(self, *, sync: bool) -> None:
        if not self._f.closed:
            if sync:
                self._f.flush()
                os.fsync(self._f.fileno())
            self._f.close()


class MemorySink:
    """A pyramid kept in memory for the session (when it cannot fit the disk cache)."""

    def __init__(self, size_hint: int) -> None:
        self.buf = bytearray(size_hint)

    def write_at(self, offset: int, data: bytes) -> None:
        end = offset + len(data)
        if end > len(self.buf):
            self.buf.extend(b"\0" * (end - len(self.buf)))
        self.buf[offset:end] = data


# ---------------------------------------------------------------------------------------------------- source
@dataclass(frozen=True)
class StreamPlan:
    """How one view is streamed in storage order."""

    key: str
    view: View | None
    rv: ResolvedView
    orientation: str
    stored_rows: int
    stored_cols: int
    contiguous: bool
    base_offset: int  # first element of the stored array (contiguous plans)
    esize: int
    dtype: str


def stream_plan(ck: Checkpoint, key: str, view: View | None) -> StreamPlan:
    meta = ck.tensor(key)
    rv = resolve_view(meta.shape, view)
    dt = lookup(meta.dtype)
    orientation = "cols" if columns_are_slow(rv, meta.shape) else "rows"
    st = storage_strides(meta.shape)
    s_r = st[rv.row_axis] if rv.row_axis is not None else 0
    s_c = st[rv.col_axis] if rv.col_axis is not None else 1
    base = sum(idx * st[a] for a, idx in rv.fixed)
    sr, sc = (rv.cols, rv.rows) if orientation == "cols" else (rv.rows, rv.cols)
    si, sj = (s_c, s_r) if orientation == "cols" else (s_r, s_c)
    # Stored element (i, j) is at base + i*si + j*sj: one C-order run iff consecutive j are adjacent and rows abut.
    contiguous = (sc == 1 or sj == 1) and (sr == 1 or si == sc)
    return StreamPlan(key, view, rv, orientation, sr, sc, contiguous, base, dt.size, dt.name)


def _read_band(
    ck: Checkpoint, p: StreamPlan, i0: int, i1: int, j0: int, j1: int, max_cells: int
) -> tuple[npt.NDArray[np.float64], bool, int]:
    """Stored-orientation values [i0, i1) x [j0, j1) as float64, whether 64-bit integers were rounded, bytes read."""
    dt = lookup(p.dtype)
    h, w = i1 - i0, j1 - j0
    if p.contiguous:
        if j0 == 0 and j1 == p.stored_cols:
            raw = ck.read_elements(p.key, p.base_offset + i0 * p.stored_cols, h * w).reshape(h, w)
        else:
            raw = np.stack([ck.read_elements(p.key, p.base_offset + i * p.stored_cols + j0, w) for i in range(i0, i1)])
        nbytes = h * w * p.esize
    else:
        if p.orientation == "cols":  # stored (i, j) = view (j, i)
            roi = ck.read_roi_raw(p.key, j0, j1, i0, i1, p.view, max_cells=max_cells)
            raw = roi.values().T
        else:
            roi = ck.read_roi_raw(p.key, i0, i1, j0, j1, p.view, max_cells=max_cells)
            raw = roi.values()
        nbytes = roi.stripe_bytes
    vals, rounded = _as_float64(raw, dt)
    return np.ascontiguousarray(vals), rounded, nbytes


# ------------------------------------------------------------------------------------------------------- build
@dataclass
class BuildResult:
    pyramid_id: str
    layout: Layout
    total_bytes: int
    bytes_read: int
    rounded_inputs: bool
    level_sha256: dict[int, str]


BAND_CELLS = 1 << 19  # cells per read band: ~50 MiB of float64 temporaries at most
BYTES_PER_BAND_CELL = 96
BYTES_PER_ROW_RECORD = 96


def build_working_set(
    rows: int, cols: int, orientation: str, read_chunk_bytes: int, band_cells: int = BAND_CELLS
) -> int:
    """Estimated peak working set of a build: one band and its temporaries, the band's records, one pending row of
    records per level (about two base rows in all), the CRC table and one read buffer."""
    lay = Layout.plan(rows, cols, orientation)
    base = lay.levels[0]
    k = max(1, band_cells // (16 * lay.stored_shape[1]))
    rows_in_band = k + 3
    crc = lay.crc_length
    return (
        band_cells * BYTES_PER_BAND_CELL
        + rows_in_band * base.stored_cols * BYTES_PER_ROW_RECORD
        + crc
        + read_chunk_bytes
    )


class _LevelWriter:
    """Writes one level's records row by row in stored orientation, with CRC32 runs and a running SHA-256."""

    def __init__(self, sink: Sink, info: LevelInfo, crc: npt.NDArray[np.uint32]) -> None:
        self.sink, self.info, self.crc = sink, info, crc
        self.row = 0
        self.sha = hashlib.sha256()

    def write(self, agg: BlockAggregates) -> None:
        rec = to_records(agg, self.info.level)
        n, w = rec.shape
        if w != self.info.stored_cols or self.row + n > self.info.stored_rows:
            raise AssertionError("pyramid level row out of plan")  # a builder bug, never data dependent
        data = rec.tobytes()
        self.sink.write_at(self.info.offset + self.row * w * self.info.record_size, data)
        self.sha.update(data)
        rb = w * self.info.record_size
        cs = CHUNK_RECORDS * self.info.record_size
        for r in range(n):
            row = data[r * rb : (r + 1) * rb]
            base = self.info.crc_offset + (self.row + r) * self.info.chunks_per_row
            for c in range(self.info.chunks_per_row):
                self.crc[base + c] = zlib.crc32(row[c * cs : (c + 1) * cs])
        self.row += n


def build_pyramid(
    ck: Checkpoint,
    key: str,
    view: View | None,
    identity: Mapping[str, Any],
    pyramid_id: str,
    sink: Sink,
    control: BuildControl | None = None,
    *,
    band_cells: int = BAND_CELLS,
) -> BuildResult:
    """One streaming pass: base records per band, parents merged as soon as both children rows exist. The source is
    re-checked after the last read; any FILE_CHANGED (or a control that raises) aborts with nothing published."""
    control = control or _NoControl()
    p = stream_plan(ck, key, view)
    lay = Layout.plan(p.rv.rows, p.rv.cols, p.orientation)
    crc = np.zeros(lay.crc_length // 4, dtype="<u4")
    writers = [_LevelWriter(sink, li, crc) for li in lay.levels]
    pending: list[BlockAggregates | None] = [None] * len(lay.levels)
    sink.write_at(0, PREAMBLE.pack(MAGIC, FILE_VERSION, PYRAMID_RECORD_FORMAT))

    def feed(i: int, rows: BlockAggregates | None, final: bool) -> None:
        """Rows of level index i (already written); merges pairs into level i+1."""
        if i + 1 >= len(lay.levels):
            return
        cur = pending[i] if rows is None else concat_rows(pending[i], rows)
        pending[i] = None
        if cur is None:
            if final:
                feed(i + 1, None, True)
            return
        n = cur.finite.shape[0]
        if n % 2 and not final:
            pending[i] = _rows(cur, n - 1, n)
            cur = _rows(cur, 0, n - 1)
        if cur.finite.shape[0]:
            parent = merge_pairs(merge_pairs(cur, 0), 1)
            writers[i + 1].write(parent)
            feed(i + 1, parent, final)
        elif final:
            feed(i + 1, None, True)

    sr, sc = p.stored_rows, p.stored_cols
    cells_cap = min(band_cells, ck.limits.roi_hard_max_cells)
    if p.contiguous:
        cells_cap = min(cells_cap, ck.limits.read_chunk_bytes // p.esize)
    if 16 * sc <= cells_cap:
        width, k = sc, max(1, cells_cap // (16 * sc))
    else:
        width, k = max(16, (cells_cap // 16) // 16 * 16), 1
    rounded = False
    bytes_read = 0
    gc = lay.levels[0].stored_cols
    for b0 in range(0, lay.levels[0].stored_rows, k):
        i0, i1 = b0 * 16, min(sr, (b0 + k) * 16)
        band = BlockAggregates.empty(-(-(i1 - i0) // 16), gc)
        for j0 in range(0, sc, width):
            j1 = min(sc, j0 + width)
            vals, rnd, nbytes = _read_band(ck, p, i0, i1, j0, j1, cells_cap)
            rounded |= rnd
            bytes_read += nbytes
            band.merge_into(base_records(vals), 0, j0 // 16)
            control.tick(nbytes)
        writers[0].write(band)
        feed(0, band, False)
    feed(0, None, True)
    ck.check()  # a change after the last read invalidates the pyramid
    for w in writers:
        if w.row != w.info.stored_rows:
            raise AssertionError("pyramid level incomplete")  # a builder bug, never data dependent

    crc_bytes = crc.tobytes()
    sink.write_at(lay.crc_offset, crc_bytes)
    level_sha = {w.info.level: w.sha.hexdigest() for w in writers}
    trailer = {
        "format": FORMAT_NAME,
        "file_version": FILE_VERSION,
        "pyramid_record_format": PYRAMID_RECORD_FORMAT,
        "numerics_version": TILE_NUMERICS_VERSION,
        "pyramid_id": pyramid_id,
        "identity": dict(identity),
        "rows": p.rv.rows,
        "cols": p.rv.cols,
        "orientation": p.orientation,
        "base_level": BASE_LEVEL,
        "chunk_records": CHUNK_RECORDS,
        "source_dtype": p.dtype,
        "rounded_inputs": rounded,
        "levels": [{**d, "sha256": level_sha[d["level"]]} for d in lay.as_json()],
        "crc_sha256": hashlib.sha256(crc_bytes).hexdigest(),
    }
    tb = _canonical(trailer).encode("utf-8")
    if len(tb) > TRAILER_MAX:
        raise fail(ErrorCode.RESOURCE_LIMIT, "pyramid trailer too large")
    sink.write_at(lay.data_bytes, tb)
    sink.write_at(
        lay.data_bytes + len(tb), FOOTER.pack(FOOTER_MAGIC, lay.data_bytes, len(tb), hashlib.sha256(tb).digest())
    )
    total = lay.data_bytes + len(tb) + FOOTER.size
    return BuildResult(pyramid_id, lay, total, bytes_read, rounded, level_sha)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


# ------------------------------------------------------------------------------------------------------ reading
class Blob(Protocol):
    size: int

    def pread(self, offset: int, length: int) -> bytes: ...


class FileBlob:
    """Positioned reads of a published pyramid file; one short-lived handle per use (eviction-friendly on Windows)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._f = open(path, "rb")  # noqa: SIM115  (closed by close())
        self.size = os.fstat(self._f.fileno()).st_size

    def pread(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.size:
            raise PyramidCorrupt("read outside the pyramid file")
        self._f.seek(offset)
        data = self._f.read(length)
        if len(data) != length:
            raise PyramidCorrupt("short read in the pyramid file")
        return data

    def close(self) -> None:
        self._f.close()


class BytesBlob:
    def __init__(self, data: bytes | bytearray) -> None:
        self.data = data
        self.size = len(data)

    def pread(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.size:
            raise PyramidCorrupt("read outside the pyramid")
        return bytes(self.data[offset : offset + length])

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class Pyramid:
    """A validated pyramid: its trailer (identity, layout) and CRC table. Tiles are read from `blob` on demand."""

    trailer: dict[str, Any]
    layout: Layout
    crc: npt.NDArray[np.uint32]

    @property
    def pyramid_id(self) -> str:
        return str(self.trailer["pyramid_id"])

    @property
    def source_dtype(self) -> str:
        return str(self.trailer["source_dtype"])

    @property
    def rounded_inputs(self) -> bool:
        return bool(self.trailer["rounded_inputs"])


def load(blob: Blob, identity: Mapping[str, Any], pyramid_id: str) -> Pyramid:
    """Validate a stored pyramid against the identity it must have; PyramidCorrupt on any mismatch."""
    if blob.size < PREAMBLE.size + FOOTER.size:
        raise PyramidCorrupt("file too small")
    magic, ver, rec = PREAMBLE.unpack(blob.pread(0, PREAMBLE.size))
    if magic != MAGIC or ver != FILE_VERSION or rec != PYRAMID_RECORD_FORMAT:
        raise PyramidCorrupt("preamble: magic, file version or record format")
    fmagic, toff, tlen, tsha = FOOTER.unpack(blob.pread(blob.size - FOOTER.size, FOOTER.size))
    if fmagic != FOOTER_MAGIC or tlen > TRAILER_MAX or toff + tlen + FOOTER.size != blob.size:
        raise PyramidCorrupt("footer")
    tb = blob.pread(toff, tlen)
    if hashlib.sha256(tb).digest() != tsha:
        raise PyramidCorrupt("trailer checksum")
    try:
        trailer = json.loads(tb.decode("utf-8"), parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError):
        raise PyramidCorrupt("trailer JSON") from None
    if not isinstance(trailer, dict):
        raise PyramidCorrupt("trailer JSON")
    expect = {"format": FORMAT_NAME, "file_version": FILE_VERSION, "pyramid_record_format": PYRAMID_RECORD_FORMAT,
              "numerics_version": TILE_NUMERICS_VERSION, "pyramid_id": pyramid_id, "base_level": BASE_LEVEL,
              "chunk_records": CHUNK_RECORDS}  # fmt: skip
    for k, v in expect.items():
        if trailer.get(k) != v:
            raise PyramidCorrupt(f"trailer field {k}")
    if trailer.get("identity") != json.loads(_canonical(dict(identity))):
        raise PyramidCorrupt("identity differs from the requested source, view or numerics")
    rows, cols, orient = trailer.get("rows"), trailer.get("cols"), trailer.get("orientation")
    if type(rows) is not int or type(cols) is not int or orient not in ("rows", "cols") or rows < 1 or cols < 1:
        raise PyramidCorrupt("dimensions")
    if type(trailer.get("rounded_inputs")) is not bool or not isinstance(trailer.get("source_dtype"), str):
        raise PyramidCorrupt("source fields")
    lay = Layout.plan(rows, cols, orient)
    levels = trailer.get("levels")
    if not isinstance(levels, list) or len(levels) != len(lay.levels):
        raise PyramidCorrupt("levels")
    for want, got in zip(lay.as_json(), levels, strict=True):
        if not isinstance(got, dict) or {k: got.get(k) for k in want} != want:
            raise PyramidCorrupt(f"level {want['level']} layout")
    if toff != lay.data_bytes:
        raise PyramidCorrupt("record sections do not match the layout")
    crc_bytes = blob.pread(lay.crc_offset, lay.crc_length)
    if hashlib.sha256(crc_bytes).hexdigest() != trailer.get("crc_sha256"):
        raise PyramidCorrupt("CRC table checksum")
    return Pyramid(trailer, lay, np.frombuffer(crc_bytes, dtype="<u4"))


def _no_constant(name: str) -> None:
    raise ValueError(name)


def _cells(start: int, n: int, block: int, total: int) -> npt.NDArray[np.int64]:
    """Real cell count along one axis of grid cells start..start+n at this block size."""
    lo = (np.arange(start, start + n, dtype=np.int64)) * block
    return np.minimum(lo + block, total) - lo


def read_records(blob: Blob, pyr: Pyramid, level: int, r0: int, r1: int, c0: int, c1: int) -> BlockAggregates:
    """Records of view-grid rows [r0, r1) x cols [c0, c1) at `level` (clamped to the top level), CRC-verified and
    validated, in view orientation."""
    lay = pyr.layout
    li = lay.level(level)
    cols_orient = lay.orientation == "cols"
    sr0, sr1, sc0, sc1 = (c0, c1, r0, r1) if cols_orient else (r0, r1, c0, c1)
    if not (0 <= sr0 < sr1 <= li.stored_rows and 0 <= sc0 < sc1 <= li.stored_cols):
        raise PyramidCorrupt("record region outside the level")  # geometry guarantees it; corrupt otherwise
    dt = record_dtype(li.level)
    rs = li.record_size
    k0, k1 = sc0 // CHUNK_RECORDS, -(-sc1 // CHUNK_RECORDS)
    lo, hi = k0 * CHUNK_RECORDS, min(li.stored_cols, k1 * CHUNK_RECORDS)
    parts = []
    for r in range(sr0, sr1):
        data = blob.pread(li.offset + (r * li.stored_cols + lo) * rs, (hi - lo) * rs)
        for k in range(k0, k1):
            a = (k * CHUNK_RECORDS - lo) * rs
            b = (min(li.stored_cols, (k + 1) * CHUNK_RECORDS) - lo) * rs
            if zlib.crc32(data[a:b]) != int(pyr.crc[li.crc_offset + r * li.chunks_per_row + k]):
                raise PyramidCorrupt(f"CRC mismatch at level {li.level} row {r}")
        parts.append(np.frombuffer(data, dtype=dt)[sc0 - lo : sc1 - lo])
    rec = np.stack(parts)
    block = 1 << li.level
    sr_total, sc_total = lay.stored_shape
    expected = np.outer(_cells(sr0, sr1 - sr0, block, sr_total), _cells(sc0, sc1 - sc0, block, sc_total))
    _validate(rec, expected)
    agg = from_records(rec)
    if cols_orient:
        agg = BlockAggregates(*(getattr(agg, f).T.copy() for f in _AGG_FIELDS))
    return agg


COUNT_SANE = 1 << 60  # no tensor holds 2^60 elements; four such counts still add up within int64


def _validate(rec: npt.NDArray[np.void], expected_cells: npt.NDArray[np.int64]) -> None:
    if any(rec[f].size and int(rec[f].max()) >= COUNT_SANE for f in COUNTS):
        raise PyramidCorrupt("record counter out of range")
    total = sum(rec[f].astype(np.int64) for f in COUNTS)
    if not np.array_equal(total, expected_cells):
        raise PyramidCorrupt("record counts do not add up to the block's cells")
    if np.any(rec["flags"] & (0xFF ^ FLAG_BITS)) or bytes(rec["pad"].tobytes()).strip(b"\0"):
        raise PyramidCorrupt("record flags or padding")
    fin = rec["finite"] > 0
    none = ~fin
    if np.any(none & ((rec["flags"] != 0) | (rec["sum"] != 0) | (rec["sum_abs"] != 0) | (rec["sum_sq"] != 0))):
        raise PyramidCorrupt("a record without finite elements carries sums or flags")
    if np.any(none & ~((rec["min"] == np.inf) & (rec["max"] == -np.inf))):
        raise PyramidCorrupt("a record without finite elements carries extrema")
    if np.any(fin & ~(np.isfinite(rec["min"]) & np.isfinite(rec["max"]) & (rec["min"] <= rec["max"]))):
        raise PyramidCorrupt("record extrema")
    for bit, f in ((1, "sum"), (2, "sum_abs"), (4, "sum_sq")):
        flagged = (rec["flags"] & bit) != 0
        if np.any(fin & (flagged == np.isfinite(rec[f]))):
            raise PyramidCorrupt(f"overflow flag of {f} does not match the stored sum")
    if np.any(fin & ((rec["sum_abs"] < 0) | (rec["sum_sq"] < 0))):
        raise PyramidCorrupt("negative magnitude sums")


def pyramid_tile(
    blob: Blob,
    pyr: Pyramid,
    *,
    level: int,
    x: int,
    y: int,
    size: int,
    aggregation: Aggregation,
    view_rows: int,
    view_cols: int,
    esize: int,
) -> Tile:
    """An exact tile at level >= 4 from a validated pyramid."""
    if level < BASE_LEVEL:
        raise fail(ErrorCode.VALIDATION_ERROR, "pyramid tiles start at level 4")
    if (view_rows, view_cols) != (pyr.layout.rows, pyr.layout.cols):
        raise PyramidCorrupt("pyramid dimensions differ from the view")
    g: TileGeometry = tile_geometry(view_rows, view_cols, level, x, y, size)
    li = pyr.layout.level(level)
    if li.level < level:  # above the top: one record covers the whole view
        r0, c0 = 0, 0
    else:
        r0, c0 = y * size, x * size
    acc = read_records(blob, pyr, level, r0, r0 + g.rows, c0, c0 + g.cols)
    return finish_tile(
        g, aggregation, acc, pyr.source_dtype, rounded=pyr.rounded_inputs,
        covered_bytes=g.covered_cells * esize, bytes_read=0,
    )  # fmt: skip


def pyramid_id_for(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(dict(identity)).encode("utf-8")).hexdigest()
