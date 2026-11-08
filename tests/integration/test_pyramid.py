"""Exact pyramids (doc 03 section 5, doc 13 section 4) against the independent reference (tests/reference_tiles.py:
own decoder, NumPy views, exact Fraction aggregation per block) and the generator's golden values. Never compares two
WeightLab paths for expected values; pyramid tiles are compared with the reference directly."""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers import write_safetensors
from tests.reference_tiles import decode_tensor, ref_tile, sum_error_bound, view2d
from weightlab import open_checkpoint
from weightlab.analysis import pyramid as P
from weightlab.analysis import tile_codec as tc
from weightlab.analysis.tile import AGGREGATIONS, BlockAggregates, Tile
from weightlab.core.errors import WeightLabError
from weightlab.formats.checkpoint import Checkpoint
from weightlab.formats.selection import View

Q = "demo.layers.0.self_attn.q_proj.weight"  # 257 x 263 F32
F32_ULP = 2.0**-23
PID = "ab" * 32
IDENT = {"kind": "pyramid", "test": True}


def build(ck: Checkpoint, key: str, view: View | None, band_cells: int = P.BAND_CELLS, control: Any = None) -> Any:
    sink = P.MemorySink(0)
    res = P.build_pyramid(ck, key, view, IDENT, PID, sink, control, band_cells=band_cells)
    blob = P.BytesBlob(bytes(sink.buf[: res.total_bytes]))
    return res, blob, P.load(blob, IDENT, PID)


def tile(blob: Any, pyr: P.Pyramid, v: np.ndarray, level: int, x: int, y: int, agg: str, size: int = 256) -> Tile:
    return P.pyramid_tile(blob, pyr, level=level, x=x, y=y, size=size, aggregation=agg,  # type: ignore[arg-type]
                          view_rows=v.shape[0], view_cols=v.shape[1], esize=4)  # fmt: skip


def check(t: Tile, ref: dict[str, Any], v: np.ndarray, level: int, x: int, y: int, size: int, agg: str) -> None:
    g = t.geometry
    assert (g.row_start, g.row_end, g.col_start, g.col_end) == ref["range"]
    np.testing.assert_array_equal(t.status, ref["status"])
    assert (t.finite_count, t.nonfinite_count) == (ref["finite"], ref["nonfinite"])
    if agg in ("min", "max"):  # exact; {+0, -0} may be either zero, so compare numerically
        np.testing.assert_array_equal(t.values, ref["values"])
    else:  # float64 summation order vs the exactly rounded reference: <= 1 float32 ulp when well conditioned
        bound, exact = sum_error_bound(v, level, x, y, size, agg)
        ok = bound <= 2.0**-25 * np.abs(exact)
        np.testing.assert_allclose(t.values[ok], ref["values"][ok], rtol=F32_ULP, atol=0, equal_nan=True)


def all_tiles(blob: Any, pyr: P.Pyramid, v: np.ndarray, levels: range, size: int = 256) -> int:
    n = 0
    for level in levels:
        span = size << level
        for y in range(-(-v.shape[0] // span)):
            for x in range(-(-v.shape[1] // span)):
                for agg in AGGREGATIONS:
                    t = tile(blob, pyr, v, level, x, y, agg, size)
                    check(t, ref_tile(v, level, x, y, size, agg), v, level, x, y, size, agg)
                    assert t.quality == "exact"
                    assert t.numeric_precision == ("lossless" if agg in ("min", "max") else "rounded")
                    n += 1
    return n


@pytest.fixture
def base(fixtures_dir: Path) -> Any:
    with open_checkpoint(fixtures_dir / "tiny_base.safetensors") as ck:
        yield ck


# --------------------------------------------------------------------------------------- values vs reference
def test_record_sizes_are_the_frozen_format() -> None:
    assert P.BASE_DTYPE.itemsize == 56
    assert P.COARSE_DTYPE.itemsize == 80
    assert [P.BASE_DTYPE.fields[f][1] for f in ("finite", "nan", "pos_inf", "neg_inf", "flags", "sum", "max")] == [
        0, 2, 4, 6, 8, 16, 48]  # fmt: skip
    assert [P.COARSE_DTYPE.fields[f][1] for f in ("finite", "neg_inf", "flags", "sum", "sum_sq", "max")] == [
        0, 24, 32, 40, 56, 72]  # fmt: skip
    assert P.BASE_DTYPE["finite"] == np.dtype("<u2")
    assert P.COARSE_DTYPE["finite"] == np.dtype("<u8")


@pytest.mark.parametrize("transpose", [False, True])
def test_odd_257x263_every_level_and_aggregation(base: Checkpoint, fixtures_dir: Path, transpose: bool) -> None:
    v = view2d(decode_tensor(fixtures_dir / "tiny_base.safetensors", Q), 0, 1, {}, transpose)
    res, blob, pyr = build(base, Q, View(0, 1, {}, transpose))
    assert res.layout.orientation == ("cols" if transpose else "rows")  # streamed in storage order
    assert res.layout.top_level == 9  # ceil(log2(263))
    assert all_tiles(blob, pyr, v, range(4, 12)) == 8 * 5
    assert all_tiles(blob, pyr, v, range(4, 7), size=7) > 40  # partial tiles at every edge


def test_rank3_slices_and_their_transposes(base: Checkpoint, fixtures_dir: Path) -> None:
    cube = decode_tensor(fixtures_dir / "tiny_base.safetensors", "demo.cube.weight")
    for view in (View(1, 2, {0: 1}, False), View(1, 2, {0: 0}, True), View(0, 2, {1: 3}, False),
                 View(0, 1, {2: 5}, True), View(2, 0, {1: 2}, False)):  # fmt: skip
        v = view2d(cube, view.row_axis, view.col_axis, dict(view.fixed_indices), view.transpose)
        _, blob, pyr = build(base, "demo.cube.weight", view)
        all_tiles(blob, pyr, v, range(4, 6))


def test_nonfinite_golden_blocks_and_all_nonfinite_cells(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    gold = expected["edge"]["nonfinite_blocks"]
    path = fixtures_dir / "tiny_edge.safetensors"
    v = view2d(decode_tensor(path, gold["key"]), 0, 1, {}, False)
    with open_checkpoint(path) as ck:
        _, blob, pyr = build(ck, gold["key"], None)
    t = tile(blob, pyr, v, 4, 0, 0, "mean")
    for k, blk in gold["level4_blocks_mean"].items():
        i, j = map(int, k.split(","))
        assert int(t.status[i, j]) == blk["status"]
        if blk["value"] is None:
            assert np.isnan(t.values[i, j])  # placeholder, never drawn
        else:
            assert t.values[i, j] == np.float32(blk["value"])
    all_tiles(blob, pyr, v, range(4, 7))


def test_counts_beyond_uint16_at_level8_and_9(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    gold = expected["edge"]["big_block"]
    path = fixtures_dir / "tiny_edge.safetensors"
    v = view2d(decode_tensor(path, gold["key"]), 0, 1, {}, False)
    with open_checkpoint(path) as ck:
        _res, blob, pyr = build(ck, gold["key"], None)
    l8 = tile(blob, pyr, v, 8, 0, 0, "mean")
    assert l8.finite_count == 2 * gold["level8_block_0_0_finite_count"] == 131072  # two blocks of 65,536 (> uint16)
    assert l8.values[0, 0] == np.float32(gold["level8_block_0_0_sum"] / 65536)
    l9 = tile(blob, pyr, v, 9, 0, 0, "mean")
    assert l9.finite_count == gold["level9_block_0_0_finite_count"]  # one block
    assert l9.values[0, 0] == np.float32(gold["level9_block_0_0_sum"] / 131072)
    rec = P.read_records(blob, pyr, 8, 0, 1, 0, 2)
    assert rec.finite.tolist() == [[65536, 65536]]
    all_tiles(blob, pyr, v, range(4, 11))


def test_half_precision_sources(fixtures_dir: Path) -> None:
    path = fixtures_dir / "tiny_dtypes.safetensors"
    with open_checkpoint(path) as ck:
        for key in ("demo.bf16.weight", "demo.f16.weight", "demo.bf16_specials", "demo.f16_specials"):
            t = decode_tensor(path, key)
            v = view2d(t, 0 if t.ndim == 2 else None, 1 if t.ndim == 2 else None, {}, False)
            _, blob, pyr = build(ck, key, None)
            all_tiles(blob, pyr, v, range(4, 6))


@pytest.mark.parametrize("key", ["p.i64", "p.u64"])
def test_64_bit_integers_are_rounded_in_pyramids(fixtures_dir: Path, expected: dict[str, Any], key: str) -> None:
    with open_checkpoint(fixtures_dir / "tiny_precision_a.safetensors") as ck:
        _, blob, pyr = build(ck, key, None)
    ints = [int(s) for s in expected["precision"]["i64" if key == "p.i64" else "u64"]]
    assert pyr.rounded_inputs
    t = P.pyramid_tile(blob, pyr, level=4, x=0, y=0, size=256, aggregation="max", view_rows=1, view_cols=len(ints),
                       esize=8)  # fmt: skip
    assert t.values[0, 0] == np.float32(float(max(ints)))
    assert t.numeric_precision == "rounded"  # min/max of rounded 64-bit integers are not exact
    assert any("2^53" in w for w in t.warnings)


def test_f64_extremes_survive_and_overflow_is_flagged(tmp_path: Path) -> None:
    a = np.zeros((32, 32))
    a[0, 0], a[0, 1], a[17, 3] = 1e300, -1e300, 1e300
    a[20:24, 20:24] = 1.5e200  # x^2 overflows float64 (sum_sq): flagged, never Inf or a finite fake
    path = write_safetensors(tmp_path / "x.safetensors", {"w": ("F64", (32, 32), a.astype("<f8").tobytes())})
    v = view2d(decode_tensor(path, "w"), 0, 1, {}, False)
    with open_checkpoint(path) as ck:
        _, blob, pyr = build(ck, "w", None)
    rec = P.read_records(blob, pyr, 4, 0, 2, 0, 2)
    assert rec.max[0, 0] == 1e300 and rec.min[0, 0] == -1e300  # noqa: PT018  (exact in float64 records)
    assert rec.flags[1, 1] & P.FLAG_BITS == 4  # SUM_SQ_OVERFLOW
    for agg in AGGREGATIONS:
        t = tile(blob, pyr, v, 4, 0, 0, agg)
        assert np.all(np.isfinite(t.values) == ((t.status & 17) == 0))
        check(t, ref_tile(v, 4, 0, 0, 256, agg), v, 4, 0, 0, 256, agg)
    rms = tile(blob, pyr, v, 5, 0, 0, "rms")  # the flag propagated to the parent
    assert rms.status[0, 0] & 16
    assert rms.numeric_precision == "overflowed"
    assert P.read_records(blob, pyr, 5, 0, 1, 0, 1).flags[0, 0] & 4


# ---------------------------------------------------------------------------------- merge algebra, counters
def test_parents_are_exact_merges_of_children(base: Checkpoint) -> None:
    _, blob, pyr = build(base, Q, None)
    for level in range(4, 9):
        child = P.read_records(blob, pyr, level, 0, -(-257 >> level), 0, -(-263 >> level))
        parent = P.read_records(blob, pyr, level + 1, 0, -(-257 >> (level + 1)), 0, -(-263 >> (level + 1)))
        merged = P.merge_pairs(P.merge_pairs(child, 0), 1)
        for f in ("finite", "nan", "pos_inf", "neg_inf", "flags"):
            np.testing.assert_array_equal(getattr(parent, f), getattr(merged, f))
        for f in ("sum", "sum_abs", "sum_sq", "min", "max"):
            np.testing.assert_array_equal(getattr(parent, f), getattr(merged, f))  # same merge order: bitwise


def test_rms_comes_from_merged_sums_of_squares(base: Checkpoint, fixtures_dir: Path) -> None:
    v = view2d(decode_tensor(fixtures_dir / "tiny_base.safetensors", Q), 0, 1, {}, False)
    _, blob, pyr = build(base, Q, None)
    child = P.read_records(blob, pyr, 8, 0, 2, 0, 2)  # 256 x 256 + edge blocks (1 row / 7 columns)
    t = tile(blob, pyr, v, 9, 0, 0, "rms")
    want = np.sqrt(child.sum_sq.sum() / child.finite.sum())
    assert t.values[0, 0] == np.float32(want)
    averaged = np.mean(np.sqrt(child.sum_sq / child.finite))  # what an average of child RMS would give
    assert abs(float(t.values[0, 0]) - averaged) > 1e-3


def test_synthetic_counters_beyond_2_to_32_merge_and_round_trip() -> None:
    big = 3_000_000_000
    agg = BlockAggregates.empty(1, 2)
    agg.finite[:] = big
    agg.nan[:] = 7
    agg.sum[:] = 1.0
    agg.sum_abs[:] = 1.0
    agg.sum_sq[:] = 1.0
    agg.min[:] = -1.0
    agg.max[:] = 1.0
    parent = P.merge_pairs(agg, 1)
    assert parent.finite.tolist() == [[6_000_000_000]]  # > 2^32, < 2^63
    rec = P.to_records(parent, 5)
    assert rec.dtype == P.COARSE_DTYPE
    assert int(rec["finite"][0, 0]) == 6_000_000_000
    assert P.from_records(rec).finite.tolist() == [[6_000_000_000]]
    with pytest.raises(WeightLabError) as e:
        P.to_records(parent, 4)  # a base record cannot hold it
    assert e.value.code.value == "RESOURCE_LIMIT"
    near = BlockAggregates.empty(1, 2)
    near.finite[:] = 2**62
    with pytest.raises(WeightLabError) as e:
        P.merge_pairs(BlockAggregates(*(np.concatenate([getattr(near, f)] * 2, 1) for f in P._AGG_FIELDS)), 1)
    assert e.value.code.value == "RESOURCE_LIMIT"  # checked addition: never wraps


def test_chunking_does_not_change_a_single_byte(base: Checkpoint) -> None:
    hashes = set()
    for band in (16, 256, 4096 + 16, 1 << 19):
        res, _, _ = build(base, Q, None, band_cells=band)
        hashes.add(tuple(sorted(res.level_sha256.items())))
    assert len(hashes) == 1  # blocks never straddle bands: identical records whatever the band size


def test_views_never_share_a_pyramid(base: Checkpoint) -> None:
    files = base.fingerprints(["demo.cube.weight"])
    ids = set()
    for view in (
        View(1, 2, {0: 0}, False),
        View(1, 2, {0: 1}, False),
        View(1, 2, {0: 0}, True),
        View(0, 2, {1: 0}, False),
    ):
        ids.add(tc.cache_key(tc.pyramid_identity(source_kind="tensor", files=files, tensor_keys=["demo.cube.weight"],
                                                 view=view)))  # fmt: skip
    assert len(ids) == 4
    ident = tc.pyramid_identity(source_kind="tensor", files=files, tensor_keys=["demo.cube.weight"],
                                view=View(1, 2, {0: 0}, False))  # fmt: skip
    other = tc.pyramid_identity(source_kind="tensor", files=files, tensor_keys=["demo.cube.weight"],
                                view=View(1, 2, {0: 1}, False))  # fmt: skip
    sink = P.MemorySink(0)
    res = P.build_pyramid(base, "demo.cube.weight", View(1, 2, {0: 0}, False), ident, tc.cache_key(ident), sink)
    blob = P.BytesBlob(bytes(sink.buf[: res.total_bytes]))
    P.load(blob, ident, tc.cache_key(ident))
    with pytest.raises(P.PyramidCorrupt):
        P.load(blob, other, tc.cache_key(ident))  # a stored pyramid only serves its own identity
    with pytest.raises(P.PyramidCorrupt):
        P.load(blob, ident, tc.cache_key(other))


# ------------------------------------------------------------------------------------------ corruption
def test_corruption_anywhere_is_detected(base: Checkpoint, fixtures_dir: Path) -> None:
    v = view2d(decode_tensor(fixtures_dir / "tiny_base.safetensors", Q), 0, 1, {}, False)
    res, blob, _pyr = build(base, Q, None)
    data = bytes(blob.data)
    lay = res.layout

    def flipped(at: int) -> P.BytesBlob:
        b = bytearray(data)
        b[at] ^= 0x01
        return P.BytesBlob(b)

    # a record byte: CRC of its run fails on the tile read
    bad = flipped(lay.levels[0].offset + 56 * 3 + 20)
    p2 = P.load(bad, IDENT, PID)  # the trailer and CRC table are intact
    with pytest.raises(P.PyramidCorrupt):
        tile(bad, p2, v, 4, 0, 0, "mean")
    for at in (0, lay.crc_offset + 1, lay.data_bytes + 5, len(data) - 3):  # preamble, CRC table, trailer, footer
        with pytest.raises(P.PyramidCorrupt):
            P.load(flipped(at), IDENT, PID)
    with pytest.raises(P.PyramidCorrupt):
        P.load(P.BytesBlob(data[:-1]), IDENT, PID)  # truncated
    with pytest.raises(P.PyramidCorrupt):
        P.load(P.BytesBlob(data + b"\0"), IDENT, PID)  # appended


def test_record_validation_rejects_inconsistent_records() -> None:
    agg = BlockAggregates.empty(1, 1)
    agg.finite[:] = 3
    agg.sum[:] = agg.sum_abs[:] = agg.sum_sq[:] = 1.0
    agg.min[:], agg.max[:] = -1.0, 1.0
    good = P.to_records(agg, 4)
    P._validate(good, np.array([[3]]))
    cases = []
    r = good.copy()
    r["finite"] = 2  # counts no longer add up to the block's cells
    cases.append(r)
    r = good.copy()
    r["flags"] = 8  # reserved flag bit
    cases.append(r)
    r = good.copy()
    r["sum"] = np.inf  # a non-finite sum without its overflow flag
    cases.append(r)
    r = good.copy()
    r["flags"] = 1  # an overflow flag with a finite sum
    cases.append(r)
    r = good.copy()
    r["min"] = 2.0  # min > max
    cases.append(r)
    for bad in cases:
        with pytest.raises(P.PyramidCorrupt):
            P._validate(bad, np.array([[3]]))


# ---------------------------------------------------------------------------- source changes, cancellation
class _Ticks:
    def __init__(self, after: int, action: Any) -> None:
        self.n, self.after, self.action = 0, after, action

    def tick(self, nbytes: int) -> None:
        self.n += 1
        if self.n == self.after:
            self.action()


def test_file_change_during_a_build_is_file_changed(tmp_path: Path) -> None:
    a = np.arange(64 * 64, dtype="<f4").reshape(64, 64)
    path = write_safetensors(tmp_path / "m.safetensors", {"w": ("F32", (64, 64), a.tobytes())})

    def touch() -> None:
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    with open_checkpoint(path) as ck:
        sink = P.MemorySink(0)
        with pytest.raises(WeightLabError) as e:
            P.build_pyramid(ck, "w", None, IDENT, PID, sink, _Ticks(2, touch), band_cells=16 * 64)
    assert e.value.code.value == "FILE_CHANGED"


def test_file_change_after_the_last_read_is_file_changed(tmp_path: Path) -> None:
    a = np.ones((32, 32), "<f4")
    path = write_safetensors(tmp_path / "m.safetensors", {"w": ("F32", (32, 32), a.tobytes())})

    def append() -> None:
        with path.open("ab") as f:
            f.write(b"\0" * 8)

    with open_checkpoint(path) as ck:
        n_reads = 2  # band of 16 rows x 32 columns: two reads in all; change after the last one
        with pytest.raises(WeightLabError) as e:
            P.build_pyramid(ck, "w", None, IDENT, PID, P.MemorySink(0), _Ticks(n_reads, append), band_cells=16 * 32)
    assert e.value.code.value == "FILE_CHANGED"


def test_a_raising_control_stops_the_build_within_one_chunk(tmp_path: Path) -> None:
    a = np.ones((256, 64), "<f4")
    path = write_safetensors(tmp_path / "m.safetensors", {"w": ("F32", (256, 64), a.tobytes())})

    class Stop(Exception):
        pass

    def stop() -> None:
        raise Stop

    with open_checkpoint(path) as ck:
        ticks = _Ticks(3, stop)
        with pytest.raises(Stop):
            P.build_pyramid(ck, "w", None, IDENT, PID, P.MemorySink(0), ticks, band_cells=16 * 64)
        assert ck.bytes_read < 256 * 64 * 4  # stopped early, not after the whole scan
    assert ticks.n == 3


def test_layout_matches_the_storage_estimate() -> None:
    lay = P.Layout.plan(8192, 8192, "rows")
    base = 8192 * 8192 // 256
    assert lay.levels[0].length == base * 56
    coarse = sum(li.length for li in lay.levels[1:])
    assert abs(coarse / 80 - base / 3) / (base / 3) < 0.01  # about one third as many coarse records
    per_cell = (lay.data_bytes) / (8192 * 8192)
    assert abs(per_cell - 0.323) < 0.01  # doc 03: 0.323 B per cell
    assert struct.calcsize("<8sQQ32s") == P.FOOTER.size == 56
