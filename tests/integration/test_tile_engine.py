"""Exact direct-read tiles (Phase 1 Task 4B foundation) against the independent reference in tests/reference_tiles.py
and the generator's golden values in examples/fixtures/expected_results.json. Never compares two WeightLab paths."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers import write_safetensors
from tests.reference_tiles import decode_tensor, ref_tile, view2d
from weightlab.analysis.tile import (
    AGGREGATIONS,
    BlockAggregates,
    TileBudget,
    compute_tile,
)
from weightlab.core.errors import WeightLabError
from weightlab.formats.checkpoint import Checkpoint, open_checkpoint
from weightlab.formats.selection import View

Q = "demo.layers.0.self_attn.q_proj.weight"  # 257 x 263 F32
F32_ULP = 2.0**-23


def check(tile: Any, ref: dict[str, Any], *, exact: bool) -> None:
    assert (tile.geometry.rows, tile.geometry.cols) == (ref["rows"], ref["cols"])
    g = tile.geometry
    assert (g.row_start, g.row_end, g.col_start, g.col_end) == ref["range"]
    np.testing.assert_array_equal(tile.status, ref["status"])
    if exact and tile.geometry.level == 0:  # one element per cell: bitwise, so -0.0 and NaN placeholders match too
        np.testing.assert_array_equal(tile.values.view(np.uint32), ref["values"].view(np.uint32))
    elif exact:  # min/max of {+0.0, -0.0} may be either zero (IEEE 754 leaves it open); equal as numbers
        np.testing.assert_array_equal(tile.values, ref["values"])
    else:  # float64 sums in a different order may move the float32 by at most one unit in the last place
        np.testing.assert_allclose(tile.values, ref["values"], rtol=F32_ULP, atol=0, equal_nan=True)
    assert (tile.finite_count, tile.nonfinite_count) == (ref["finite"], ref["nonfinite"])


@pytest.fixture
def base(fixtures_dir: Path) -> Any:
    with open_checkpoint(fixtures_dir / "tiny_base.safetensors") as ck:
        yield ck


@pytest.fixture
def q_view(fixtures_dir: Path) -> np.ndarray:
    return decode_tensor(fixtures_dir / "tiny_base.safetensors", Q)


# ----------------------------------------------------------------------------- L0 exact values, L1-L3 aggregation
def test_l0_tiles_are_the_exact_source_cells(base: Checkpoint, q_view: np.ndarray) -> None:
    for x, y in ((0, 0), (1, 0), (0, 1), (1, 1)):
        t = compute_tile(base, Q, None, level=0, x=x, y=y, aggregation="mean")
        check(t, ref_tile(q_view, 0, x, y, 256, "mean"), exact=True)
        assert t.numeric_precision == "lossless"
        assert t.quality == "exact"
        assert t.value_precision == "float32_visualization"
    corner = compute_tile(base, Q, None, level=0, x=1, y=1, aggregation="mean")
    assert corner.values.shape == (1, 7)  # golden level0_tile_1_1_shape
    assert corner.values.tolist() == [[float(np.float32(v)) for v in q_view[256, 256:263]]]


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_l1_to_l3_match_independent_block_aggregation(
    base: Checkpoint, q_view: np.ndarray, level: int, aggregation: str
) -> None:
    t = compute_tile(base, Q, None, level=level, x=0, y=0, aggregation=aggregation)  # type: ignore[arg-type]
    check(t, ref_tile(q_view, level, 0, 0, 256, aggregation), exact=aggregation in ("min", "max"))
    assert t.numeric_precision == ("lossless" if aggregation in ("min", "max") else "rounded")


def test_partial_edge_blocks_are_count_weighted_golden(base: Checkpoint, expected: dict[str, Any]) -> None:
    for name, level in (("level1", 1), ("level3", 3)):
        gold = expected["tiles_257x263"][name]
        t = compute_tile(base, Q, None, level=level, x=0, y=0, aggregation="mean")
        assert list(t.values.shape) == gold["output_shape"]
        assert t.values[0, 0] == np.float32(gold["first_block_mean"])
        assert t.values[-1, -1] == np.float32(gold["last_block_mean"])  # a 1 x 7 block at L3, 1 x 1 at L1
    # The L3 corner block holds 1 row x 7 columns: its mean is over those 7 cells, not over 64.
    l3 = expected["tiles_257x263"]["level3"]
    assert l3["last_block_count"] == 7


def test_edge_tiles_of_the_odd_shape(base: Checkpoint, q_view: np.ndarray) -> None:
    for level, x, y, size in (
        (0, 1, 0, 256),
        (1, 0, 0, 100),
        (1, 1, 1, 100),
        (2, 0, 0, 32),
        (2, 1, 1, 32),
        (3, 0, 0, 16),
        (3, 2, 2, 16),
    ):
        for agg in ("mean", "rms", "max"):
            t = compute_tile(base, Q, None, level=level, x=x, y=y, size=size, aggregation=agg)  # type: ignore[arg-type]
            check(t, ref_tile(q_view, level, x, y, size, agg), exact=agg == "max")


def test_rms_comes_from_sums_of_squares_not_averaged_rms(tmp_path: Path) -> None:
    # One 2x2 block (L1) of [[3, 4], [0, 0]] padded... use a 2x4 tensor: L1 blocks [[3,4],[0,0]] and [[1,1],[1,1]].
    a = np.array([[3, 0, 1, 1], [4, 0, 1, 1]], dtype=np.float32)
    path = write_safetensors(tmp_path / "rms.safetensors", {"w": ("F32", (2, 4), a.tobytes())})
    with open_checkpoint(path) as ck:
        l2 = compute_tile(ck, "w", None, level=2, x=0, y=0, aggregation="rms")
        l1 = compute_tile(ck, "w", None, level=1, x=0, y=0, aggregation="rms")
    true_rms = math.sqrt((9 + 16 + 1 + 1 + 1 + 1) / 8)
    assert l2.values[0, 0] == np.float32(true_rms)
    averaged = float(np.mean(l1.values))  # what averaging the L1 RMS values would give
    assert not math.isclose(averaged, true_rms, rel_tol=1e-3)


def test_chunking_does_not_change_the_tile(base: Checkpoint, q_view: np.ndarray) -> None:
    # Rectangles that split blocks mid-way (37 cells, 263 = one view row) and ones that hold many blocks.
    refs = {agg: ref_tile(q_view, 2, 0, 0, 256, agg) for agg in AGGREGATIONS}
    for band in (37, 263, 1000, 1 << 18):
        budget = TileBudget(band_cells=band)
        for agg in AGGREGATIONS:
            t = compute_tile(base, Q, None, level=2, x=0, y=0, aggregation=agg, budget=budget)
            check(t, refs[agg], exact=agg in ("min", "max"))


# ----------------------------------------------------------------------------------------- non-finite and extremes
def test_nonfinite_golden_status_masks(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    gold = expected["edge"]["nonfinite_blocks"]
    with open_checkpoint(fixtures_dir / "tiny_edge.safetensors") as ck:
        l0 = compute_tile(ck, gold["key"], None, level=0, x=0, y=0, aggregation="mean")
        for k, cell in gold["level0_cells"].items():
            r, c = map(int, k.split(","))
            assert l0.status[r, c] == cell["status"], k
            if cell["value"] is None:
                assert math.isnan(l0.values[r, c])
            else:
                assert l0.values[r, c] == np.float32(cell["value"])
        l4 = compute_tile(ck, gold["key"], None, level=4, x=0, y=0, aggregation="mean")
        for k, blk in gold["level4_blocks_mean"].items():
            r, c = map(int, k.split(","))
            assert l4.status[r, c] == blk["status"], k
            if blk["value"] is None:
                assert math.isnan(l4.values[r, c])
            else:
                assert l4.values[r, c] == np.float32(blk["value"]), k
        assert l4.status.tolist() == [[0, 6], [15, 3]]  # incl. an all-non-finite block with all three kinds
        view = decode_tensor(fixtures_dir / "tiny_edge.safetensors", gold["key"])
        for level in (0, 1, 2, 3, 4, 5):
            for agg in AGGREGATIONS:
                t = compute_tile(ck, gold["key"], None, level=level, x=0, y=0, aggregation=agg)  # type: ignore[arg-type]
                check(t, ref_tile(view, level, 0, 0, 256, agg), exact=agg in ("min", "max") or level == 0)


def test_large_blocks_count_beyond_uint16(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    gold = expected["edge"]["big_block"]
    with open_checkpoint(fixtures_dir / "tiny_edge.safetensors") as ck:
        l8 = compute_tile(ck, gold["key"], None, level=8, x=0, y=0, aggregation="mean")
        l9 = compute_tile(ck, gold["key"], None, level=9, x=0, y=0, aggregation="mean")
    assert l8.values.shape == (1, 2)
    assert l8.values[0, 0] == np.float32(gold["level8_block_0_0_sum"] / gold["level8_block_0_0_finite_count"])
    assert l9.finite_count == gold["level9_block_0_0_finite_count"] == 131072
    assert l9.values[0, 0] == np.float32(gold["level9_block_0_0_sum"] / 131072)


def test_f64_extremes_are_withheld_never_drawn_as_inf(tmp_path: Path) -> None:
    big = 1.7976931348623157e308
    a = np.array([[1e300, -1e300, big, big], [5e-324, -0.0, -big, -big]], dtype="<f8")
    path = write_safetensors(tmp_path / "x.safetensors", {"w": ("F64", (2, 4), a.tobytes())})
    with open_checkpoint(path) as ck:
        l0 = compute_tile(ck, "w", None, level=0, x=0, y=0, aggregation="max")
        l1 = {agg: compute_tile(ck, "w", None, level=1, x=0, y=0, aggregation=agg) for agg in AGGREGATIONS}
        l2 = compute_tile(ck, "w", None, level=2, x=0, y=0, aggregation="mean")
    # L0: beyond float32 -> NOT_REPRESENTABLE (16) with a NaN placeholder; tiny values are drawable.
    assert l0.status.tolist() == [[16, 16, 16, 16], [0, 0, 16, 16]]
    assert np.isnan(l0.values[l0.status == 16]).all()
    assert l0.values[1, 1] == 0.0
    assert math.copysign(1, l0.values[1, 1]) == -1.0
    assert l0.numeric_precision == "overflowed"
    assert l0.warnings
    # L1 block (0, 1) = [[big, big], [-big, -big]]: the exact mean is 0. Whether float64 gets there depends on the
    # summation order; the cell is either exactly 0 or withheld (16, NaN placeholder), never Inf or another number.
    m = l1["mean"]
    assert (m.status[0, 1], float(m.values[0, 1])) == (0, 0.0) or (m.status[0, 1] == 16 and math.isnan(m.values[0, 1]))
    assert l1["rms"].status[0, 1] == 16  # the sum of squares overflows: withheld
    assert l1["min"].status[0, 1] == 16  # -big is exact in float64 but beyond float32
    # L1 block (0, 0) = [[1e300, -1e300], [5e-324, -0.0]]: exact mean 1.25e-324 -> float64 0 -> drawable 0.
    assert l1["mean"].status[0, 0] == 0
    assert l1["mean"].values[0, 0] == 0.0
    for agg, t in l1.items():
        assert not np.any(np.isinf(t.values)), agg
    # Whole tensor: exact mean ~1.25e-324, but big + big may overflow first, depending on the order: 0 or withheld.
    assert (l2.status[0, 0], float(l2.values[0, 0])) == (0, 0.0) or (
        l2.status[0, 0] == 16 and math.isnan(l2.values[0, 0])
    )
    assert not (l2.status & 1).any()

    # Whole-tensor block of 2 x 4 elements at L2 including 1e300 * ... squares: the sum of squares overflows.
    with open_checkpoint(path) as ck:
        rms = compute_tile(ck, "w", None, level=2, x=0, y=0, aggregation="rms")
    assert rms.status.tolist() == [[16]]
    assert math.isnan(rms.values[0, 0])
    assert rms.numeric_precision == "overflowed"


def test_an_overflowed_intermediate_is_withheld_even_if_the_exact_mean_fits() -> None:
    """Merging partial records in this order: big, big (sum overflows), then -big, -big. The exact mean is 0, but a
    float64 accumulator went to Inf on the way: the cell is withheld (16), never drawn as 0 or Inf."""
    big = 1.7976931348623157e308
    acc = BlockAggregates.empty(1, 1)
    for v in (big, big, -big, -big):
        acc.merge_into(BlockAggregates.of_values(np.array([[v]]), 0, 0, 1), 0, 0)
    assert acc.flags[0, 0] & 1  # SUM_OVERFLOW stays set
    values, status = acc.display("mean")
    assert status.tolist() == [[16]]
    assert math.isnan(values[0, 0])
    assert acc.display("max")[1].tolist() == [[16]]  # big itself is beyond float32


def test_overflow_flags_in_the_aggregate_record() -> None:
    big = 1.7976931348623157e308
    agg = BlockAggregates.of_values(np.array([[big, big]]), 0, 0, 2)
    assert agg.flags[0, 0] == 1 | 2 | 4  # SUM, SUM_ABS and SUM_SQ overflow
    signed = BlockAggregates.of_values(np.array([[big, big], [-big, -big]]), 0, 0, 2)
    assert signed.flags[0, 0] & (2 | 4) == 6  # |x| and x^2 always overflow; the signed sum may cancel
    sq = BlockAggregates.of_values(np.array([[1e200, 1e200]]), 0, 0, 2)
    assert sq.flags[0, 0] == 4  # only the sum of squares overflows
    v, s = sq.display("mean")
    assert v[0, 0] != v[0, 0]
    assert s[0, 0] == 16
    # Merging two finite halves whose sum overflows sets the flag on the merged record.
    acc = BlockAggregates.empty(1, 1)
    acc.merge_into(BlockAggregates.of_values(np.array([[big]]), 0, 0, 1), 0, 0)
    assert acc.flags[0, 0] == 4  # big^2 overflows; big itself and |big| do not
    acc.merge_into(BlockAggregates.of_values(np.array([[big]]), 0, 0, 1), 0, 0)
    assert acc.flags[0, 0] == 1 | 2 | 4
    assert acc.finite[0, 0] == 2


@pytest.mark.parametrize(("key", "values"), [("p.i64", "i64"), ("p.u64", "u64")])
def test_64_bit_integers_are_labelled_rounded(
    fixtures_dir: Path, expected: dict[str, Any], key: str, values: str
) -> None:
    with open_checkpoint(fixtures_dir / "tiny_precision_a.safetensors") as ck:
        t = compute_tile(ck, key, None, level=0, x=0, y=0, aggregation="max")
    ints = [int(s) for s in expected["precision"][values]]
    assert t.values.tolist() == [[float(np.float32(float(v))) for v in ints]]
    assert t.numeric_precision == "rounded"
    assert any("2^53" in w for w in t.warnings)
    with open_checkpoint(fixtures_dir / "tiny_precision_a.safetensors") as ck:
        small = compute_tile(ck, "p.i32", None, level=0, x=0, y=0, aggregation="max")
    assert small.numeric_precision == "lossless"


@pytest.mark.parametrize("key", ["demo.bf16.weight", "demo.f16.weight", "demo.bf16_specials", "demo.f16_specials"])
def test_half_precision_decoding(fixtures_dir: Path, key: str) -> None:
    path = fixtures_dir / "tiny_dtypes.safetensors"
    view = view2d(decode_tensor(path, key), 0, 1, {}, False)
    with open_checkpoint(path) as ck:
        for level in (0, 1, 2):
            for agg in AGGREGATIONS:
                t = compute_tile(ck, key, None, level=level, x=0, y=0, aggregation=agg)  # type: ignore[arg-type]
                check(t, ref_tile(view, level, 0, 0, 256, agg), exact=agg in ("min", "max") or level == 0)


# ------------------------------------------------------------------------------------------ views: slices, transpose
def test_rank3_slices_and_transpose(base: Checkpoint, fixtures_dir: Path) -> None:
    cube = decode_tensor(fixtures_dir / "tiny_base.safetensors", "demo.cube.weight")  # [2, 4, 8]
    views = [
        (1, 2, {0: 1}, False),
        (1, 2, {0: 0}, True),
        (0, 2, {1: 3}, False),
        (2, 0, {1: 2}, False),
        (0, 1, {2: 7}, True),
    ]
    for ra, ca, fixed, tr in views:
        ref_view = view2d(cube, ra, ca, fixed, tr)
        for level in (0, 1, 2):
            for agg in AGGREGATIONS:
                t = compute_tile(
                    base, "demo.cube.weight", View(ra, ca, fixed, tr), level=level, x=0, y=0, aggregation=agg
                )  # type: ignore[arg-type]
                check(t, ref_tile(ref_view, level, 0, 0, 256, agg), exact=agg in ("min", "max") or level == 0)
    # Fixture expectations: slice means of axis 0 = 0 and 1.
    gold = {0: 15.5, 1: 47.5}
    for i, mean in gold.items():
        t = compute_tile(base, "demo.cube.weight", View(1, 2, {0: i}, False), level=3, x=0, y=0, aggregation="mean")
        assert t.values[0, 0] == np.float32(mean)


def test_transposed_view_of_the_odd_tensor(base: Checkpoint, q_view: np.ndarray) -> None:
    tv = View(0, 1, {}, True)
    for level, x, y in ((0, 1, 0), (0, 0, 1), (1, 0, 0), (3, 0, 0)):
        for agg in ("mean", "min"):
            t = compute_tile(base, Q, tv, level=level, x=x, y=y, aggregation=agg)  # type: ignore[arg-type]
            check(t, ref_tile(q_view.T, level, x, y, 256, agg), exact=agg == "min" or level == 0)


def test_rank0_rank1_and_bool(base: Checkpoint, fixtures_dir: Path) -> None:
    s = compute_tile(base, "demo.scalar", None, level=0, x=0, y=0, aggregation="mean")
    assert s.values.tolist() == [[3.5]]
    vec = decode_tensor(fixtures_dir / "tiny_base.safetensors", "demo.vector.weight")
    for level in (0, 1, 2):
        t = compute_tile(base, "demo.vector.weight", None, level=level, x=0, y=0, aggregation="mean")
        check(t, ref_tile(vec.reshape(1, -1), level, 0, 0, 256, "mean"), exact=level == 0)
    with open_checkpoint(fixtures_dir / "tiny_precision_a.safetensors") as ck:
        b = compute_tile(ck, "p.bool", None, level=0, x=0, y=0, aggregation="mean")
        b1 = compute_tile(ck, "p.bool", None, level=1, x=0, y=0, aggregation="mean")
    assert b.values.tolist() == [[1.0, 0.0]]
    assert b1.values.tolist() == [[0.5]]


# ------------------------------------------------------------------------------------------ change, limits, budgets
def test_source_change_during_reads_is_file_changed(
    tmp_path: Path, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tiny_base.safetensors"
    shutil.copy(fixtures_dir / "tiny_base.safetensors", path)
    calls = {"n": 0}
    real = Checkpoint.read_roi_raw

    def read_then_change(self: Checkpoint, *args: Any, **kwargs: Any) -> Any:
        out = real(self, *args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:
            with path.open("ab") as f:
                f.write(b"\0")
        return out

    monkeypatch.setattr(Checkpoint, "read_roi_raw", read_then_change)
    with open_checkpoint(path) as ck:
        with pytest.raises(WeightLabError) as e:
            compute_tile(ck, Q, None, level=0, x=0, y=0, aggregation="mean", budget=TileBudget(band_cells=4096))
        assert e.value.code.value == "FILE_CHANGED"
        assert calls["n"] == 2  # stopped at the next read, nothing returned


def test_change_after_the_last_read_is_file_changed(
    tmp_path: Path, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tiny_base.safetensors"
    shutil.copy(fixtures_dir / "tiny_base.safetensors", path)
    real = Checkpoint.read_roi_raw

    def read_then_change(self: Checkpoint, *args: Any, **kwargs: Any) -> Any:
        out = real(self, *args, **kwargs)
        with path.open("ab") as f:
            f.write(b"\0")
        return out

    monkeypatch.setattr(Checkpoint, "read_roi_raw", read_then_change)
    with open_checkpoint(path) as ck, pytest.raises(WeightLabError) as e:
        compute_tile(ck, "demo.scalar", None, level=0, x=0, y=0, aggregation="mean")  # one read only
    assert e.value.code.value == "FILE_CHANGED"


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"level": 0, "x": 2, "y": 0}, "INVALID_ROI"),  # starts beyond column 263
        ({"level": 0, "x": 0, "y": 2}, "INVALID_ROI"),
        ({"level": 0, "x": 2**53, "y": 0}, "INVALID_ROI"),  # beyond 2^53-1
        ({"level": 0, "x": 2**53 - 1, "y": 0}, "INVALID_ROI"),
        ({"level": 0, "x": -1, "y": 0}, "INVALID_ROI"),
        ({"level": 53, "x": 0, "y": 0}, "VALIDATION_ERROR"),
        ({"level": -1, "x": 0, "y": 0}, "VALIDATION_ERROR"),
        ({"level": True, "x": 0, "y": 0}, "VALIDATION_ERROR"),
        ({"level": 1.0, "x": 0, "y": 0}, "VALIDATION_ERROR"),
        ({"level": 0, "x": 0, "y": 0, "size": 0}, "VALIDATION_ERROR"),
        ({"level": 0, "x": 0, "y": 0, "size": 257}, "VALIDATION_ERROR"),
        ({"level": 0, "x": 0, "y": 0, "aggregation": "median"}, "VALIDATION_ERROR"),
    ],
)
def test_invalid_geometry_is_rejected(base: Checkpoint, kwargs: dict[str, Any], code: str) -> None:
    before = base.bytes_read
    with pytest.raises(WeightLabError) as e:
        compute_tile(base, Q, None, **{"aggregation": "mean", **kwargs})
    assert e.value.code.value == code
    assert base.bytes_read == before  # rejected before any read


def test_invalid_views_are_rejected(base: Checkpoint) -> None:
    for view in (View(1, 2, {}, False), View(1, 1, {0: 0}, False), View(1, 2, {0: 2}, False)):
        with pytest.raises(WeightLabError) as e:
            compute_tile(base, "demo.cube.weight", view, level=0, x=0, y=0, aggregation="mean")
        assert e.value.code.value == "INVALID_ROI"


def test_budgets_are_enforced_before_reading(base: Checkpoint) -> None:
    before = base.bytes_read
    cases = [
        TileBudget(direct_tile_max_bytes=65536 * 4 - 1),  # L0 tile of F32 covers 256 KiB
        TileBudget(max_read_bytes=1000),
        TileBudget(memory_bytes=1 << 20),
        TileBudget(max_tile_cells=100),
    ]
    for b in cases:
        with pytest.raises(WeightLabError) as e:
            compute_tile(base, Q, None, level=0, x=0, y=0, aggregation="mean", budget=b)
        assert e.value.code.value == "RESOURCE_LIMIT", b
        assert base.bytes_read == before, b
    # Exactly at the direct budget is allowed.
    t = compute_tile(
        base, Q, None, level=0, x=0, y=0, aggregation="mean", budget=TileBudget(direct_tile_max_bytes=65536 * 4)
    )
    assert t.covered_bytes == 65536 * 4


def test_direct_budget_refuses_tiles_that_need_a_pyramid(tmp_path: Path) -> None:
    # A 4096 x 4096 F32 tensor (64 MiB, sparse on disk): an L4 tile covers it all = 64 MiB; an 8 MiB budget refuses
    # L3 (2048 x 2048 x 4 = 16 MiB) without reading and without falling back to sampling.
    path = tmp_path / "wide.safetensors"
    header = b'{"w":{"dtype":"F32","shape":[4096,4096],"data_offsets":[0,67108864]}}'
    header += b" " * (-len(header) % 8)
    with path.open("wb") as f:
        f.write(len(header).to_bytes(8, "little") + header)
        f.truncate(8 + len(header) + 67108864)
    budget = TileBudget.from_settings(
        {"direct_tile_max_mib": 8, "max_tile_cells": 65536, "memory_budget_mib": 1024, "max_active_tiles": 2}
    )
    with open_checkpoint(path) as ck:
        before = ck.bytes_read  # the header, read at open
        with pytest.raises(WeightLabError) as e:
            compute_tile(ck, "w", None, level=3, x=0, y=0, aggregation="mean", budget=budget)
        assert e.value.code.value == "RESOURCE_LIMIT"
        assert "pyramid" in e.value.detail
        assert ck.bytes_read == before
        t = compute_tile(ck, "w", None, level=2, x=0, y=0, aggregation="mean", budget=budget)  # 1024^2 x 4 = 4 MiB
        assert t.values.shape == (256, 256)
        assert not t.values.any()
        # Rows of 1024 of 4096 columns: the 12 KiB gaps between them are below coalesce_gap_bytes, so whole stripes are
        # read (fewer syscalls); the total stays within the read budget of READ_AMPLIFICATION x the direct limit.
        assert t.covered_bytes <= t.bytes_read <= budget.max_read_bytes


def test_transposed_reads_stay_near_the_covered_bytes(tmp_path: Path) -> None:
    """Reads run along the storage-contiguous axis, so a transposed or sliced view does not multiply I/O."""
    a = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    path = write_safetensors(tmp_path / "t.safetensors", {"w": ("F32", (1024, 1024), a.tobytes())})
    with open_checkpoint(path) as ck:
        plain = compute_tile(ck, "w", None, level=2, x=0, y=0, aggregation="mean")
        trans = compute_tile(ck, "w", View(0, 1, {}, True), level=2, x=0, y=0, aggregation="mean")
    assert plain.bytes_read == trans.bytes_read == a.nbytes
    np.testing.assert_array_equal(trans.values, plain.values.T)


# ------------------------------------------------------------------------------------------ high levels, bounded
@pytest.mark.parametrize("level", [16, 32, 52])
@pytest.mark.parametrize("shape", [(8, 8), (3, 300), (300, 3)])
def test_tiny_tensors_at_huge_levels_use_memory_of_the_tensor_not_the_block(
    tmp_path: Path, level: int, shape: tuple[int, int]
) -> None:
    """A level-L block is 2^L x 2^L cells (level 16: 4 Gi cells, 32 GiB as float64). The engine must reduce only
    the tensor's own cells: the traced allocation peak stays within a few times the tensor, whatever the level."""
    import tracemalloc

    rng = np.random.default_rng(level)
    a = rng.standard_normal(shape).astype("<f4")
    a.flat[1] = np.nan
    a.flat[2] = -np.inf
    path = write_safetensors(tmp_path / "t.safetensors", {"w": ("F32", shape, a.tobytes())})
    v = view2d(decode_tensor(path, "w"), 0, 1, {}, False)
    with open_checkpoint(path) as ck:
        for agg in AGGREGATIONS:
            tracemalloc.start()
            try:
                tile = compute_tile(ck, "w", None, level=level, x=0, y=0, aggregation=agg,
                                    budget=TileBudget(band_cells=97))  # fmt: skip  # several rectangles, odd phases
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            check(tile, ref_tile(v, level, 0, 0, 256, agg), exact=agg in ("min", "max"))
            assert tile.status.tolist() == [[2 | 8]]  # HAS_NAN | HAS_NEG_INF; the finite aggregate is drawable
            assert (tile.finite_count, tile.nonfinite_count) == (a.size - 2, 2)
            assert peak < 1024 * 1024, f"level {level} {agg}: traced peak {peak} bytes"


def test_large_blocks_over_partial_rectangles_match_the_reference(tmp_path: Path) -> None:
    """Blocks larger than the read rectangles (block-size-independent reduction path) with every phase offset."""
    rng = np.random.default_rng(7)
    a = (rng.standard_normal((70, 90)) * 10.0 ** rng.integers(-3, 4, (70, 90))).astype("<f8")
    a[5, 5], a[60, 80], a[33, 1] = np.nan, np.inf, -0.0
    path = write_safetensors(tmp_path / "t.safetensors", {"w": ("F64", (70, 90), a.tobytes())})
    v = view2d(decode_tensor(path, "w"), 0, 1, {}, False)
    with open_checkpoint(path) as ck:
        for level in (5, 6, 7, 20):
            for band in (13, 64, 1000):
                for agg in AGGREGATIONS:
                    span = 3 << level
                    for y in range(-(-70 // span)):
                        for x in range(-(-90 // span)):
                            t = compute_tile(ck, "w", None, level=level, x=x, y=y, size=3, aggregation=agg,
                                             budget=TileBudget(band_cells=band))  # fmt: skip
                            check(t, ref_tile(v, level, x, y, 3, agg), exact=agg in ("min", "max"))
