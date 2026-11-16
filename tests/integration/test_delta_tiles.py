"""Delta heatmap tiles, pyramids and row-sampled previews (ROADMAP P1.1; doc 03 section 5, doc 09 sections 3-4)
against the independent reference: `ref_delta` (B - A per element in exact rational arithmetic, its own decoder)
aggregated block by block by `ref_tile`, and the generator's golden values. Never compares two WeightLab paths for
expected values."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers import write_safetensors
from tests.reference_tiles import aggregate, decode_tensor, ref_delta, ref_tile, view2d
from weightlab import open_checkpoint
from weightlab.analysis import pyramid as P
from weightlab.analysis import sampling
from weightlab.analysis import tile_codec as tc
from weightlab.analysis.readers import DeltaReader, delta_values
from weightlab.analysis.tile import AGGREGATIONS, Tile, compute_source_tile
from weightlab.core.errors import WeightLabError
from weightlab.formats.selection import View

F32_ULP = 2.0**-23
PID = "cd" * 32
TRANSFORMS = ("b_minus_a", "abs_b_minus_a", "relative_abs")


def ident(transform: str, view: View | None = None) -> dict[str, Any]:
    return tc.pyramid_identity(source_kind="delta", files=[], tensor_keys=["w"], view=view,
                               operation={"transform": transform})  # fmt: skip


def check(t: Tile, ref: dict[str, Any], agg: str) -> None:
    g = t.geometry
    assert (g.row_start, g.row_end, g.col_start, g.col_end) == ref["range"]
    np.testing.assert_array_equal(t.status, ref["status"])
    assert (t.finite_count, t.nonfinite_count) == (ref["finite"], ref["nonfinite"])
    if agg in ("min", "max") or g.level == 0:  # one exactly rounded value per cell
        np.testing.assert_array_equal(t.values, ref["values"])
    else:  # float64 summation order vs the exactly rounded reference: <= 1 float32 ulp (well-conditioned data)
        np.testing.assert_allclose(t.values, ref["values"], rtol=F32_ULP, atol=0, equal_nan=True)


def pair(tmp_path: Path, a: np.ndarray, b: np.ndarray, dta: str = "F32", dtb: str = "F32") -> tuple[Path, Path]:
    np_dt = {"F32": "<f4", "F64": "<f8", "F16": "<f2"}
    pa = write_safetensors(tmp_path / "a.safetensors", {"w": (dta, a.shape, a.astype(np_dt[dta]).tobytes())})
    pb = write_safetensors(tmp_path / "b.safetensors", {"w": (dtb, b.shape, b.astype(np_dt[dtb]).tobytes())})
    return pa, pb


def synthetic(shape: tuple[int, ...], seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """A and B = A + an update, with NaN / +Inf / -Inf on either side and in both."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape).astype(np.float32)
    b = (a + 0.05 * rng.standard_normal(shape)).astype(np.float32)
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    flat_a[3], flat_b[7], flat_a[11], flat_b[11] = np.nan, np.inf, -np.inf, -np.inf  # inf - inf = NaN
    flat_b[19] = np.nan
    flat_a[23] = np.inf
    return a, b


def refs(pa: Path, pb: Path, transform: str, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    return ref_delta(decode_tensor(pa, "w"), decode_tensor(pb, "w"), transform, eps)


def view_of(arr: np.ndarray, view: View | None) -> np.ndarray:
    if view is None:  # the default view: a matrix as stored, a vector or scalar as one row
        return arr if arr.ndim == 2 else view2d(arr, None, None, {}, False)
    return view2d(arr, view.row_axis, view.col_axis, dict(view.fixed_indices), view.transpose)


# --------------------------------------------------------------------------------------------- direct tiles
def test_golden_rms_of_the_delta_is_not_the_difference_of_rms(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    gold = expected["rms_delta_vs_difference_of_rms"]
    with open_checkpoint(fixtures_dir / "tiny_base.safetensors") as a, \
            open_checkpoint(fixtures_dir / "tiny_ft.safetensors") as b:  # fmt: skip
        src = DeltaReader(a, b, gold["key"], "b_minus_a", 1e-8)
        whole = compute_source_tile(src, None, level=3, x=0, y=0, aggregation="rms")  # one cell: the 8x8 tensor
        l0 = compute_source_tile(src, None, level=0, x=0, y=0, aggregation="mean")
    assert whole.values.shape == (1, 1)
    assert math.isclose(float(whole.values[0, 0]), gold["rms_of_delta"], rel_tol=F32_ULP)
    assert not math.isclose(float(whole.values[0, 0]), gold["rms_ft_minus_rms_base"], rel_tol=0.1)
    np.testing.assert_array_equal(l0.values, np.array(expected["delta_full"], np.float32))
    assert l0.numeric_precision == "rounded"  # derived: float64 arithmetic, never lossless
    assert l0.bytes_read == 2 * 8 * 8 * 4  # both inputs read, nothing else


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("view", [None, View(0, 1, {}, True)], ids=["plain", "transposed"])
def test_every_level_and_aggregation_matches_the_reference(tmp_path: Path, transform: str, view: View | None) -> None:
    a, b = synthetic((37, 45))
    pa, pb = pair(tmp_path, a, b, "F32", "F16")  # mixed dtypes: both promoted to float64
    d, over = refs(pa, pb, transform, 0.5)
    v, ov = view_of(d, view), view_of(over, view)
    with open_checkpoint(pa) as ca, open_checkpoint(pb) as cb:
        src = DeltaReader(ca, cb, "w", transform, 0.5)
        for level in range(6):
            span = 16 << level
            for y in range(-(-v.shape[0] // span)):
                for x in range(-(-v.shape[1] // span)):
                    for agg in AGGREGATIONS:
                        t = compute_source_tile(src, view, level=level, x=x, y=y, size=16, aggregation=agg)
                        check(t, ref_tile(v, level, x, y, 16, agg, ov), agg)
                        assert t.source_dtype == "F32"
                        assert t.numeric_precision == "rounded"


def test_rank3_slices_are_paired_cell_by_cell(tmp_path: Path) -> None:
    a, b = synthetic((3, 5, 7), seed=9)
    pa, pb = pair(tmp_path, a, b)
    d, over = refs(pa, pb, "b_minus_a")
    with open_checkpoint(pa) as ca, open_checkpoint(pb) as cb:
        src = DeltaReader(ca, cb, "w", "b_minus_a", 1e-8)
        for view in (View(1, 2, {0: 1}, False), View(2, 0, {1: 3}, True), View(0, 2, {1: 4}, False)):
            v, ov = view_of(d, view), view_of(over, view)
            for level in (0, 1, 2):
                for agg in AGGREGATIONS:
                    t = compute_source_tile(src, view, level=level, x=0, y=0, aggregation=agg)
                    check(t, ref_tile(v, level, 0, 0, 256, agg, ov), agg)


def test_overflowed_deltas_are_withheld_never_an_inf(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    pr = expected["precision"]
    pa, pb = fixtures_dir / "tiny_precision_a.safetensors", fixtures_dir / "tiny_precision_b.safetensors"
    d, over = ref_delta(decode_tensor(pa, "p.f64"), decode_tensor(pb, "p.f64"), "b_minus_a")
    i = pr["f64_delta_overflow_index"]
    assert over.tolist() == [j == i for j in range(over.size)]  # the reference sees the fixture's overflow
    v, ov = view_of(d, None), view_of(over, None)
    with open_checkpoint(pa) as ca, open_checkpoint(pb) as cb:
        src = DeltaReader(ca, cb, "p.f64", "b_minus_a", 1e-8)
        l0 = compute_source_tile(src, None, level=0, x=0, y=0, aggregation="mean")
        assert int(l0.status[0, i]) == 16  # NOT_REPRESENTABLE: a finite element whose value is withheld
        assert math.isnan(float(l0.values[0, i]))  # placeholder, never drawn
        assert l0.numeric_precision == "overflowed"
        assert any("NOT_REPRESENTABLE" in w for w in l0.warnings)
        assert l0.finite_count == 8 and l0.nonfinite_count == 0  # noqa: PT018
        for level in range(4):
            for agg in AGGREGATIONS:
                t = compute_source_tile(src, None, level=level, x=0, y=0, aggregation=agg)
                check(t, ref_tile(v, level, 0, 0, 256, agg, ov), agg)
        f32 = DeltaReader(ca, cb, "p.f32_overflow_delta", "b_minus_a", 1e-8)
        t32 = compute_source_tile(f32, None, level=0, x=0, y=0, aggregation="mean")
    want = float(pr["f32_overflow_delta_b_minus_a_f64"])  # exact in float64, beyond float32: NOT_REPRESENTABLE
    assert want > float(np.finfo(np.float32).max)
    assert 16 in t32.status.tolist()[0]
    assert t32.numeric_precision == "overflowed"


def test_an_overflow_that_does_not_determine_an_extreme_leaves_it_exact(tmp_path: Path) -> None:
    big = float(np.finfo(np.float64).max)
    a = np.array([[-big, 0.0, 5.0, big]])
    b = np.array([[big, 1.0, 2.0, -big]])  # deltas: +overflow, 1, -3, -overflow
    pa, pb = pair(tmp_path, a, b, "F64", "F64")
    with open_checkpoint(pa) as ca, open_checkpoint(pb) as cb:
        src = DeltaReader(ca, cb, "w", "b_minus_a", 1e-8)
        three = {agg: compute_source_tile(src, View(0, 1, {}, False), level=0, x=0, y=0, aggregation=agg)
                 for agg in AGGREGATIONS}  # fmt: skip
        assert three["mean"].status.tolist() == [[16, 0, 0, 16]]
        # Level 1 pairs columns (0, 1) = (+overflow, 1) and (2, 3) = (-3, -overflow).
        t = {agg: compute_source_tile(src, View(0, 1, {}, False), level=1, x=0, y=0, aggregation=agg)
             for agg in AGGREGATIONS}  # fmt: skip
    assert t["min"].status.tolist() == [[0, 16]] and float(t["min"].values[0, 0]) == 1.0  # noqa: PT018
    assert t["max"].status.tolist() == [[16, 0]] and float(t["max"].values[0, 1]) == -3.0  # noqa: PT018
    for agg in ("mean", "mean_abs", "rms"):
        assert t[agg].status.tolist() == [[16, 16]]
        assert t[agg].numeric_precision == "overflowed"


def test_non_finite_pairs_follow_ieee_and_are_counted_by_kind(tmp_path: Path) -> None:
    a = np.array([[1.0, np.inf, np.inf, np.nan, 2.0, -np.inf]], np.float32)
    b = np.array([[np.inf, 1.0, np.inf, 1.0, 2.0, 3.0]], np.float32)
    pa, pb = pair(tmp_path, a, b)
    with open_checkpoint(pa) as ca, open_checkpoint(pb) as cb:
        t = compute_source_tile(DeltaReader(ca, cb, "w", "b_minus_a", 1e-8), None, level=0, x=0, y=0,
                                aggregation="mean")  # fmt: skip
        r = compute_source_tile(DeltaReader(ca, cb, "w", "relative_abs", 1e-8), None, level=0, x=0, y=0,
                                aggregation="mean")  # fmt: skip
    # +Inf, -Inf, NaN (inf - inf), NaN, 0, +Inf
    assert t.status.tolist() == [[5, 9, 3, 3, 0, 5]]
    assert (t.finite_count, t.nonfinite_count) == (1, 5)
    # |B - A| / max(|A|, eps): inf/1, inf/inf = NaN, NaN, NaN, 0, inf/inf = NaN
    assert r.status.tolist() == [[5, 3, 3, 3, 0, 3]]


def test_mismatched_shapes_are_incompatible(tmp_path: Path) -> None:
    pa = write_safetensors(tmp_path / "a.safetensors", {"w": ("F32", (2, 3), np.zeros(6, "<f4").tobytes())})
    pb = write_safetensors(tmp_path / "b.safetensors", {"w": ("F32", (3, 2), np.zeros(6, "<f4").tobytes())})
    with open_checkpoint(pa) as ca, open_checkpoint(pb) as cb, pytest.raises(WeightLabError) as e:
        DeltaReader(ca, cb, "w", "b_minus_a", 1e-8)
    assert e.value.code.value == "INCOMPATIBLE_TENSOR"


def test_the_roi_arithmetic_is_the_tile_arithmetic() -> None:
    a = np.array([1.0, -2.0, 3.0, np.finfo(np.float64).max, 0.0])
    b = np.array([1.5, 2.0, 3.0, -np.finfo(np.float64).max, 1e-12])
    for transform in TRANSFORMS:
        d, over = delta_values(a, b, transform, 1e-8)
        rd, rover = ref_delta(a, b, transform, 1e-8)
        np.testing.assert_array_equal(over, rover)
        np.testing.assert_array_equal(d[~over], rd[~rover])


# ------------------------------------------------------------------------------------------------ pyramids
def build(src: DeltaReader, view: View | None, transform: str, band_cells: int = P.BAND_CELLS,
          control: Any = None) -> tuple[Any, P.BytesBlob, P.Pyramid]:  # fmt: skip
    sink = P.MemorySink(0)
    res = P.build_source_pyramid(src, view, ident(transform, view), PID, sink, control, band_cells=band_cells)
    blob = P.BytesBlob(bytes(sink.buf[: res.total_bytes]))
    return res, blob, P.load(blob, ident(transform, view), PID)


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("view", [None, View(0, 1, {}, True)], ids=["plain", "transposed"])
def test_delta_pyramid_tiles_match_the_reference(tmp_path: Path, transform: str, view: View | None) -> None:
    a, b = synthetic((300, 70), seed=3)
    pa, pb = pair(tmp_path, a, b)
    d, over = refs(pa, pb, transform, 0.25)
    v, ov = view_of(d, view), view_of(over, view)
    with open_checkpoint(pa) as ca, open_checkpoint(pb) as cb:
        src = DeltaReader(ca, cb, "w", transform, 0.25)
        res, blob, pyr = build(src, view, transform)
        assert res.bytes_read == 2 * 300 * 70 * 4
        assert pyr.derived
        small, *_ = build(DeltaReader(ca, cb, "w", transform, 0.25), view, transform, band_cells=16 * 70)
        assert small.level_sha256 == res.level_sha256  # banding never changes a byte
    for level in range(4, 10):
        span = 8 << level
        for y in range(-(-v.shape[0] // span)):
            for x in range(-(-v.shape[1] // span)):
                for agg in AGGREGATIONS:
                    t = P.pyramid_tile(blob, pyr, level=level, x=x, y=y, size=8, aggregation=agg,
                                       view_rows=v.shape[0], view_cols=v.shape[1], esize=8)  # fmt: skip
                    check(t, ref_tile(v, level, x, y, 8, agg, ov), agg)
                    assert t.numeric_precision == "rounded"  # derived even for min / max


def test_delta_pyramid_with_overflow_withholds_and_validates(tmp_path: Path) -> None:
    big = float(np.finfo(np.float64).max)
    rng = np.random.default_rng(1)
    a = rng.standard_normal((40, 50))
    b = a + 0.1
    a[5, 7], b[5, 7] = -big, big  # +overflow
    a[33, 41], b[33, 41] = big, -big  # -overflow
    pa, pb = pair(tmp_path, a, b, "F64", "F64")
    d, over = refs(pa, pb, "b_minus_a")
    assert int(over.sum()) == 2
    with open_checkpoint(pa) as ca, open_checkpoint(pb) as cb:
        _, blob, pyr = build(DeltaReader(ca, cb, "w", "b_minus_a", 1e-8), None, "b_minus_a")
    for level in range(4, 8):
        for agg in AGGREGATIONS:
            t = P.pyramid_tile(blob, pyr, level=level, x=0, y=0, size=256, aggregation=agg, view_rows=40,
                               view_cols=50, esize=16)  # fmt: skip
            check(t, ref_tile(d, level, 0, 0, 256, agg, over), agg)
    rec = P.to_records(P.base_records(d[:16, :16], over[:16, :16]), P.BASE_LEVEL)
    cells = np.full(rec.shape, 256, np.int64)
    P._validate(rec, cells, derived=True)
    with pytest.raises(P.PyramidCorrupt, match="extrema"):  # a stored tensor never has an infinite extreme
        P._validate(rec, cells, derived=False)


def test_a_change_of_either_input_during_a_delta_build_is_file_changed(tmp_path: Path) -> None:
    a = np.arange(64 * 64, dtype="<f4").reshape(64, 64)
    pa, pb = pair(tmp_path, a, a + 1)

    def touch() -> None:
        st = pb.stat()
        os.utime(pb, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    class Ticks:
        n = 0

        def tick(self, nbytes: int) -> None:
            self.n += 1
            if self.n == 2:
                touch()

    with open_checkpoint(pa) as ca, open_checkpoint(pb) as cb, pytest.raises(WeightLabError) as e:
        build(DeltaReader(ca, cb, "w", "b_minus_a", 1e-8), None, "b_minus_a", band_cells=16 * 64, control=Ticks())
    assert e.value.code.value == "FILE_CHANGED"


def test_a_raising_control_stops_a_delta_build_early(tmp_path: Path) -> None:
    a = np.ones((256, 64), "<f4")
    pa, pb = pair(tmp_path, a, a * 2)

    class Stop(Exception):
        pass

    class Ticks:
        n = 0

        def tick(self, nbytes: int) -> None:
            self.n += 1
            if self.n == 3:
                raise Stop

    with open_checkpoint(pa) as ca, open_checkpoint(pb) as cb:
        with pytest.raises(Stop):
            build(DeltaReader(ca, cb, "w", "b_minus_a", 1e-8), None, "b_minus_a", band_cells=16 * 64,
                  control=Ticks())  # fmt: skip
        assert ca.bytes_read < 256 * 64 * 4 and cb.bytes_read < 256 * 64 * 4  # noqa: PT018


# ------------------------------------------------------------------------------------------- row sampling
def test_row_sampled_delta_previews_pair_the_selected_rows(tmp_path: Path) -> None:
    a, b = synthetic((64, 40), seed=4)
    pa, pb = pair(tmp_path, a, b)
    d, over = refs(pa, pb, "abs_b_minus_a")
    assert not over.any()
    with open_checkpoint(pa) as ca, open_checkpoint(pb) as cb:
        src = DeltaReader(ca, cb, "w", "abs_b_minus_a", 1e-8)
        plan = sampling.plan_sample(src, "w", None, level=4, x=0, y=0, size=256, rows_per_block=4, seed=None,
                                    max_read_bytes=1 << 20, max_cells=1 << 20)  # fmt: skip
        assert plan.read_bytes == plan.rows_read * 40 * 8  # rows of both inputs
        t = sampling.sampled_tile(src, "w", None, level=4, x=0, y=0, size=256, aggregation="mean", plan=plan,
                                  pyramid_id=PID)  # fmt: skip
    assert t.quality == "row_sampled"
    assert t.covered_bytes == 64 * 40 * 8
    for i in range(4):
        rows = [16 * i + off for off in sampling.select_rows(16, 4, policy="stride")]
        for j in range(3):
            blk = [float(x) for x in d[rows, 16 * j : 16 * (j + 1)].ravel()]
            want = aggregate(blk, "mean")
            if want is None:
                assert t.status[i, j] & 1
            else:
                assert math.isclose(float(t.values[i, j]), float(np.float32(want)), rel_tol=F32_ULP)
