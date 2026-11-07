"""Property tests: any tile (level, coordinates, size, aggregation) of any view of a random F32/F64 tensor with NaN,
+/-Inf, -0.0 and extreme values equals the independent reference (tests/reference_tiles.py), whatever the read band
size; the tiles of one level partition the view exactly."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.helpers import write_safetensors
from tests.reference_tiles import decode_tensor, ref_tile, view2d
from weightlab import open_checkpoint
from weightlab.analysis.tile import AGGREGATIONS, TileBudget, compute_tile, tile_geometry
from weightlab.formats.selection import View

SPECIAL = [np.nan, np.inf, -np.inf, -0.0, 0.0, 1e300, -1e300, 5e-324, 3.4e38, 3.5e38]


@st.composite
def cases(draw: st.DrawFn) -> dict[str, object]:
    dtype = draw(st.sampled_from(["F32", "F64"]))
    rank = draw(st.integers(1, 3))
    shape = tuple(draw(st.lists(st.integers(1, 23), min_size=rank, max_size=rank)))
    n = int(np.prod(shape))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n) * 10.0 ** rng.integers(-3, 4, n)
    k = draw(st.integers(0, min(n, 8)))
    for i in rng.choice(n, size=k, replace=False):
        a[i] = SPECIAL[int(rng.integers(len(SPECIAL)))]
    with np.errstate(over="ignore"):
        data = a.astype("<f4" if dtype == "F32" else "<f8").tobytes()
    if rank == 1:
        view = None
    else:
        ra, ca = draw(st.permutations(range(rank)))[:2]
        fixed = {ax: draw(st.integers(0, shape[ax] - 1)) for ax in range(rank) if ax not in (ra, ca)}
        view = View(ra, ca, fixed, draw(st.booleans()))
    level = draw(st.integers(0, 4))
    size = draw(st.integers(1, 9))
    band = draw(st.integers(1, 64))
    agg = draw(st.sampled_from(AGGREGATIONS))
    return {
        "dtype": dtype,
        "shape": shape,
        "data": data,
        "view": view,
        "level": level,
        "size": size,
        "band": band,
        "agg": agg,
    }


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(cases())
def test_any_tile_matches_the_reference(case: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = write_safetensors(Path(d) / "t.safetensors", {"w": (case["dtype"], case["shape"], case["data"])})  # type: ignore[dict-item]
        t = decode_tensor(path, "w")
        view: View | None = case["view"]  # type: ignore[assignment]
        v = (
            view2d(t, None, None, {}, False)
            if view is None
            else view2d(t, view.row_axis, view.col_axis, dict(view.fixed_indices), view.transpose)
        )
        level, size = int(case["level"]), int(case["size"])  # type: ignore[call-overload]
        g0 = tile_geometry(v.shape[0], v.shape[1], level, 0, 0, size)
        span = size << level
        covered = 0
        with open_checkpoint(path) as ck:
            for y in range(-(-v.shape[0] // span)):
                for x in range(-(-v.shape[1] // span)):
                    tile = compute_tile(ck, "w", view, level=level, x=x, y=y, size=size, aggregation=case["agg"],  # type: ignore[arg-type]
                                        budget=TileBudget(band_cells=int(case["band"])))  # type: ignore[call-overload]  # fmt: skip
                    ref = ref_tile(v, level, x, y, size, str(case["agg"]))
                    np.testing.assert_array_equal(tile.status, ref["status"])
                    np.testing.assert_allclose(tile.values, ref["values"], rtol=2.0**-23, atol=0, equal_nan=True)
                    assert (tile.finite_count, tile.nonfinite_count) == (ref["finite"], ref["nonfinite"])
                    covered += tile.geometry.covered_cells
        assert covered == v.shape[0] * v.shape[1]  # the tiles of one level partition the view
        assert g0.row_start == 0
        assert g0.col_start == 0
