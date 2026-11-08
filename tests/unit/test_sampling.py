"""Row-sampled previews (doc 05 section 5): deterministic selection (stride, seeded SplitMix64 rows) checked against
an independent re-implementation built on the fixture generator's own `splitmix64`, golden selections pinned as
literals (so every CI OS must reproduce them), labels, budgets, and values against the reference aggregation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers import ROOT, write_safetensors
from tests.reference_tiles import _status, aggregate, decode_tensor, view2d
from weightlab import open_checkpoint
from weightlab.analysis import sampling as S
from weightlab.analysis import tile_codec as tc
from weightlab.analysis.tile import AGGREGATIONS
from weightlab.core.errors import WeightLabError
from weightlab.formats.selection import View


def _generator() -> Any:
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "generate_demo_fixtures.py")
    assert spec is not None and spec.loader is not None  # noqa: PT018
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _generator()


def ref_mix(pyramid_id: str, level: int, x: int, y: int, seed: int) -> int:
    raw = json.dumps({"level": level, "pyramid_id": pyramid_id, "seed": seed, "x": x, "y": y},
                     separators=(",", ":")).encode()  # fmt: skip
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def ref_select(n: int, k: int, mix: int | None, block_row: int) -> list[int]:
    """Independent: stride strata, or Floyd's k-subset with rejection-sampled bounded draws."""
    if mix is None:
        return [((2 * j + 1) * n) // (2 * k) for j in range(k)]
    chosen: set[int] = set()
    draw = 0
    for j in range(n - k, n):
        bound = j + 1
        limit = 2**64 - (2**64 % bound)
        while True:
            r = GEN.splitmix64(mix ^ ((block_row << 20) | draw))
            draw += 1
            if r < limit:
                break
        t = r % bound
        chosen.add(j if t in chosen else t)
    return sorted(chosen)


# ---------------------------------------------------------------------------------------------- selection
def test_splitmix64_is_the_generators_function() -> None:
    for x in (0, 1, 2**63, 2**64 - 1, 0x9E3779B97F4A7C15, 123456789):
        assert S.splitmix64(x) == GEN.splitmix64(x)
    assert S.splitmix64(0) == 0xE220A8397B1DCDAF  # published SplitMix64 test vector


def test_golden_selections_are_reproduced_on_every_platform() -> None:
    mix = S.seed_mix("0" * 64, 9, 0, 0, 42)
    assert mix == ref_mix("0" * 64, 9, 0, 0, 42)
    assert mix == 0xBE714A054E6DF2AA  # pinned from the independent reference above
    assert S.select_rows(512, 8, policy="splitmix64_rows", mix=mix, block_row=0) == [
        147, 156, 164, 373, 412, 414, 489, 501]  # fmt: skip
    assert S.select_rows(512, 8, policy="splitmix64_rows", mix=mix, block_row=3) == [
        34, 39, 113, 166, 168, 228, 272, 376]  # fmt: skip
    assert S.select_rows(512, 8, policy="stride") == [32, 96, 160, 224, 288, 352, 416, 480]
    assert S.select_rows(37, 5, policy="stride") == [3, 11, 18, 25, 33]


@pytest.mark.parametrize("n", [1, 2, 16, 37, 512, 2**20, 2**52])
@pytest.mark.parametrize("k", [1, 2, 8, 256])
def test_selection_matches_the_independent_reference(n: int, k: int) -> None:
    if k > n:
        return
    for seed in (0, 1, 2**32 - 1):
        mix = S.seed_mix("ab" * 32, 6, 1, 2, seed)
        for block_row in (0, 1, 777):
            got = S.select_rows(n, k, policy="splitmix64_rows", mix=mix, block_row=block_row)
            assert got == ref_select(n, k, mix, block_row)
            assert len(set(got)) == k
            assert all(0 <= r < n for r in got)
    assert S.select_rows(n, k, policy="stride") == ref_select(n, k, None, 0)


def test_fixed_seeds_repeat_and_distinct_seeds_differ() -> None:
    pid = "cd" * 32
    a = [S.select_rows(4096, 8, policy="splitmix64_rows", mix=S.seed_mix(pid, 12, 0, 0, 7), block_row=b)
         for b in range(16)]  # fmt: skip
    again = [S.select_rows(4096, 8, policy="splitmix64_rows", mix=S.seed_mix(pid, 12, 0, 0, 7), block_row=b)
             for b in range(16)]  # fmt: skip
    other = [S.select_rows(4096, 8, policy="splitmix64_rows", mix=S.seed_mix(pid, 12, 0, 0, 8), block_row=b)
             for b in range(16)]  # fmt: skip
    assert a == again
    assert a != other
    assert sum(x != y for x, y in zip(a, other, strict=True)) >= 15
    other_source = S.seed_mix("ce" * 32, 12, 0, 0, 7)  # a different file fingerprint / key / view
    assert S.select_rows(4096, 8, policy="splitmix64_rows", mix=other_source, block_row=0) != a[0]


def test_seeded_selection_is_uniform_enough() -> None:
    counts = np.zeros(10, int)
    mix = S.seed_mix("ef" * 32, 3, 0, 0, 1)
    for b in range(2000):
        for r in S.select_rows(10, 3, policy="splitmix64_rows", mix=mix, block_row=b):
            counts[r] += 1
    expected = 2000 * 3 / 10
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 27.9  # chi-square, 9 degrees of freedom, p = 0.001


# ---------------------------------------------------------------------------------------- sampled tiles
def ref_sampled(v: np.ndarray, level: int, x: int, y: int, size: int, agg: str, k: int, mix: int | None) -> Any:
    b = 1 << level
    span = size * b
    r0, c0 = y * span, x * span
    r1, c1 = min(v.shape[0], r0 + span), min(v.shape[1], c0 + span)
    rows, cols = -(-(r1 - r0) // b), -(-(c1 - c0) // b)
    values = np.full((rows, cols), np.nan, np.float32)
    status = np.zeros((rows, cols), np.uint8)
    seen = 0
    for i in range(rows):
        top = r0 + i * b
        n = min(r1, top + b) - top
        picked = [top + o for o in ref_select(n, min(k, n), mix, top // b)]
        for j in range(cols):
            blk = [float(q) for r in picked for q in v[r, c0 + j * b : min(c1, c0 + (j + 1) * b)]]
            seen += len(blk)
            val = aggregate(blk, agg)
            st = _status(blk, val if val is not None else (None if not any(map(math.isfinite, blk)) else math.inf))
            status[i, j] = st
            if st & 17 == 0 and val is not None:
                values[i, j] = np.float32(val)
    return values, status, seen


@pytest.fixture
def tall(tmp_path: Path) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((300, 100)).astype("<f4")
    a[5, 7], a[40, 9], a[41, 70], a[299, 99] = np.nan, np.inf, -np.inf, np.nan
    path = write_safetensors(tmp_path / "t.safetensors", {"w": ("F32", (300, 100), a.tobytes())})
    return path, view2d(decode_tensor(path, "w"), 0, 1, {}, False)


@pytest.mark.parametrize("seed", [None, 0, 99])
def test_sampled_tiles_equal_the_reference_aggregation_of_the_selected_rows(
    tall: tuple[Path, np.ndarray], seed: int | None
) -> None:
    path, v = tall
    pid = "12" * 32
    with open_checkpoint(path) as ck:
        for level, size, x, y in ((5, 256, 0, 0), (4, 3, 1, 2), (6, 2, 0, 1), (4, 256, 0, 0)):
            for k in (1, 3, 8):
                plan = S.plan_sample(ck, "w", None, level=level, x=x, y=y, size=size, rows_per_block=k, seed=seed,
                                     max_read_bytes=1 << 30, max_cells=1 << 30)  # fmt: skip
                if plan.complete:
                    continue
                for agg in AGGREGATIONS:
                    t = S.sampled_tile(ck, "w", None, level=level, x=x, y=y, size=size, aggregation=agg, plan=plan,
                                       pyramid_id=pid, band_cells=37)  # fmt: skip
                    mix = None if seed is None else S.seed_mix(pid, level, x, y, seed)
                    values, status, seen = ref_sampled(v, level, x, y, size, agg, k, mix)
                    np.testing.assert_array_equal(t.status, status)
                    np.testing.assert_allclose(t.values, values, rtol=2.0**-23, atol=0, equal_nan=True)
                    assert t.finite_count + t.nonfinite_count == seen < t.geometry.covered_cells
                    assert t.quality == "row_sampled"
                    assert t.sampled_rows == k
                    assert t.sampling_policy == ("stride" if seed is None else "splitmix64_rows")
                    assert t.seed == seed
                    assert t.numeric_precision in ("rounded", "overflowed")  # never lossless, never exact
                    assert any("not exact" in w for w in t.warnings)
                    meta = tc.metadata(t, cache_key="0" * 64, source={"kind": "tensor"}, view=View(0, 1, {}, False))
                    tc.validate_metadata(meta, t.geometry.rows, t.geometry.cols)


def test_sampled_metadata_never_claims_exact_coverage() -> None:
    base = {"schema_version": 2, "rows": 1, "cols": 1, "row_start": 0, "row_end": 32, "col_start": 0, "col_end": 32,
            "level": 5, "quality": "row_sampled", "value_precision": "float32_visualization", "sampled_rows": 4,
            "sampling_policy": "stride", "seed": None, "finite_count": 128, "nonfinite_count": 0}  # fmt: skip
    tc.validate_metadata(base, 1, 1)
    for bad in ({"finite_count": 1024}, {"sampled_rows": None}, {"sampling_policy": None},
                {"sampling_policy": "splitmix64_rows"}, {"seed": 3}, {"quality": "sampled"}):  # fmt: skip
        with pytest.raises(WeightLabError):
            tc.validate_metadata({**base, **bad}, 1, 1)
    with pytest.raises(WeightLabError):
        tc.validate_metadata({**base, "quality": "exact"}, 1, 1)  # an exact tile with sampling fields


def test_budgets_are_checked_before_any_read(tall: tuple[Path, np.ndarray]) -> None:
    path, _ = tall
    with open_checkpoint(path) as ck:
        before = ck.bytes_read
        with pytest.raises(WeightLabError) as e:
            S.plan_sample(ck, "w", None, level=5, x=0, y=0, size=256, rows_per_block=8, seed=None,
                          max_read_bytes=1000, max_cells=1 << 30)  # fmt: skip
        assert e.value.code.value == "RESOURCE_LIMIT"
        auto = S.plan_sample(ck, "w", None, level=5, x=0, y=0, size=256, rows_per_block=None, seed=None,
                             max_read_bytes=10 * 100 * 4 * 2, max_cells=1 << 30)  # fmt: skip
        assert auto.rows_per_block == 2  # the default (8) halved until the reads fit
        assert ck.bytes_read == before


def test_a_selection_of_every_row_is_reported_complete(tall: tuple[Path, np.ndarray]) -> None:
    path, _ = tall
    with open_checkpoint(path) as ck:
        plan = S.plan_sample(ck, "w", None, level=4, x=0, y=1, size=3, rows_per_block=16, seed=None,
                             max_read_bytes=1 << 30, max_cells=1 << 30)  # fmt: skip
        assert plan.complete  # 16 of 16 rows per block: that is an exact scan, never labelled row_sampled
        with pytest.raises(ValueError, match="exact"):
            S.sampled_tile(ck, "w", None, level=4, x=0, y=1, size=3, aggregation="mean", plan=plan, pyramid_id="0" * 64)


def test_transposed_views_are_refused_for_row_sampling(tall: tuple[Path, np.ndarray]) -> None:
    path, _ = tall
    with open_checkpoint(path) as ck, pytest.raises(WeightLabError) as e:
        S.plan_sample(ck, "w", View(0, 1, {}, True), level=5, x=0, y=0, size=256, rows_per_block=2, seed=None,
                      max_read_bytes=1 << 30, max_cells=1 << 30)  # fmt: skip
    assert e.value.code.value == "RESOURCE_LIMIT"
