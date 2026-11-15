"""Exact tensor histograms (ADR-021) against an independent count.

The reference decodes the file with struct/json (tests/helpers.py) and counts every value against the published
edges with plain Python comparisons, so the numpy path under test (searchsorted, bincount, frexp) is never reused.
"""

from __future__ import annotations

import itertools
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers import reference_array, write_safetensors
from weightlab import WeightLabError, open_checkpoint
from weightlab.analysis import histogram as hmod
from weightlab.analysis.histogram import Binning, histogram, int_edges
from weightlab.core.limits import Limits

F64_MAX = 1.7976931348623157e308
SUB = 5e-324


def _enc(dtype: str, values: list[Any]) -> bytes:
    if dtype == "BF16":
        return b"".join(struct.pack("<I", struct.unpack("<I", struct.pack("<f", v))[0])[2:] for v in values)
    storage = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4", "I8": "|i1", "U64": "<u8",
               "U8": "|u1", "BOOL": "|u1"}[dtype]  # fmt: skip
    return np.array(values, dtype=storage).tobytes()


def _write(tmp: Path, dtype: str, values: list[Any], name: str = "t") -> Path:
    return write_safetensors(tmp / f"{name}.safetensors", {"w": (dtype, (len(values),), _enc(dtype, values))})


def _ref_values(path: Path, dtype: str) -> list[Any]:
    arr = reference_array(path, "w").reshape(-1)
    if dtype == "BF16":
        return [struct.unpack("<f", struct.pack("<I", int(b) << 16))[0] for b in arr]
    if dtype in ("F16", "F32", "F64"):
        return [float(x) for x in arr]
    return [int(x) for x in arr]


def _ref(values: list[Any], h: dict[str, Any]) -> dict[str, Any]:
    """Independent count of finite values against the published edges."""
    b = h["bins"]
    edges = [int(e) if isinstance(e, str) else e for e in b["edges"]]
    lo, hi = edges[0], (edges[-1] if b["last_bin_closed"] else edges[-1] - 1)
    counts, below, above = [0] * (len(edges) - 1), 0, 0
    for v in values:
        if isinstance(v, float) and not math.isfinite(v):
            continue
        if v < lo:
            below += 1
        elif v > hi:
            above += 1
        else:
            i = 0
            while i + 1 < len(counts) and v >= edges[i + 1]:
                i += 1
            counts[i] += 1
    return {"counts": counts, "below": below, "above": above}


def _check(path: Path, dtype: str, binning: Binning, **limits: int) -> dict[str, Any]:
    values = _ref_values(path, dtype)
    with open_checkpoint(path, limits=Limits(**limits)) as ck:
        h = histogram(ck, "w", binning)
    floats = [v for v in values if isinstance(v, float)]
    finite = [v for v in values if not (isinstance(v, float) and not math.isfinite(v))]
    c = h["counts"]
    assert c["elements"] == len(values)
    assert c["finite"] == len(finite)
    assert c["nan"] == sum(1 for v in floats if math.isnan(v))
    assert c["pos_inf"] == sum(1 for v in floats if v == math.inf)
    assert c["neg_inf"] == sum(1 for v in floats if v == -math.inf)
    assert c["zeros"] == sum(1 for v in finite if v == 0)
    assert c["negative_zeros"] == sum(1 for v in floats if v == 0 and math.copysign(1, v) < 0)
    assert c["positives"] == sum(1 for v in finite if v > 0)
    assert c["negatives"] == sum(1 for v in finite if v < 0)
    if finite:

        def num(x):
            return int(x) if isinstance(x, str) else x

        assert num(h["min"]) == min(finite)
        assert num(h["max"]) == max(finite)
    if h["bins"] is not None:
        b = h["bins"]
        assert {k: b[k] for k in ("counts", "below", "above")} == _ref(values, h)
        assert b["below"] + sum(b["counts"]) + b["above"] == c["finite"]
        edges = [int(e) if isinstance(e, str) else e for e in b["edges"]]
        assert all(x < y for x, y in itertools.pairwise(edges)) or len(set(edges)) == 1
        assert len(b["counts"]) <= binning.bins
    if h["magnitude"] is not None:
        ref: dict[int, list[int]] = {}
        for v in finite:
            if v != 0:
                e = math.frexp(abs(v))[1] - 1
                assert 2.0**e <= abs(v) < 2.0 ** (e + 1) if e < 1023 else 2.0**e <= abs(v)
                ref.setdefault(e, [0, 0])[0 if v > 0 else 1] += 1
        assert [(r["exponent"], r["positive"], r["negative"]) for r in h["magnitude"]["buckets"]] == [
            (e, p, n) for e, (p, n) in sorted(ref.items())
        ]
    return h


SPECIALS = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 3.25, math.nan, math.inf, -math.inf, 1e-3, -7.0, 7.0]


@pytest.mark.parametrize("dtype", ["F64", "F32", "F16", "BF16"])
@pytest.mark.parametrize("mode", ["min_max", "symmetric"])
@pytest.mark.parametrize("bins", [1, 3, 7, 64])
def test_float_specials_match_independent_count(tmp_path: Path, dtype: str, mode: str, bins: int) -> None:
    rng = np.random.default_rng(7)
    values = SPECIALS + [float(x) for x in rng.normal(0, 2, 500)]
    h = _check(_write(tmp_path, dtype, values), dtype, Binning(bins, mode))  # type: ignore[arg-type]
    assert h["counts"]["negative_zeros"] == 1
    assert h["value_domain"] == "float"
    assert h["coverage"] == "complete"
    if mode == "symmetric":
        assert h["bins"]["lo"] == -h["bins"]["hi"]


def test_extreme_float64_range_does_not_overflow(tmp_path: Path) -> None:
    values = [-F64_MAX, F64_MAX, SUB, -SUB, 0.0, 1.0, -1e300, 1e-300]
    for mode in ("min_max", "symmetric"):
        h = _check(_write(tmp_path, "F64", values), "F64", Binning(16, mode))  # type: ignore[arg-type]
        assert all(math.isfinite(e) for e in h["bins"]["edges"])
        assert h["bins"]["edges"][0] == -F64_MAX
        assert h["bins"]["edges"][-1] == F64_MAX
    mag = {r["exponent"] for r in h["magnitude"]["buckets"]}
    assert -1074 in mag
    assert 1023 in mag


def test_high_precision_edges_are_exact(tmp_path: Path) -> None:
    """Values one ulp apart around a bin edge land on the side the published edge says."""
    x = 1.0
    values = [x, np.nextafter(x, 2), np.nextafter(x, 0), 2.0, 0.0]
    h = _check(_write(tmp_path, "F64", [float(v) for v in values]), "F64", Binning(2))
    assert h["bins"]["edges"] == [0.0, 1.0, 2.0]
    assert h["bins"]["counts"] == [2, 3]  # [0, 1) holds 0 and 1-ulp; [1, 2] holds 1, 1+ulp and 2


def test_narrow_range_gets_fewer_bins(tmp_path: Path) -> None:
    a = 1.0
    values = [a, float(np.nextafter(a, 2)), float(np.nextafter(np.nextafter(a, 2), 2))]
    h = _check(_write(tmp_path, "F64", values), "F64", Binning(64))
    assert len(h["bins"]["counts"]) == 2
    assert h["bins"]["counts"] == [1, 2]


def test_constant_tensor_has_one_closed_bin(tmp_path: Path) -> None:
    h = _check(_write(tmp_path, "F32", [0.25] * 9), "F32", Binning(64))
    assert h["bins"]["edges"] == [0.25, 0.25]
    assert h["bins"]["counts"] == [9]
    z = _check(_write(tmp_path, "F32", [0.0, -0.0, 0.0], "z"), "F32", Binning(8, "symmetric"))
    assert z["bins"]["edges"] == [0.0, 0.0]
    assert z["bins"]["counts"] == [3]
    assert z["min"] == 0.0
    assert math.copysign(1, z["min"]) == 1


def test_no_finite_values_has_no_bins(tmp_path: Path) -> None:
    h = _check(_write(tmp_path, "F32", [math.nan, math.inf, -math.inf]), "F32", Binning())
    assert h["bins"] is None
    assert h["min"] is None
    assert h["magnitude"]["buckets"] == []
    e = write_safetensors(tmp_path / "e.safetensors", {"w": ("F32", (0, 4), b"")})
    with open_checkpoint(e) as ck:
        h = histogram(ck, "w", Binning())
    assert h["counts"]["elements"] == 0
    assert h["bins"] is None


def test_explicit_range_counts_outside_values_and_reads_once(tmp_path: Path) -> None:
    values = [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, math.nan]
    p = _write(tmp_path, "F32", values)
    h = _check(p, "F32", Binning(4, "explicit", -1.0, 1.0))
    assert h["bins"]["below"] == 1
    assert h["bins"]["above"] == 1
    assert h["bins"]["counts"] == [1, 1, 1, 2]  # [-1,-.5) [-.5,0) [0,.5) [.5,1]
    read: list[int] = []
    with open_checkpoint(p) as ck:
        histogram(ck, "w", Binning(4, "explicit", -1.0, 1.0), tick=read.append)
        assert sum(read) == len(values) * 4
        read.clear()
        histogram(ck, "w", Binning(4), tick=read.append)
        assert sum(read) == 2 * len(values) * 4  # range pass, then bins


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("I64", [-(2**63), 2**63 - 1, 0, 1, -1, 2**53 + 1, -(2**53) - 1]),
        ("U64", [0, 2**64 - 1, 2**63, 1, 2**53 + 1]),
        ("I32", list(range(-50, 51))),
        ("I8", [-128, 127, 0, 0, -1, 1, 5]),
        ("U8", [0, 255, 7, 7, 7]),
        ("BOOL", [0, 1, 1, 0, 1]),
    ],
)
@pytest.mark.parametrize("mode", ["min_max", "symmetric"])
@pytest.mark.parametrize("bins", [1, 2, 10, 300])
def test_integers_match_independent_count(tmp_path: Path, dtype: str, values: list[int], mode: str, bins: int) -> None:
    h = _check(_write(tmp_path, dtype, values), dtype, Binning(bins, mode))  # type: ignore[arg-type]
    assert h["value_domain"] == "integer"
    assert h["magnitude"] is None
    assert h["bins"]["last_bin_closed"] is False
    for v in (h["min"], h["max"], *h["bins"]["edges"]):
        assert (isinstance(v, int) and abs(v) <= 2**53 - 1) or isinstance(v, str)


def test_small_integer_range_gets_one_bin_per_value(tmp_path: Path) -> None:
    h = _check(_write(tmp_path, "BOOL", [0, 1, 1]), "BOOL", Binning(64))
    assert h["bins"]["edges"] == [0, 1, 2]
    assert h["bins"]["counts"] == [1, 2]
    assert h["counts"]["zeros"] == 1
    assert h["counts"]["positives"] == 2
    assert int_edges(-3, 3, 64) == [-3, -2, -1, 0, 1, 2, 3, 4]
    assert int_edges(0, 9, 3) == [0, 4, 8, 10]


def test_integer_explicit_range(tmp_path: Path) -> None:
    p = _write(tmp_path, "U8", [0, 3, 200, 255])
    h = _check(p, "U8", Binning(4, "explicit", -10, 10))
    assert h["bins"]["below"] == 0
    assert h["bins"]["above"] == 2
    with pytest.raises(WeightLabError, match="integer lo and hi"), open_checkpoint(p) as ck:
        histogram(ck, "w", Binning(4, "explicit", 0.5, 10))


def test_many_chunks_equal_one_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(3)
    values = [float(x) for x in rng.standard_t(2, 5000)] + SPECIALS
    p = _write(tmp_path, "F32", values)
    whole = _check(p, "F32", Binning(50))
    monkeypatch.setattr(hmod, "CHUNK_ELEMENTS", 97)
    assert _check(p, "F32", Binning(50), read_chunk_bytes=64) == whole


def test_a_raising_tick_stops_the_scan(tmp_path: Path) -> None:
    p = _write(tmp_path, "F32", [float(i) for i in range(4000)])

    class Stop(Exception):
        pass

    calls = 0

    def tick(_: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise Stop

    with open_checkpoint(p, limits=Limits(read_chunk_bytes=1024)) as ck, pytest.raises(Stop):
        histogram(ck, "w", Binning(), tick=tick)
    assert calls == 2


@pytest.mark.parametrize(
    "kw",
    [{"bins": 0}, {"bins": 1025}, {"range": "explicit"}, {"range": "explicit", "lo": 1.0, "hi": 0.0},
     {"range": "explicit", "lo": 0.0, "hi": math.inf}, {"lo": 0.0, "hi": 1.0}],
)  # fmt: skip
def test_invalid_binning(kw: dict[str, Any]) -> None:
    with pytest.raises(WeightLabError):
        Binning(**kw)
