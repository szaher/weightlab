"""Exact streaming value histogram of one tensor (ROADMAP P0.2, ADR-021).

Every element is read once per pass, in bounded chunks, and counted exactly; nothing is sampled or estimated.

* Classes. NaN, +Inf and -Inf are counted separately and never binned. Zeros (with negative zeros as a subset),
  positives and negatives are counted for every dtype. BOOL counts False as zero and True as positive.
* Range. `min_max` bins [min, max] of the finite values, `symmetric` bins [-m, m] with m = max(|min|, |max|), and
  `explicit` bins the client's [lo, hi]. The value range comes from a first pass, so `min_max` and `symmetric`
  read the tensor twice; `explicit` reads it once. Finite values outside [lo, hi] are counted in `below` / `above`
  and never dropped, so `below + sum(counts) + above` is always the finite count.
* Float edges. e_i = lo * (1 - i/n) + hi * (i/n) in float64 (no overflow even for [-max, max]), then forced
  monotone, clipped to [lo, hi] and de-duplicated, so a range narrower than n representable steps gets fewer bins.
  The published edges *define* the bins: bin i holds e_i <= x < e_(i+1), and the last bin also holds x = hi.
  A constant tensor (lo == hi) has the single closed bin [lo, lo]. Negative zero equals zero for binning.
* Integer edges (I*, U*, BOOL) are exact integers: width w = ceil((hi - lo + 1) / n), e_i = lo + i*w and a final
  edge hi + 1; every bin is half-open, so a range of at most n values gets one bin per value.
* Magnitude (float dtypes): finite nonzero values bucketed by binary exponent, 2^e <= |x| < 2^(e+1), positive and
  negative separately, from `frexp` (exact, subnormals included). It is data-independent and shows tails and
  extreme ranges that equal-width bins compress into one bar.

Decoded values are exact (formats/dtypes.py), and each bin is decided by comparison against the published float64
or integer edges, so an independent count against the same edges gives the same numbers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from weightlab.core.errors import ErrorCode, fail
from weightlab.core.limits import MAX_SAFE_INTEGER
from weightlab.formats.checkpoint import Checkpoint
from weightlab.formats.dtypes import lookup

HISTOGRAM_VERSION = 1
MAX_BINS = 1024
DEFAULT_BINS = 64
CHUNK_ELEMENTS = 1 << 19  # 512 Ki elements: 4 MiB of float64 per decoded chunk, a few temporaries of that size
MIB = 1024 * 1024
WORKING_SET = 64 * MIB  # tests/analysis/test_histogram.py measures the peak against this
_E_MIN, _E_MAX = -1073, 1024  # frexp exponents of finite nonzero float64 (5e-324 .. max); bucket = e - 1
_FLOAT_RULE = "bin i holds e[i] <= x < e[i+1]; the last bin also holds x = e[n]"
_INT_RULE = "bin i holds e[i] <= x < e[i+1] (integers; the last edge is hi + 1)"
_MAG_RULE = "bucket e holds finite nonzero x with 2^e <= |x| < 2^(e+1)"

RangeMode = Literal["min_max", "symmetric", "explicit"]
Tick = Callable[[int], None]  # bytes read after each chunk; raising stops the scan (cancellation, wall time)


@dataclass(frozen=True)
class Binning:
    bins: int = DEFAULT_BINS
    range: RangeMode = "min_max"
    lo: float | int | None = None  # explicit only; integers are kept exact for integer tensors
    hi: float | int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.bins <= MAX_BINS:
            raise fail(ErrorCode.VALIDATION_ERROR, f"bins must be within 1..{MAX_BINS}")
        if self.range == "explicit":
            if self.lo is None or self.hi is None:
                raise fail(ErrorCode.VALIDATION_ERROR, "an explicit range needs lo and hi")
            if not (_finite(self.lo) and _finite(self.hi) and self.lo <= self.hi):
                raise fail(ErrorCode.VALIDATION_ERROR, "an explicit range needs finite lo <= hi")
        elif self.lo is not None or self.hi is not None:
            raise fail(ErrorCode.VALIDATION_ERROR, "lo and hi are only for an explicit range")

    def identity(self) -> dict[str, Any]:
        return {"bins": self.bins, "range": self.range, "lo": self.lo, "hi": self.hi}


def _finite(v: float | int) -> bool:
    try:
        return math.isfinite(float(v))
    except OverflowError:
        return False


def json_num(v: int | float) -> int | float | str:
    """A float as a JSON number; an integer as a number when it is a safe JavaScript integer, else a string."""
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        i = int(v)
        return i if abs(i) <= MAX_SAFE_INTEGER else str(i)
    return float(v)


def validate_for(binning: Binning, dtype: str) -> None:
    """Raise VALIDATION_ERROR when `binning` cannot apply to `dtype` (non-integer explicit bounds of an integer
    tensor), so a job can be refused before it is queued."""
    if binning.range == "explicit" and lookup(dtype).kind != "float":
        _int_range(binning, None, None)


def scan_bytes(ck: Checkpoint, key: str, binning: Binning) -> int:
    """Bytes this histogram reads: one pass for an explicit range, two otherwise."""
    meta = ck.tensor(key)
    return meta.numel * lookup(meta.dtype).size * (1 if binning.range == "explicit" else 2)


# -- edges -------------------------------------------------------------------------------------
def float_edges(lo: float, hi: float, n: int) -> npt.NDArray[np.float64]:
    lo, hi = float(lo) + 0.0, float(hi) + 0.0  # -0.0 -> 0.0
    if lo == hi:
        return np.array([lo, hi])
    t = np.arange(n + 1, dtype=np.float64) / n
    with np.errstate(over="ignore", invalid="ignore"):
        e = lo * (1.0 - t) + hi * t
    e[0], e[-1] = lo, hi
    e = np.maximum.accumulate(np.clip(e, lo, hi))
    return np.unique(e)


def int_edges(lo: int, hi: int, n: int) -> list[int]:
    w = -(-(hi - lo + 1) // n)
    return [*range(lo, hi + 1, w), hi + 1]


def _int_range(binning: Binning, vmin: int | None, vmax: int | None) -> tuple[int, int] | None:
    if binning.range == "explicit":
        assert binning.lo is not None
        assert binning.hi is not None
        out = []
        for v in (binning.lo, binning.hi):
            if (isinstance(v, float) and not v.is_integer()) or abs(int(v)) > MAX_SAFE_INTEGER:
                raise fail(ErrorCode.VALIDATION_ERROR, "an explicit range of an integer tensor needs integer lo and "
                           f"hi within +-{MAX_SAFE_INTEGER}")  # fmt: skip
            out.append(int(v))
        return out[0], out[1]
    if vmin is None or vmax is None:
        return None
    if binning.range == "symmetric":
        m = max(abs(vmin), abs(vmax))
        return -m, m
    return vmin, vmax


def _float_range(binning: Binning, vmin: float | None, vmax: float | None) -> tuple[float, float] | None:
    if binning.range == "explicit":
        assert binning.lo is not None
        assert binning.hi is not None
        return float(binning.lo), float(binning.hi)
    if vmin is None or vmax is None:
        return None
    if binning.range == "symmetric":
        m = max(abs(vmin), abs(vmax))
        return -m, m
    return vmin, vmax


# -- accumulation ------------------------------------------------------------------------------
class _Bins:
    """Counts finite values against fixed edges: below lo, inside (by interior edge), above hi."""

    def __init__(self, edges: Any, lo: Any, hi: Any, integer: bool, dtype: np.dtype[Any] | None) -> None:
        self.edges, self.lo, self.hi, self.integer = edges, lo, hi, integer
        self.counts = np.zeros(max(len(edges) - 1, 0), dtype=np.int64)
        self.below = 0
        self.above = 0
        interior = list(edges[1:-1])
        if integer:
            assert dtype is not None
            info = np.iinfo(dtype)
            # An interior edge at or below the dtype's minimum is <= every value; one above its maximum, none.
            self.offset = sum(1 for e in interior if e <= info.min)
            self.interior = np.array([e for e in interior if info.min < e <= info.max], dtype=dtype)
            self.lo_in = lo > info.min  # otherwise nothing can be below
            self.hi_in = hi < info.max
        else:
            self.offset = 0
            self.interior = np.asarray(interior, dtype=np.float64)
            self.lo_in = self.hi_in = True

    def add(self, x: npt.NDArray[Any]) -> None:
        if not x.size:
            return
        below = x < self.lo if self.lo_in else np.zeros(x.shape, dtype=np.bool_)
        above = x > self.hi if self.hi_in else np.zeros(x.shape, dtype=np.bool_)
        nb, na = int(np.count_nonzero(below)), int(np.count_nonzero(above))
        self.below += nb
        self.above += na
        inside = x[~(below | above)] if nb or na else x
        if not inside.size or not self.counts.size:
            return
        idx = np.searchsorted(self.interior, inside, side="right") + self.offset
        self.counts += np.bincount(idx, minlength=self.counts.size)

    def view(self, mode: RangeMode, requested: int) -> dict[str, Any]:
        return {
            "range": mode,
            "requested_bins": requested,
            "lo": json_num(self.lo),
            "hi": json_num(self.hi),
            "edges": [json_num(e) for e in self.edges],
            "counts": [int(c) for c in self.counts],
            "below": self.below,
            "above": self.above,
            "last_bin_closed": not self.integer,
            "rule": _INT_RULE if self.integer else _FLOAT_RULE,
        }


@dataclass
class _Stats:
    elements: int = 0
    nan: int = 0
    pos_inf: int = 0
    neg_inf: int = 0
    zeros: int = 0
    negative_zeros: int = 0
    positives: int = 0
    negatives: int = 0
    vmin: Any = None
    vmax: Any = None
    mag_pos: npt.NDArray[np.int64] | None = None
    mag_neg: npt.NDArray[np.int64] | None = None

    @property
    def finite(self) -> int:
        return self.elements - self.nan - self.pos_inf - self.neg_inf

    def add_float(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Count one chunk; return its finite values."""
        self.elements += x.size
        fin = np.isfinite(x)
        nfin = int(np.count_nonzero(fin))
        if nfin != x.size:
            nan = np.isnan(x)
            self.nan += int(np.count_nonzero(nan))
            inf = ~(fin | nan)
            self.pos_inf += int(np.count_nonzero(inf & (x > 0)))
            self.neg_inf += int(np.count_nonzero(inf & (x < 0)))
            x = x[fin]
        if not x.size:
            return x
        zero = x == 0
        nz = int(np.count_nonzero(zero))
        self.zeros += nz
        if nz:
            self.negative_zeros += int(np.count_nonzero(np.signbit(x[zero])))
        pos = x > 0
        npos = int(np.count_nonzero(pos))
        self.positives += npos
        self.negatives += x.size - nz - npos
        lo, hi = float(x.min()) + 0.0, float(x.max()) + 0.0
        self.vmin = lo if self.vmin is None else min(self.vmin, lo)
        self.vmax = hi if self.vmax is None else max(self.vmax, hi)
        if self.mag_pos is None or self.mag_neg is None:
            self.mag_pos = np.zeros(_E_MAX - _E_MIN + 1, dtype=np.int64)
            self.mag_neg = np.zeros(_E_MAX - _E_MIN + 1, dtype=np.int64)
        if nz != x.size:
            nonzero = x[~zero] if nz else x
            _, e = np.frexp(np.abs(nonzero))
            idx = e.astype(np.int64) - _E_MIN
            neg = nonzero < 0
            self.mag_pos += np.bincount(idx[~neg], minlength=self.mag_pos.size)
            self.mag_neg += np.bincount(idx[neg], minlength=self.mag_neg.size)
        return x

    def add_int(self, x: npt.NDArray[Any]) -> None:
        self.elements += x.size
        if not x.size:
            return
        nz = int(np.count_nonzero(x == 0))
        npos = int(np.count_nonzero(x > 0))
        self.zeros += nz
        self.positives += npos
        self.negatives += x.size - nz - npos
        lo, hi = int(x.min()), int(x.max())
        self.vmin = lo if self.vmin is None else min(self.vmin, lo)
        self.vmax = hi if self.vmax is None else max(self.vmax, hi)

    def magnitude(self) -> list[dict[str, int]]:
        if self.mag_pos is None or self.mag_neg is None:
            return []
        rows = np.flatnonzero(self.mag_pos + self.mag_neg)
        return [
            {"exponent": int(i) + _E_MIN - 1, "positive": int(self.mag_pos[i]), "negative": int(self.mag_neg[i])}
            for i in rows
        ]


# -- the scan ----------------------------------------------------------------------------------
def _chunks(ck: Checkpoint, key: str, tick: Tick | None) -> Iterator[npt.NDArray[Any]]:
    meta = ck.tensor(key)
    esize = lookup(meta.dtype).size
    chunk = min(ck.limits.read_chunk_bytes, CHUNK_ELEMENTS * esize)
    chunk -= chunk % esize
    for c in ck.iter_flat_chunks(key, chunk):
        if c.dtype == np.bool_:
            c = c.view(np.uint8)
        yield c
        if tick is not None:
            tick(c.size * esize)


def histogram(ck: Checkpoint, key: str, binning: Binning, tick: Tick | None = None) -> dict[str, Any]:
    """The exact histogram of tensor `key` (see the module docstring). `tick(n)` runs after every chunk."""
    meta = ck.tensor(key)
    dt = lookup(meta.dtype)
    integer = dt.kind != "float"
    st = _Stats()
    bins: _Bins | None = None
    np_dtype: np.dtype[Any] | None = None
    if integer:
        np_dtype = np.dtype(np.uint8 if dt.kind == "bool" else np.uint64 if dt.kind == "uint" else np.int64)

    def make_bins(rng: tuple[Any, Any] | None) -> _Bins | None:
        if rng is None:
            return None
        lo, hi = rng
        if integer:
            return _Bins(int_edges(lo, hi, binning.bins), lo, hi, True, np_dtype)
        e = float_edges(lo, hi, binning.bins)
        return _Bins(e, float(e[0]), float(e[-1]), False, None)

    explicit = binning.range == "explicit"
    if explicit:
        bins = make_bins(_int_range(binning, None, None) if integer else _float_range(binning, None, None))
    for c in _chunks(ck, key, tick):
        if integer:
            st.add_int(c)
            if bins is not None:
                bins.add(c)
        else:
            f = st.add_float(c)
            if bins is not None:
                bins.add(f)
    if not explicit:
        rng = _int_range(binning, st.vmin, st.vmax) if integer else _float_range(binning, st.vmin, st.vmax)
        bins = make_bins(rng)
        if bins is not None:
            for c in _chunks(ck, key, tick):
                if integer:
                    bins.add(c)
                else:
                    fin = c[np.isfinite(c)]
                    bins.add(fin)
    ck.check()
    finite = st.finite
    if bins is not None:
        assert bins.below + int(bins.counts.sum()) + bins.above == finite, "histogram lost values"
    counts = {
        "elements": st.elements,
        "finite": finite,
        "zeros": st.zeros,
        "negative_zeros": st.negative_zeros,
        "positives": st.positives,
        "negatives": st.negatives,
        "nan": st.nan,
        "pos_inf": st.pos_inf,
        "neg_inf": st.neg_inf,
    }
    return {
        "histogram_version": HISTOGRAM_VERSION,
        "tensor_key": key,
        "dtype": meta.dtype,
        "shape": list(meta.shape),
        "coverage": "complete",
        "value_domain": "integer" if integer else "float",
        "counts": counts,
        "min": None if st.vmin is None else json_num(st.vmin),
        "max": None if st.vmax is None else json_num(st.vmax),
        "bins": None if bins is None else bins.view(binning.range, binning.bins),
        "magnitude": None if integer else {"rule": _MAG_RULE, "buckets": st.magnitude()},
        "method": (
            "exact streaming count of every element (no sampling); "
            + ("one pass, explicit range" if explicit else "two passes: value range, then bins")
        ),
    }
