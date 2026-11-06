"""Overflow-safe streaming accumulators for change metrics (doc 05 sections 3-4).

Sums of squares and dot products are kept as `mantissa * 2**exponent` with power-of-two chunk
scaling (LAPACK dnrm2 idea, but exact scaling), so a finite input never overflows an L2 norm or a
cosine; only a final value outside float64 range is reported as overflowed (None).
Every element is visited once; results are float64-rounded (`numeric_precision = rounded`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass
class ExpSum:
    """Non-negative-or-signed running sum stored as m * 2**e with 0.5 <= |m| < 1 (or m == 0)."""

    m: float = 0.0
    e: int = 0

    def add(self, mantissa: float, exponent: int) -> None:
        if mantissa == 0.0:
            return
        if self.m == 0.0:
            self.m, self.e = mantissa, exponent
        elif exponent > self.e:
            self.m = math.ldexp(self.m, self.e - exponent) + mantissa
            self.e = exponent
        else:
            self.m += math.ldexp(mantissa, exponent - self.e)
        fm, fe = math.frexp(self.m)
        self.m, self.e = fm, self.e + fe

    def merge(self, other: ExpSum) -> None:
        self.add(other.m, other.e)

    def add_squares(self, x: F64) -> None:
        if x.size == 0:
            return
        peak = float(np.max(np.abs(x)))
        if peak == 0.0:
            return
        k = math.frexp(peak)[1]
        y = np.ldexp(x, -k)
        self.add(float(np.dot(y, y)), 2 * k)

    def add_products(self, a: F64, b: F64) -> None:
        if a.size == 0:
            return
        pa, pb = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
        if pa == 0.0 or pb == 0.0:
            return
        ka, kb = math.frexp(pa)[1], math.frexp(pb)[1]
        self.add(float(np.dot(np.ldexp(a, -ka), np.ldexp(b, -kb))), ka + kb)

    def value(self) -> float | None:
        return _ldexp(self.m, self.e)

    def sqrt(self) -> float | None:
        """sqrt of a non-negative sum, overflow-safe."""
        if self.m == 0.0:
            return 0.0
        m, e = (self.m, self.e) if self.e % 2 == 0 else (self.m * 2.0, self.e - 1)
        return _ldexp(math.sqrt(m), e // 2)


def _scaled(x: F64, peak: float) -> tuple[F64, int] | None:
    """x * 2**-k with 2**(k-1) <= peak < 2**k (exact power-of-two scaling), or None if all zero."""
    if peak == 0.0 or x.size == 0:
        return None
    k = math.frexp(peak)[1]
    return np.ldexp(x, -k), k


def _ldexp(m: float, e: int) -> float | None:
    try:
        v = math.ldexp(m, e)
    except OverflowError:
        return None
    return v if math.isfinite(v) else None


def sqrt_ratio(num: ExpSum, den: ExpSum) -> float | None:
    """sqrt(num / den) without forming either value; None when den is zero or the result overflows."""
    if den.m == 0.0:
        return None
    if num.m == 0.0:
        return 0.0
    m, e = num.m / den.m, num.e - den.e
    if e % 2:
        m, e = m * 2.0, e - 1
    return _ldexp(math.sqrt(m), e // 2)


def cosine(dot: ExpSum, ssq_a: ExpSum, ssq_b: ExpSum) -> float | None:
    """dot / (||a|| ||b||); None for a zero norm."""
    if ssq_a.m == 0.0 or ssq_b.m == 0.0:
        return None
    m = dot.m / math.sqrt(ssq_a.m * ssq_b.m)
    e2 = 2 * dot.e - ssq_a.e - ssq_b.e  # exponent of the result, doubled
    if e2 % 2:
        m, e2 = m * math.sqrt(2.0), e2 - 1
    v = _ldexp(m, e2 // 2)
    return None if v is None else max(-1.0, min(1.0, v))


@dataclass
class PairStats:
    """Paired statistics of A (reference) and B, with delta = B - A in float64.

    Pairs where either input is non-finite are excluded from every finite-pair metric and counted
    in `invalid_pairs`. A finite pair whose delta overflows float64 is counted in `delta_overflow`
    and makes every delta metric `overflowed` (withheld), never a silent Inf.
    """

    elements: int = 0
    finite_pairs: int = 0
    invalid_pairs: int = 0
    nan_a: int = 0
    pos_inf_a: int = 0
    neg_inf_a: int = 0
    nan_b: int = 0
    pos_inf_b: int = 0
    neg_inf_b: int = 0
    delta_overflow: int = 0
    changed: int = 0
    max_abs: float = 0.0
    ssq_delta: ExpSum = field(default_factory=ExpSum)
    ssq_a: ExpSum = field(default_factory=ExpSum)
    ssq_b: ExpSum = field(default_factory=ExpSum)
    dot_ab: ExpSum = field(default_factory=ExpSum)

    def _count_nonfinite(self, arr: F64, finite: npt.NDArray[np.bool_], prefix: str) -> None:
        bad = arr[~finite]
        nan = int(np.isnan(bad).sum())
        pos = int((bad > 0).sum())
        setattr(self, f"nan_{prefix}", getattr(self, f"nan_{prefix}") + nan)
        setattr(self, f"pos_inf_{prefix}", getattr(self, f"pos_inf_{prefix}") + pos)
        setattr(self, f"neg_inf_{prefix}", getattr(self, f"neg_inf_{prefix}") + bad.size - nan - pos)

    def update(self, a: F64, b: F64, delta: F64 | None = None) -> None:
        """Feed one aligned chunk. `delta`, when given (LoRA), is the exact-as-computed B - A.

        Fast path: an all-finite chunk (the usual case) skips every mask; each input is scaled by
        a power of two once and reused for its sum of squares and for the dot product.
        """
        n = int(a.size)
        self.elements += n
        fa, fb = np.isfinite(a), np.isfinite(b)
        all_a, all_b = bool(fa.all()), bool(fb.all())
        if not all_a:
            self._count_nonfinite(a, fa, "a")
        if not all_b:
            self._count_nonfinite(b, fb, "b")
        ok: npt.NDArray[np.bool_] | None = None
        if not (all_a and all_b):
            ok = fa & fb
        if delta is not None:
            fd = np.isfinite(delta)
            if not fd.all():
                ok = fd if ok is None else ok & fd
        if ok is not None:
            a, b = a[ok], b[ok]
            if delta is not None:
                delta = delta[ok]
        n_ok = int(a.size)
        self.finite_pairs += n_ok
        self.invalid_pairs += n - n_ok
        if n_ok == 0:
            return
        peak_a, peak_b = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
        if delta is None:
            if max(peak_a, peak_b) >= 2.0**1022:  # only then can B - A overflow float64
                with np.errstate(over="ignore", invalid="ignore"):
                    d = b - a
                over = ~np.isfinite(d)
                n_over = int(over.sum())
                if n_over:
                    self.delta_overflow += n_over
                    d = d[~over]
            else:
                d = b - a
        else:
            d = delta
        self.changed += int(np.count_nonzero(d))
        peak_d = float(np.max(np.abs(d))) if d.size else 0.0
        self.max_abs = max(self.max_abs, peak_d)
        ya = _scaled(a, peak_a)
        yb = _scaled(b, peak_b)
        if ya is not None:
            self.ssq_a.add(float(np.dot(ya[0], ya[0])), 2 * ya[1])
        if yb is not None:
            self.ssq_b.add(float(np.dot(yb[0], yb[0])), 2 * yb[1])
        if ya is not None and yb is not None:
            self.dot_ab.add(float(np.dot(ya[0], yb[0])), ya[1] + yb[1])
        yd = _scaled(d, peak_d)
        if yd is not None:
            self.ssq_delta.add(float(np.dot(yd[0], yd[0])), 2 * yd[1])

    # -- derived metrics (None = not defined or withheld) ---------------------------------
    @property
    def delta_withheld(self) -> bool:
        return self.delta_overflow > 0

    def delta_l2(self) -> float | None:
        return None if self.delta_withheld else self.ssq_delta.sqrt()

    def delta_rms(self) -> float | None:
        if self.delta_withheld or self.finite_pairs == 0:
            return None
        return sqrt_ratio(self.ssq_delta, ExpSum(*math.frexp(float(self.finite_pairs))))

    def relative_l2(self, eps_norm: float) -> float | None:
        if self.delta_withheld or self.finite_pairs == 0:
            return None
        norm_a = self.ssq_a.sqrt()
        if norm_a is not None and norm_a < eps_norm:
            return _ratio_or_none(self.delta_l2(), eps_norm)
        return sqrt_ratio(self.ssq_delta, self.ssq_a)

    def near_zero_reference(self, eps_norm: float) -> bool:
        norm_a = self.ssq_a.sqrt()
        return norm_a is not None and norm_a < eps_norm

    def cosine(self) -> float | None:
        return cosine(self.dot_ab, self.ssq_a, self.ssq_b)

    def max_abs_delta(self) -> float | None:
        if self.delta_withheld or self.finite_pairs == 0:
            return None
        return self.max_abs


def _ratio_or_none(num: float | None, den: float) -> float | None:
    if num is None:
        return None
    v = num / den
    return v if math.isfinite(v) else None


@dataclass
class Rollup:
    """Aggregate of several tensors' paired stats (layer, role or whole model)."""

    tensors: int = 0
    elements: int = 0
    finite_pairs: int = 0
    invalid_pairs: int = 0
    withheld: int = 0  # member tensors whose delta overflowed
    max_abs: float = 0.0
    ssq_delta: ExpSum = field(default_factory=ExpSum)
    ssq_a: ExpSum = field(default_factory=ExpSum)

    def add(self, s: PairStats, ssq_delta: ExpSum | None = None) -> None:
        self.tensors += 1
        self.elements += s.elements
        self.finite_pairs += s.finite_pairs
        self.invalid_pairs += s.invalid_pairs
        if s.delta_withheld:
            self.withheld += 1
        else:
            self.ssq_delta.merge(ssq_delta if ssq_delta is not None else s.ssq_delta)
            self.max_abs = max(self.max_abs, s.max_abs)
        self.ssq_a.merge(s.ssq_a)

    def delta_l2(self) -> float | None:
        return None if self.withheld else self.ssq_delta.sqrt()

    def delta_rms(self) -> float | None:
        if self.withheld or self.finite_pairs == 0:
            return None
        return sqrt_ratio(self.ssq_delta, ExpSum(*math.frexp(float(self.finite_pairs))))

    def relative_l2(self) -> float | None:
        return None if self.withheld else sqrt_ratio(self.ssq_delta, self.ssq_a)
