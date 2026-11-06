"""Independent exact reference for change metrics: Python Fractions (every float64 is an exact
rational) and Decimal square roots at 60 digits. Shares no code with weightlab.analysis."""

from __future__ import annotations

import math
from collections.abc import Iterable
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 60
F64_MAX = Fraction(1.7976931348623157e308)


def _sqrt(q: Fraction) -> float | None:
    if q == 0:
        return 0.0
    d = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
    v = float(d)
    return v if math.isfinite(v) else None


def pair_reference(a: Iterable[float], b: Iterable[float], eps_norm: float = 1e-12) -> dict[str, object]:
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if math.isfinite(x) and math.isfinite(y)]
    fa = [Fraction(x) for x, _ in pairs]
    fb = [Fraction(y) for _, y in pairs]
    d = [y - x for x, y in zip(fa, fb, strict=True)]
    overflow = any(abs(v) > F64_MAX for v in d)
    ssq_d = sum((v * v for v in d), Fraction(0))
    ssq_a = sum((v * v for v in fa), Fraction(0))
    ssq_b = sum((v * v for v in fb), Fraction(0))
    dot = sum((x * y for x, y in zip(fa, fb, strict=True)), Fraction(0))
    n = len(pairs)
    norm_a = _sqrt(ssq_a)
    out: dict[str, object] = {
        "finite_pairs": n,
        "overflow": overflow,
        "changed": sum(1 for v in d if v != 0),
        "norm_a": norm_a,
        "norm_b": _sqrt(ssq_b),
    }
    if overflow:
        out.update(delta_l2=None, delta_rms=None, relative_l2=None, max_abs_delta=None)
    else:
        out["delta_l2"] = _sqrt(ssq_d)
        out["delta_rms"] = _sqrt(ssq_d / n) if n else None
        if n == 0:
            out["relative_l2"] = None
        elif norm_a is not None and norm_a < eps_norm:
            v = _sqrt(ssq_d)
            out["relative_l2"] = None if v is None or not math.isfinite(v / eps_norm) else v / eps_norm
        else:
            out["relative_l2"] = _sqrt(ssq_d / ssq_a)
        out["max_abs_delta"] = float(max((abs(v) for v in d), default=Fraction(0))) if n else None
    if ssq_a == 0 or ssq_b == 0:
        out["cosine"] = None
    else:
        c = (Decimal(dot.numerator) / Decimal(dot.denominator)) / (
            (Decimal(ssq_a.numerator) / Decimal(ssq_a.denominator)).sqrt()
            * (Decimal(ssq_b.numerator) / Decimal(ssq_b.denominator)).sqrt()
        )
        out["cosine"] = max(-1.0, min(1.0, float(c)))
    return out


def close(got: float | None, want: float | None, rtol: float = 1e-12, atol: float = 0.0) -> bool:
    if got is None or want is None:
        return got is None and want is None
    return math.isclose(got, want, rel_tol=rtol, abs_tol=atol)
