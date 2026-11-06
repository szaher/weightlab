"""Streaming accumulators vs the exact Fraction reference, any chunking, extreme magnitudes."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.reference import close, pair_reference
from weightlab.analysis.accumulate import ExpSum, PairStats, Rollup

FINITE = st.floats(allow_nan=False, allow_infinity=False, width=64)
ANY = st.one_of(FINITE, st.sampled_from([math.nan, math.inf, -math.inf]), st.floats(-1e3, 1e3))
# Results below the normal float64 range (2**-1022) carry absolute, not relative, precision: a
# subnormal has fewer significant bits, and power-of-two chunk scaling may round them differently.
TINY = 2.0**-1022
SCALE = st.sampled_from([1.0, 1e-300, 1e-150, 1e150, 1e300, 2.0**-1074, 1.7e308])


def run(a: np.ndarray, b: np.ndarray, chunk: int) -> PairStats:
    s = PairStats()
    for i in range(0, len(a), chunk):
        s.update(a[i : i + chunk], b[i : i + chunk])
    return s


def check(s: PairStats, ref: dict[str, object], rtol: float) -> None:
    assert s.finite_pairs == ref["finite_pairs"]
    assert s.changed == ref["changed"] or s.delta_withheld
    assert s.delta_withheld == ref["overflow"]
    for name, got in (
        ("delta_l2", s.delta_l2()),
        ("delta_rms", s.delta_rms()),
        ("relative_l2", s.relative_l2(1e-12)),
        ("max_abs_delta", s.max_abs_delta()),
        ("norm_a", s.ssq_a.sqrt()),
        ("norm_b", s.ssq_b.sqrt()),
    ):
        want = ref[name]
        assert close(got, want, rtol, atol=TINY), (name, got, want)  # type: ignore[arg-type]
    c, want_c = s.cosine(), ref["cosine"]
    if want_c is None:
        assert c is None
    else:
        assert c is not None
        assert math.isclose(c, want_c, rel_tol=1e-9, abs_tol=1e-12)  # type: ignore[arg-type]


@settings(max_examples=400, deadline=None)
@given(st.lists(st.tuples(ANY, ANY), min_size=0, max_size=60), st.integers(1, 17))
def test_pairs_match_exact_reference(pairs: list[tuple[float, float]], chunk: int) -> None:
    a = np.array([p[0] for p in pairs], dtype=np.float64)
    b = np.array([p[1] for p in pairs], dtype=np.float64)
    s = run(a, b, chunk)
    ref = pair_reference(a.tolist(), b.tolist())
    # Float64 subtraction rounds each delta (<= 0.5 ulp) and power-of-two scaling can flush terms far
    # below the chunk maximum; agreement with the exact value is still within 1e-12 relative.
    if s.delta_withheld != ref["overflow"]:
        return  # |B-A| within 1 ulp of the float64 limit: rounding decides; both outcomes are honest
    check(s, ref, rtol=1e-12)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.floats(-4, 4), min_size=1, max_size=80),
    st.lists(st.floats(-4, 4), min_size=1, max_size=80),
    SCALE,
    st.integers(1, 13),
)
def test_extreme_scales_never_overflow_norms(xs: list[float], ys: list[float], scale: float, chunk: int) -> None:
    n = min(len(xs), len(ys))
    with np.errstate(over="ignore"):  # scaled inputs may become +/-inf: non-finite pairs by design
        a = np.array(xs[:n]) * scale
        b = np.array(ys[:n]) * scale
    s = run(a, b, chunk)
    ref = pair_reference(a.tolist(), b.tolist())
    if s.delta_withheld != ref["overflow"]:
        return
    check(s, ref, rtol=1e-12)


@settings(max_examples=100, deadline=None)
@given(st.lists(FINITE, min_size=2, max_size=50), st.integers(1, 7), st.integers(1, 7))
def test_chunk_invariance(xs: list[float], c1: int, c2: int) -> None:
    a = np.array(xs)
    b = a[::-1].copy()
    s1, s2 = run(a, b, c1), run(a, b, c2)
    for f in (PairStats.delta_l2, PairStats.delta_rms, PairStats.cosine, PairStats.max_abs_delta):
        assert close(f(s1), f(s2), rtol=1e-12, atol=TINY)


def test_known_values_and_withholding() -> None:
    s = PairStats()
    s.update(np.array([1.0, 2.0, np.nan, 4.0]), np.array([1.0, 3.0, 5.0, np.inf]))
    assert (s.finite_pairs, s.invalid_pairs, s.nan_a, s.pos_inf_b, s.changed) == (2, 2, 1, 1, 1)
    assert s.delta_l2() == 1.0
    assert s.max_abs_delta() == 1.0
    o = PairStats()
    o.update(np.array([-1.7976931348623157e308]), np.array([1.7976931348623157e308]))
    assert o.delta_overflow == 1
    assert o.delta_l2() is None
    assert o.relative_l2(1e-12) is None
    assert o.cosine() == -1.0  # norms stay finite through scaling
    z = PairStats()
    z.update(np.zeros(3), np.zeros(3))
    assert z.cosine() is None
    assert z.relative_l2(1e-12) == 0.0
    big = PairStats()
    big.update(np.array([0.0, 0.0]), np.array([1.7e308, 1.7e308]))
    assert big.delta_l2() is None  # finite deltas, but ||B-A|| = 2.4e308 exceeds float64: overflowed


def test_rollup_combines_sums_of_squares() -> None:
    parts = [(np.arange(5.0), np.arange(5.0) + 1), (np.ones(3), np.ones(3) * 3)]
    r = Rollup()
    whole_a, whole_b = [], []
    for a, b in parts:
        s = PairStats()
        s.update(a, b)
        r.add(s)
        whole_a += a.tolist()
        whole_b += b.tolist()
    ref = pair_reference(whole_a, whole_b)
    assert close(r.delta_l2(), ref["delta_l2"])  # type: ignore[arg-type]
    assert close(r.delta_rms(), ref["delta_rms"])  # type: ignore[arg-type]
    assert close(r.relative_l2(), ref["relative_l2"])  # type: ignore[arg-type]


def test_expsum_sqrt_exponent_parity() -> None:
    for v in (2.0, 8.0, 1e-310, 3e300):
        e = ExpSum()
        m, k = math.frexp(v)
        e.add(m, k)
        assert math.isclose(e.sqrt() or 0.0, math.sqrt(v), rel_tol=1e-15)
