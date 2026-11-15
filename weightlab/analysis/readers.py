"""Value readers for heatmap tiles, pyramids and row-sampled previews (doc 03 section 5, doc 09 sections 3-4).

A reader yields float64 values of one source for a view rectangle or a C-order element run, with an overflow mask,
whether any input was rounded on the way (64-bit integers beyond 2^53) and the source bytes read:

* `TensorReader`: one stored tensor; the values are the source values (never an overflow).
* `DeltaReader`: one matched comparison entry; the same cells of A and B are read and combined elementwise in
  float64 by `delta_values` (the paired pass of doc 03 section 5; never derived from A and B aggregates). A value
  whose inputs are finite but whose computation overflows float64 is reported in the overflow mask; aggregation
  counts it as a finite element whose value cannot be represented, so every aggregate that depends on it is
  NOT_REPRESENTABLE, never a source Inf. A pair with a non-finite input gives the IEEE result of the computation,
  which is always non-finite (counted by its kind) and is the tile's `invalid_pair_count`.

`delta_values` is also the arithmetic of POST /tensors/roi for `delta` sources, so tiles and exact values agree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from weightlab.core.errors import ErrorCode, fail
from weightlab.core.limits import MAX_SAFE_INTEGER, Limits
from weightlab.formats.base import Consistency
from weightlab.formats.checkpoint import Checkpoint
from weightlab.formats.dtypes import DType, lookup
from weightlab.formats.selection import View

F64 = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]

Transform = Literal["b_minus_a", "abs_b_minus_a", "relative_abs"]
TRANSFORMS: tuple[Transform, ...] = ("b_minus_a", "abs_b_minus_a", "relative_abs")


def as_float64(vals: npt.NDArray[Any], dt: DType) -> tuple[F64, bool]:
    """Values for aggregation and whether any was rounded on the way (64-bit integers beyond 2^53)."""
    if dt.kind == "float":
        return vals.astype(np.float64, copy=False), False
    if dt.kind == "bool":
        return vals.astype(np.float64), False
    rounded = False
    if dt.size == 8 and vals.size:
        big = vals > MAX_SAFE_INTEGER if dt.kind == "uint" else (vals > MAX_SAFE_INTEGER) | (vals < -MAX_SAFE_INTEGER)
        rounded = bool(big.any())
    return vals.astype(np.float64), rounded


def delta_values(a: F64, b: F64, transform: str, epsilon: float) -> tuple[F64, Mask]:
    """Elementwise transform of B - A in float64 and the mask of values that overflowed although both inputs were
    finite. `abs_b_minus_a` = |B - A|; `relative_abs` = |B - A| / max(|A|, epsilon) (the denominator rule of
    relative_l2, doc 05)."""
    if transform not in TRANSFORMS:
        raise fail(ErrorCode.VALIDATION_ERROR, f"transform must be one of {', '.join(TRANSFORMS)}")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        d = b - a
        if transform == "abs_b_minus_a":
            d = np.abs(d)
        elif transform == "relative_abs":
            d = np.abs(d) / np.maximum(np.abs(a), epsilon)
    return d, np.isfinite(a) & np.isfinite(b) & ~np.isfinite(d)


@dataclass(frozen=True)
class Values:
    v: F64
    overflow: Mask | None  # None: no value overflowed (always for a tensor source)
    rounded: bool
    nbytes: int

    def transposed(self) -> Values:
        return Values(self.v.T, None if self.overflow is None else self.overflow.T, self.rounded, self.nbytes)


class Reader(Protocol):
    key: str
    shape: tuple[int, ...]
    source_dtype: str  # the dtype a tile reports (A's for a delta, as /tensors/roi does)
    esize: int  # source bytes read per view cell (summed over the inputs)
    max_esize: int  # the widest input element (sizes single positioned reads)
    derived: bool  # float64 arithmetic of the inputs: never `lossless`
    limits: Limits

    def roi_read_bytes(self, r0: int, r1: int, c0: int, c1: int, view: View | None, *, max_cells: int) -> int: ...

    def roi(
        self, r0: int, r1: int, c0: int, c1: int, view: View | None, *, max_cells: int,
        consistency: Consistency = Consistency.BEST_EFFORT,
    ) -> Values: ...  # fmt: skip

    def elements(self, start: int, count: int) -> Values: ...

    def check(self) -> None: ...

    def input_bytes(self) -> int: ...

    def input_digests(self, tick: Callable[[int], None] | None = None) -> Any: ...


class TensorReader:
    derived = False

    def __init__(self, ck: Checkpoint, key: str) -> None:
        meta = ck.tensor(key)
        self.ck, self.key, self.shape = ck, key, meta.shape
        self.dt = lookup(meta.dtype)
        self.source_dtype = self.dt.name
        self.esize = self.max_esize = self.dt.size
        self.limits = ck.limits

    def roi_read_bytes(self, r0: int, r1: int, c0: int, c1: int, view: View | None, *, max_cells: int) -> int:
        return self.ck.roi_read_bytes(self.key, r0, r1, c0, c1, view, max_cells=max_cells)

    def roi(
        self, r0: int, r1: int, c0: int, c1: int, view: View | None, *, max_cells: int,
        consistency: Consistency = Consistency.BEST_EFFORT,
    ) -> Values:  # fmt: skip
        raw = self.ck.read_roi_raw(self.key, r0, r1, c0, c1, view, max_cells=max_cells, consistency=consistency)
        vals, rounded = as_float64(raw.values(), self.dt)
        return Values(vals, None, rounded, raw.stripe_bytes)

    def elements(self, start: int, count: int) -> Values:
        vals, rounded = as_float64(self.ck.read_elements(self.key, start, count), self.dt)
        return Values(vals, None, rounded, count * self.esize)

    def check(self) -> None:
        self.ck.check()

    def input_bytes(self) -> int:
        return self.ck.input_bytes([self.key])

    def input_digests(self, tick: Callable[[int], None] | None = None) -> Any:
        return self.ck.input_digests([self.key], tick=tick)


def _tighter(a: Limits, b: Limits) -> Limits:
    if a == b:
        return a
    return replace(a, **{f.name: min(getattr(a, f.name), getattr(b, f.name)) for f in fields(Limits)})


class DeltaReader:
    """Paired reads of one tensor key in checkpoints A and B (same shape: a matched, analyzable entry)."""

    derived = True

    def __init__(self, a: Checkpoint, b: Checkpoint, key: str, transform: str, epsilon: float) -> None:
        self.a, self.b = TensorReader(a, key), TensorReader(b, key)
        if self.a.shape != self.b.shape:
            raise fail(ErrorCode.INCOMPATIBLE_TENSOR, f"A and B shapes differ ({self.a.shape} vs {self.b.shape})")
        if transform not in TRANSFORMS:
            raise fail(ErrorCode.VALIDATION_ERROR, f"transform must be one of {', '.join(TRANSFORMS)}")
        self.key, self.shape = key, self.a.shape
        self.transform, self.epsilon = transform, float(epsilon)
        self.source_dtype = self.a.source_dtype
        self.esize = self.a.esize + self.b.esize
        self.max_esize = max(self.a.esize, self.b.esize)
        self.limits = _tighter(self.a.limits, self.b.limits)

    def _pair(self, va: Values, vb: Values) -> Values:
        d, overflow = delta_values(va.v, vb.v, self.transform, self.epsilon)
        return Values(d, overflow if overflow.any() else None, va.rounded or vb.rounded, va.nbytes + vb.nbytes)

    def roi_read_bytes(self, r0: int, r1: int, c0: int, c1: int, view: View | None, *, max_cells: int) -> int:
        return self.a.roi_read_bytes(r0, r1, c0, c1, view, max_cells=max_cells) + self.b.roi_read_bytes(
            r0, r1, c0, c1, view, max_cells=max_cells
        )

    def roi(
        self, r0: int, r1: int, c0: int, c1: int, view: View | None, *, max_cells: int,
        consistency: Consistency = Consistency.BEST_EFFORT,
    ) -> Values:  # fmt: skip
        va = self.a.roi(r0, r1, c0, c1, view, max_cells=max_cells, consistency=consistency)
        vb = self.b.roi(r0, r1, c0, c1, view, max_cells=max_cells, consistency=consistency)
        return self._pair(va, vb)

    def elements(self, start: int, count: int) -> Values:
        return self._pair(self.a.elements(start, count), self.b.elements(start, count))

    def check(self) -> None:
        self.a.check()
        self.b.check()

    def input_bytes(self) -> int:
        return self.a.input_bytes() + self.b.input_bytes()

    def input_digests(self, tick: Callable[[int], None] | None = None) -> Any:
        return {"a": self.a.input_digests(tick), "b": self.b.input_digests(tick)}
