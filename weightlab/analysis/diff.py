"""Checkpoint-pair change analysis (FR-023): one bounded streaming pass per matched tensor pair.

Memory: at most `chunk_elements` elements of each input are live (plus float64 temporaries), so
peak RSS is independent of model size. Orientation is always B - A, A = base (reference).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from weightlab.analysis.accumulate import PairStats
from weightlab.analysis.compat import Entry
from weightlab.core.errors import ErrorCode, fail
from weightlab.formats.base import Consistency
from weightlab.formats.checkpoint import Checkpoint

NUMERICS_VERSION = 2  # 2: exact 64-bit integer deltas, epsilon policy in rollups, LoRA overflow tracking
DEFAULT_CHUNK_ELEMENTS = 1 << 18  # 256 Ki elements per side: ~16 MiB of float64 temporaries (doc 11: 16-64 MiB)
DEFAULT_EPS_NORM = 1e-12

Progress = Callable[[int, int, str], None]  # (bytes done, bytes total, current key)


class Cancelled(Exception):
    """The run was cancelled; no partial result may be reported as complete."""


@dataclass
class RunControl:
    cancel: threading.Event = field(default_factory=threading.Event)
    progress: Progress | None = None
    done_bytes: int = 0
    total_bytes: int = 0

    def tick(self, nbytes: int, key: str) -> None:
        if self.cancel.is_set():
            raise Cancelled
        self.done_bytes += nbytes
        if self.progress is not None:
            self.progress(self.done_bytes, self.total_bytes, key)


WIDE_INTS = ("I64", "U64")  # the only dtypes whose values float64 cannot hold exactly


def exact_int_delta(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """B - A for integer chunks, exact before a single rounding to float64.

    Each value is split as x = hi * 2**32 + lo (lo in [0, 2**32)); hi and lo differences fit in
    int64 for every I64/U64 input, and both convert to float64 exactly, so `dh * 2**32 + dl` is the
    correctly rounded exact difference. A nonzero integer difference is never rounded to zero
    (2**53 vs 2**53 + 1 gives 1.0, not 0.0)."""

    def split(x: npt.NDArray[Any]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        if x.dtype == np.uint64:
            return (x >> np.uint64(32)).astype(np.int64), (x & np.uint64(0xFFFFFFFF)).astype(np.int64)
        y = x.astype(np.int64, copy=False)
        return y >> 32, y & 0xFFFFFFFF

    ha, la = split(a)
    hb, lb = split(b)
    return (hb - ha).astype(np.float64) * 4294967296.0 + (lb - la).astype(np.float64)


def pair_stats(
    a: Checkpoint, b: Checkpoint, entry: Entry, control: RunControl, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS
) -> PairStats:
    """Stream both tensors in lockstep (same element ranges) and accumulate paired stats."""
    assert entry.a is not None
    assert entry.b is not None
    ka, kb = entry.a, entry.b
    if ka.numel != kb.numel:
        raise fail(ErrorCode.INCOMPATIBLE_TENSOR, f"{entry.key}: element counts differ")
    size_a, size_b = ka.byte_length // max(ka.numel, 1), kb.byte_length // max(kb.numel, 1)
    max_elems = min(a.limits.read_chunk_bytes // max(size_a, 1), b.limits.read_chunk_bytes // max(size_b, 1))
    n = max(1, min(chunk_elements, max_elems))
    stats = PairStats()
    it_a = a.iter_flat_chunks(entry.key, n * max(size_a, 1), consistency=Consistency.BEST_EFFORT)
    it_b = b.iter_flat_chunks(entry.key, n * max(size_b, 1), consistency=Consistency.BEST_EFFORT)
    wide = ka.dtype in WIDE_INTS or kb.dtype in WIDE_INTS
    for ca, cb in zip(it_a, it_b, strict=True):
        # I64/U64: norms and cosine use float64-rounded values (doc 05: `rounded`), but the delta
        # is formed exactly first so changed counts and max |B - A| never lose a difference.
        delta = exact_int_delta(ca, cb) if wide else None
        stats.update(np.asarray(ca, dtype=np.float64), np.asarray(cb, dtype=np.float64), delta)
        control.tick(ca.size * size_a + cb.size * size_b, entry.key)
    return stats
