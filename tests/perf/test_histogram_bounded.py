"""ADR-021: a tensor histogram's working memory is bounded by `histogram.WORKING_SET` and does not grow with the
tensor. Measured with tracemalloc (numpy reports its buffers to it) on tensors 8x apart in size, for the widest
decode expansion (I8 / BF16 -> 8 / 4 bytes per element) and the most temporaries (float, symmetric range).
WEIGHTLAB_SKIP_PERF=1 skips."""

from __future__ import annotations

import os
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tests.helpers import write_safetensors
from weightlab import open_checkpoint
from weightlab.analysis.histogram import WORKING_SET, Binning, histogram

MIB = 1024 * 1024


def _peak(path: Path, binning: Binning) -> int:
    with open_checkpoint(path) as ck:
        tracemalloc.start()
        try:
            h = histogram(ck, "w", binning)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert h["counts"]["elements"] == ck.tensor("w").numel
    return peak


@pytest.mark.perf
@pytest.mark.skipif(os.environ.get("WEIGHTLAB_SKIP_PERF") == "1", reason="WEIGHTLAB_SKIP_PERF=1")
@pytest.mark.parametrize(("dtype", "storage"), [("BF16", "<u2"), ("I8", "|i1"), ("F32", "<f4")])
def test_histogram_memory_is_bounded_and_size_independent(
    tmp_path: Path, record_property: Callable[[str, object], None], dtype: str, storage: str
) -> None:
    rng = np.random.default_rng(1)
    peaks = []
    for n in (1 << 21, 1 << 24):
        raw = rng.integers(0, 2**15 if storage == "<u2" else 100, n).astype(storage)  # finite BF16 bit patterns
        p = write_safetensors(tmp_path / f"{dtype}_{n}.safetensors", {"w": (dtype, (n,), raw.tobytes())})
        peaks.append(_peak(p, Binning(1024, "symmetric")))
        p.unlink()
    record_property(f"{dtype}_peaks", peaks)
    assert max(peaks) < WORKING_SET, f"peak {max(peaks) / MIB:.1f} MiB above the {WORKING_SET // MIB} MiB reservation"
    assert peaks[1] - peaks[0] < 4 * MIB, peaks  # 8x the elements, the same working set
