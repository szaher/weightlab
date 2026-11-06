"""G0.5: `weightlab diff` memory does not grow with the compared pair. Runs
scripts/run_diff_benchmark.py (fresh processes) on two sparse BF16 pairs 8x apart in size and checks
exact metrics plus bounded, size-independent peak memory. The counter is `weightlab.core.memory`
(macOS phys_footprint peak, Linux VmHWM, Windows peak working set), not ru_maxrss: on macOS the
resident peak includes freed allocations kept by the allocator (CI runners showed 250+ MiB with
~60 MiB in use), and on Linux ru_maxrss is inherited from the forking parent.
WEIGHTLAB_SKIP_PERF=1 skips."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.helpers import ROOT

MIB = 1024 * 1024


@pytest.mark.perf
@pytest.mark.skipif(os.environ.get("WEIGHTLAB_SKIP_PERF") == "1", reason="WEIGHTLAB_SKIP_PERF=1")
def test_diff_memory_independent_of_pair_size(tmp_path: Path, record_property: Callable[[str, object], None]) -> None:
    out = tmp_path / "bench.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_diff_benchmark.py"), "--dir", str(tmp_path / "w"),
         "--sizes-gib", "0.125,1", "--no-dense", "--json", str(out)],
        capture_output=True, text=True, timeout=600, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out.read_text())
    small, large = res["cases"]
    for c in (small, large):
        record_property(f"{c['case']}_mem_growth", c["mem_growth_bytes"])
        record_property(f"{c['case']}_resident_peak", c["resident_peak"])
        assert c["mem_after_import"] > 0  # a probe that reads 0 must not pass a bound
        assert c["metrics_correct"]
        assert c["report_complete"]
        assert c["bytes_read"] >= c["pair_bytes"]  # every byte of both sides was compared
        assert c["mem_growth_bytes"] < 128 * MIB  # chunk buffers + float64 scratch, not the tensor
    assert large["pair_bytes"] == 8 * small["pair_bytes"]
    assert large["mem_growth_bytes"] - small["mem_growth_bytes"] < 16 * MIB
