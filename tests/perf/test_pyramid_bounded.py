"""Pyramids, sampling, jobs and caches at CI scale (scripts/pyramid_benchmark.py --small): in a fresh process the
whole HTTP flow keeps the peak memory bounded, pyramid tiles equal the independent NumPy reference, the sampled preview
follows its documented rows, cancellation is prompt and publishes nothing, and LRU eviction respects the disk cap.
WEIGHTLAB_SKIP_PERF=1 skips."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import ROOT

MIB = 1024 * 1024


@pytest.mark.perf
@pytest.mark.skipif(os.environ.get("WEIGHTLAB_SKIP_PERF") == "1", reason="WEIGHTLAB_SKIP_PERF=1")
def test_pyramid_jobs_and_caches_are_bounded_and_exact(tmp_path: Path) -> None:
    out = tmp_path / "pyramid.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pyramid_benchmark.py"), "--small", "--dir", str(tmp_path),
         "--json", str(out)],
        capture_output=True, text=True, timeout=1800, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr.replace("\0", "")[-3000:]
    res = json.loads(out.read_text())
    evidence = os.environ.get("WEIGHTLAB_EVIDENCE_DIR")
    if evidence:
        Path(evidence, "pyramid_benchmark.json").write_text(out.read_text(), encoding="utf-8")

    assert res["peak_growth_bytes"] < 256 * MIB  # far below the 1 GiB backend budget
    assert res["phases"]["dense_build"]["bytes_scanned"] == res["dense_bytes"]  # one pass, nothing re-read
    for key, acc in res["accuracy"].items():
        assert acc["quality"] == "exact", key
        assert acc["max_ulps"] <= 1, key  # against the float64 NumPy reference of the file bytes
    s = res["sampled"]
    assert s["quality"] == "row_sampled"
    assert s["matches_pattern"]
    assert s["elements_read"] < s["covered_elements"]
    c = res["cancel"]
    assert c["state"] == "canceled"
    assert c["result"] is None
    assert c["tmp_files_left"] == 0
    assert c["after_cancel_status"] == 202  # never a false exact tile
    assert c["latency_ms"] < 2000
    e = res["eviction"]
    assert e["max_disk_bytes"] <= e["cap_bytes"]
    assert (e["reuse_hits"], e["reuse_misses"]) == (3, 2)  # LRU: the three most recent reused, the oldest rebuilt
    lb = res["large_build"]
    assert lb["state"] == "completed"
    assert lb["tile_quality"] == "exact"
    assert lb["matches_pattern"]
