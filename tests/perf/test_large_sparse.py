"""AC-002: a 10 GB+ file is cataloged and bounded ROIs are read without RSS proportional to its size.
Runs scripts/run_benchmarks.py in a fresh process. WEIGHTLAB_SKIP_PERF=1 skips;
WEIGHTLAB_BIG_FILE_GIB overrides the size (default 12)."""

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
def test_oversized_sparse_file_bounded_rss(tmp_path: Path, record_property: Callable[[str, object], None]) -> None:
    out = tmp_path / "bench.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_benchmarks.py"), "--dir", str(tmp_path), "--json", str(out)],
        capture_output=True, text=True, timeout=900, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out.read_text())
    for k in ("file_bytes", "rss_after_import", "rss_after_catalog", "rss_after_roi", "rss_after_scan", "scan_bytes"):
        record_property(k, res[k])
    evidence = os.environ.get("WEIGHTLAB_EVIDENCE_DIR")
    if evidence:
        Path(evidence, "benchmark.json").write_text(out.read_text(), encoding="utf-8")

    assert res["file_bytes"] >= 10 * 10**9 or os.environ.get("WEIGHTLAB_BIG_FILE_GIB")
    assert res["roi_correct"]
    assert res["rss_after_roi"] - res["rss_after_import"] < 16 * MIB  # catalog + exact ROIs
    assert res["rss_after_scan"] - res["rss_after_import"] < 128 * MIB  # 1 GiB streamed in 4 MiB chunks
    assert res["rss_after_scan"] < res["file_bytes"] / 50
