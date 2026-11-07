"""Bounded tile memory and I/O on a large sparse checkpoint (Phase 1 Task 4B foundation): L0-L4 direct tiles of a
4 GiB F32 tensor in a fresh process keep the peak memory flat, read only the covered bytes, find the known far-corner
values, and an L5 tile (256 MiB covered) is refused instead of read or sampled. WEIGHTLAB_SKIP_PERF=1 skips."""

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
def test_direct_tiles_have_bounded_memory_and_reads(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    out = tmp_path / "tiles.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "tile_benchmark.py"), "--dir", str(tmp_path), "--json", str(out)],
        capture_output=True, text=True, timeout=1800, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out.read_text())
    evidence = os.environ.get("WEIGHTLAB_EVIDENCE_DIR")
    if evidence:
        Path(evidence, "tile_benchmark.json").write_text(out.read_text(), encoding="utf-8")
    record_property("peak_growth_bytes", res["peak_growth_bytes"])

    assert res["file_bytes"] >= 4 * 1024**3 - 64 * MIB
    # The engine's working set is ~35 MiB by its own estimate (band + output records + one read chunk); the peak
    # stays far below the 64 MiB a single L4 tile covers and unrelated to the 4 GiB file.
    assert res["peak_growth_bytes"] < 128 * MIB
    for t in res["tiles"]:
        assert t["shape"] == [256, 256], t["name"]
        assert t["bytes_read"] == t["covered_bytes"], t["name"]  # full-width stripes: no read amplification here
    by_name = {t["name"]: t for t in res["tiles"]}
    assert by_name["L0 far corner"]["corner_value"] == 64.0  # the known pattern at the last cell
    assert by_name["L3 far corner"]["corner_value"] == 64.0  # max of the last 8 x 8 block
    assert by_name["L4 (64 MiB, the direct limit)"]["covered_bytes"] == 64 * MIB
    assert res["l5"].startswith("RESOURCE_LIMIT")
    assert "pyramid" in res["l5"]
