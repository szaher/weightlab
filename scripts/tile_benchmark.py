#!/usr/bin/env python3
"""Tile engine memory and I/O benchmark (Phase 1 Task 4B foundation; doc 11 B02 subset, direct tiles only).

    uv run python scripts/tile_benchmark.py --dir /tmp/wl --json out.json [--size-gib 4]

Writes a sparse SafeTensors file holding one square F32 tensor of about `--size-gib` GiB, with a known pattern in
the far-corner region, then computes L0-L4 tiles in a *fresh* process and reports, per tile: wall time, covered
bytes, bytes read, and the process's peak memory (weightlab.core.memory: macOS phys_footprint peak, Linux VmHWM,
Windows PeakWorkingSetSize). L5 must be refused with RESOURCE_LIMIT (it would need the exact pyramid).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY = "bench.weight"

CHILD = r"""
import json, sys, time
import numpy as np
from weightlab.core.memory import peak_memory
from weightlab.formats.checkpoint import open_checkpoint
from weightlab.formats.selection import View
from weightlab.analysis.tile import compute_tile, TileBudget
from weightlab.api.config import DEFAULT_SETTINGS
from weightlab.core.errors import WeightLabError

path, key, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
budget = TileBudget.from_settings(DEFAULT_SETTINGS)
out = {"memory_metric": peak_memory().metric, "peak_after_import": peak_memory().peak, "tiles": []}
with open_checkpoint(path) as ck:
    out["peak_after_open"] = peak_memory().peak
    far = (n - 1) // 256  # last L0 tile index
    cases = [
        ("L0 origin", None, 0, 0, 0, "mean"),
        ("L0 far corner", None, 0, far, far, "max"),
        ("L1", None, 1, 0, 0, "mean"),
        ("L2", None, 2, 1, 1, "rms"),
        ("L3", None, 3, 0, 0, "mean"),
        ("L3 far corner", None, 3, (n - 1) // 2048, (n - 1) // 2048, "max"),
        ("L3 transposed", View(0, 1, {}, True), 3, 1, 0, "min"),
        ("L4 (64 MiB, the direct limit)", None, 4, 0, 0, "mean_abs"),
    ]
    for name, view, level, x, y, agg in cases:
        t0 = time.perf_counter()
        t = compute_tile(ck, key, view, level=level, x=x, y=y, aggregation=agg, budget=budget)
        dt = time.perf_counter() - t0
        out["tiles"].append({"name": name, "level": level, "x": x, "y": y, "aggregation": agg, "seconds": dt,
                             "shape": list(t.values.shape), "covered_bytes": t.covered_bytes, "bytes_read": t.bytes_read,
                             "peak_after": peak_memory().peak, "corner_value": float(t.values[-1, -1]),
                             "corner_status": int(t.status[-1, -1])})
    try:
        compute_tile(ck, key, None, level=5, x=0, y=0, aggregation="mean", budget=budget)
        out["l5"] = "computed (unexpected)"
    except WeightLabError as e:
        out["l5"] = f"{e.code.value}: {e.detail}"
    out["bytes_read_total"] = ck.bytes_read
print(json.dumps(out))
"""


def make_sparse(path: Path, size_gib: float) -> int:
    """n x n F32 tensor, sparse except a ramp in its last 8 x 8 cells (row-major value r*8+c+1)."""
    n = int(math.sqrt(size_gib * 2**30 / 4)) // 2048 * 2048
    header = json.dumps({KEY: {"dtype": "F32", "shape": [n, n], "data_offsets": [0, n * n * 4]}}).encode()
    header += b" " * (-len(header) % 8)
    base = 8 + len(header)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)) + header)
        if sys.platform == "win32":
            subprocess.run(["fsutil", "sparse", "setflag", str(path)], check=False, capture_output=True)
        f.truncate(base + n * n * 4)
        for r in range(8):
            f.seek(base + ((n - 8 + r) * n + (n - 8)) * 4)
            f.write(struct.pack("<8f", *[r * 8 + c + 1 for c in range(8)]))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--size-gib", type=float, default=4.0)
    args = ap.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    path = args.dir / "tile_bench.safetensors"
    n = make_sparse(path, args.size_gib)
    try:
        env = {"PYTHONPATH": str(ROOT)}
        proc = subprocess.run([sys.executable, "-c", CHILD, str(path), KEY, str(n)], capture_output=True, text=True,
                              timeout=1800, check=False, env={**__import__("os").environ, **env})  # fmt: skip
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode
        res = json.loads(proc.stdout)
    finally:
        path.unlink(missing_ok=True)
    res.update(n=n, file_bytes=n * n * 4, platform=sys.platform)
    peak = max(t["peak_after"] for t in res["tiles"])
    res["peak_growth_bytes"] = peak - res["peak_after_open"]
    text = json.dumps(res, indent=2)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
