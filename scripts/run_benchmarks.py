#!/usr/bin/env python3
"""Phase 0 memory benchmark (AC-002, B01 subset): catalog and exact ROIs on a sparse file larger
than RAM-friendly sizes, in a fresh process, reporting peak RSS.

    python scripts/run_benchmarks.py --dir /tmp/wl --size-gib 12 --json out.json

The file holds one BF16 tensor; known bit patterns are written at a few places so the ROI result
is checked, not just its shape. On NTFS the file is marked sparse with fsutil first.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLS = 65536
KEY = "big.bf16.weight"

CHILD = r"""
import json, os, sys, time
t0 = time.perf_counter()

from weightlab.core.memory import peak_memory

def peak_rss() -> int:  # name kept for the G0 report fields; see weightlab.core.memory for the counter
    return peak_memory().peak

import numpy as np
from weightlab import open_checkpoint
from weightlab.analysis.roi import encode_roi, ValueEncoding
from weightlab.formats.selection import View
path, key, prow, pcol, scan_bytes = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out = {"memory_metric": peak_memory().metric, "rss_after_import": peak_rss()}
t = time.perf_counter()
ck = open_checkpoint(path)
meta = ck.tensor(key)
out.update(catalog_seconds=time.perf_counter() - t, rss_after_catalog=peak_rss(), shape=list(meta.shape),
           dtype=meta.dtype, tensor_bytes=meta.byte_length)
t = time.perf_counter()
roi = ck.read_roi_raw(key, prow, prow + 1, pcol, pcol + 16)
out["roi_hex"] = encode_roi(roi, ValueEncoding.HEX_BITS).values[0]
out["roi_values"] = encode_roi(roi).values[0]
rows, cols = meta.shape
last = ck.read_roi_raw(key, rows - 1, rows, cols - 1, cols)
out["last_hex"] = encode_roi(last, ValueEncoding.HEX_BITS).values[0][0]
col = ck.read_roi_raw(key, 0, 64, pcol, pcol + 1, View(0, 1, {}, transpose=False))  # 64 rows x 1 col, strided
out["column_stripe_bytes"] = col.stripe_bytes
out.update(roi_seconds=time.perf_counter() - t, rss_after_roi=peak_rss())
t = time.perf_counter()
done = 0
nonzero = 0
for chunk in ck.iter_flat_chunks(key, 4 * 1024 * 1024):
    nonzero += int(np.count_nonzero(chunk))
    done += chunk.size * 2
    if done >= scan_bytes:
        break
out.update(scan_bytes=done, scan_seconds=time.perf_counter() - t, scan_nonzero=nonzero, rss_after_scan=peak_rss(),
           bytes_read=ck.bytes_read)
ck.close()
out["total_seconds"] = time.perf_counter() - t0
print(json.dumps(out))
"""


def bf16_bits(i: int) -> int:
    return 0x3F80 + i  # 1.0, 1.0078125, ... exact BF16 patterns


def make_sparse(path: Path, size_gib: float) -> tuple[int, int, int]:
    rows = int(size_gib * 2**30) // (COLS * 2)
    header = json.dumps({KEY: {"dtype": "BF16", "shape": [rows, COLS], "data_offsets": [0, rows * COLS * 2]}},
                        separators=(",", ":")).encode()  # fmt: skip
    header += b" " * (-len(header) % 8)
    data_start = 8 + len(header)
    total = data_start + rows * COLS * 2
    prow, pcol = rows * 7 // 9, 100
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)) + header)
    if sys.platform == "win32":
        subprocess.run(["fsutil", "sparse", "setflag", str(path)], check=False, capture_output=True)
    with path.open("r+b") as f:
        f.truncate(total)
        f.seek(data_start + (prow * COLS + pcol) * 2)
        f.write(b"".join(struct.pack("<H", bf16_bits(i)) for i in range(16)))
        f.seek(total - 2)
        f.write(struct.pack("<H", 0xC000))  # -2.0
    return prow, pcol, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--size-gib", type=float, default=float(os.environ.get("WEIGHTLAB_BIG_FILE_GIB", "12")))
    ap.add_argument("--scan-gib", type=float, default=1.0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    path = args.dir / "big_sparse.safetensors"
    t = time.perf_counter()
    prow, pcol, total = make_sparse(path, args.size_gib)
    create_s = time.perf_counter() - t
    try:
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", CHILD, str(path), KEY, str(prow), str(pcol),
                               str(int(args.scan_gib * 2**30))], capture_output=True, text=True, env=env, check=False)  # fmt: skip
        if proc.returncode:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode
        res = json.loads(proc.stdout)
        allocated = getattr(path.stat(), "st_blocks", 0) * 512 or None
    finally:
        path.unlink(missing_ok=True)
    expected_hex = [f"0x{bf16_bits(i):04x}" for i in range(16)]
    res.update(
        file_bytes=total, file_allocated_bytes=allocated, create_seconds=create_s, platform=platform.platform(),
        python=platform.python_version(), machine=platform.machine(),
        roi_correct=res["roi_hex"] == expected_hex and res["last_hex"] == "0xc000",
        rss_growth_bytes=res["rss_after_scan"] - res["rss_after_import"],
    )  # fmt: skip
    text = json.dumps(res, indent=2)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if res["roi_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
