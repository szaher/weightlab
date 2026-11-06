#!/usr/bin/env python3
"""Phase 0.5 change-report benchmark (G0.5): memory stays bounded as the compared pair grows, and
real checkpoints/adapters run end to end. Every case runs `weightlab diff` in a fresh process
that reports its own peak RSS, elapsed time and bytes read.

    python scripts/run_diff_benchmark.py --dir /tmp/wl --sizes-gib 0.25,1,4 --json out.json \
        [--pair BASE DERIVED] [--adapter BASE ADAPTER_DIR]

Synthetic cases: one BF16 tensor per side, sparse (zero pages) except for known values at three
positions in B, so the expected metrics are exact: delta_l2 = sqrt(sum d^2), max_abs_delta = max |d|.
Sparse zero pages come from the page cache, so synthetic throughput measures compute, not disk.

Dense cases (unless --no-dense): B05, a 1 GiB BF16 pair of seeded random values; B06, an 8192x8192
BF16 base with rank-8 and rank-32 LoRA adapters. Their expected delta_l2 is recomputed here with
plain NumPy (row blocks in float64), independently of WeightLab's streaming accumulators.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COLS = 8192
KEY = "big.bf16.weight"
# (row fraction, col, BF16 bits, value): 1.0, -2.0, 0.5
POKES = [(1 / 7, 3, 0x3F80, 1.0), (4 / 9, 100, 0xC000, -2.0), (1.0, COLS - 1, 0x3F00, 0.5)]

CHILD = r"""
import json, sys, time
t0 = time.perf_counter()

def peak_rss() -> int:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return int(c.PeakWorkingSetSize)
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)

from weightlab.cli import main
from weightlab.formats.checkpoint import Checkpoint
read = [0]
_close = Checkpoint.close
def close(self):
    read[0] += self.bytes_read
    _close(self)
Checkpoint.close = close
out = {"rss_after_import": peak_rss()}
t = time.perf_counter()
code = main(sys.argv[1:])
out.update(exit_code=code, diff_seconds=time.perf_counter() - t, rss_peak=peak_rss(), bytes_read=read[0],
           total_seconds=time.perf_counter() - t0)
print(json.dumps(out))
"""


def make_sparse(path: Path, size_gib: float, pokes: bool) -> int:
    rows = int(size_gib * 2**30) // (COLS * 2)
    header = json.dumps({KEY: {"dtype": "BF16", "shape": [rows, COLS], "data_offsets": [0, rows * COLS * 2]}},
                        separators=(",", ":")).encode()  # fmt: skip
    header += b" " * (-len(header) % 8)
    start = 8 + len(header)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)) + header)
    if sys.platform == "win32":
        subprocess.run(["fsutil", "sparse", "setflag", str(path)], check=False, capture_output=True)
    with path.open("r+b") as f:
        f.truncate(start + rows * COLS * 2)
        if pokes:
            for frac, col, bits, _ in POKES:
                row = min(rows - 1, int(rows * frac))
                f.seek(start + (row * COLS + col) * 2)
                f.write(struct.pack("<H", bits))
    return rows * COLS


def write_tensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Minimal SafeTensors writer (BF16 from float32 by truncation is avoided: values are made
    BF16-exact first, so the file holds exactly the array given)."""
    header: dict[str, object] = {}
    off = 0
    for k, a in tensors.items():
        n = a.size * (2 if a.dtype == np.uint16 else a.itemsize)
        header[k] = {"dtype": "BF16" if a.dtype == np.uint16 else "F32", "shape": list(a.shape),
                     "data_offsets": [off, off + n]}  # fmt: skip
        off += n
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for a in tensors.values():
            f.write(np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<"), copy=False).tobytes())


def bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 bit patterns (round to nearest even; inputs are finite)."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def dense_pair(workdir: Path, size_gib: float) -> dict[str, object]:
    """Written in row blocks so the harness itself stays small; expected ||B-A|| accumulated per block."""
    rng = np.random.default_rng(5)
    cols, block = 4096, 1024
    rows = int(size_gib / 2 * 2**30) // (cols * 2) // block * block
    d = workdir / "b05"
    files = [d / "a" / "model.safetensors", d / "b" / "model.safetensors"]
    h = json.dumps({"w": {"dtype": "BF16", "shape": [rows, cols], "data_offsets": [0, rows * cols * 2]}}).encode()
    h += b" " * (-len(h) % 8)
    ssq = 0.0
    handles = []
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        handles.append(f.open("wb"))
        handles[-1].write(struct.pack("<Q", len(h)) + h)
    try:
        for _ in range(rows // block):
            base = rng.standard_normal((block, cols), dtype=np.float32) * 0.02
            a_bits = bf16(base)
            b_bits = bf16(base + rng.standard_normal((block, cols), dtype=np.float32) * 0.001)
            handles[0].write(a_bits.astype("<u2").tobytes())
            handles[1].write(b_bits.astype("<u2").tobytes())
            delta = (from_bf16(b_bits) - from_bf16(a_bits)).ravel()
            ssq += float(np.dot(delta, delta))
    finally:
        for fh in handles:
            fh.close()
    try:
        report = workdir / "reports" / "b05_dense_pair.json"
        res = diff([str(d / "a"), str(d / "b")], report)
        got = json.loads(report.read_text(encoding="utf-8"))["tensors"][0]["metrics"]["delta_l2"]["value"]
    finally:
        for f in files:
            f.unlink(missing_ok=True)
            f.parent.rmdir()
        d.rmdir()
    pair_bytes = 2 * rows * cols * 2
    res.update(case=f"b05_dense_bf16_pair_{size_gib}gib", pair_bytes=pair_bytes, delta_l2=got,
               expected_delta_l2=math.sqrt(ssq), metrics_correct=math.isclose(got, math.sqrt(ssq), rel_tol=1e-9),
               throughput_mib_s_per_side=pair_bytes / 2 / float(res["diff_seconds"]) / 2**20,  # type: ignore[arg-type]
               rss_growth_bytes=int(res["rss_peak"]) - int(res["rss_after_import"]))  # type: ignore[call-overload]  # fmt: skip
    return res


def lora_b06(workdir: Path, rank: int, n: int = 8192) -> dict[str, object]:
    rng = np.random.default_rng(rank)
    d = workdir / f"b06_r{rank}"
    w = np.concatenate([bf16(rng.standard_normal((1024, n), dtype=np.float32) * 0.02) for _ in range(n // 1024)])
    write_tensors(d / "base" / "model.safetensors", {"model.layers.0.mlp.proj.weight": w})
    del w
    la = rng.standard_normal((rank, n), dtype=np.float32) * 0.01
    lb = rng.standard_normal((n, rank), dtype=np.float32) * 0.01
    alpha = 2 * rank
    write_tensors(d / "adapter" / "adapter_model.safetensors",
                  {"base_model.model.model.layers.0.mlp.proj.lora_A.weight": la,
                   "base_model.model.model.layers.0.mlp.proj.lora_B.weight": lb})  # fmt: skip
    (d / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": rank, "lora_alpha": alpha, "target_modules": ["proj"], "bias": "none",
         "base_model_name_or_path": "b06-synthetic-base"}), encoding="utf-8")  # fmt: skip
    scale = alpha / rank
    a64, b64 = la.astype(np.float64), lb.astype(np.float64)
    ssq = sum(float(np.sum((scale * (b64[i : i + 1024] @ a64)) ** 2)) for i in range(0, n, 1024))
    try:
        report = workdir / "reports" / f"b06_r{rank}.json"
        res = diff([str(d / "base"), "--adapter", str(d / "adapter")], report)
        row = json.loads(report.read_text(encoding="utf-8"))["tensors"][0]
    finally:
        for f in (d / "base" / "model.safetensors", d / "adapter" / "adapter_model.safetensors",
                  d / "adapter" / "adapter_config.json"):  # fmt: skip
            f.unlink(missing_ok=True)
        for sub in (d / "base", d / "adapter", d):
            sub.rmdir()
    got = row["metrics"]["delta_l2"]["value"]
    res.update(case=f"b06_lora_{n}x{n}_rank{rank}", status=row["status"], gmac=row["lora"]["gmac"],
               delta_l2=got, expected_delta_l2=math.sqrt(ssq),
               metrics_correct=row["status"] == "analyzed" and math.isclose(got, math.sqrt(ssq), rel_tol=1e-9),
               rss_growth_bytes=int(res["rss_peak"]) - int(res["rss_after_import"]))  # type: ignore[call-overload]  # fmt: skip
    return res


def diff(args: list[str], report: Path) -> dict[str, object]:
    env = dict(os.environ, PYTHONPATH=str(ROOT), WEIGHTLAB_REPORT_TIMESTAMP="2026-11-06T00:00:00Z")
    proc = subprocess.run([sys.executable, "-c", CHILD, "diff", *args, "--json", str(report), "--quiet"],
                          capture_output=True, text=True, env=env, check=False)  # fmt: skip
    if proc.returncode:
        raise SystemExit(f"benchmark child failed ({proc.returncode}):\n{proc.stderr}")
    res: dict[str, object] = json.loads(proc.stdout)
    if res["exit_code"] not in (0, 3):  # 3 = report written but incomplete
        raise SystemExit(f"weightlab diff exited {res['exit_code']}:\n{proc.stderr}")
    r = json.loads(report.read_text(encoding="utf-8"))
    lin = r["compatibility"]["lineage"]
    res.update(
        report_complete=r["complete"], compatibility_class=lin["compatibility_class"],
        transfer_eligibility=lin["transfer_eligibility"], tensors_reported=len(r["tensors"]),
        completeness=r["completeness"], not_compared=len(r["compatibility"]["not_compared"]),
    )  # fmt: skip
    return res


def synthetic(workdir: Path, size_gib: float, chunk: int | None) -> dict[str, object]:
    a, b = workdir / f"s{size_gib}" / "a.safetensors", workdir / f"s{size_gib}" / "b.safetensors"
    elements = make_sparse(a, size_gib, pokes=False)
    make_sparse(b, size_gib, pokes=True)
    try:
        extra = ["--chunk-elements", str(chunk)] if chunk else []
        report = workdir / "reports" / f"synthetic_{size_gib}gib.json"  # never inside an input directory
        res = diff([str(a), str(b), *extra], report)
        row = json.loads(report.read_text(encoding="utf-8"))["tensors"][0]["metrics"]
    finally:
        a.unlink(missing_ok=True)
        b.unlink(missing_ok=True)
        a.parent.rmdir()
    want_l2 = math.sqrt(sum(v * v for *_, v in POKES))
    pair_bytes = 2 * elements * 2
    secs = float(res["diff_seconds"])  # type: ignore[arg-type]
    res.update(
        case=f"synthetic_sparse_bf16_{size_gib}gib", elements_per_side=elements, pair_bytes=pair_bytes,
        throughput_mib_s_per_side=pair_bytes / 2 / secs / 2**20,
        rss_growth_bytes=int(res["rss_peak"]) - int(res["rss_after_import"]),  # type: ignore[call-overload]
        delta_l2=row["delta_l2"]["value"], expected_delta_l2=want_l2,
        max_abs_delta=row["max_abs_delta"]["value"], expected_max_abs_delta=2.0,
        metrics_correct=math.isclose(row["delta_l2"]["value"], want_l2, rel_tol=1e-15)
        and row["max_abs_delta"]["value"] == 2.0 and res["bytes_read"] >= pair_bytes,
    )  # fmt: skip
    return res


def real(workdir: Path, name: str, args: list[str], inputs: list[Path]) -> dict[str, object]:
    res = diff(args, workdir / "reports" / f"{name}.json")
    size = sum(f.stat().st_size for p in inputs for f in ([p] if p.is_file() else p.glob("*.safetensors")))
    res.update(case=name, input_safetensors_bytes=size,
               rss_growth_bytes=int(res["rss_peak"]) - int(res["rss_after_import"]))  # type: ignore[call-overload]  # fmt: skip
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--sizes-gib", default="0.25,1,4")
    ap.add_argument("--chunk-elements", type=int)
    ap.add_argument("--pair", nargs=2, type=Path, metavar=("BASE", "DERIVED"))
    ap.add_argument("--adapter", nargs=2, type=Path, metavar=("BASE", "ADAPTER_DIR"))
    ap.add_argument("--no-dense", action="store_true", help="skip the dense B05 pair and B06 LoRA cases")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    (args.dir / "reports").mkdir(parents=True, exist_ok=True)
    cases = [synthetic(args.dir, float(s), args.chunk_elements) for s in args.sizes_gib.split(",") if s]
    if not args.no_dense:
        cases.append(dense_pair(args.dir, 1.0))
        cases += [lora_b06(args.dir, r) for r in (8, 32)]
    if args.pair:
        cases.append(real(args.dir, "real_checkpoint_pair", [str(p) for p in args.pair], list(args.pair)))
    if args.adapter:
        base, ad = args.adapter
        cases.append(real(args.dir, "real_lora_adapter", [str(base), "--adapter", str(ad)], [base, ad]))
    synth = [c for c in cases if str(c["case"]).startswith("synthetic")]
    growth = [int(c["rss_growth_bytes"]) for c in synth]  # type: ignore[call-overload]
    out = {
        "benchmark": "weightlab diff (G0.5)",
        "platform": platform.platform(), "python": platform.python_version(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": cases,
        "summary": {
            "synthetic_all_correct": all(bool(c["metrics_correct"]) for c in synth),
            "dense_all_correct": all(bool(c["metrics_correct"]) for c in cases if str(c["case"]).startswith("b0")),
            "synthetic_rss_growth_range_bytes": [min(growth), max(growth)] if growth else None,
            "synthetic_size_ratio": (max(c["pair_bytes"] for c in synth) / min(c["pair_bytes"] for c in synth))  # type: ignore[type-var,operator]
            if synth else None,
        },
    }  # fmt: skip
    text = json.dumps(out, indent=2)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    summary = out["summary"]
    return 0 if summary["synthetic_all_correct"] and summary["dense_all_correct"] else 1  # type: ignore[index]


if __name__ == "__main__":
    sys.exit(main())
