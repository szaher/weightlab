#!/usr/bin/env python3
"""Exact pyramid, row sampling, jobs and cache benchmark (Phase 1 Task 4B, PR #7; doc 11 B02 / B12 subset).

    uv run python scripts/pyramid_benchmark.py --dir /tmp/wl --json out.json [--small]

Writes (1) a dense 8192 x 8192 BF16 tensor (128 MiB) with deterministic values, (2) a sparse BF16 tensor above
`pyramid_exact_max_mib` (36864 x 16384, 1.125 GiB; row r holds r mod 251 in its first 64 columns) and (3) four
6144 x 6144 BF16 tensors (72 MiB each) for cache eviction. A *fresh* child process drives the real HTTP stack (FastAPI behind the
Guard, via Starlette's TestClient: no socket, nothing leaves the process) and reports wall times, bytes, job states,
cache statistics and the process's peak memory (weightlab.core.memory) after every phase:

* cold exact pyramid build of the dense tensor (202 -> job -> completed), then cold and warm tile latencies at
  L4 / L5 / L8 / L13, and direct L0 / L3 tiles; memory-cache hit ratio;
* a row-sampled preview of the large sparse tensor (auto mode, above the limit);
* cancellation of an exact build of the large tensor during its scan (latency from POST cancel to `canceled`,
  nothing published);
* disk-cache eviction and reuse with tile_cache_disk_mib lowered so three of the four small pyramids fit;
* the explicit exact build of the large sparse tensor (above the auto limit) with its peak memory.

The parent checks accuracy against an independent NumPy reference of the dense tensor (float64 block means and
maxima computed from the file bytes) and the known pattern of the sparse one. `--small` shrinks every size (CI).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CHILD = r"""
import json, math, sys, time, statistics
from pathlib import Path
from fastapi.testclient import TestClient
from weightlab.core.memory import peak_memory
from weightlab.api.app import create_app
from weightlab.api.config import ServerConfig

work, small = Path(sys.argv[1]), sys.argv[2] == "1"
T = "b" * 43
H = {"Authorization": "Bearer " + T}
P = "/api/v1"
V = {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}
out = {"memory_metric": peak_memory().metric, "phases": {}}
def mark(name, **kw):
    out["phases"][name] = {"peak_bytes": peak_memory().peak, **kw}

cfg = ServerConfig(port=8765, data_dir=work / "data", token=T)
with TestClient(create_app(cfg), base_url="http://127.0.0.1:8765") as c:
    def call(m, path, **kw):
        r = c.request(m, P + path, headers={**H, **kw.pop("headers", {})}, **kw)
        return r
    if small:
        assert call("PATCH", "/settings", json={"direct_tile_max_mib": 8, "pyramid_exact_max_mib": 64}).status_code == 200
    mark("start")
    pj = call("POST", "/projects", json={"name": "bench"}).json()["id"]
    def imp(name):
        m = call("POST", f"/projects/{pj}/models", json={"entry_path": str(work / name / "model.safetensors"),
                                                          "selected_root": str(work / name)}).json()
        t = call("GET", f"/models/{m['id']}/tensors").json()["items"][0]
        return {"kind": "tensor", "model_id": m["id"], "tensor_id": t["id"]}
    dense, sparse = imp("dense"), imp("sparse")
    smalls = [imp(f"s{i}") for i in range(4)]
    mark("imported")
    def tile(src, level, x=0, y=0, agg="mean", mode="auto", accept=None, **q):
        body = {"source": src, "view": V, "tile": {"level": level, "x": x, "y": y, "size": 256},
                "aggregation": agg, "quality": {"mode": mode, **q}}
        t0 = time.perf_counter()
        r = call("POST", "/tiles", json=body, headers={"Accept": accept} if accept else {})
        return r, time.perf_counter() - t0
    def wait(job_id, until=("completed", "failed", "canceled", "interrupted"), timeout=3600):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            j = call("GET", f"/jobs/{job_id}").json()
            if j["state"] in until:
                return j
            time.sleep(0.01)
        raise SystemExit("job timeout")

    # -- cold build of the dense tensor -------------------------------------------------------------
    r, dt = tile(dense, 5)  # L5 covers more than direct_tile_max_mib: the exact pyramid
    assert r.status_code == 202, (r.status_code, r.headers.get("content-type"))
    job = r.json()
    t0 = time.perf_counter()
    done = wait(job["job_id"])
    build_s = time.perf_counter() - t0
    assert done["state"] == "completed", done
    mark("dense_build", seconds=build_s, bytes_scanned=done["bytes_scanned"], estimated_bytes=done["estimated_bytes"],
         request_s=dt)
    tiles = {}
    lat = {}
    for level, x, y, agg in ((4, 0, 0, "mean"), (4, 0, 0, "max"), (5, 0, 0, "rms"), (8, 0, 0, "mean"),
                             (13, 0, 0, "mean"), (0, 3, 5, "mean"), (3, 0, 1, "mean_abs")):
        cold = []
        warm = []
        for i in range(12):
            r, dt = tile(dense, level, x, y, agg)
            assert r.status_code == 200, (r.status_code, r.text[:300] if r.headers.get('content-type','').startswith('application/json') else '')
            (cold if i == 0 else warm).append(dt)
        rows, cols = int(r.headers["x-weightlab-rows"]), int(r.headers["x-weightlab-cols"])
        vals = list(memoryview(r.content[: rows * cols * 4]).cast("f"))
        key = f"L{level}_{x}_{y}_{agg}"
        tiles[key] = {"rows": rows, "cols": cols, "values": vals, "quality": r.headers["x-weightlab-quality"],
                      "row_start": int(r.headers["x-weightlab-row-start"]),
                      "col_start": int(r.headers["x-weightlab-col-start"])}
        lat[key] = {"cold_ms": cold[0] * 1e3, "warm_p50_ms": statistics.median(warm) * 1e3,
                    "warm_p95_ms": sorted(warm)[int(0.95 * (len(warm) - 1))] * 1e3}
    out["dense_tiles"] = tiles
    out["latency"] = lat
    mark("dense_tiles")

    # -- row-sampled preview of the large source -------------------------------------------------------
    lvl = 9 if not small else 6
    r, dt = tile(sparse, lvl)
    assert r.status_code == 200, (r.status_code, r.text[:300] if r.headers.get('content-type','').startswith('application/json') else '')
    rows, cols = int(r.headers["x-weightlab-rows"]), int(r.headers["x-weightlab-cols"])
    meta = call("GET", f"/tiles/{r.headers['x-weightlab-cache-key']}/metadata").json()
    r2, dt2 = tile(sparse, lvl)
    out["sampled"] = {"level": lvl, "seconds": dt, "warm_seconds": dt2, "quality": r.headers["x-weightlab-quality"],
                      "sampled_rows": int(r.headers["x-weightlab-sampled-rows"]), "policy": meta["sampling_policy"],
                      "elements_read": meta["finite_count"] + meta["nonfinite_count"],
                      "covered_elements": (meta["row_end"] - meta["row_start"]) * (meta["col_end"] - meta["col_start"]),
                      "row_start": meta["row_start"], "row_end": meta["row_end"],
                      "rows": rows, "cols": cols, "values_col0": list(memoryview(r.content[: rows * cols * 4]).cast("f"))[::cols]}
    mark("sampled")

    # -- cancellation during the large scan --------------------------------------------------------------
    r, _ = tile(sparse, lvl, mode="exact")
    assert r.status_code == 202, (r.status_code, r.text[:300] if r.headers.get('content-type','').startswith('application/json') else '')
    jid = r.json()["job_id"]
    wait(jid, ("running",))
    while call("GET", f"/jobs/{jid}").json()["progress"] < 0.05:
        time.sleep(0.002)
    before = call("GET", f"/jobs/{jid}").json()
    t0 = time.perf_counter()
    c_ = call("POST", f"/jobs/{jid}/cancel").json()
    if c_["state"] != "canceled":
        c_ = wait(jid)
    cancel_s = time.perf_counter() - t0
    pyr_dir = work / "data" / "cache" / "pyramids"
    out["cancel"] = {"latency_ms": cancel_s * 1e3, "state": c_["state"], "progress_at_cancel": before["progress"],
                     "bytes_scanned": c_["bytes_scanned"], "result": c_["result"],
                     "tmp_files_left": len(list(pyr_dir.glob("*.tmp-*")))}
    r, _ = tile(sparse, lvl, mode="exact")
    out["cancel"]["after_cancel_status"] = r.status_code  # 202: never a false exact tile
    if r.status_code == 202:
        call("POST", f"/jobs/{r.json()['job_id']}/cancel")
    mark("cancel")

    # -- disk cache eviction and reuse ---------------------------------------------------------------
    from weightlab.analysis.pyramid import estimate_bytes
    shape = (2048, 2304) if small else (6144, 6144)
    top = 12 if small else 13  # one tile covering the whole tensor (> direct_tile_max_mib)
    reserve = estimate_bytes(*shape)
    cap_mib = int(3.5 * reserve) // 2**20  # whole MiB: room for three reservations, not for four pyramids
    assert 3 * reserve <= cap_mib * 2**20, (reserve, cap_mib)
    assert call("PATCH", "/settings", json={"tile_cache_disk_mib": cap_mib}).status_code == 200
    evict = {"cap_mib": cap_mib, "reservation_bytes": reserve, "events": []}
    def request(i, agg, phase):
        r, dt = tile(smalls[i], top, agg=agg)
        ev = {"phase": phase, "tensor": i, "agg": agg, "status": r.status_code, "ms": dt * 1e3}
        if r.status_code == 202:
            t0 = time.perf_counter()
            assert wait(r.json()["job_id"])["state"] == "completed"
            ev["build_ms"] = (time.perf_counter() - t0) * 1e3
            r, dt = tile(smalls[i], top, agg=agg)
            ev["after"], ev["after_ms"] = r.status_code, dt * 1e3
        ev["disk_bytes"] = sum(p.stat().st_size for p in pyr_dir.glob("*.wlpyr"))
        ev["pyramids_on_disk"] = len(list(pyr_dir.glob("*.wlpyr")))
        evict["events"].append(ev)
    for i in range(4):  # cold builds; the fourth evicts the least recently used one
        request(i, "mean", "build")
    for i in (3, 2, 1, 0, 3):  # new aggregations, so the memory tile cache cannot answer: disk hit or rebuild
        request(i, "rms" if i != 3 or len([e for e in evict["events"] if e["tensor"] == 3]) < 2 else "max", "reuse")
    evict["max_disk_bytes"] = max(e["disk_bytes"] for e in evict["events"])
    evict["cap_bytes"] = cap_mib * 2**20
    reuse = [e for e in evict["events"] if e["phase"] == "reuse"]
    evict["reuse_hits"] = sum(e["status"] == 200 for e in reuse)
    evict["reuse_misses"] = sum(e["status"] == 202 for e in reuse)
    out["eviction"] = evict
    assert call("PATCH", "/settings", json={"tile_cache_disk_mib": 1024}).status_code == 200
    mark("eviction")

    # -- explicit exact build above the auto limit ---------------------------------------------------
    r, _ = tile(sparse, lvl, mode="exact")
    assert r.status_code == 202, (r.status_code, r.text[:300] if r.headers.get('content-type','').startswith('application/json') else '')
    t0 = time.perf_counter()
    done = wait(r.json()["job_id"])
    out["large_build"] = {"seconds": time.perf_counter() - t0, "state": done["state"],
                          "bytes_scanned": done["bytes_scanned"],
                          "pyramid_bytes": max(p.stat().st_size for p in pyr_dir.glob("*.wlpyr"))}
    r, dt = tile(sparse, lvl, mode="exact")
    out["large_build"]["tile_status"] = r.status_code
    out["large_build"]["tile_quality"] = r.headers.get("x-weightlab-quality")
    out["large_build"]["tile_ms"] = dt * 1e3
    rows, cols = int(r.headers["x-weightlab-rows"]), int(r.headers["x-weightlab-cols"])
    out["large_build"]["values_col0"] = list(memoryview(r.content[: rows * cols * 4]).cast("f"))[::cols]
    mark("large_build")
    svc = c.app.app.state.tiles  # the Guard wraps the FastAPI app
    st = svc.pyramids.stats()
    out["cache"] = {"memory_tile_hits": svc.memory.hits, "memory_tile_misses": svc.memory.misses,
                    "memory_tile_hit_ratio": svc.memory.hits / max(1, svc.memory.hits + svc.memory.misses),
                    "memory_tile_bytes": svc.memory.bytes_used, "memory_tile_entries": len(svc.memory),
                    "pyramid_hits": st.hits, "pyramid_misses": st.misses, "pyramid_evictions": st.evictions,
                    "pyramid_corrupt": st.corrupt, "disk_bytes": st.disk_bytes, "disk_entries": st.disk_entries}
out["peak_bytes"] = peak_memory().peak
print(json.dumps(out))
"""


def bf16_bytes(a: np.ndarray) -> bytes:
    """Round-to-nearest-even float32 -> BF16 bit patterns (deterministic, NumPy-version independent)."""
    u = a.astype("<f4").view("<u4").astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")
    return r.tobytes()


def write(path: Path, key: str, shape: tuple[int, int], data: bytes | None) -> None:
    header = json.dumps({key: {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, shape[0] * shape[1] * 2]}})
    hb = header.encode() + b" " * (-len(header) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        if data is not None:
            f.write(data)
            return
        if sys.platform == "win32":
            subprocess.run(["fsutil", "sparse", "setflag", str(path)], check=False, capture_output=True)
        base = 8 + len(hb)
        f.truncate(base + shape[0] * shape[1] * 2)
        row = shape[1] * 2
        for r in range(shape[0]):
            f.seek(base + r * row)
            f.write(bf16_bytes(np.full(64, r % 251, np.float32)))


def decode_bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    bits = np.frombuffer(raw[8 + n :], "<u2").astype(np.uint32) << 16
    return bits.view(np.float32).reshape(shape)


def reference(dense: np.ndarray, level: int, x: int, y: int, agg: str) -> np.ndarray:
    b = 1 << level
    r0, c0 = y * 256 * b, x * 256 * b
    sub = dense[r0 : r0 + 256 * b, c0 : c0 + 256 * b].astype(np.float64)
    rows, cols = -(-sub.shape[0] // b), -(-sub.shape[1] // b)
    out = np.empty((rows, cols))
    for i in range(rows):
        blk_rows = sub[i * b : (i + 1) * b]
        for j in range(cols):
            blk = blk_rows[:, j * b : (j + 1) * b]
            out[i, j] = {"mean": np.mean, "max": np.max, "mean_abs": lambda a: np.mean(np.abs(a)),
                         "rms": lambda a: math.sqrt(np.mean(a * a))}[agg](blk)  # fmt: skip
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    ap.add_argument(
        "--small", action="store_true", help="CI sizes: 4096x2048 dense, 72 MiB sparse, 2048x2304 cache set"
    )
    args = ap.parse_args()
    work = args.dir
    work.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(work / "data", ignore_errors=True)  # a fresh data directory and cache every run
    dense_shape = (4096, 2048) if args.small else (8192, 8192)
    sparse_shape = (4608, 8192) if args.small else (36864, 16384)
    small_shape = (2048, 2304) if args.small else (6144, 6144)  # each above direct_tile_max_mib
    rng = np.random.default_rng(20261107)
    dense = (rng.standard_normal(dense_shape, dtype=np.float32) * 0.02).astype(np.float32)
    write(work / "dense" / "model.safetensors", "dense.weight", dense_shape, bf16_bytes(dense))
    write(work / "sparse" / "model.safetensors", "sparse.weight", sparse_shape, None)
    for i in range(4):
        a = (rng.standard_normal(small_shape, dtype=np.float32) * (i + 1)).astype(np.float32)
        write(work / f"s{i}" / "model.safetensors", "s.weight", small_shape, bf16_bytes(a))
    del dense
    try:
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        proc = subprocess.run([sys.executable, "-c", CHILD, str(work), "1" if args.small else "0"],
                              capture_output=True, text=True, timeout=3600, check=False, env=env)  # fmt: skip
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
            return proc.returncode
        res = json.loads(proc.stdout)
        decoded = decode_bf16(work / "dense" / "model.safetensors", dense_shape)
        acc = {}
        for key, t in res.pop("dense_tiles").items():
            level, x, y, agg = key[1:].split("_", 3)
            ref = reference(decoded, int(level), int(x), int(y), agg).astype(np.float32)
            got = np.array(t["values"], np.float32).reshape(t["rows"], t["cols"])
            rel = np.abs(got.astype(np.float64) - ref) / np.maximum(np.abs(ref.astype(np.float64)), 1e-30)
            acc[key] = {"quality": t["quality"], "cells": int(got.size), "max_rel_error": float(rel.max()),
                        "max_ulps": int(np.max(np.abs(got.view(np.int32).astype(np.int64)
                                                      - ref.view(np.int32).astype(np.int64))))}  # fmt: skip
        res["accuracy"] = acc
        s = res["sampled"]
        k = s["sampled_rows"]
        lvl = s["level"]
        b = 1 << lvl
        want = []
        for i in range(s["rows"]):
            top = s["row_start"] + i * b
            n = min(s["row_end"], top + b) - top
            picked = [top + ((2 * j + 1) * n) // (2 * min(k, n)) for j in range(min(k, n))]  # stride policy
            want.append(float(np.float32(np.mean([r % 251 for r in picked]) * 64 / min(b, sparse_shape[1]))))
        s["matches_pattern"] = bool(np.allclose(s.pop("values_col0"), want, rtol=1e-6))
        lb = res["large_build"]
        exact_col0 = [float(np.float32(np.mean([r % 251 for r in range(i * b, min(sparse_shape[0], (i + 1) * b))])
                                       * 64 / min(b, sparse_shape[1]))) for i in range(s["rows"])]  # fmt: skip
        lb["matches_pattern"] = bool(np.allclose(lb.pop("values_col0"), exact_col0, rtol=1e-6))
    finally:
        for name in ("dense", "sparse", *(f"s{i}" for i in range(4))):
            (work / name / "model.safetensors").unlink(missing_ok=True)
    res.update(platform=sys.platform, dense_shape=dense_shape, sparse_shape=sparse_shape, small_shape=small_shape,
               sparse_bytes=sparse_shape[0] * sparse_shape[1] * 2, dense_bytes=dense_shape[0] * dense_shape[1] * 2)  # fmt: skip
    base = res["phases"]["imported"]["peak_bytes"]
    res["peak_growth_bytes"] = res["peak_bytes"] - base
    text = json.dumps(res, indent=2)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
