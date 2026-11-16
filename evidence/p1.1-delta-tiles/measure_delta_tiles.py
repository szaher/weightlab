#!/usr/bin/env python3
"""Measure delta heatmap tiles (ROADMAP P1.1) on the pinned real checkpoints and check them against NumPy.

    uv run --with psutil python evidence/p1.1-delta-tiles/measure_delta_tiles.py \
        --models ~/weightlab-models --out evidence/p1.1-delta-tiles/results

Uses a real `weightlab start` process (fresh data directory, loopback, bearer token; the `Server` of
evidence/alpha2-compare/measure_real_compare.py) and the checkpoints fetched by
evidence/phase1-real-models/fetch_models.py (checked against their pinned SHA-256 first). For each base/Instruct pair:
create a comparison, then for the embedding (the largest matched matrix) and one MLP projection:

* a direct L3 tile (exact, 2048 x 2048 cells of both inputs);
* a cold tile that needs the exact delta pyramid: the 202 `pyramid_build` job (best effort) with wall time,
  `estimated_bytes` and the server's peak RSS while it runs;
* the exact tile from the finished pyramid (JSON), checked cell by cell against block means of B - A computed here
  from the raw file bytes (own BF16 decoder, float64, streamed by row block).

`direct_tile_max_mib` is set to its minimum (8) so the MLP projection also takes the pyramid path. RSS is sampled
every 50 ms with psutil: an observation on this machine, not a memory guarantee.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alpha2-compare"))
from measure_real_compare import MIB, PAIRS, Server, log, sha256

BUDGET = {"max_working_set_mib": 1024, "max_threads": 1, "max_wall_seconds": 1800}
KEYS = ["model.embed_tokens.weight", "model.layers.0.mlp.down_proj.weight"]
VIEW = {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}


def header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def ref_block_means(pa: Path, pb: Path, key: str, level: int) -> np.ndarray:
    """Block means of B - A over 2^level x 2^level blocks, from the file bytes (BF16/F32), streamed by block row."""
    (ha, oa), (hb, ob) = header(pa), header(pb)
    ea, eb = ha[key], hb[key]
    rows, cols = ea["shape"]
    blk = 1 << level

    def rows_of(path: Path, e: dict[str, Any], base: int, r0: int, r1: int) -> np.ndarray:
        size = {"BF16": 2, "F16": 2, "F32": 4}[e["dtype"]]
        with path.open("rb") as f:
            f.seek(base + e["data_offsets"][0] + r0 * cols * size)
            raw = f.read((r1 - r0) * cols * size)
        if e["dtype"] == "BF16":
            v = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            v = np.frombuffer(raw, "<f2" if e["dtype"] == "F16" else "<f4")
        return v.astype(np.float64).reshape(r1 - r0, cols)

    out = np.full((-(-rows // blk), -(-cols // blk)), np.nan)
    for i, r0 in enumerate(range(0, rows, blk)):
        r1 = min(rows, r0 + blk)
        d = rows_of(pb, eb, ob, r0, r1) - rows_of(pa, ea, oa, r0, r1)
        for j, c0 in enumerate(range(0, cols, blk)):
            x = d[:, c0 : c0 + blk]
            x = x[np.isfinite(x)]
            if x.size:
                out[i, j] = x.mean()
    return out


def tile_body(src: dict[str, Any], level: int, mode: str = "auto") -> dict[str, Any]:
    return {"source": src, "view": VIEW, "tile": {"level": level, "x": 0, "y": 0, "size": 256},
            "aggregation": "mean", "quality": {"mode": mode}}  # fmt: skip


def json_level(rows: int, cols: int) -> int:
    level = 4
    while -(-rows // (1 << level)) > 128 or -(-cols // (1 << level)) > 128:
        level += 1
    return level


def timed_tile(s: Server, body: dict[str, Any]) -> tuple[int, int, float, float]:
    """A binary tile (the UI's format; JSON is limited to 128 x 128 cells): status, body bytes, seconds, peak RSS."""
    s.peak = s.rss()
    t0 = time.monotonic()
    req = urllib.request.Request(
        f"http://127.0.0.1:{s.port}/api/v1/tiles", data=json.dumps(body).encode(), method="POST",
        headers={"Host": f"127.0.0.1:{s.port}", "Authorization": f"Bearer {s.token}",
                 "Content-Type": "application/json", "Accept": "application/vnd.weightlab.tile.v2"},
    )  # fmt: skip
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            st, n = r.status, len(r.read())
    except urllib.error.HTTPError as e:
        st, n = e.code, len(e.read())
    return st, n, round(time.monotonic() - t0, 3), round(s.peak / MIB, 1)


def build(s: Server, src: dict[str, Any], level: int) -> dict[str, Any]:
    """The cold tile request (202) and its pyramid_build job, polled to completion."""
    s.peak = s.rss()
    before = s.rss()
    t0 = time.monotonic()
    job = s.json("POST", "/tiles", tile_body(src, level), expect=202)
    while job["state"] in ("created", "queued", "running"):
        time.sleep(0.1)
        job = s.json("GET", f"/jobs/{job['job_id']}")
    if job["state"] != "completed":
        raise AssertionError(f"pyramid_build ended {job['state']}: {job.get('error')}")
    return {"consistency": "best_effort", "wall_seconds": round(time.monotonic() - t0, 2),
            "estimated_bytes": job["estimated_bytes"], "rss_before_mib": round(before / MIB, 1),
            "peak_rss_mib": round(s.peak / MIB, 1), "level_requested": level}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    models: Path = args.models.expanduser().resolve()
    fetched = {(e["model"], e["file"]): e for e in json.loads((models / "fetched.json").read_text())}
    names = sorted({n for p in PAIRS for n in p})
    for n in names:
        log(f"verifying {n} SHA-256")
        if sha256(models / n / "model.safetensors") != fetched[(n, "model.safetensors")]["sha256"]:
            raise SystemExit(f"{n}: SHA-256 differs from the pinned manifest")
    results: dict[str, Any] = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
            "python": platform.python_version(),
        },
        "budget": BUDGET,
        "settings": {"direct_tile_max_mib": 8},
        "pairs": [],
    }
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="wl-delta-") as tmp:
        s = Server(Path(tmp) / "data")
        try:
            results["idle_rss_mib"] = round(s.rss() / MIB, 1)
            results["weightlab_version"] = s.json("GET", "/health").get("version")
            s.json("PATCH", "/settings", {"direct_tile_max_mib": 8})
            project = s.json("POST", "/projects", {"name": "real-delta"}, expect=201)["id"]
            ids = {}
            for n in names:
                f = models / n / "model.safetensors"
                body = {"entry_path": str(f), "selected_root": str(f.parent), "display_name": n}
                ids[n] = s.json("POST", f"/projects/{project}/models", body, expect=201)["id"]
            for a, b in PAIRS:
                log(f"pair {a} -> {b}")
                cmp_ = s.json("POST", "/comparisons", {"project_id": project, "model_a_id": ids[a],
                                                       "model_b_id": ids[b], "user_attestation": False,
                                                       "policy": "descriptive_only"}, expect=201)  # fmt: skip
                items = s.json("GET", f"/comparisons/{cmp_['id']}/entries?limit=1000")["items"]
                pair: dict[str, Any] = {"a": a, "b": b, "class": cmp_["lineage"]["compatibility_class"],
                                        "tensors": []}  # fmt: skip
                for key in KEYS:
                    e = next(x for x in items if x["raw_key"] == key)
                    src = {"kind": "delta", "comparison_id": cmp_["id"], "entry_id": e["entry_id"],
                           "transform": "b_minus_a"}  # fmt: skip
                    rows, cols = e["shape_a"]
                    t: dict[str, Any] = {"key": key, "shape": [rows, cols], "dtypes": [e["dtype_a"], e["dtype_b"]]}
                    st, nbytes, sec, peak = timed_tile(s, tile_body(src, 3))
                    t["direct_l3"] = {"status": st, "body_bytes": nbytes, "seconds": sec, "peak_rss_mib": peak}
                    if st != 200:
                        failures.append(f"{a} {key}: direct L3 tile gave {st}")
                    level = json_level(rows, cols)
                    t["build"] = m = build(s, src, level)
                    log(f"  {key}: build {m['wall_seconds']} s, peak RSS {m['peak_rss_mib']} MiB")
                    doc = s.json("POST", "/tiles", tile_body(src, level))
                    meta = doc["metadata"]
                    got = np.array([[np.nan if v is None else v for v in row] for row in doc["values"]])
                    want = ref_block_means(models / a / "model.safetensors", models / b / "model.safetensors", key,
                                           level)  # fmt: skip
                    ok = np.allclose(got, want.astype(np.float32), rtol=2e-6, atol=0, equal_nan=True)
                    worst = float(np.nanmax(np.abs(got - want) / np.maximum(np.abs(want), 1e-30)))
                    t["exact_tile"] = {"level": level, "quality": meta["quality"], "cells": got.size,
                                       "finite_count": meta["finite_count"],
                                       "invalid_pair_count": meta["invalid_pair_count"],
                                       "matches_numpy": bool(ok), "max_rel_error": worst}  # fmt: skip
                    if not ok or meta["quality"] != "exact":
                        failures.append(f"{a} {key}: exact tile differs from NumPy (max rel {worst:.3g})")
                    log(f"  {key} L{level} exact tile vs NumPy: {'OK' if ok else 'MISMATCH'} (max rel {worst:.2g})")
                    pair["tensors"].append(t)
                results["pairs"].append(pair)
            results["final_rss_mib"] = round(s.rss() / MIB, 1)
        finally:
            s.stop()
    results["failures"] = failures
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"delta_tiles_{platform.system().lower()}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    log(f"wrote {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
