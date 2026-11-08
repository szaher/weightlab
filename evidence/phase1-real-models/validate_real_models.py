"""Phase 1 real-checkpoint validation (PR #9): WeightLab against real Qwen2.5 SafeTensors, checked by an
independent reader.

    uv run --with torch --with safetensors --with psutil \
        python evidence/phase1-real-models/validate_real_models.py \
        --models ~/weightlab-models --out evidence/phase1-real-models/results [--tier initial|all]

What it does, with a real `weightlab start` process (fresh data directory, loopback, bearer token):

1. Hashes every checkpoint (SHA-256, size, mtime) before and after: nothing may change.
2. Imports each model through the API; checks every tensor's shape and dtype against the reference reader.
3. Exact ROI: random and edge windows of real attention/MLP/embedding/norm tensors, plain and transposed, compared
   value by value (float64 identity) with the reference reader (`safetensors` + PyTorch, independent of
   WeightLab's reader). Tolerance: exact (0).
4. Tiles: several levels x all five aggregations x first/middle/last tiles, through the quality decision table
   (direct reads and pyramid jobs). Reference: float64 block aggregates of the reference tensor, rounded to float32.
   Tolerance: mean / mean_abs / rms <= 1 float32 ulp (float64 summation order), min / max exact, status bits exact.
5. Pyramids: cold build (time, bytes scanned), cold and warm tile retrieval, warm retrieval after a server restart
   (disk cache), disk footprint, and the cap after lowering tile_cache_disk_mib.
6. Row-sampled previews (pyramid_exact_max_mib lowered to 64 MiB): stride-sampled tiles against the reference
   computed from the independently selected rows (tests/unit/test_sampling.py:ref_select).
7. Job lifecycle: cancellation latency, nothing published after cancel.
8. Source mutation on an APFS clone of a checkpoint (never the original): FILE_CHANGED for tiles and ROI.
9. `weightlab diff` on base/Instruct pairs: compatibility, unmatched tensors, top differences (no causal claims).
10. Peak RSS of the server process (sampled every 50 ms) against the 1 GiB backend target.

Outputs (in --out): numerical_report.json, benchmark.json, diff_<pair>/ and a run log. Model weights are never
written to --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from tests.unit.test_sampling import ref_select  # noqa: E402  (independent sampling reference)

MIB = 1024 * 1024
F32_ULP = 2.0**-23
TILE = "application/vnd.weightlab.tile.v2"
URL_RE = re.compile(r"http://127\.0\.0\.1:(\d+)/#token=([A-Za-z0-9_-]{43,})")
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG.append(line)
    print(line, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256(path)}


# ------------------------------------------------------------------------------------------------- server
class Server:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        cmd = [
            sys.executable,
            "-c",
            "from weightlab.cli import main; raise SystemExit(main())",
            "start",
            "--port",
            "0",
            "--data-dir",
            str(data_dir),
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO)
        out = ""
        deadline = time.monotonic() + 90
        assert self.proc.stdout is not None
        while not (m := URL_RE.search(out)):
            line = self.proc.stdout.readline()
            if (not line and self.proc.poll() is not None) or time.monotonic() > deadline:
                raise RuntimeError(f"weightlab start failed:\n{out}")
            out += line
        self.port, self.token = int(m.group(1)), m.group(2)
        threading.Thread(target=self._drain, daemon=True).start()
        self.ps = psutil.Process(self.proc.pid)
        self.peak_rss = 0
        self.rss_samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        threading.Thread(target=self._sample, daemon=True).start()
        for _ in range(200):
            try:
                if self.call("GET", "/health", auth=False)[0] == 200:
                    break
            except OSError:
                time.sleep(0.1)

    def _drain(self) -> None:
        assert self.proc.stdout is not None
        for _ in self.proc.stdout:
            pass

    def _sample(self) -> None:
        t0 = time.monotonic()
        while not self._stop.is_set():
            try:
                rss = self.ps.memory_info().rss
            except psutil.Error:
                return
            self.peak_rss = max(self.peak_rss, rss)
            if not self.rss_samples or time.monotonic() - self.rss_samples[-1][0] - t0 >= 1:
                self.rss_samples.append((time.monotonic() - t0, rss))
            time.sleep(0.05)

    def rss(self) -> int:
        return int(self.ps.memory_info().rss)

    def call(
        self, method: str, path: str, body: Any = None, *, accept: str = "application/json", auth: bool = True
    ) -> tuple[int, dict[str, str], bytes]:
        data = None if body is None else json.dumps(body).encode()
        h = {"Host": f"127.0.0.1:{self.port}", "Accept": accept}
        if auth:
            h["Authorization"] = f"Bearer {self.token}"
        if data is not None:
            h["Content-Type"] = "application/json"
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/v1{path}", data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()

    def json(self, method: str, path: str, body: Any = None, expect: int | tuple[int, ...] = 200) -> Any:
        st, _, raw = self.call(method, path, body)
        ok = (expect,) if isinstance(expect, int) else expect
        if st not in ok:
            raise AssertionError(f"{method} {path} -> {st}: {raw[:500]!r}")
        return json.loads(raw) if raw else None

    def stop(self) -> None:
        self._stop.set()
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# ------------------------------------------------------------------------------------------------- reference
class Reference:
    """The independent reader: safetensors' own parser + PyTorch BF16 -> float64 conversion."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._f = safe_open(str(path), "pt")
        self._cache: dict[str, np.ndarray] = {}

    def keys(self) -> list[str]:
        return list(self._f.keys())

    def meta(self, key: str) -> tuple[list[int], str]:
        s = self._f.get_slice(key)
        return list(s.get_shape()), s.get_dtype()

    def f64(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache = {key: self._f.get_tensor(key).to(torch.float64).numpy()}  # keep one tensor in memory
        return self._cache[key]


def view2d(a: np.ndarray, transpose: bool = False) -> np.ndarray:
    v = a.reshape(1, -1) if a.ndim == 1 else a
    return v.T if transpose else v


def ref_blocks(region: np.ndarray, block: int, agg: str) -> np.ndarray:
    """float64 aggregates of every block (edge blocks clipped), rounded once to float32."""
    rows = math.ceil(region.shape[0] / block)
    cols = math.ceil(region.shape[1] / block)
    pad = np.full((rows * block, cols * block), np.nan)
    pad[: region.shape[0], : region.shape[1]] = region
    b = pad.reshape(rows, block, cols, block).transpose(0, 2, 1, 3).reshape(rows, cols, block * block)
    with np.errstate(all="ignore"):
        if agg == "mean":
            out = np.nanmean(b, axis=2)
        elif agg == "mean_abs":
            out = np.nanmean(np.abs(b), axis=2)
        elif agg == "rms":
            out = np.sqrt(np.nanmean(b * b, axis=2))
        elif agg == "min":
            out = np.nanmin(b, axis=2)
        else:
            out = np.nanmax(b, axis=2)
    return out.astype(np.float32)


def ulps(got: np.ndarray, ref: np.ndarray) -> float:
    g = got.astype(np.float64)
    r = ref.astype(np.float64)
    sp = np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64)
    sp[sp == 0] = np.finfo(np.float32).smallest_subnormal
    return float(np.max(np.abs(g - r) / sp)) if g.size else 0.0


# ------------------------------------------------------------------------------------------------- checks
class Run:
    def __init__(self, server: Server) -> None:
        self.s = server
        self.report: dict[str, Any] = {"roi": [], "tiles": [], "sampled": [], "shape_dtype": {}, "failures": []}
        self.bench: dict[str, Any] = {"pyramids": [], "tile_latency_ms": {}, "import": [], "roi_latency_ms": []}

    def fail(self, what: str) -> None:
        log(f"FAIL {what}")
        self.report["failures"].append(what)

    def view(self, rank: int, transpose: bool = False) -> dict[str, Any]:
        if rank < 2:
            return {"row_axis": None, "col_axis": None, "fixed_indices": {}, "transpose": False}
        return {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": transpose}

    # -- tiles
    def tile(
        self, src: dict[str, str], view: dict[str, Any], level: int, x: int, y: int, agg: str, mode: str = "auto"
    ) -> tuple[int, dict[str, str], bytes, float]:
        t0 = time.perf_counter()
        st, h, raw = self.s.call(
            "POST",
            "/tiles",
            {
                "source": src,
                "view": view,
                "tile": {"level": level, "x": x, "y": y, "size": 256},
                "aggregation": agg,
                "quality": {"mode": mode},
            },
            accept=TILE,
        )
        return st, h, raw, (time.perf_counter() - t0) * 1000

    def wait_job(self, job: dict[str, Any], timeout: float = 1800) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            j = self.s.json("GET", f"/jobs/{job['job_id']}")
            if j["state"] not in ("created", "queued", "running"):
                return j  # type: ignore[no-any-return]
            time.sleep(0.05)
        raise AssertionError("job did not finish")

    def exact_tile(
        self, src: dict[str, str], view: dict[str, Any], level: int, x: int, y: int, agg: str, label: str
    ) -> tuple[dict[str, str], bytes]:
        st, h, raw, ms = self.tile(src, view, level, x, y, agg)
        if st == 202:
            job = json.loads(raw)
            t0 = time.perf_counter()
            done = self.wait_job(job)
            build_s = time.perf_counter() - t0
            if done["state"] != "completed":
                raise AssertionError(f"pyramid job {done['state']}: {done.get('error')}")
            st, h, raw, cold = self.tile(src, view, level, x, y, agg)
            _, _, _, warm = self.tile(src, view, level, x, y, agg)
            self.bench["pyramids"].append({"tensor": label, "transpose": view.get("transpose", False), "build_s": round(build_s, 3), "bytes_scanned": done.get("bytes_scanned"),
                                            "estimated_bytes": done.get("estimated_bytes"), "first_tile_after_build_ms": round(cold, 2), "warm_tile_ms": round(warm, 2)})  # fmt: skip
            log(
                f"  pyramid {label}: built in {build_s:.2f} s, scanned {done.get('bytes_scanned', 0) / MIB:.1f} MiB; tile {cold:.1f} ms cold, {warm:.1f} ms warm"
            )
        else:
            self.bench["tile_latency_ms"].setdefault(f"L{level}", []).append(round(ms, 2))
        if st != 200 or h.get("content-type") != TILE:
            raise AssertionError(f"tile {label} L{level} ({x},{y}) {agg}: HTTP {st} {raw[:300]!r}")
        return h, raw

    def check_tile(
        self,
        ref: Reference,
        key: str,
        src: dict[str, str],
        level: int,
        x: int,
        y: int,
        agg: str,
        transpose: bool = False,
    ) -> None:
        a = view2d(ref.f64(key), transpose)
        h, raw = self.exact_tile(src, self.view(ref.f64(key).ndim, transpose), level, x, y, agg, key)
        rows, cols = int(h["x-weightlab-rows"]), int(h["x-weightlab-cols"])
        r0, r1 = int(h["x-weightlab-row-start"]), int(h["x-weightlab-row-end"])
        c0, c1 = int(h["x-weightlab-col-start"]), int(h["x-weightlab-col-end"])
        span = 256 * 2**level
        exp = (y * span, min(a.shape[0], y * span + span), x * span, min(a.shape[1], x * span + span))
        if (r0, r1, c0, c1) != exp or h["x-weightlab-quality"] != "exact" or len(raw) != rows * cols * 5:
            self.fail(
                f"tile geometry {key} L{level} ({x},{y}): {(r0, r1, c0, c1)} vs {exp}, quality {h['x-weightlab-quality']}"
            )
            return
        vals = np.frombuffer(raw, "<f4", rows * cols).reshape(rows, cols)
        status = np.frombuffer(raw, np.uint8, rows * cols, rows * cols * 4).reshape(rows, cols)
        expect = ref_blocks(a[r0:r1, c0:c1], 2**level, agg)
        err = ulps(vals, expect) if np.all(status == 0) else float("inf")
        limit = 0.0 if agg in ("min", "max") else 1.0
        ok = err <= limit
        self.report["tiles"].append(
            {
                "tensor": key,
                "transpose": transpose,
                "level": level,
                "x": x,
                "y": y,
                "aggregation": agg,
                "cells": rows * cols,
                "max_ulps": err,
                "limit_ulps": limit,
                "ok": ok,
            }
        )
        if not ok:
            self.fail(f"tile values {key} L{level} ({x},{y}) {agg}: {err} ulps > {limit}")

    # -- ROI
    def check_roi(
        self, ref: Reference, key: str, src: dict[str, str], bounds: tuple[int, int, int, int], transpose: bool = False
    ) -> None:
        a = view2d(ref.f64(key), transpose)
        r0, r1, c0, c1 = bounds
        t0 = time.perf_counter()
        body = self.s.json(
            "POST",
            "/tensors/roi",
            {
                "source": src,
                "view": self.view(ref.f64(key).ndim, transpose),
                "roi": {"r0": r0, "r1": r1, "c0": c0, "c1": c1},
                "value_encoding": "auto",
            },
        )
        self.bench["roi_latency_ms"].append(round((time.perf_counter() - t0) * 1000, 2))
        got = np.array(body["values"], dtype=np.float64)
        exp = a[r0:r1, c0:c1]
        same = got.shape == exp.shape and bool(np.all((got == exp) & (np.signbit(got) == np.signbit(exp))))
        maxerr = float(np.max(np.abs(got - exp))) if got.shape == exp.shape else float("inf")
        rec = {"tensor": key, "transpose": transpose, "roi": [r0, r1, c0, c1], "values": int(exp.size), "max_abs_error": maxerr, "bit_exact": same,
               "precision": body["numeric_precision"], "encoding": body["value_encoding"], "dtype": body["source_dtype"]}  # fmt: skip
        self.report["roi"].append(rec)
        if not same or body["numeric_precision"] != "lossless":
            self.fail(
                f"ROI {key} {bounds} transpose={transpose}: max error {maxerr}, precision {body['numeric_precision']}"
            )


def tensors_by_key(s: Server, model_id: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    offset = 0
    while True:
        page = s.json("GET", f"/models/{model_id}/tensors?limit=200&offset={offset}")
        for t in page["items"]:
            out[t["raw_key"]] = t
        offset += len(page["items"])
        if offset >= page["total"] or not page["items"]:
            return out


def top_level(rows: int, cols: int) -> int:
    level = 0
    while 256 * 2**level < max(rows, cols):
        level += 1
    return level


def import_model(run: Run, project: str, name: str, path: Path) -> dict[str, Any]:
    s = run.s
    t0 = time.perf_counter()
    model = s.json(
        "POST", f"/projects/{project}/models", {"entry_path": str(path), "selected_root": str(path.parent)}, expect=201
    )
    import_s = time.perf_counter() - t0
    run.bench["import"].append(
        {"model": name, "seconds": round(import_s, 3), "tensor_count": model["tensor_count"], "status": model["status"]}
    )
    log(
        f"{name}: imported in {import_s:.2f} s, {model['tensor_count']} tensors, status {model['status']}, warnings {model.get('warnings')}"
    )
    if model["status"] != "ready":
        run.fail(f"{name} imported with status {model['status']}")
    return model  # type: ignore[no-any-return]


def validate_model(run: Run, model: dict[str, Any], name: str, path: Path, rng: random.Random) -> None:
    s = run.s
    ref = Reference(path)
    cat = tensors_by_key(s, model["id"])
    mism = []
    for k in ref.keys():  # noqa: SIM118 (Reference.keys is a method, not a dict)
        shape, dt = ref.meta(k)
        t = cat.get(k)
        want_dt = {"BF16": "BF16", "F16": "F16", "F32": "F32"}.get(dt, dt)
        if t is None or t["shape"] != shape or t["dtype"] != want_dt:
            mism.append(k)
    roles: dict[str, int] = {}
    for t in cat.values():
        roles[str(t["role"])] = roles.get(str(t["role"]), 0) + 1
    run.report["shape_dtype"][name] = {"tensors": len(ref.keys()), "catalog": len(cat), "mismatches": mism, "roles": roles,
                                        "dtypes": sorted({t["dtype"] for t in cat.values()})}  # fmt: skip
    if mism or len(cat) != len(ref.keys()):
        run.fail(f"{name}: catalog differs from the reference for {mism[:5]}")
    # Role filter (browse layers): the catalog query the UI uses.
    q = s.json("GET", f"/models/{model['id']}/tensors?role=attention.q&limit=200")
    log(f"  role filter attention.q: {q['total']} tensors; roles {roles}")

    layers = 1 + max(int(m.group(1)) for k in cat if (m := re.search(r"layers\.(\d+)\.", k)))
    mid = layers // 2
    picks = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        f"model.layers.{mid}.mlp.down_proj.weight",
        f"model.layers.{mid}.mlp.gate_proj.weight",
        f"model.layers.{layers - 1}.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.norm.weight",
    ]
    for key in picks:
        t = cat[key]
        src = {"kind": "tensor", "model_id": model["id"], "tensor_id": t["id"]}
        a = view2d(ref.f64(key))
        R, C = a.shape
        log(f"  {key}: {R} x {C} {t['dtype']}")
        # Exact ROI: random windows, the last corner, and (2D) a transposed window.
        for _ in range(4):
            h, w = min(64, R), min(64, C)
            r0, c0 = rng.randrange(0, R - h + 1), rng.randrange(0, C - w + 1)
            run.check_roi(ref, key, src, (r0, r0 + h, c0, c0 + w))
        run.check_roi(ref, key, src, (R - min(64, R), R, C - min(64, C), C))
        if t["rank"] == 2:
            run.check_roi(ref, key, src, (C - 32, C, R - 32, R), transpose=True)
        # Tiles: several levels, every aggregation, first / middle / last tile.
        top = top_level(R, C)
        levels = sorted({0, 3, 4, min(6, top), top} & set(range(top + 1)))
        for level in levels:
            span = 256 * 2**level
            nx, ny = math.ceil(C / span), math.ceil(R / span)
            coords = sorted({(0, 0), (nx // 2, ny // 2), (nx - 1, ny - 1)})
            for agg in ("mean", "mean_abs", "rms", "min", "max"):
                for x, y in coords:
                    run.check_tile(ref, key, src, level, x, y, agg)
        if key.endswith("q_proj.weight"):
            for agg in ("mean_abs", "max"):
                run.check_tile(ref, key, src, top_level(C, R), 0, 0, agg, transpose=True)


def sampled_checks(run: Run, model: dict[str, Any], path: Path) -> None:
    """Row-sampled previews of the embedding with pyramid_exact_max_mib lowered to the minimum (64 MiB)."""
    s = run.s
    s.json("PATCH", "/settings", {"pyramid_exact_max_mib": 64})
    ref = Reference(path)
    cat = tensors_by_key(s, model["id"])
    key = "model.embed_tokens.weight"
    a = ref.f64(key)
    src = {"kind": "tensor", "model_id": model["id"], "tensor_id": cat[key]["id"]}
    view = {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}
    for level in (8, 9, top_level(*a.shape)):
        st, h, raw, ms = run.tile(src, view, level, 0, 0, "mean")
        if st == 200 and h["x-weightlab-quality"] == "exact":
            log(f"  sampled L{level}: exact (pyramid already cached), not a sampling check")
            continue
        if st != 200 or h["x-weightlab-quality"] != "row_sampled":
            run.fail(f"sampled tile L{level}: HTTP {st} quality {h.get('x-weightlab-quality')} {raw[:200]!r}")
            continue
        k = int(h["x-weightlab-sampled-rows"])
        rows, cols = int(h["x-weightlab-rows"]), int(h["x-weightlab-cols"])
        r0, r1 = int(h["x-weightlab-row-start"]), int(h["x-weightlab-row-end"])
        c0, c1 = int(h["x-weightlab-col-start"]), int(h["x-weightlab-col-end"])
        b = 2**level
        vals = np.frombuffer(raw, "<f4", rows * cols).reshape(rows, cols)
        exp = np.empty((rows, cols), np.float32)
        for i in range(rows):
            lo = r0 + i * b
            n = min(r1, lo + b) - lo
            sel = [lo + j for j in ref_select(n, min(k, n), None, i)]
            exp[i] = [np.float32(np.mean(a[sel, c0 + j * b : min(c1, c0 + (j + 1) * b)])) for j in range(cols)]
        err = ulps(vals, exp)
        meta = s.json("GET", f"/tiles/{h['x-weightlab-cache-key']}/metadata")
        rec = {
            "tensor": key,
            "level": level,
            "sampled_rows_per_block": k,
            "policy": meta["sampling_policy"],
            "latency_ms": round(ms, 2),
            "max_ulps": err,
            "ok": err <= 1.0,
        }
        run.report["sampled"].append(rec)
        log(
            f"  sampled L{level}: {k} rows/block ({meta['sampling_policy']}), {ms:.1f} ms, {err} ulps vs reference rows"
        )
        if err > 1.0:
            run.fail(f"sampled tile L{level}: {err} ulps")
    s.json("PATCH", "/settings", {"pyramid_exact_max_mib": 1024})


def lifecycle_checks(run: Run, project: str, model: dict[str, Any], path: Path, work: Path) -> dict[str, Any]:
    s = run.s
    out: dict[str, Any] = {}
    cat = tensors_by_key(s, model["id"])
    # Cancellation: a new pyramid (transposed embedding view) canceled while it runs.
    t = cat["model.embed_tokens.weight"]
    src = {"kind": "tensor", "model_id": model["id"], "tensor_id": t["id"]}
    cache = s.data_dir / "cache" / "pyramids"
    before = {p.name for p in cache.glob("*.wlpyr")}
    job = s.json("POST", "/jobs", {"kind": "pyramid_build", "source": src, "view": {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": True},
                                   "budget": {"max_working_set_mib": 1024, "max_threads": 2, "max_wall_seconds": 600}}, expect=202)  # fmt: skip
    while (j := s.json("GET", f"/jobs/{job['job_id']}"))["state"] in ("created", "queued", "running") and j[
        "progress"
    ] < 0.05:
        time.sleep(0.01)
    t0 = time.perf_counter()
    c = s.json("POST", f"/jobs/{job['job_id']}/cancel")
    lat = (time.perf_counter() - t0) * 1000
    after = {p.name for p in cache.glob("*.wlpyr")}
    tmp = list(cache.glob("*.tmp-*"))
    out["cancel"] = {
        "state": c["state"],
        "latency_ms": round(lat, 2),
        "progress_at_cancel": c["progress"],
        "result": c["result"],
        "new_pyramids": sorted(after - before),
        "tmp_files": len(tmp),
    }
    log(
        f"  cancel: {c['state']} in {lat:.1f} ms at {c['progress']:.0%}; new pyramids {sorted(after - before)}, tmp {len(tmp)}"
    )
    if c["state"] != "canceled" or c["result"] is not None or after - before or tmp:
        run.fail(f"cancellation: {out['cancel']}")
    # Mutation on an APFS clone (or a copy) of the checkpoint: the original is never touched.
    clone_dir = work / "mutation"
    clone_dir.mkdir()
    clone = clone_dir / "model.safetensors"
    if sys.platform == "darwin":
        subprocess.run(["cp", "-c", str(path), str(clone)], check=True)
    else:
        shutil.copy2(path, clone)
    m2 = s.json(
        "POST",
        f"/projects/{project}/models",
        {"entry_path": str(clone), "selected_root": str(clone_dir), "display_name": "mutation-clone"},
        expect=201,
    )
    t2 = tensors_by_key(s, m2["id"])["model.layers.0.self_attn.q_proj.weight"]
    src2 = {"kind": "tensor", "model_id": m2["id"], "tensor_id": t2["id"]}
    st, _, _, _ = run.tile(
        src2, {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}, 0, 0, 0, "mean"
    )
    st_t = time.time() + 30
    os.utime(clone, (st_t, st_t))
    st_after, _, raw_after, _ = run.tile(
        src2, {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}, 0, 0, 0, "max"
    )
    roi_st, _, roi_raw = s.call(
        "POST",
        "/tensors/roi",
        {
            "source": src2,
            "view": {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False},
            "roi": {"r0": 0, "r1": 4, "c0": 0, "c1": 4},
            "value_encoding": "auto",
        },
    )
    code = lambda raw: json.loads(raw)["error"]["code"] if raw else None  # noqa: E731
    out["mutation"] = {
        "before": st,
        "tile_after": st_after,
        "tile_code": code(raw_after),
        "roi_after": roi_st,
        "roi_code": code(roi_raw),
    }
    log(f"  mutation (clone): tile {st} -> {st_after} {code(raw_after)}, ROI {roi_st} {code(roi_raw)}")
    if st != 200 or st_after != 409 or roi_st != 409 or code(raw_after) != "FILE_CHANGED":
        run.fail(f"mutation: {out['mutation']}")
    clone.unlink()
    return out


def disk_cap_check(run: Run) -> dict[str, Any]:
    """Lowering tile_cache_disk_mib evicts at once (LRU); the cap holds on disk."""
    s = run.s
    cache = s.data_dir / "cache" / "pyramids"
    # Disk cap: lowering tile_cache_disk_mib evicts at once.
    used = sum(p.stat().st_size for p in cache.glob("*.wlpyr"))
    cap_mib = max(1, used // MIB // 2)
    s.json("PATCH", "/settings", {"tile_cache_disk_mib": cap_mib})
    after_cap = sum(p.stat().st_size for p in cache.glob("*.wlpyr"))
    res = {"before_bytes": used, "cap_mib": cap_mib, "after_bytes": after_cap, "ok": after_cap <= cap_mib * MIB}
    log(f"  disk cap: {used / MIB:.1f} MiB -> cap {cap_mib} MiB -> {after_cap / MIB:.1f} MiB")
    if after_cap > cap_mib * MIB:
        run.fail(f"disk cap: {res}")
    s.json("PATCH", "/settings", {"tile_cache_disk_mib": 1024})
    return res


def restart_warm(data_dir: Path, project: str, model: dict[str, Any], run: Run) -> dict[str, Any]:
    """Warm retrieval from the disk pyramid cache in a new server process."""
    s2 = Server(data_dir)
    try:
        cat = tensors_by_key(s2, model["id"])
        t = cat["model.embed_tokens.weight"]
        src = {"kind": "tensor", "model_id": model["id"], "tensor_id": t["id"]}
        r2 = Run(s2)
        st, h, _, ms = r2.tile(
            src,
            {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False},
            top_level(*t["shape"]),
            0,
            0,
            "mean",
        )
        st2, h2, _, ms2 = r2.tile(
            src,
            {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False},
            top_level(*t["shape"]),
            0,
            0,
            "mean",
        )
        res = {
            "status": st,
            "quality": h.get("x-weightlab-quality"),
            "first_after_restart_ms": round(ms, 2),
            "repeat_ms": round(ms2, 2),
            "peak_rss_mib": round(s2.peak_rss / MIB, 1),
        }
        log(f"  after restart: HTTP {st} {h.get('x-weightlab-quality')} in {ms:.1f} ms (disk cache), then {ms2:.1f} ms")
        if (
            st != 200
            or h.get("x-weightlab-quality") != "exact"
            or st2 != 200
            or h2.get("x-weightlab-quality") != "exact"
        ):
            run.fail(f"restart warm retrieval: {res}")
        return res
    finally:
        s2.stop()


def run_diff(base: Path, derived: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", "from weightlab.cli import main; raise SystemExit(main())", "diff", str(base), str(derived), "--out", str(out), "--quiet", "--top", "15"],
                       capture_output=True, text=True, cwd=REPO)  # fmt: skip
    secs = time.perf_counter() - t0
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    for f in out.iterdir():  # absolute paths are not published
        if f.suffix in (".json", ".md", ".csv"):
            f.write_text(f.read_text().replace(str(base.parent.parent), "<models>"))
    return {
        "exit": p.returncode,
        "seconds": round(secs, 2),
        "stderr_tail": p.stderr[-500:],
        "summary": {k: rep[k] for k in rep if k in ("compatibility", "summary", "counts", "unmatched", "verdict")},
    }


def machine() -> dict[str, Any]:
    info: dict[str, Any] = {"os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}", "machine": platform.machine(), "python": platform.python_version(),
                            "cpu_count": os.cpu_count(), "ram_gib": round(psutil.virtual_memory().total / 2**30, 1)}  # fmt: skip
    if sys.platform == "darwin":
        info["cpu"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        d = subprocess.run(["diskutil", "info", "/"], capture_output=True, text=True).stdout
        info["storage"] = "SSD (internal, Apple Fabric)" if "Solid State:               Yes" in d else "unknown"
    info["torch"] = torch.__version__
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tier", choices=["initial", "all"], default="initial")
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args()
    models_dir = args.models.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((HERE / "models.json").read_text())
    pins = [m for m in spec["models"] if args.tier == "all" or m["tier"] == "initial"]
    pins = [m for m in pins if (models_dir / m["name"] / "model.safetensors").exists()]
    rng = random.Random(args.seed)
    before = {m["name"]: fingerprint(models_dir / m["name"] / "model.safetensors") for m in pins}
    for m in pins:
        want = m["files"]["model.safetensors"]["sha256"]
        if before[m["name"]]["sha256"] != want:
            sys.exit(f"{m['name']}: checkpoint hash {before[m['name']]['sha256']} is not the pinned {want}")
    work = Path(tempfile.mkdtemp(prefix="wl-real-"))
    data_dir = work / "data"
    t_all = time.perf_counter()
    s = Server(data_dir)
    run = Run(s)
    run.bench["idle_rss_mib_start"] = round(s.rss() / MIB, 1)
    models: dict[str, dict[str, Any]] = {}
    lifecycle: dict[str, Any] = {}
    try:
        project = s.json("POST", "/projects", {"name": "real-models"}, expect=201)["id"]
        for m in pins:
            models[m["name"]] = import_model(run, project, m["name"], models_dir / m["name"] / "model.safetensors")
        # Row-sampled previews first: no pyramid exists yet, so above the (lowered) limit Auto must sample.
        first = pins[0]["name"]
        sampled_checks(run, models[first], models_dir / first / "model.safetensors")
        for m in pins:
            log(f"{m['name']}: exact ROI and tiles")
            validate_model(run, models[m["name"]], m["name"], models_dir / m["name"] / "model.safetensors", rng)
        run.bench["disk_cache_bytes_after_validation"] = sum(
            p.stat().st_size for p in (data_dir / "cache" / "pyramids").glob("*.wlpyr")
        )
        time.sleep(2)
        run.bench["rss_mib_after_validation"] = round(s.rss() / MIB, 1)
    finally:
        run.bench["peak_rss_mib_validation"] = round(s.peak_rss / MIB, 1)
        run.bench["rss_timeline_mib"] = [(round(t, 1), round(r / MIB, 1)) for t, r in s.rss_samples]
        s.stop()
    # A new process serves the pyramids from the disk cache.
    lifecycle["restart"] = restart_warm(data_dir, "", models[first], run)
    s = Server(data_dir)
    run.s = s
    try:
        largest = max(pins, key=lambda m: m["files"]["model.safetensors"]["size"])["name"]
        lifecycle.update(
            lifecycle_checks(run, project, models[largest], models_dir / largest / "model.safetensors", work)
        )
        lifecycle["disk_cap"] = disk_cap_check(run)
        run.bench["peak_rss_mib_lifecycle"] = round(s.peak_rss / MIB, 1)
    finally:
        s.stop()
    run.bench["peak_rss_mib"] = max(
        run.bench["peak_rss_mib_validation"], run.bench["peak_rss_mib_lifecycle"], lifecycle["restart"]["peak_rss_mib"]
    )
    diffs = {}
    for a, b in (("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"), ("Qwen2.5-Math-1.5B", "Qwen2.5-Math-1.5B-Instruct")):
        if a in before and b in before:
            log(f"diff {a} vs {b}")
            diffs[f"{a}__{b}"] = run_diff(
                models_dir / a / "model.safetensors", models_dir / b / "model.safetensors", args.out / f"diff_{a}__{b}"
            )
            log(f"  exit {diffs[f'{a}__{b}']['exit']} in {diffs[f'{a}__{b}']['seconds']} s")
    after = {m["name"]: fingerprint(models_dir / m["name"] / "model.safetensors") for m in pins}
    unchanged = before == after
    if not unchanged:
        run.fail("a checkpoint changed during validation")
    shutil.rmtree(work, ignore_errors=True)

    tiles = run.report["tiles"]
    rois = run.report["roi"]
    summary = {
        "models": [m["name"] for m in pins],
        "checkpoints_unchanged": unchanged,
        "roi_windows": len(rois),
        "roi_values_compared": sum(r["values"] for r in rois),
        "roi_max_abs_error": max((r["max_abs_error"] for r in rois), default=None),
        "tiles_checked": len(tiles),
        "tile_cells_compared": sum(t["cells"] for t in tiles),
        "tile_max_ulps_mean_meanabs_rms": max(
            (t["max_ulps"] for t in tiles if t["aggregation"] in ("mean", "mean_abs", "rms")), default=None
        ),
        "tile_max_ulps_min_max": max(
            (t["max_ulps"] for t in tiles if t["aggregation"] in ("min", "max")), default=None
        ),
        "sampled_tiles": len(run.report["sampled"]),
        "failures": run.report["failures"],
        "verdict": "PASS" if not run.report["failures"] else "FAIL",
        "wall_seconds": round(time.perf_counter() - t_all, 1),
    }
    numerical = {"tolerances": {"roi": "exact: float64 identity incl. sign of zero (BF16 -> float64 is exact)",
                                "tiles_mean_mean_abs_rms": "<= 1 float32 ulp of the float64 reference rounded to float32",
                                "tiles_min_max": "exact (0 ulp)", "sampled": "<= 1 float32 ulp vs the reference on independently selected rows"},
                 "summary": summary, "fingerprints_before": before, "fingerprints_after": after, **run.report}  # fmt: skip
    bench = {"machine": machine(), "settings": "defaults (memory_budget_mib 1024, direct_tile_max_mib 64, pyramid_exact_max_mib 1024, tile_cache_disk_mib 1024)",
             **run.bench, "lifecycle": lifecycle, "diff": diffs}  # fmt: skip
    for name, lat in list(bench["tile_latency_ms"].items()):
        bench["tile_latency_ms"][name] = {
            "n": len(lat),
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "max": max(lat),
        }
    lat = bench.pop("roi_latency_ms")
    bench["roi_latency_ms"] = {
        "n": len(lat),
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
        "max": max(lat),
    }
    (args.out / "numerical_report.json").write_text(json.dumps(numerical, indent=1, default=str))
    (args.out / "benchmark.json").write_text(json.dumps(bench, indent=1, default=str))
    (args.out / "run.log").write_text("\n".join(LOG) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
