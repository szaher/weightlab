"""POST /tiles, GET /tiles/{cache_key}/metadata and /jobs over HTTP (doc 09 sections 3 and 5): every branch of the
quality decision table, binary and JSON payloads against the frozen schemas, 202 jobs with deduplication, polling,
idempotent cancellation, retries after completion, sampled labels, invalid requests, cross-project isolation, file
changes, corrupt cache entries, project deletion, restart and offline operation.

Settings are lowered to their contract minimums (direct_tile_max_mib 8, pyramid_exact_max_mib 64) so small synthetic
tensors exercise every branch: `mid` (1024 x 2304 F32, 9 MiB) needs a pyramid at L4; `big` (4096 x 4128 F32 sparse,
64.5 MiB) is above the pyramid limit."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import struct
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import PORT, TOKEN, VIEW2D, import_model, tensors_by_key
from tests.api.contract import Api, component
from tests.helpers import write_safetensors
from tests.unit.test_sampling import ref_select
from weightlab.analysis import sampling
from weightlab.api.app import create_app
from weightlab.api.config import ServerConfig

H = {"Authorization": f"Bearer {TOKEN}"}
BIN = "application/vnd.weightlab.tile.v2"
MID = (1024, 2304)
BIG = (4096, 4128)


def write_big(path: Path) -> None:
    """Sparse F32 tensor: row r holds the value r in its first 32 columns, zeros elsewhere."""
    rows, cols = BIG
    header = json.dumps({"big.weight": {"dtype": "F32", "shape": [rows, cols], "data_offsets": [0, rows * cols * 4]}})
    hb = header.encode() + b" " * (-len(header) % 8)
    base = 8 + len(hb)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        if sys.platform == "win32":
            import subprocess

            # fsutil is a Windows system tool (as in scripts/tile_benchmark.py); without it the file is just dense
            subprocess.run(["fsutil", "sparse", "setflag", str(path)], check=False, capture_output=True)  # noqa: S607
        f.truncate(base + rows * cols * 4)
        for r in range(rows):
            f.seek(base + r * cols * 4)
            f.write(np.full(32, r, "<f4").tobytes())


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tilemodels")
    rng = np.random.default_rng(11)
    mid = rng.standard_normal(MID).astype("<f4")
    mid[3, 5] = np.nan
    mid[700, 2000] = np.inf
    (root / "mid").mkdir()
    write_safetensors(root / "mid" / "model.safetensors", {"mid.weight": ("F32", MID, mid.tobytes())})
    np.save(root / "mid.npy", mid)
    (root / "big").mkdir()
    write_big(root / "big" / "model.safetensors")
    return root


class Env:
    def __init__(self, client: TestClient, api: Api, root: Path) -> None:
        self.client, self.api, self.root = client, api, root
        api.patch("/settings", json_body={"direct_tile_max_mib": 8, "pyramid_exact_max_mib": 64}, expect=200)
        self.project = api.post("/projects", json_body={"name": "tiles"}, expect=201)["id"]
        self.mid = import_model(api, self.project, root / "mid", "model.safetensors")["id"]
        self.big = import_model(api, self.project, root / "big", "model.safetensors")["id"]
        self.mid_t = tensors_by_key(api, self.mid)["mid.weight"]["id"]
        self.big_t = tensors_by_key(api, self.big)["big.weight"]["id"]

    def req(self, which: str, level: int, x: int = 0, y: int = 0, size: int = 256, agg: str = "mean",
            mode: str = "auto", **quality: Any) -> dict[str, Any]:  # fmt: skip
        model, tensor = (self.mid, self.mid_t) if which == "mid" else (self.big, self.big_t)
        return {"source": {"kind": "tensor", "model_id": model, "tensor_id": tensor}, "view": VIEW2D,
                "tile": {"level": level, "x": x, "y": y, "size": size}, "aggregation": agg,
                "quality": {"mode": mode, **quality}}  # fmt: skip

    def raw(self, body: dict[str, Any], accept: str | None = None) -> Any:
        hdrs = {**H, **({"Accept": accept} if accept else {})}
        return self.client.post("/api/v1/tiles", json=body, headers=hdrs)

    def binary(self, body: dict[str, Any]) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
        r = self.raw(body)
        assert r.status_code == 200, r.text[:500]
        assert r.headers["content-type"] == BIN
        rows, cols = int(r.headers["x-weightlab-rows"]), int(r.headers["x-weightlab-cols"])
        assert len(r.content) == rows * cols * 5
        assert int(r.headers["content-length"]) == rows * cols * 5
        values = np.frombuffer(r.content, "<f4", rows * cols).reshape(rows, cols)
        status = np.frombuffer(r.content, np.uint8, rows * cols, rows * cols * 4).reshape(rows, cols)
        assert not np.any(status & 0xE0)
        assert np.array_equal(np.isfinite(values), (status & 17) == 0)
        meta = self.api.get("/tiles/{cache_key}/metadata", cache_key=r.headers["x-weightlab-cache-key"], expect=200)
        component("TileMetadata").validate(meta)
        assert meta["quality"] == r.headers["x-weightlab-quality"]
        assert (meta["rows"], meta["cols"], meta["level"]) == (rows, cols, body["tile"]["level"])
        assert meta["source"] == body["source"]
        return r, values, status, meta

    def job(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.api.post("/tiles", json_body=body, expect=202)  # type: ignore[no-any-return]

    def finish(self, job_id: str, t: float = 60) -> dict[str, Any]:
        end = time.monotonic() + t
        while time.monotonic() < end:
            j = self.api.get("/jobs/{job_id}", job_id=job_id, expect=200)
            if j["state"] not in ("created", "queued", "running"):
                return j  # type: ignore[no-any-return]
            time.sleep(0.02)
        raise AssertionError("job did not finish")


@pytest.fixture
def env(client: TestClient, api: Api, models: Path) -> Env:
    return Env(client, api, models)


def mid_ref(models: Path, level: int, agg: str = "mean") -> np.ndarray:
    """Independent block aggregation of the dense array: each block is sliced from its real cells only (no
    padding to 2^level, which at level 20 would be terabytes)."""
    a = np.load(models / "mid.npy").astype(np.float64)
    b = 1 << level
    r, c = -(-MID[0] // b), -(-MID[1] // b)
    out = np.empty((r, c))
    for i in range(r):
        for j in range(c):
            blk = a[i * b : (i + 1) * b, j * b : (j + 1) * b]
            fin = blk[np.isfinite(blk)]
            out[i, j] = fin.mean() if agg == "mean" else fin.max()
    return out


# ---------------------------------------------------------------------------------------- decision table
def test_direct_reads_are_exact_and_row_sampling_is_refused(env: Env, models: Path) -> None:
    r, values, _status, meta = env.binary(env.req("mid", 3))  # L0-L3: always direct
    assert r.headers["x-weightlab-quality"] == "exact"
    assert "x-weightlab-sampled-rows" not in r.headers
    assert meta["sampled_rows"] is None and meta["seed"] is None  # noqa: PT018
    assert meta["finite_count"] + meta["nonfinite_count"] == 1024 * 2048  # every covered element
    np.testing.assert_allclose(values[:128, :256], mid_ref(models, 3)[:128, :256].astype(np.float32), rtol=2e-7)
    _, v2, _, _ = env.binary(env.req("mid", 3, mode="exact"))
    assert np.array_equal(values, v2, equal_nan=True)
    e = env.api.post("/tiles", json_body=env.req("mid", 3, mode="row_sampled"), expect=422)
    assert e["error"]["code"] == "VALIDATION_ERROR"
    assert "direct read" in e["error"]["message"]
    small = env.req("mid", 4, x=1, y=0, size=8)  # covered 128 x 128 cells: direct at L4 too
    assert env.binary(small)[0].headers["x-weightlab-quality"] == "exact"


def test_cold_pyramid_gives_a_deduplicated_job_then_exact_tiles(env: Env, models: Path) -> None:
    body = env.req("mid", 4)
    first = env.job(body)
    assert first["kind"] == "pyramid_build"
    assert first["deduplicated"] is False
    assert first["state"] in ("queued", "running", "completed")
    assert first["estimated_bytes"] == MID[0] * MID[1] * 4
    second = env.job({**body, "quality": {"mode": "exact"}})  # exact joins the same pending job
    if second["state"] != "completed":
        assert second["job_id"] == first["job_id"]
        assert second["deduplicated"] is True
    e = env.api.post("/tiles", json_body=env.req("mid", 4, mode="row_sampled"), expect=422)
    assert "pyramid_exact_max_mib" in e["error"]["message"]  # below the limit sampling is never offered
    done = env.finish(first["job_id"])
    assert done["state"] == "completed"
    assert done["progress"] == 1.0
    assert set(done["result"]) == {"kind", "pyramid_id"}
    r, values, status, meta = env.binary(body)
    assert r.headers["x-weightlab-quality"] == "exact"
    assert meta["finite_count"] == MID[0] * MID[1] - 2
    assert meta["nonfinite_count"] == 2
    np.testing.assert_allclose(values, mid_ref(models, 4).astype(np.float32), rtol=2.0**-22, equal_nan=True)
    assert int(status[0, 0]) == 2  # the block with the NaN: finite aggregate, HAS_NAN
    for level in (5, 9, 11, 20):  # coarser levels and above the top, all from the same pyramid
        _, v, _, m = env.binary(env.req("mid", level, agg="max"))
        np.testing.assert_array_equal(v, mid_ref(models, level, "max").astype(np.float32))
        assert m["quality"] == "exact"
    e = env.api.post("/tiles", json_body=env.req("mid", 4, mode="row_sampled"), expect=422)
    assert "pyramid is ready" in e["error"]["message"]


def test_json_tiles_are_contract_valid_and_equal_the_binary_form(env: Env) -> None:
    job = env.job(env.req("mid", 4))
    env.finish(job["job_id"])
    body = env.req("mid", 4, x=1, y=0, size=128)
    doc = env.api.post("/tiles", json_body=body, headers={"Accept": "application/json"}, expect=200)  # TileJson
    _, values, status, meta = env.binary(body)
    assert doc["status"] == status.tolist()
    assert doc["values"] == [[float(v) if s & 17 == 0 else None for v, s in zip(vr, sr, strict=True)]
                             for vr, sr in zip(values.tolist(), status.tolist(), strict=True)]  # fmt: skip
    assert doc["metadata"] == meta
    e = env.api.post("/tiles", json_body=env.req("mid", 4), headers={"Accept": "application/json"}, expect=406)
    assert e["error"]["code"] == "NOT_ACCEPTABLE"  # 64 x 144: a dimension above 128
    e = env.api.post("/tiles", json_body=body, headers={"Accept": "text/html"}, expect=406)
    assert env.raw(body, accept="application/json;q=0.5, application/vnd.weightlab.tile.v2").headers[
        "content-type"] == BIN  # fmt: skip


def test_large_sources_get_labelled_row_sampled_previews(env: Env) -> None:
    r, values, _status, meta = env.binary(env.req("big", 5))
    assert r.headers["x-weightlab-quality"] == "row_sampled"
    k = int(r.headers["x-weightlab-sampled-rows"])
    assert meta["sampled_rows"] == k == 2  # the default (8) reduced to fit the 8 MiB sampling budget
    assert meta["sampling_policy"] == "stride"
    assert meta["seed"] is None
    assert meta["finite_count"] + meta["nonfinite_count"] == 128 * k * BIG[1] < 4096 * BIG[1]
    assert any("not exact" in w for w in meta["warnings"])
    expected = [np.mean([32 * i + o for o in ref_select(32, k, None, 0)]) for i in range(128)]
    np.testing.assert_array_equal(values[:, 0], np.float32(expected))  # column block 0 holds the row index
    assert not values[:, 1:].any()
    seeded = env.req("big", 5, mode="row_sampled", seed=7, sampled_rows_per_block=3)
    _, v7, _, m7 = env.binary(seeded)
    assert (m7["sampling_policy"], m7["seed"], m7["sampled_rows"]) == ("splitmix64_rows", 7, 3)
    _, v7b, _, _ = env.binary(seeded)
    assert np.array_equal(v7, v7b)  # reproducible
    _, v8, _, _ = env.binary(env.req("big", 5, mode="row_sampled", seed=8, sampled_rows_per_block=3))
    assert not np.array_equal(v7, v8)  # another seed, other rows
    pid = m7["cache_key"]
    assert pid != meta["cache_key"]
    exact = env.job(env.req("big", 5, mode="exact"))  # exact is never sampled: it queues the pyramid
    assert exact["kind"] == "pyramid_build"
    c = env.api.post("/jobs/{job_id}/cancel", job_id=exact["job_id"], expect=200)
    assert c["state"] in ("canceled", "completed")
    if c["state"] == "canceled":
        assert c["result"] is None
        again = env.job(env.req("big", 5, mode="exact"))  # waiters are back to 202-on-request, never exact
        assert again["job_id"] != exact["job_id"]
        env.api.post("/jobs/{job_id}/cancel", job_id=again["job_id"], expect=200)
        assert env.binary(env.req("big", 5))[0].headers["x-weightlab-quality"] == "row_sampled"


def test_row_sampling_budget_refusal_is_explicit(env: Env) -> None:
    e = env.api.post("/tiles", json_body=env.req("big", 5, mode="row_sampled", sampled_rows_per_block=32),
                     expect=413)  # fmt: skip
    assert e["error"]["code"] == "RESOURCE_LIMIT"


# ----------------------------------------------------------------------------------------------- jobs API
def test_jobs_api_polls_cancels_idempotently_and_validates(env: Env) -> None:
    budget = {"max_working_set_mib": 512, "max_threads": 1, "max_wall_seconds": 120}
    src = {"kind": "tensor", "model_id": env.mid, "tensor_id": env.mid_t}
    j = env.api.post("/jobs", json_body={"kind": "pyramid_build", "source": src, "view": VIEW2D, "budget": budget,
                                          "consistency": "verified"}, expect=202)  # fmt: skip
    assert j["estimated_bytes"] >= MID[0] * MID[1] * 4 * 3  # the scan plus two full hashes of the file
    done = env.finish(j["job_id"])
    assert done["state"] == "completed"
    assert done["bytes_scanned"] >= MID[0] * MID[1] * 4 * 3
    again = env.api.post("/jobs/{job_id}/cancel", job_id=j["job_id"], expect=200)
    assert again["state"] == "completed"  # cancel after completion changes nothing
    assert again["result"] == done["result"]
    over = env.api.post("/jobs", json_body={"kind": "pyramid_build", "source": src, "view": VIEW2D,
                                             "budget": {**budget, "max_threads": 64}}, expect=422)  # fmt: skip
    assert over["error"]["code"] == "RESOURCE_LIMIT"
    assert over["error"]["details"]["max_threads"] == 2
    for kind in ("tensor_stats", "compare_stats"):
        e = env.api.post("/jobs", json_body={"kind": kind, "source": src, "budget": budget}, expect=422)
        assert "not available" in e["error"]["message"]
    e = env.api.post("/jobs", json_body={"kind": "file_hash", "model_id": env.mid, "budget": budget}, expect=422)
    assert "not available" in e["error"]["message"]
    e = env.api.post("/jobs", json_body={"kind": "pyramid_build", "source": src, "budget": budget}, expect=422)
    assert "view" in e["error"]["message"]
    env.api.post("/jobs", json_body={"kind": "pyramid_build", "source": src, "view": VIEW2D, "budget": budget,
                                      "quality": {}}, expect=422)  # fmt: skip
    env.api.get("/jobs/{job_id}", job_id="job_doesnotexist", expect=404)
    env.api.post("/jobs/{job_id}/cancel", job_id="job_doesnotexist", expect=404)


def test_concurrent_identical_tile_requests_create_one_job(env: Env) -> None:
    body = env.req("big", 6, mode="exact")
    out: list[Any] = []
    threads = [threading.Thread(target=lambda: out.append(env.raw(body))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    jobs = [r.json() for r in out]
    assert all(r.status_code == 202 for r in out)
    for j in jobs:
        component("Job").validate(j)
    assert len({j["job_id"] for j in jobs}) == 1
    assert sum(j["deduplicated"] for j in jobs) == 7
    env.api.post("/jobs/{job_id}/cancel", job_id=jobs[0]["job_id"], expect=200)


# ----------------------------------------------------------------------------------- invalid and isolated
@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"tile": {"level": 53, "x": 0, "y": 0, "size": 256}}, 422),
        ({"tile": {"level": 0, "x": 0, "y": 0, "size": 512}}, 422),
        ({"tile": {"level": 0, "x": 2**53, "y": 0, "size": 256}}, 422),
        ({"tile": {"level": 0, "x": 99, "y": 0, "size": 256}}, 422),  # outside the view: INVALID_ROI
        ({"quality": {"mode": "per_cell"}}, 422),
        ({"quality": {}}, 422),
        ({"aggregation": "median"}, 422),
        ({"extra": 1}, 422),
        ({"view": {"row_axis": 0, "col_axis": 0, "fixed_indices": {}, "transpose": False}}, 422),
    ],
)
def test_invalid_tile_requests_get_the_error_envelope(env: Env, change: dict[str, Any], code: int) -> None:
    e = env.api.post("/tiles", json_body={**env.req("mid", 0), **change}, expect=code)
    assert e["error"]["code"] in ("VALIDATION_ERROR", "INVALID_ROI")


def test_derived_sources_are_refused_not_substituted(env: Env) -> None:
    body = {**env.req("mid", 0), "source": {"kind": "delta", "comparison_id": "cmp_x", "entry_id": "ent_x",
                                            "transform": "b_minus_a"}}  # fmt: skip
    e = env.api.post("/tiles", json_body=body, expect=422)
    assert "not available" in e["error"]["message"]


def test_cross_project_ids_are_not_found(env: Env, models: Path) -> None:
    other = env.api.post("/projects", json_body={"name": "other"}, expect=201)["id"]
    om = import_model(env.api, other, models / "mid", "model.safetensors")["id"]
    ot = tensors_by_key(env.api, om)["mid.weight"]["id"]
    mixed = {**env.req("mid", 0), "source": {"kind": "tensor", "model_id": env.mid, "tensor_id": ot}}
    if ot != env.mid_t:
        env.api.post("/tiles", json_body=mixed, expect=404)
    job = {"kind": "pyramid_build", "source": mixed["source"], "view": VIEW2D,
           "budget": {"max_working_set_mib": 512, "max_threads": 1, "max_wall_seconds": 60}}  # fmt: skip
    if ot != env.mid_t:
        env.api.post("/jobs", json_body=job, expect=404)
    env.api.post("/tiles", json_body={**mixed, "source": {**mixed["source"], "model_id": "mod_nothere"}}, expect=404)


# ----------------------------------------------------------------------------- changes, corruption, deletion
def test_a_changed_source_is_never_served_from_old_results(env: Env, models: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "mid2"
    shutil.copytree(models / "mid", copy)
    m = import_model(env.api, env.project, copy, "model.safetensors")["id"]
    t = tensors_by_key(env.api, m)["mid.weight"]["id"]
    body = {**env.req("mid", 4), "source": {"kind": "tensor", "model_id": m, "tensor_id": t}}
    env.finish(env.job(body)["job_id"])
    env.binary(body)
    st = (copy / "model.safetensors").stat()
    os.utime(copy / "model.safetensors", ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    e = env.api.post("/tiles", json_body=body, expect=409)
    assert e["error"]["code"] == "FILE_CHANGED"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows ctime is the creation time: doc 03's documented limit")
def test_a_same_size_rewrite_with_mtime_restored_is_file_changed(env: Env, models: Path, tmp_path: Path) -> None:
    """Doc 03: only a same-size rewrite preserving BOTH mtime_ns and ctime_ns may go undetected. Rewriting one
    payload byte in place and restoring mtime_ns still moves ctime_ns, so cached tiles, the pyramid and ROI reads
    of the old bytes must all be refused (found by the real-checkpoint validation, PR #9)."""
    import shutil

    copy = tmp_path / "mid3"
    shutil.copytree(models / "mid", copy)
    path = copy / "model.safetensors"
    m = import_model(env.api, env.project, copy, "model.safetensors")["id"]
    t = tensors_by_key(env.api, m)["mid.weight"]["id"]
    src = {"kind": "tensor", "model_id": m, "tensor_id": t}
    body = {**env.req("mid", 4), "source": src}
    env.finish(env.job(body)["job_id"])
    env.binary(body)  # served from the pyramid
    direct = {**env.req("mid", 0), "source": src}
    env.binary(direct)  # direct tile, now in the memory tile cache
    roi = {"source": src, "view": VIEW2D, "roi": {"r0": 0, "r1": 2, "c0": 0, "c1": 2}, "value_encoding": "auto"}
    env.api.post("/tensors/roi", json_body=roi, expect=200)

    before = path.stat()
    time.sleep(0.02)
    n = int.from_bytes(path.read_bytes()[:8], "little")
    with path.open("r+b") as f:  # same size, same inode: one payload byte flipped in place
        f.seek(8 + n)
        b = f.read(1)
        f.seek(8 + n)
        f.write(bytes([b[0] ^ 0x01]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns, after.st_ino) == (before.st_size, before.st_mtime_ns, before.st_ino)
    if after.st_ctime_ns == before.st_ctime_ns:
        pytest.skip("this filesystem did not move ctime_ns")

    for req in (body, direct):
        e = env.api.post("/tiles", json_body=req, expect=409)
        assert e["error"]["code"] == "FILE_CHANGED"
    e = env.api.post("/tensors/roi", json_body=roi, expect=409)
    assert e["error"]["code"] == "FILE_CHANGED"
    listed = env.api.get("/projects/{project_id}/models", project_id=env.project, expect=200)
    rows = listed["items"] if isinstance(listed, dict) else listed
    assert next(x for x in rows if x["id"] == m)["status"] == "changed"


def test_rows_imported_before_ctime_was_recorded_keep_working(env: Env, config: ServerConfig) -> None:
    """Migration 0002 leaves ctime_ns NULL for earlier imports: those models still open (size, mtime_ns and file ID
    are compared as before) instead of all turning FILE_CHANGED after an upgrade."""
    import sqlite3

    c = sqlite3.connect(config.data_dir / "weightlab.sqlite3")
    with c:
        c.execute("UPDATE model_files SET ctime_ns = NULL WHERE model_id = ?", (env.mid,))
    c.close()
    env.binary(env.req("mid", 0))


def test_a_corrupt_cached_pyramid_is_rebuilt_not_served(env: Env, config: ServerConfig) -> None:
    body = env.req("mid", 4)
    env.finish(env.job(body)["job_id"])
    files = list((config.data_dir / "cache" / "pyramids").glob("*.wlpyr"))
    assert len(files) == 1
    data = bytearray(files[0].read_bytes())
    data[-20] ^= 0xFF
    files[0].write_bytes(bytes(data))
    rebuilt = env.job(body)  # missing again: a new build, never the corrupt data
    assert env.finish(rebuilt["job_id"])["state"] == "completed"
    assert env.binary(body)[3]["quality"] == "exact"


def test_project_deletion_purges_its_cache_and_never_touches_checkpoints(env: Env, config: ServerConfig) -> None:
    env.finish(env.job(env.req("mid", 4))["job_id"])
    src = env.root / "mid" / "model.safetensors"
    before = (hashlib.sha256(src.read_bytes()).hexdigest(), src.stat().st_mtime_ns)
    running = env.job(env.req("big", 7, mode="exact"))
    cache = config.data_dir / "cache" / "pyramids"
    assert list(cache.glob("*.wlpyr"))
    env.api.delete("/projects/{project_id}", project_id=env.project, expect=204)
    assert not list(cache.glob("*.wlpyr"))
    assert not list(cache.glob("*.tmp-*"))
    assert (hashlib.sha256(src.read_bytes()).hexdigest(), src.stat().st_mtime_ns) == before
    env.api.get("/jobs/{job_id}", job_id=running["job_id"], expect=404)  # deleted with the project


def test_pyramids_survive_a_restart_and_interrupted_jobs_never_claim_exact(models: Path, tmp_path: Path) -> None:
    cfg = ServerConfig(port=PORT, data_dir=tmp_path / "data", token=TOKEN)
    with TestClient(create_app(cfg), base_url=f"http://127.0.0.1:{PORT}") as c:
        env = Env(c, Api(c, TOKEN), models)
        env.finish(env.job(env.req("mid", 4))["job_id"])
        long = env.job(env.req("big", 8, mode="exact"))
        project, big, big_t, mid, mid_t = env.project, env.big, env.big_t, env.mid, env.mid_t
    with TestClient(create_app(cfg), base_url=f"http://127.0.0.1:{PORT}") as c:
        api = Api(c, TOKEN)
        j = api.get("/jobs/{job_id}", job_id=long["job_id"], expect=200)
        assert j["state"] in ("interrupted", "completed", "queued", "running")
        assert (j["result"] is None) == (j["state"] != "completed")
        warm = {"source": {"kind": "tensor", "model_id": mid, "tensor_id": mid_t}, "view": VIEW2D,
                "tile": {"level": 4, "x": 0, "y": 0, "size": 256}, "aggregation": "mean",
                "quality": {"mode": "auto"}}  # fmt: skip
        r = c.post("/api/v1/tiles", json=warm, headers=H)
        assert r.status_code == 200  # the disk pyramid outlived the process
        assert r.headers["x-weightlab-quality"] == "exact"
        if j["state"] == "interrupted":
            cold = {**warm, "source": {"kind": "tensor", "model_id": big, "tensor_id": big_t},
                    "tile": {"level": 8, "x": 0, "y": 0, "size": 256}, "quality": {"mode": "exact"}}  # fmt: skip
            again = api.post("/tiles", json_body=cold, expect=202)
            assert again["job_id"] != long["job_id"]
            api.post("/jobs/{job_id}/cancel", job_id=again["job_id"], expect=200)
        assert project


def test_tiles_and_jobs_work_offline(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*a: Any, **k: Any) -> Any:
        raise OSError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    env.finish(env.job(env.req("mid", 4))["job_id"])
    env.binary(env.req("mid", 4))
    env.binary(env.req("big", 5))


def test_sampling_selection_is_the_documented_function() -> None:
    assert sampling.select_rows(32, 2, policy="stride") == ref_select(32, 2, None, 0) == [8, 24]


@pytest.fixture
def no_threads_left() -> Iterator[None]:
    before = {t.name for t in threading.enumerate()}
    yield
    time.sleep(0.05)
    stray = {t.name for t in threading.enumerate() if t.name.startswith("weightlab-job")} - before
    assert not stray


def test_app_shutdown_leaves_no_worker_threads(models: Path, tmp_path: Path, no_threads_left: None) -> None:
    cfg = ServerConfig(port=PORT, data_dir=tmp_path / "data", token=TOKEN)
    with TestClient(create_app(cfg), base_url=f"http://127.0.0.1:{PORT}") as c:
        env = Env(c, Api(c, TOKEN), models)
        env.job(env.req("big", 8, mode="exact"))


# ----------------------------------------------------------------------------- PR #7 review: placement and sharing
def _job_body(model: str, tensor: str, mib: int) -> dict[str, Any]:
    budget = {"max_working_set_mib": mib, "max_threads": 1, "max_wall_seconds": 60}
    return {"kind": "pyramid_build", "source": {"kind": "tensor", "model_id": model, "tensor_id": tensor},
            "view": VIEW2D, "budget": budget}  # fmt: skip


def _finished(api: Api, job_id: str, t: float = 60) -> dict[str, Any]:
    end = time.monotonic() + t
    while time.monotonic() < end:
        j = api.get("/jobs/{job_id}", job_id=job_id, expect=200)
        if j["state"] not in ("created", "queued", "running"):
            return j  # type: ignore[no-any-return]
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_low_disk_never_lets_a_job_exceed_its_memory_budget(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pyramid fits tile_cache_disk_mib, so the job is admitted for disk placement; at run time the disk has
    no free space. Holding the pyramid in memory instead would exceed max_working_set_mib: the job must fail with
    RESOURCE_LIMIT and allocate nothing, not silently build a session-memory pyramid."""
    from weightlab import open_checkpoint
    from weightlab.analysis import pyramid as P
    from weightlab.cache import PyramidCache

    shape = (8192, 8192)  # BF16, sparse zeros: a ~19 MiB pyramid, so memory placement cannot fit a 64 MiB budget
    header = json.dumps(
        {"z.weight": {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, shape[0] * shape[1] * 2]}}
    )
    hb = header.encode() + b" " * (-len(header) % 8)
    (tmp_path / "z").mkdir()
    with (tmp_path / "z" / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        f.truncate(8 + len(hb) + shape[0] * shape[1] * 2)
    model = import_model(env.api, env.project, tmp_path / "z", "model.safetensors")["id"]
    tensor = tensors_by_key(env.api, model)["z.weight"]["id"]
    with open_checkpoint(tmp_path / "z" / "model.safetensors") as ck:
        ws = P.build_working_set(*shape, "rows", ck.limits.read_chunk_bytes)
    nbytes = P.estimate_bytes(*shape)
    tight = max(64, -(-ws // (1024 * 1024)) + 1)  # 64 MiB is the contract minimum
    assert ws <= tight * 1024 * 1024 < ws + nbytes  # disk placement fits the budget, memory placement does not
    monkeypatch.setattr(PyramidCache, "_free_space_ok", lambda self, n: False)
    svc = env.client.app.app.state.tiles  # type: ignore[attr-defined]  # inside the security Guard

    job = env.api.post("/jobs", json_body=_job_body(model, tensor, tight), expect=202)
    done = _finished(env.api, job["job_id"])
    assert done["state"] == "failed", done
    assert done["error"]["code"] == "RESOURCE_LIMIT"
    assert "memory budget" in done["error"]["message"]
    assert done["result"] is None
    st = svc.pyramids.stats()
    assert (st.session_entries, st.disk_entries, st.disk_reserved) == (0, 0, 0)

    roomy = env.api.post("/jobs", json_body=_job_body(model, tensor, 512), expect=202)
    assert _finished(env.api, roomy["job_id"])["state"] == "completed"  # re-admitted with the pyramid's size
    st = svc.pyramids.stats()
    assert (st.session_entries, st.disk_entries) == (1, 0)


def test_two_projects_one_checkpoint_two_workers_share_one_build(
    models: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pyramids are content-addressed, so two projects importing the same checkpoint need the same pyramid. With
    two workers both jobs run at once: the second waits for the first build instead of failing with CONFLICT, both
    complete, one build ran and both projects own the result. Deleting the building project cancels its job; the
    other project's job then builds the pyramid itself."""
    import weightlab.api.tiles as tiles_mod

    real = tiles_mod.pyr.build_pyramid
    calls: list[str] = []
    release = threading.Event()

    def gated(*a: Any, **k: Any) -> Any:
        ctl = a[6] if len(a) > 6 else k["control"]
        calls.append(ctl.job_id)
        if len(calls) == 1:  # hold the first build until the test has both jobs running side by side
            while not release.wait(0.01):
                ctl.check()
        return real(*a, **k)

    monkeypatch.setattr(tiles_mod.pyr, "build_pyramid", gated)
    cfg = ServerConfig(port=PORT, data_dir=tmp_path / "data", token=TOKEN)
    with TestClient(create_app(cfg), base_url=f"http://127.0.0.1:{PORT}") as c:
        Api(c, TOKEN).patch("/settings", json_body={"max_heavy_jobs": 2}, expect=200)
    for delete_builder in (False, True):
        calls.clear()
        release.clear()
        with TestClient(create_app(cfg), base_url=f"http://127.0.0.1:{PORT}") as c:  # two workers from the start
            api = Api(c, TOKEN)
            envs = [Env(c, api, models) for _ in range(2)]
            assert envs[0].project != envs[1].project
            svc = c.app.app.state.tiles  # type: ignore[attr-defined]
            jobs = [e.job(e.req("mid", 4))["job_id"] for e in envs]
            assert jobs[0] != jobs[1]  # separate jobs: each project owns, sees and cancels only its own
            end = time.monotonic() + 10
            while time.monotonic() < end and not (
                calls and all(api.get("/jobs/{job_id}", job_id=j, expect=200)["state"] == "running" for j in jobs)
            ):
                time.sleep(0.01)
            time.sleep(0.3)  # the second job reaches the reservation while the first build holds it
            states = [api.get("/jobs/{job_id}", job_id=j, expect=200)["state"] for j in jobs]
            assert states == ["running", "running"], states  # aff7a28: the second one failed with CONFLICT
            builder = 0 if calls[0] == jobs[0] else 1
            other = 1 - builder
            if delete_builder:
                api.delete("/projects/{project_id}", project_id=envs[builder].project, expect=204)
                done = _finished(api, jobs[other])
                assert done["state"] == "completed", done  # the surviving project built it itself
                assert calls[1] == jobs[other]
                owners = svc.pyramids.stats().owners
                assert [sorted(v) for v in owners.values()] == [[envs[other].project]]
            else:
                release.set()
                done = [_finished(api, j) for j in jobs]
                assert [d["state"] for d in done] == ["completed", "completed"], done
                assert done[0]["result"] == done[1]["result"]
                assert len(calls) == 1  # one build, shared
                owners = svc.pyramids.stats().owners
                assert [sorted(v) for v in owners.values()] == [sorted(e.project for e in envs)]
            for e in envs if not delete_builder else [envs[other]]:
                _, _, _, meta = e.binary(e.req("mid", 4))
                assert meta["quality"] == "exact"
            release.set()
            for e in envs:  # independent data dir per round: clean up so the next round builds again
                with contextlib.suppress(AssertionError):
                    api.delete("/projects/{project_id}", project_id=e.project, expect=204)
