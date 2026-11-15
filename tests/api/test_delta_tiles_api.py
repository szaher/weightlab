"""Delta heatmap tiles over HTTP (ROADMAP P1.1; doc 09 sections 3-4): POST /tiles and pyramid_build jobs for
`kind: delta` sources. Every branch of the quality decision table (direct, pyramid job then exact, row-sampled above
the limit), binary and JSON payloads against the frozen schemas, the comparison's caveats and invalid pairs, the
identity (transform, epsilon, both sides' files), incompatible entries, changed inputs and cross-project isolation.

Expected values are computed here from the stored arrays with NumPy (independent of the tile engine)."""

from __future__ import annotations

import json
import os
import shutil
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import TOKEN, VIEW1D, VIEW2D, import_model
from tests.api.contract import Api, component
from tests.api.test_roi_api import compare, entries
from tests.helpers import write_safetensors

H = {"Authorization": f"Bearer {TOKEN}"}
BIN = "application/vnd.weightlab.tile.v2"
MID = (1024, 2304)
BIG = (4096, 4128)
BUDGET = {"max_working_set_mib": 512, "max_threads": 1, "max_wall_seconds": 60}


def write_sparse(path: Path, offset: float) -> None:
    """4096 x 4128 F32 (64.5 MiB): row r holds r + offset in its first 32 columns, zeros elsewhere."""
    rows, cols = BIG
    header = json.dumps({"big.weight": {"dtype": "F32", "shape": [rows, cols], "data_offsets": [0, rows * cols * 4]}})
    hb = header.encode() + b" " * (-len(header) % 8)
    base = 8 + len(hb)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        f.truncate(base + rows * cols * 4)
        for r in range(rows):
            f.seek(base + r * cols * 4)
            f.write(np.full(32, r + offset, "<f4").tobytes())


@pytest.fixture(scope="module")
def pairs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("deltamodels")
    rng = np.random.default_rng(21)
    a = rng.standard_normal(MID).astype("<f4")
    b = (a + 0.01 * rng.standard_normal(MID)).astype("<f4")
    a[3, 5] = np.nan
    b[700, 2000] = np.inf
    for name, arr in (("mid_a", a), ("mid_b", b)):
        (root / name).mkdir()
        write_safetensors(root / name / "model.safetensors", {"mid.weight": ("F32", MID, arr.tobytes())})
        np.save(root / f"{name}.npy", arr)
    for name, off in (("big_a", 0.0), ("big_b", 0.5)):
        (root / name).mkdir()
        write_sparse(root / name / "model.safetensors", off)
    return root


class Env:
    def __init__(self, client: TestClient, api: Api, root: Path) -> None:
        self.client, self.api, self.root = client, api, root
        api.patch("/settings", json_body={"direct_tile_max_mib": 8, "pyramid_exact_max_mib": 64}, expect=200)
        self.project = api.post("/projects", json_body={"name": "delta"}, expect=201)["id"]

    def pair(self, a: str, b: str, key: str, root: Path | None = None) -> dict[str, Any]:
        """A b_minus_a delta source of `key` between two imported models (and, under "_entry", its entry)."""
        root = root or self.root
        ma = import_model(self.api, self.project, root / a, "model.safetensors")
        mb = import_model(self.api, self.project, root / b, "model.safetensors")
        c = compare(self.api, self.project, ma, mb)
        e = next(x for x in entries(self.api, c["id"], limit=100)["items"] if x["raw_key"] == key)
        return {"kind": "delta", "comparison_id": c["id"], "entry_id": e["entry_id"], "transform": "b_minus_a"}

    def raw(self, body: dict[str, Any], accept: str | None = None) -> Any:
        return self.client.post("/api/v1/tiles", json=body, headers={**H, **({"Accept": accept} if accept else {})})

    def binary(self, body: dict[str, Any]) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
        r = self.raw(body)
        assert r.status_code == 200, r.text[:500]
        assert r.headers["content-type"] == BIN
        rows, cols = int(r.headers["x-weightlab-rows"]), int(r.headers["x-weightlab-cols"])
        assert len(r.content) == rows * cols * 5
        values = np.frombuffer(r.content, "<f4", rows * cols).reshape(rows, cols)
        status = np.frombuffer(r.content, np.uint8, rows * cols, rows * cols * 4).reshape(rows, cols)
        meta = self.api.get("/tiles/{cache_key}/metadata", cache_key=r.headers["x-weightlab-cache-key"], expect=200)
        component("TileMetadata").validate(meta)
        assert meta["source"] == body["source"]
        assert meta["invalid_pair_count"] == meta["nonfinite_count"]
        return r, values, status, meta

    def finish(self, job_id: str, t: float = 60) -> dict[str, Any]:
        end = time.monotonic() + t
        while time.monotonic() < end:
            j = self.api.get("/jobs/{job_id}", job_id=job_id, expect=200)
            if j["state"] not in ("created", "queued", "running"):
                return j  # type: ignore[no-any-return]
            time.sleep(0.02)
        raise AssertionError("job did not finish")


@pytest.fixture
def env(client: TestClient, api: Api, pairs: Path) -> Env:
    return Env(client, api, pairs)


def body(src: dict[str, Any], level: int, view: dict[str, Any] = VIEW2D, size: int = 256, agg: str = "mean",
         mode: str = "auto", x: int = 0, y: int = 0, **quality: Any) -> dict[str, Any]:  # fmt: skip
    return {"source": src, "view": view, "tile": {"level": level, "x": x, "y": y, "size": size}, "aggregation": agg,
            "quality": {"mode": mode, **quality}}  # fmt: skip


def block_mean(d: np.ndarray, level: int) -> np.ndarray:
    b = 1 << level
    r, c = -(-d.shape[0] // b), -(-d.shape[1] // b)
    out = np.full((r, c), np.nan)
    for i in range(r):
        for j in range(c):
            blk = d[i * b : (i + 1) * b, j * b : (j + 1) * b]
            fin = blk[np.isfinite(blk)]
            if fin.size:
                out[i, j] = fin.mean()
    return out


# ------------------------------------------------------------------------------------------- direct reads
def test_direct_delta_tiles_are_the_elementwise_difference(
    env: Env, models_dir: Path, expected: dict[str, Any]
) -> None:
    ma = import_model(env.api, env.project, models_dir, "tiny_base.safetensors")
    mb = import_model(env.api, env.project, models_dir, "tiny_ft.safetensors")
    c = compare(env.api, env.project, ma, mb)
    items = entries(env.api, c["id"], limit=100)["items"]
    up = next(x for x in items if x["raw_key"] == expected["tensor_key"])
    src = {"kind": "delta", "comparison_id": c["id"], "entry_id": up["entry_id"], "transform": "b_minus_a"}
    want = (np.array(expected["ft_full"]) - np.array(expected["base_full"])).astype(np.float32)

    _, values, status, meta = env.binary(body(src, 0))
    np.testing.assert_array_equal(values, want)
    assert not status.any()
    assert (meta["finite_count"], meta["nonfinite_count"], meta["invalid_pair_count"]) == (64, 0, 0)
    assert meta["quality"] == "exact"
    doc = env.raw(body(src, 0), accept="application/json").json()
    component("TileJson").validate(doc)
    assert doc["values"] == want.tolist()

    absb = env.binary(body({**src, "transform": "abs_b_minus_a"}, 0))
    np.testing.assert_array_equal(absb[1], np.abs(want))
    rel = env.binary(body({**src, "transform": "relative_abs", "epsilon": 0.5}, 0))
    base = np.array(expected["base_full"])
    np.testing.assert_array_equal(rel[1], (np.abs(want.astype(np.float64)) / np.maximum(np.abs(base), 0.5))
                                  .astype(np.float32))  # fmt: skip
    assert any("relative_abs = |B - A| / max(|A|, 0.5)" in w for w in rel[3]["warnings"])
    keys = {absb[0].headers["x-weightlab-cache-key"], rel[0].headers["x-weightlab-cache-key"],
            env.raw(body(src, 0)).headers["x-weightlab-cache-key"]}  # fmt: skip
    assert len(keys) == 3  # the transform is part of the identity
    eps2 = env.raw(body({**src, "transform": "relative_abs", "epsilon": 0.25}, 0)).headers["x-weightlab-cache-key"]
    assert eps2 != rel[0].headers["x-weightlab-cache-key"]  # so is epsilon, for relative_abs
    b_eps = env.raw(body({**src, "epsilon": 0.25}, 0)).headers["x-weightlab-cache-key"]
    assert b_eps == env.raw(body(src, 0)).headers["x-weightlab-cache-key"]  # but not where it is unused

    nf = next(x for x in items if x["raw_key"] == "demo.special_nonfinite.weight")
    _, _, st, m = env.binary(body({**src, "entry_id": nf["entry_id"]}, 0))
    assert m["invalid_pair_count"] == int(np.count_nonzero(st & 1)) > 0


def test_overflowed_deltas_are_withheld_over_http(env: Env, models_dir: Path, expected: dict[str, Any]) -> None:
    ma = import_model(env.api, env.project, models_dir, "tiny_precision_a.safetensors")
    mb = import_model(env.api, env.project, models_dir, "tiny_precision_b.safetensors")
    c = compare(env.api, env.project, ma, mb)
    e = next(x for x in entries(env.api, c["id"], limit=100)["items"] if x["raw_key"] == "p.f64")
    src = {"kind": "delta", "comparison_id": c["id"], "entry_id": e["entry_id"], "transform": "b_minus_a"}
    doc = env.raw(body(src, 0, view=VIEW1D), accept="application/json").json()
    component("TileJson").validate(doc)
    i = expected["precision"]["f64_delta_overflow_index"]
    assert doc["values"][0][i] is None
    assert doc["status"][0][i] == 16
    assert doc["metadata"]["invalid_pair_count"] == 0  # both inputs finite: an overflow, not an invalid pair
    assert any("NOT_REPRESENTABLE" in w for w in doc["metadata"]["warnings"])


def test_incompatible_entries_and_unverified_lineage(env: Env, models_dir: Path) -> None:
    a = import_model(env.api, env.project, models_dir, "tiny_base.safetensors")
    x = import_model(env.api, env.project, models_dir, "compat/shape_mismatch.safetensors")
    c = compare(env.api, env.project, a, x)
    assert c["lineage"]["compatibility_class"] == "X"
    bad = entries(env.api, c["id"], status=["shape_mismatch"])["items"][0]
    src = {"kind": "delta", "comparison_id": c["id"], "entry_id": bad["entry_id"], "transform": "b_minus_a"}
    assert env.api.post("/tiles", json_body=body(src, 0), expect=409)["error"]["code"] == "INCOMPATIBLE_TENSOR"
    job = {"kind": "pyramid_build", "source": src, "view": VIEW2D, "budget": BUDGET}
    assert env.api.post("/jobs", json_body=job, expect=409)["error"]["code"] == "INCOMPATIBLE_TENSOR"
    ok = next(e for e in entries(env.api, c["id"], status=["matched"])["items"] if len(e["shape_a"]) == 2)
    meta = env.binary(body({**src, "entry_id": ok["entry_id"]}, 0))[3]
    assert any(w.startswith("Class X: raw deltas may be meaningless") for w in meta["warnings"])


# --------------------------------------------------------------------------------------------- pyramids
def test_cold_delta_pyramid_is_a_paired_job_then_exact_tiles(env: Env, pairs: Path) -> None:
    src = env.pair("mid_a", "mid_b", "mid.weight")
    first = env.api.post("/tiles", json_body=body(src, 4), expect=202)
    assert first["kind"] == "pyramid_build"
    assert first["estimated_bytes"] == MID[0] * MID[1] * 8  # both inputs are read once
    second = env.api.post("/jobs", json_body={"kind": "pyramid_build", "source": src, "view": VIEW2D,
                                              "consistency": "best_effort", "budget": BUDGET}, expect=202)  # fmt: skip
    if second["state"] != "completed":
        assert second["job_id"] == first["job_id"]
        assert second["deduplicated"] is True
    assert env.finish(first["job_id"])["state"] == "completed"
    _, values, status, meta = env.binary(body(src, 4))
    assert meta["quality"] == "exact"
    d = np.load(pairs / "mid_b.npy").astype(np.float64) - np.load(pairs / "mid_a.npy").astype(np.float64)
    want = block_mean(d, 4)
    drawable = (status & 17) == 0
    np.testing.assert_allclose(values[drawable], want[: values.shape[0], : values.shape[1]][drawable], rtol=2e-7)
    assert meta["invalid_pair_count"] == 2  # the NaN of A and the +Inf of B
    assert status[0, 0] & 2 and status[700 >> 4, 2000 >> 4] & 4  # noqa: PT018  (their blocks keep their kind)
    cmp_ = env.api.get("/comparisons/{comparison_id}", comparison_id=src["comparison_id"], expect=200)
    entry = env.api.get("/comparisons/{comparison_id}/entries", comparison_id=src["comparison_id"],
                        expect=200)["items"][0]  # fmt: skip
    of_a = {"kind": "tensor", "model_id": cmp_["model_a_id"], "tensor_id": entry["tensor_id_a"]}
    tensor_job = env.api.post("/tiles", json_body=body(of_a, 4), expect=202)  # A's own pyramid is a different one
    assert tensor_job["job_id"] != first["job_id"]
    env.finish(tensor_job["job_id"])


def test_delta_sources_above_the_pyramid_limit_are_row_sampled_or_jobs(env: Env) -> None:
    src = env.pair("big_a", "big_b", "big.weight")
    r, values, status, meta = env.binary(body(src, 6))
    assert meta["quality"] == "row_sampled"
    assert r.headers["x-weightlab-quality"] == "row_sampled"
    assert meta["sampled_rows"] and meta["sampling_policy"] == "stride"  # noqa: PT018
    assert any("row-sampled preview" in w for w in meta["warnings"])
    assert np.all(values[(status & 17) == 0] >= 0)  # every delta is 0 or +0.5
    job = env.api.post("/tiles", json_body=body(src, 6, mode="exact"), expect=202)
    assert job["kind"] == "pyramid_build"
    env.api.post("/jobs/{job_id}/cancel", job_id=job["job_id"], expect=200)


def test_a_changed_input_is_never_served_from_old_results(env: Env, pairs: Path, tmp_path: Path) -> None:
    for name in ("mid_a", "mid_b"):
        shutil.copytree(pairs / name, tmp_path / name)
    src = env.pair("mid_a", "mid_b", "mid.weight", root=tmp_path)
    env.finish(env.api.post("/tiles", json_body=body(src, 4), expect=202)["job_id"])
    env.binary(body(src, 4))
    env.binary(body(src, 0, size=8))
    f = tmp_path / "mid_b" / "model.safetensors"
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    for b in (body(src, 4), body(src, 0, size=8)):
        assert env.api.post("/tiles", json_body=b, expect=409)["error"]["code"] == "FILE_CHANGED"


def test_an_entry_of_another_comparison_is_not_found(env: Env) -> None:
    src = env.pair("mid_a", "mid_b", "mid.weight")
    other = env.api.post("/projects", json_body={"name": "other"}, expect=201)["id"]
    oa = import_model(env.api, other, env.root / "mid_a", "model.safetensors")
    ob = import_model(env.api, other, env.root / "mid_b", "model.safetensors")
    oc = compare(env.api, other, oa, ob)
    mixed = {**src, "comparison_id": oc["id"]}  # this project's entry under another project's comparison
    assert env.api.post("/tiles", json_body=body(mixed, 0, size=8), expect=404)["error"]["code"] == "NOT_FOUND"
    job = {"kind": "pyramid_build", "source": mixed, "view": VIEW2D, "budget": BUDGET}
    assert env.api.post("/jobs", json_body=job, expect=404)["error"]["code"] == "NOT_FOUND"
