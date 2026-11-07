"""PR #3 review regressions: stale adapter configs, stale shard indexes, atomic relink.

Each test fails against ea76a920 (the reviewed head) and passes with the fixes.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import TOKEN, import_model, tensors_by_key
from tests.api.contract import Api
from tests.helpers import write_safetensors
from weightlab.api.config import ServerConfig

VIEW2D = {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}
VIEW1D = {"row_axis": None, "col_axis": 0, "fixed_indices": {}, "transpose": False}


def _roi_body(source: dict[str, Any], view: dict[str, Any] = VIEW2D) -> dict[str, Any]:
    return {"source": source, "view": view, "roi": {"r0": 0, "r1": 1, "c0": 0, "c1": 1}}


def _rewrite_keeping_stat(path: Path, data: bytes) -> None:
    """Replace the content in place; same size and mtime when `data` has the original length."""
    st = path.stat()
    path.write_bytes(data)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


# -- adapter_config.json ----------------------------------------------------------------------
@pytest.mark.parametrize("keep_stat", [False, True], ids=["mtime_changes", "same_size_and_mtime"])
def test_adapter_config_edit_is_file_changed_for_both_adapter_sources(
    api: Api, project: str, models_dir: Path, keep_stat: bool
) -> None:
    base = import_model(api, project, models_dir, "tiny_base.safetensors")
    d = models_dir / "tiny_lora"
    body = {"base_model_id": base["id"], "adapter_directory": str(d)}
    ad = api.post("/projects/{project_id}/adapters", project_id=project, json_body=body, expect=201)
    mod = next(m for m in ad["modules"] if m["adapter_key_prefix"].endswith(".up_proj"))
    assert mod["native_scale"] == 2.0  # lora_alpha 4 / r 2
    sources = [
        {"kind": "lora_delta", "adapter_id": ad["id"], "module_id": mod["module_id"], "experiment_multiplier": 1.0},
        {"kind": "virtual_effective", "adapter_id": ad["id"], "module_id": mod["module_id"],
         "experiment_multiplier": 1.0, "research_mode": False},
    ]  # fmt: skip
    for src in sources:
        api.post("/tensors/roi", json_body=_roi_body(src), expect=200)

    cfg = d / "adapter_config.json"
    text = cfg.read_text(encoding="utf-8")
    edited = text.replace('"lora_alpha": 4', '"lora_alpha": 8')  # same length: the native scale doubles
    assert edited != text
    assert len(edited) == len(text)
    if keep_stat:
        _rewrite_keeping_stat(cfg, edited.encode())
    else:
        cfg.write_text(edited, encoding="utf-8")
    for src in sources:
        err = api.post("/tensors/roi", json_body=_roi_body(src), expect=409)
        assert err["error"]["code"] == "FILE_CHANGED", src["kind"]


# -- sharded index ----------------------------------------------------------------------------
def _sharded(root: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    t = {k: rng.standard_normal((2, 3)).astype(np.float32) for k in ("a.weight", "b.weight", "c.weight")}
    write_safetensors(root / "m" / "s1.safetensors",
                      {k: ("F32", t[k].shape, t[k].tobytes()) for k in ("a.weight", "b.weight")})  # fmt: skip
    write_safetensors(
        root / "m" / "s2.safetensors", {"c.weight": ("F32", t["c.weight"].shape, t["c.weight"].tobytes())}
    )
    _write_index(root, {"a.weight": "s1.safetensors", "b.weight": "s1.safetensors", "c.weight": "s2.safetensors"})
    return t


def _write_index(root: Path, weight_map: dict[str, str]) -> Path:
    p = root / "m" / "model.safetensors.index.json"
    p.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    return p


def test_index_only_edit_that_keeps_the_shard_set_is_file_changed(api: Api, project: str, tmp_path: Path) -> None:
    """The reviewer's scenario: drop one key from the index; both shards are still referenced, unchanged."""
    _sharded(tmp_path)
    m = import_model(api, project, tmp_path, "m")
    tensors = tensors_by_key(api, m["id"])
    src = {"kind": "tensor", "model_id": m["id"], "tensor_id": tensors["a.weight"]["id"]}
    api.post("/tensors/roi", json_body=_roi_body(src), expect=200)
    _write_index(tmp_path, {"a.weight": "s1.safetensors", "c.weight": "s2.safetensors"})
    assert api.get("/projects/{project_id}/models", project_id=project, expect=200)[0]["status"] == "changed"
    for key in ("a.weight", "b.weight"):
        body = _roi_body({"kind": "tensor", "model_id": m["id"], "tensor_id": tensors[key]["id"]})
        assert api.post("/tensors/roi", json_body=body, expect=409)["error"]["code"] == "FILE_CHANGED", key


@pytest.mark.parametrize("edit", ["reordered", "key_renamed"])
def test_same_size_same_mtime_index_rewrite_is_file_changed(api: Api, project: str, tmp_path: Path, edit: str) -> None:
    """No stat-level difference at all: only the index bytes differ. `reordered` still parses (same mapping);
    `key_renamed` maps a key that is not in the shard, which the reader would reject as INVALID_FORMAT."""
    _sharded(tmp_path)
    m = import_model(api, project, tmp_path, "m")
    tensors = tensors_by_key(api, m["id"])
    idx = tmp_path / "m" / "model.safetensors.index.json"
    original = idx.read_bytes()
    if edit == "reordered":
        doc = json.loads(original)
        doc["weight_map"] = dict(reversed(list(doc["weight_map"].items())))
        new = json.dumps(doc).encode()
    else:
        new = original.replace(b'"b.weight"', b'"x.weight"')
    assert new != original
    assert len(new) == len(original)
    _rewrite_keeping_stat(idx, new)
    body = _roi_body({"kind": "tensor", "model_id": m["id"], "tensor_id": tensors["a.weight"]["id"]})
    assert api.post("/tensors/roi", json_body=body, expect=409)["error"]["code"] == "FILE_CHANGED"


def test_index_is_recorded_and_relink_checks_its_content(api: Api, project: str, tmp_path: Path) -> None:
    _sharded(tmp_path)
    m = import_model(api, project, tmp_path, "m")
    assert m["status"] == "ready"
    new_root = tmp_path / "moved"
    shutil.copytree(tmp_path / "m", new_root / "m", copy_function=shutil.copy2)  # same sizes and mtimes
    idx = new_root / "m" / "model.safetensors.index.json"
    _rewrite_keeping_stat(idx, idx.read_bytes().replace(b'"b.weight"', b'"x.weight"'))
    body = {"entry_path": str(new_root / "m"), "selected_root": str(new_root)}
    err = api.post("/models/{model_id}/relink", model_id=m["id"], json_body=body, expect=409)
    assert err["error"]["code"] == "FILE_CHANGED"


# -- relink atomicity -------------------------------------------------------------------------
def test_relink_is_one_transaction(
    api: Api, project: str, models_dir: Path, tmp_path: Path, config: ServerConfig
) -> None:
    """Inject a failure into the second statement of relink (the file-identity update). Before the fix the path
    update had already committed, leaving the model at the new path with the old file identities."""
    m = import_model(api, project, models_dir, "tiny_base.safetensors")
    new_root = tmp_path / "moved"
    new_root.mkdir()
    shutil.move(str(models_dir / "tiny_base.safetensors"), str(new_root / "tiny_base.safetensors"))
    db = config.data_dir / "weightlab.sqlite3"
    c = sqlite3.connect(db)
    c.execute("CREATE TRIGGER fail_file_ids BEFORE UPDATE OF file_id ON model_files "
              "BEGIN SELECT RAISE(ABORT, 'injected failure'); END")  # fmt: skip
    c.commit()
    before = c.execute("SELECT entry_path, selected_root, status FROM models WHERE id = ?", (m["id"],)).fetchone()
    c.close()
    body = {"entry_path": str(new_root / "tiny_base.safetensors"), "selected_root": str(new_root)}
    raw = TestClient(api.client.app, base_url=str(api.client.base_url), raise_server_exceptions=False)
    r = raw.post(f"/api/v1/models/{m['id']}/relink", json=body, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 500
    c = sqlite3.connect(db)
    after = c.execute("SELECT entry_path, selected_root, status FROM models WHERE id = ?", (m["id"],)).fetchone()
    c.execute("DROP TRIGGER fail_file_ids")
    c.commit()
    c.close()
    assert after == before
    api.post("/models/{model_id}/relink", model_id=m["id"], json_body=body, expect=200)  # works once it can


# -- admission control ------------------------------------------------------------------------
def test_slow_request_bodies_count_towards_in_flight(tmp_path: Path) -> None:
    """Before the fix the slot was taken after the body was read, so requests stalled mid-body were not counted."""
    import asyncio

    from tests.api.conftest import PORT
    from weightlab.api.config import API_PREFIX
    from weightlab.api.security import Guard

    async def echo_app(scope: Any, receive: Any, send: Any) -> None:
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    guard = Guard(echo_app, ServerConfig(port=PORT, data_dir=tmp_path, token=TOKEN, max_in_flight=2))
    headers = [(b"host", f"127.0.0.1:{PORT}".encode()), (b"authorization", f"Bearer {TOKEN}".encode())]
    scope = {"type": "http", "method": "POST", "path": f"{API_PREFIX}/projects", "headers": headers}

    async def call(receive: Any) -> int:
        sent: list[Any] = []

        async def send(m: Any) -> None:
            sent.append(m)

        await guard(dict(scope), receive, send)
        return int(sent[0]["status"])

    async def scenario() -> tuple[int, list[int], int, int]:
        release = asyncio.Event()

        def slow_body() -> Any:
            first = True

            async def receive() -> Any:
                nonlocal first
                if first:
                    first = False
                    return {"type": "http.request", "body": b'{"na', "more_body": True}
                await release.wait()  # the rest of the body never arrives until released
                return {"type": "http.request", "body": b'me": "a"}', "more_body": False}

            return receive

        async def empty() -> Any:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def oversized() -> Any:
            return {"type": "http.request", "body": b"x" * (2 * 1024 * 1024), "more_body": False}

        stalled = [asyncio.create_task(call(slow_body())) for _ in range(2)]
        await asyncio.sleep(0.05)
        third = await call(empty)  # both slots are held by stalled bodies
        release.set()
        done = await asyncio.gather(*stalled)
        too_big = await call(oversized)  # an error while reading the body releases its slot
        after = await call(empty)
        return third, done, too_big, after

    third, done, too_big, after = asyncio.run(scenario())
    assert third == 429
    assert done == [200, 200]
    assert too_big == 413
    assert after == 200
    assert guard._in_flight == 0


# -- health -----------------------------------------------------------------------------------
def test_health_does_not_run_a_full_integrity_scan(
    api: Api, config: ServerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import weightlab.db.schema as schema

    def forbidden(conn: Any) -> Any:
        raise AssertionError("integrity_check on the unauthenticated health route")

    monkeypatch.setattr(schema, "integrity", forbidden)
    traced: list[str] = []
    real_connect = schema.connect

    def tracing_connect(path: Any) -> sqlite3.Connection:
        conn = real_connect(path)
        conn.set_trace_callback(traced.append)
        return conn

    monkeypatch.setattr(schema, "connect", tracing_connect)
    monkeypatch.setattr("weightlab.db.connect", tracing_connect)
    assert api.get("/health", expect=200)["status"] == "ok"
    assert traced
    assert not [q for q in traced if "integrity_check" in q or "foreign_key_check" in q]

    c = sqlite3.connect(config.data_dir / "weightlab.sqlite3")  # schema behind the latest -> degraded
    c.execute("DELETE FROM schema_migrations")
    c.commit()
    c.close()
    assert api.get("/health", expect=200)["status"] == "degraded"
