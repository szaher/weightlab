"""Projects, model import in place, catalog, relink, stale fingerprints (doc 09 s1/1.1/s9, doc 12 s3/s8)."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

import pytest

from tests.api.conftest import import_model, tensors_by_key
from tests.api.contract import Api
from weightlab import open_checkpoint


def snapshot(root: Path) -> dict[str, tuple[str, int]]:
    return {
        str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_project_crud(api: Api) -> None:
    assert api.get("/projects", expect=200) == []
    p = api.post("/projects", json_body={"name": "Ünïcode ⚙ project"}, expect=201)
    assert api.get("/projects/{project_id}", project_id=p["id"], expect=200) == p
    assert api.get("/projects", expect=200) == [p]
    api.delete("/projects/{project_id}", project_id=p["id"], expect=204)
    api.get("/projects/{project_id}", project_id=p["id"], expect=404)
    api.delete("/projects/{project_id}", project_id=p["id"], expect=404)
    api.post("/projects", json_body={"name": ""}, expect=422)
    api.post("/projects", json_body={"name": "x" * 121}, expect=422)


def test_import_single_file_and_catalog(api: Api, project: str, models_dir: Path) -> None:
    m = import_model(api, project, models_dir, "tiny_base.safetensors")
    assert m["format"] == "safetensors"
    assert m["status"] == "ready"
    assert m["fingerprint_method"] == "size_mtime"
    ref = {t.key: t for t in open_checkpoint(models_dir / "tiny_base.safetensors").metadata()}  # independent of the DB
    assert m["tensor_count"] == len(ref)
    cat = tensors_by_key(api, m["id"])
    assert set(cat) == set(ref)
    for key, t in cat.items():
        r = ref[key]
        assert (tuple(t["shape"]), t["dtype"], t["byte_length"]) == (r.shape, r.dtype, r.byte_length)
        assert (t["file_offset_begin"], t["file_offset_end"]) == (r.data_offset, r.data_offset + r.byte_length)
        assert t["shard_relpath"] == "tiny_base.safetensors"
        assert t["numel"] == r.numel
        assert t["role_is_heuristic"] is True
    up = cat["demo.layers.0.mlp.up_proj.weight"]
    assert (up["role"], up["layer_index"]) == ("mlp.up", 0)
    one = api.get("/models/{model_id}/tensors/{tensor_id}", model_id=m["id"], tensor_id=up["id"], expect=200)
    assert one == up
    assert api.get("/projects/{project_id}/models", project_id=project, expect=200) == [m]


def test_catalog_filters_sort_and_pagination(api: Api, project: str, models_dir: Path) -> None:
    m = import_model(api, project, models_dir, "tiny_base.safetensors")
    get = lambda **params: api.get("/models/{model_id}/tensors", model_id=m["id"], params=params, expect=200)  # noqa: E731
    everything = get(limit=1000)
    assert everything["total"] == 9
    assert get(q="MLP")["total"] == 2  # case-insensitive substring
    assert get(rank=3)["total"] == 1
    assert get(rank=0)["items"][0]["raw_key"] == "demo.scalar"
    assert get(min_numel=64)["total"] == sum(1 for t in everything["items"] if t["numel"] >= 64)
    assert {t["role"] for t in get(role=["mlp.up", "mlp.down"])["items"]} == {"mlp.up", "mlp.down"}
    assert get(layer=0)["total"] == 3
    desc = get(sort="numel", order="desc", limit=1000)["items"]
    assert [t["numel"] for t in desc] == sorted((t["numel"] for t in desc), reverse=True)
    pages = [get(limit=4, offset=o)["items"] for o in (0, 4, 8)]
    assert [t["id"] for pg in pages for t in pg] == [t["id"] for t in everything["items"]]
    for bad in (
        {"limit": 0},
        {"limit": 1001},
        {"sort": "shape"},
        {"order": "up"},
        {"dtype": ["F8_E4M3"]},
        {"offset": -1},
        {"q": "x" * 257},
    ):
        api.get("/models/{model_id}/tensors", model_id=m["id"], params=bad, expect=422)


def test_utf8_special_keys_never_in_urls(api: Api, project: str, models_dir: Path) -> None:
    m = import_model(api, project, models_dir, "tiny_edge.safetensors")
    cat = tensors_by_key(api, m["id"])
    key = "edge.ünïcødé/κλειδί.weight"
    assert key in cat
    tid = cat[key]["id"]
    assert tid.isascii()
    assert "/" not in tid
    got = api.get("/models/{model_id}/tensors/{tensor_id}", model_id=m["id"], tensor_id=tid, expect=200)
    assert got["raw_key"] == key
    assert api.get("/models/{model_id}/tensors", model_id=m["id"], params={"q": "ΚΛΕΙΔΊ"}, expect=200)["total"] == 1


def test_import_sharded_with_relpaths(api: Api, project: str, models_dir: Path) -> None:
    m = import_model(api, project, models_dir, "tiny_sharded")
    assert m["format"] == "sharded_safetensors"
    rels = {t["shard_relpath"] for t in tensors_by_key(api, m["id"]).values()}
    assert rels == {"tiny_sharded/model-00001-of-00002.safetensors", "tiny_sharded/model-00002-of-00002.safetensors"}
    assert all(not os.path.isabs(r) for r in rels)


@pytest.mark.parametrize(
    ("entry", "root", "status", "code"),
    [
        ("tiny_base.safetensors", "relative", 422, "VALIDATION_ERROR"),
        ("nope.safetensors", None, 404, "NOT_FOUND"),
        ("malformed/corrupt_json.safetensors", None, 422, "INVALID_FORMAT"),
        # Frozen fixtures: lexical ".." in an index is INVALID_FORMAT (doc 12 says PATH_OUTSIDE_ROOT; recorded
        # discrepancy). Real escapes (symlinks, entry outside root) are PATH_OUTSIDE_ROOT, tested below.
        ("malformed/path_traversal", None, 422, "INVALID_FORMAT"),
        ("malformed/missing_shard", None, 422, "MISSING_SHARD"),
        ("compat", None, 422, "INVALID_FORMAT"),
    ],
)
def test_import_errors(
    api: Api, project: str, models_dir: Path, entry: str, root: str | None, status: int, code: str
) -> None:
    body = {"entry_path": str(models_dir / entry), "selected_root": str(models_dir)}
    if root == "relative":
        body = {"entry_path": entry, "selected_root": "."}
    err = api.post("/projects/{project_id}/models", project_id=project, json_body=body, expect=status)
    assert err["error"]["code"] == code
    assert str(models_dir) not in str(err)  # no path disclosure


def test_entry_outside_selected_root(api: Api, project: str, models_dir: Path) -> None:
    body = {"entry_path": str(models_dir / "tiny_base.safetensors"), "selected_root": str(models_dir / "compat")}
    err = api.post("/projects/{project_id}/models", project_id=project, json_body=body, expect=403)
    assert err["error"]["code"] == "PATH_OUTSIDE_ROOT"


def test_symlink_escape_refused(api: Api, project: str, models_dir: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.safetensors"
    shutil.copy(models_dir / "tiny_base.safetensors", outside)
    inner = models_dir / "link.safetensors"
    try:
        inner.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not permitted")
    err = import_model(api, project, models_dir, "link.safetensors", expect=403)
    assert err["error"]["code"] == "PATH_OUTSIDE_ROOT"


def test_pickle_checkpoints_are_never_loaded(api: Api, project: str, models_dir: Path) -> None:
    pt = models_dir / "model.pt"
    pt.write_bytes(b"\x80\x04\x95" + b"\0" * 64)  # pickle protocol 4 prefix
    err = import_model(api, project, models_dir, "model.pt", expect=422)
    assert err["error"]["code"] == "INVALID_FORMAT"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX FIFOs")
def test_fifo_is_refused_without_blocking(api: Api, project: str, models_dir: Path) -> None:
    os.mkfifo(models_dir / "pipe.safetensors")
    err = import_model(api, project, models_dir, "pipe.safetensors", expect=422)
    assert err["error"]["code"] == "INVALID_FORMAT"


def test_cross_project_ids_are_404(api: Api, project: str, models_dir: Path) -> None:
    other = api.post("/projects", json_body={"name": "other"}, expect=201)["id"]
    m = import_model(api, project, models_dir, "tiny_base.safetensors")
    body = {"project_id": other, "model_a_id": m["id"], "model_b_id": m["id"]}
    assert api.post("/comparisons", json_body=body, expect=404)["error"]["code"] == "NOT_FOUND"
    body = {"base_model_id": m["id"], "adapter_directory": str(models_dir / "tiny_lora")}
    api.post("/projects/{project_id}/adapters", project_id=other, json_body=body, expect=404)
    # Contract gap: the list routes declare no 404, but doc 09's ownership rule requires it for unknown projects.
    r = api.client.get("/api/v1/projects/prj_nope/models", headers=api.headers)
    assert r.status_code == 404


def test_project_delete_leaves_checkpoints_byte_identical(api: Api, project: str, models_dir: Path) -> None:
    before = snapshot(models_dir)
    m = import_model(api, project, models_dir, "tiny_base.safetensors")
    import_model(api, project, models_dir, "tiny_sharded")
    api.delete("/projects/{project_id}", project_id=project, params={"purge_cache": True}, expect=204)
    api.get("/models/{model_id}/tensors", model_id=m["id"], expect=404)
    assert snapshot(models_dir) == before


def _move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def test_moved_model_is_disconnected_then_relinked(api: Api, project: str, models_dir: Path, tmp_path: Path) -> None:
    m = import_model(api, project, models_dir, "tiny_sharded")
    new_root = tmp_path / "moved"
    _move(models_dir / "tiny_sharded", new_root / "tiny_sharded")
    assert api.get("/projects/{project_id}/models", project_id=project, expect=200)[0]["status"] == "disconnected"
    t = next(iter(tensors_by_key(api, m["id"]).values()))
    roi = {
        "source": {"kind": "tensor", "model_id": m["id"], "tensor_id": t["id"]},
        "view": {"row_axis": None, "col_axis": 0, "fixed_indices": {}, "transpose": False}
        if t["rank"] == 1
        else {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False},
        "roi": {"r0": 0, "r1": 1, "c0": 0, "c1": 1},
    }
    assert api.post("/tensors/roi", json_body=roi, expect=409)["error"]["code"] == "FILE_CHANGED"
    body = {"entry_path": str(new_root / "tiny_sharded"), "selected_root": str(new_root)}
    relinked = api.post("/models/{model_id}/relink", model_id=m["id"], json_body=body, expect=200)
    assert relinked["status"] == "ready"
    api.post("/tensors/roi", json_body=roi, expect=200)


def test_relink_refused_after_content_change(api: Api, project: str, models_dir: Path, tmp_path: Path) -> None:
    m = import_model(api, project, models_dir, "tiny_base.safetensors")
    new_root = tmp_path / "moved"
    _move(models_dir / "tiny_base.safetensors", new_root / "tiny_base.safetensors")
    with (new_root / "tiny_base.safetensors").open("ab") as f:
        f.write(b"\0")
    body = {"entry_path": str(new_root / "tiny_base.safetensors"), "selected_root": str(new_root)}
    err = api.post("/models/{model_id}/relink", model_id=m["id"], json_body=body, expect=409)
    assert err["error"]["code"] == "FILE_CHANGED"
    # relinking to a different model with the same file name is refused too
    shutil.copy(models_dir / "tiny_ft.safetensors", new_root / "tiny_base.safetensors")
    api.post("/models/{model_id}/relink", model_id=m["id"], json_body=body, expect=409)


def test_stale_fingerprint_marks_changed_and_refuses_reads(api: Api, project: str, models_dir: Path) -> None:
    m = import_model(api, project, models_dir, "tiny_base.safetensors")
    up = tensors_by_key(api, m["id"])["demo.layers.0.mlp.up_proj.weight"]
    roi = {
        "source": {"kind": "tensor", "model_id": m["id"], "tensor_id": up["id"]},
        "view": {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False},
        "roi": {"r0": 0, "r1": 1, "c0": 0, "c1": 1},
    }
    api.post("/tensors/roi", json_body=roi, expect=200)
    p = models_dir / "tiny_base.safetensors"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # same bytes, new mtime
    assert api.get("/projects/{project_id}/models", project_id=project, expect=200)[0]["status"] == "changed"
    assert api.post("/tensors/roi", json_body=roi, expect=409)["error"]["code"] == "FILE_CHANGED"


def test_error_payloads_never_disclose_paths(api: Api, project: str, models_dir: Path, tmp_path: Path) -> None:
    for entry in (
        "nope.safetensors",
        "malformed/corrupt_json.safetensors",
        "malformed/path_traversal",
        "malformed/windows_drive_path",
        "malformed/absolute_path",
    ):
        import_model(api, project, models_dir, entry, expect=api_status(entry))
    assert api.seen_error_bodies
    for body in api.seen_error_bodies:
        assert str(tmp_path) not in body
        assert os.path.expanduser("~") not in body


def api_status(entry: str) -> int:
    return {"nope.safetensors": 404}.get(entry, 422)
