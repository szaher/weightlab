"""Bookmarks (typed SavedView, project ownership) and settings (round trip, hard bounds, read-only fields)."""

from __future__ import annotations

from pathlib import Path

from tests.api.conftest import VIEW2D, import_model, tensors_by_key
from tests.api.contract import Api


def test_bookmark_lifecycle_and_ownership(api: Api, project: str, models_dir: Path) -> None:
    m = import_model(api, project, models_dir, "tiny_base.safetensors")
    tid = tensors_by_key(api, m["id"])["demo.layers.0.mlp.up_proj.weight"]["id"]
    view = {"source": {"kind": "tensor", "model_id": m["id"], "tensor_id": tid}, "view": VIEW2D,
            "roi": {"r0": 0, "r1": 2, "c0": 0, "c1": 2}, "color": {"mode": "symmetric"}}  # fmt: skip
    bm = api.post("/projects/{project_id}/bookmarks", project_id=project, json_body={"view": view, "note": "spike"},
                  expect=201)  # fmt: skip
    assert bm["view"] == view
    assert bm["note"] == "spike"
    assert api.get("/projects/{project_id}/bookmarks", project_id=project, expect=200) == [bm]
    other = api.post("/projects", json_body={"name": "other"}, expect=201)["id"]
    # another project's IDs: 404, never 403 (no existence oracle)
    api.post("/projects/{project_id}/bookmarks", project_id=other, json_body={"view": view}, expect=404)
    api.delete("/projects/{project_id}/bookmarks/{bookmark_id}", project_id=other, bookmark_id=bm["id"], expect=404)
    bad_tensor = {**view, "source": {**view["source"], "tensor_id": "tns_unknown"}}
    api.post("/projects/{project_id}/bookmarks", project_id=project, json_body={"view": bad_tensor}, expect=404)
    api.post("/projects/{project_id}/bookmarks", project_id=project,
             json_body={"view": {**view, "zoom": 3}}, expect=422)  # fmt: skip
    api.delete("/projects/{project_id}/bookmarks/{bookmark_id}", project_id=project, bookmark_id=bm["id"], expect=204)
    assert api.get("/projects/{project_id}/bookmarks", project_id=project, expect=200) == []


def test_settings_round_trip_and_bounds(api: Api) -> None:
    s = api.get("/settings", expect=200)
    assert (s["offline"], s["telemetry_enabled"], s["roi_max_cells"]) == (True, False, 4096)
    s2 = api.patch("/settings", json_body={"max_threads": 4, "export_absolute_paths": True}, expect=200)
    assert (s2["max_threads"], s2["export_absolute_paths"]) == (4, True)
    assert api.get("/settings", expect=200) == s2
    for bad in ({}, {"max_threads": 65}, {"memory_budget_mib": 255}, {"telemetry_enabled": True}, {"offline": False},
                {"max_tile_cells": 1024}, {"max_threads": None}, {"max_threads": "4"}, {"tile_size": 128}):  # fmt: skip
        assert api.patch("/settings", json_body=bad, expect=422)["error"]["code"] == "VALIDATION_ERROR"
    assert api.get("/settings", expect=200) == s2  # rejected patches changed nothing


def test_settings_persist_across_restarts(api: Api, config: object) -> None:
    from fastapi.testclient import TestClient

    from tests.api.conftest import PORT, TOKEN
    from tests.api.contract import Api as A
    from weightlab.api.app import create_app

    api.patch("/settings", json_body={"roi_max_cells": 1000}, expect=200)
    with TestClient(create_app(config), base_url=f"http://127.0.0.1:{PORT}") as c2:  # type: ignore[arg-type]
        assert A(c2, TOKEN).get("/settings", expect=200)["roi_max_cells"] == 1000
