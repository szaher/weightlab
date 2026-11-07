"""POST /tensors/roi for all four source kinds, plus comparisons and adapters that feed them.

Expected values come from the fixture generator (examples/fixtures/expected_results.json) or plain NumPy on
hand-built inputs, never from WeightLab."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.api.conftest import VIEW1D, VIEW2D, import_model, tensors_by_key
from tests.api.contract import Api
from tests.helpers import write_safetensors

UP = "demo.layers.0.mlp.up_proj.weight"


def roi(
    api: Api,
    source: dict[str, Any],
    r: tuple[int, int, int, int],
    view: dict[str, Any] = VIEW2D,
    expect: int = 200,
    **extra: Any,
) -> Any:
    body = {"source": source, "view": view, "roi": dict(zip(("r0", "r1", "c0", "c1"), r, strict=True)), **extra}
    return api.post("/tensors/roi", json_body=body, expect=expect)


def tensor_src(model: dict[str, Any], tensors: dict[str, Any], key: str) -> dict[str, Any]:
    return {"kind": "tensor", "model_id": model["id"], "tensor_id": tensors[key]["id"]}


# -- raw tensors ------------------------------------------------------------------------------
def test_tensor_roi_matches_generator(api: Api, project: str, models_dir: Path, expected: dict[str, Any]) -> None:
    m = import_model(api, project, models_dir, "tiny_base.safetensors")
    t = tensors_by_key(api, m["id"])
    got = roi(api, tensor_src(m, t, UP), (0, 8, 0, 8))
    assert got["numeric_precision"] == "lossless"
    assert got["value_encoding"] == "float_json"
    assert got["values"] == expected["base_full"]
    tr = roi(api, tensor_src(m, t, UP), (2, 4, 1, 6), view={**VIEW2D, "transpose": True})
    assert tr["shape"] == [2, 5]
    assert tr["values"] == [[expected["base_full"][c][r] for c in range(1, 6)] for r in range(2, 4)]
    hexed = roi(api, tensor_src(m, t, UP), (0, 1, 0, 2), value_encoding="hex_bits")
    want = [f"0x{np.float32(v).view(np.uint32):08x}" for v in expected["base_full"][0][:2]]
    assert hexed["values"] == [want]


def test_tensor_roi_nonfinite_mask(api: Api, project: str, models_dir: Path, expected: dict[str, Any]) -> None:
    m = import_model(api, project, models_dir, "tiny_edge.safetensors")
    t = tensors_by_key(api, m["id"])
    cells = expected["edge"]["nonfinite_blocks"]["level0_cells"]
    got = roi(api, tensor_src(m, t, "edge.nonfinite_blocks.weight"), (0, 32, 0, 32))
    mask = got["nonfinite_mask"]
    kinds = {3: "nan", 5: "pos_inf", 9: "neg_inf"}
    for rc, cell in cells.items():
        r, c = map(int, rc.split(","))
        if cell["status"] == 0:
            assert got["values"][r][c] == cell["value"]
            assert mask[r][c] is None
        else:
            assert got["values"][r][c] is None
            assert mask[r][c] == kinds[cell["status"]]


def test_lossless_precision_fixture(api: Api, project: str, models_dir: Path, expected: dict[str, Any]) -> None:
    pr = expected["precision"]
    m = import_model(api, project, models_dir, "tiny_precision_a.safetensors")
    t = tensors_by_key(api, m["id"])
    for key, want, enc in (("p.i64", pr["i64"], "int_decimal_string"), ("p.u64", pr["u64"], "int_decimal_string")):
        got = roi(api, tensor_src(m, t, key), (0, 1, 0, len(want)), view=VIEW1D)
        assert got["value_encoding"] == enc
        assert got["values"] == [want]  # U64 max, I64 min, 2^53 +/- 1: exact decimal strings
    f64 = roi(api, tensor_src(m, t, "p.f64"), (0, 1, 0, 8), view=VIEW1D)
    assert f64["values"] == [[float(x) for x in pr["f64_a"]]]
    raw = api.client.post(
        "/api/v1/tensors/roi",
        headers=api.headers,
        json={"source": tensor_src(m, t, "p.f64"), "view": VIEW1D, "roi": {"r0": 0, "r1": 1, "c0": 4, "c1": 5}},
    ).text
    assert "-0.0" in raw  # signed zero survives on the wire
    assert math.copysign(1, f64["values"][0][4]) == -1
    assert roi(api, tensor_src(m, t, "p.i32"), (0, 1, 0, 2), view=VIEW1D)["values"] == [[int(x) for x in pr["i32"]]]
    assert roi(api, tensor_src(m, t, "p.bool"), (0, 1, 0, 2), view=VIEW1D)["values"] == [pr["bool"]]


def test_roi_limits_and_validation(api: Api, project: str, models_dir: Path) -> None:
    m = import_model(api, project, models_dir, "tiny_base.safetensors")
    t = tensors_by_key(api, m["id"])
    q = tensor_src(m, t, "demo.layers.0.self_attn.q_proj.weight")  # 257 x 263
    assert roi(api, q, (0, 64, 0, 65), expect=413)["error"]["code"] == "RESOURCE_LIMIT"  # 4160 > 4096 cells
    api.patch("/settings", json_body={"roi_max_cells": 16}, expect=200)
    roi(api, q, (0, 4, 0, 4))
    roi(api, q, (0, 4, 0, 5), expect=413)
    assert roi(api, q, (0, 2, 0, 300), expect=422)["error"]["code"] == "INVALID_ROI"  # outside the view
    roi(api, q, (0, 1, 0, 2**53), expect=422)  # beyond the JS safe integer: schema rejection, before any read
    cube = tensor_src(m, t, "demo.cube.weight")
    roi(api, cube, (0, 1, 0, 1), expect=422)  # rank 3 needs fixed_indices
    assert roi(
        api, cube, (0, 2, 0, 2), view={"row_axis": 0, "col_axis": 2, "fixed_indices": {"1": 3}, "transpose": False}
    )["shape"] == [2, 2]
    other = api.post("/projects", json_body={"name": "o"}, expect=201)["id"]
    m2 = import_model(api, other, models_dir, "tiny_ft.safetensors")
    roi(api, {"kind": "tensor", "model_id": m2["id"], "tensor_id": t[UP]["id"]}, (0, 1, 0, 1), expect=404)


# -- delta ------------------------------------------------------------------------------------
def compare(api: Api, project: str, a: dict[str, Any], b: dict[str, Any], **extra: Any) -> Any:
    body = {"project_id": project, "model_a_id": a["id"], "model_b_id": b["id"], **extra}
    return api.post("/comparisons", json_body=body, expect=201)


def entries(api: Api, cmp_id: str, **params: Any) -> Any:
    return api.get("/comparisons/{comparison_id}/entries", comparison_id=cmp_id, params=params, expect=200)


def test_comparison_and_delta_roi(api: Api, project: str, models_dir: Path, expected: dict[str, Any]) -> None:
    a = import_model(api, project, models_dir, "tiny_base.safetensors")
    b = import_model(api, project, models_dir, "tiny_ft.safetensors")
    c = compare(api, project, a, b)
    want = expected["compat_expected_class"]["tiny_base vs tiny_ft"]
    assert c["lineage"]["compatibility_class"] == want["class"]
    assert c["counts"]["matched"] == 9
    assert api.get("/comparisons/{comparison_id}", comparison_id=c["id"], expect=200) == c
    page = entries(api, c["id"], limit=1000)
    assert page["total"] == 9
    up = next(e for e in page["items"] if e["raw_key"] == UP)
    assert up["tensor_id_a"] == tensors_by_key(api, a["id"])[UP]["id"]
    src = {"kind": "delta", "comparison_id": c["id"], "entry_id": up["entry_id"], "transform": "b_minus_a"}
    d = roi(api, src, (0, 8, 0, 8))
    assert d["numeric_precision"] == "rounded"
    base, ft = np.array(expected["base_full"]), np.array(expected["ft_full"])
    assert d["values"] == (ft - base).tolist()
    ab = roi(api, {**src, "transform": "abs_b_minus_a"}, (0, 8, 0, 8))
    assert ab["values"] == np.abs(ft - base).tolist()
    rel = roi(api, {**src, "transform": "relative_abs", "epsilon": 0.5}, (0, 8, 0, 8))
    assert rel["values"] == (np.abs(ft - base) / np.maximum(np.abs(base), 0.5)).tolist()
    p1 = entries(api, c["id"], limit=4, offset=0)["items"] + entries(api, c["id"], limit=4, offset=4)["items"]
    assert [e["entry_id"] for e in p1] == [e["entry_id"] for e in page["items"][:8]]
    roi(api, {**src, "transform": "b_minus_a"}, (0, 1, 0, 1), value_encoding="hex_bits", expect=422)


def test_delta_overflow_is_null_with_overflow_mask(
    api: Api, project: str, models_dir: Path, expected: dict[str, Any]
) -> None:
    pr = expected["precision"]
    a = import_model(api, project, models_dir, "tiny_precision_a.safetensors")
    b = import_model(api, project, models_dir, "tiny_precision_b.safetensors")
    c = compare(api, project, a, b)
    e = next(x for x in entries(api, c["id"], limit=100)["items"] if x["raw_key"] == "p.f64")
    d = roi(
        api,
        {"kind": "delta", "comparison_id": c["id"], "entry_id": e["entry_id"], "transform": "b_minus_a"},
        (0, 1, 0, 8),
        view=VIEW1D,
    )
    i = pr["f64_delta_overflow_index"]
    assert d["numeric_precision"] == "overflowed"
    assert d["values"][0][i] is None
    assert d["nonfinite_mask"][0][i] == "overflow"
    for j, want in enumerate(pr["f64_delta_b_minus_a"]):
        if j != i:
            assert d["values"][0][j] == float(want)
    e32 = next(x for x in entries(api, c["id"], limit=100)["items"] if x["raw_key"] == "p.f32_overflow_delta")
    d32 = roi(
        api,
        {"kind": "delta", "comparison_id": c["id"], "entry_id": e32["entry_id"], "transform": "b_minus_a"},
        (0, 1, 0, 2),
        view=VIEW1D,
    )
    assert float(pr["f32_overflow_delta_b_minus_a_f64"]) in d32["values"][0]  # exact in float64, not overflowed


def test_incompatible_entries_and_classes(api: Api, project: str, models_dir: Path) -> None:
    a = import_model(api, project, models_dir, "tiny_base.safetensors")
    x = import_model(api, project, models_dir, "compat/shape_mismatch.safetensors")
    c = compare(api, project, a, x)
    assert c["lineage"]["compatibility_class"] == "X"
    bad = entries(api, c["id"], status=["shape_mismatch"])
    assert bad["total"] == 1
    src = {"kind": "delta", "comparison_id": c["id"], "entry_id": bad["items"][0]["entry_id"], "transform": "b_minus_a"}
    assert roi(api, src, (0, 1, 0, 1), expect=409)["error"]["code"] == "INCOMPATIBLE_TENSOR"
    untied = import_model(api, project, models_dir, "compat/untied_reference")
    tied = import_model(api, project, models_dir, "compat/tied_embeddings")
    ct = compare(api, project, untied, tied, user_attestation=True)
    assert ct["lineage"]["compatibility_class"] == "L"  # attestation never upgrades beyond L
    te = entries(api, ct["id"], status=["tied"])["items"][0]
    assert (te["raw_key"], te["tied_to"], te["tie_evidence"]) == (
        "lm_head.weight",
        "model.embed_tokens.weight",
        "config_and_known_mapping",
    )
    # Contract gap: no 422 declared for an invalid status filter; the API still rejects it.
    r = api.client.get(f"/api/v1/comparisons/{ct['id']}/entries", params={"status": "bogus"}, headers=api.headers)
    assert r.status_code == 422


# -- LoRA -------------------------------------------------------------------------------------
def adapter(api: Api, project: str, base: dict[str, Any], directory: Path, expect: int = 201) -> Any:
    body = {"base_model_id": base["id"], "adapter_directory": str(directory)}
    return api.post("/projects/{project_id}/adapters", project_id=project, json_body=body, expect=expect)


@pytest.mark.parametrize("name", ["tiny_lora", "tiny_lora_rslora"])
def test_lora_import_and_roi(api: Api, project: str, models_dir: Path, expected: dict[str, Any], name: str) -> None:
    base = import_model(api, project, models_dir, "tiny_base.safetensors")
    ad = adapter(api, project, base, models_dir / name)
    assert ad["status"] == "ready"
    assert ad["lineage"]["compatibility_class"] == "V"  # proven by construction (generator registry)
    assert api.get("/projects/{project_id}/adapters", project_id=project, expect=200) == [ad]
    bt = tensors_by_key(api, base["id"])
    for module, spec in expected["lora"][name].items():
        mod = next(m for m in ad["modules"] if m["adapter_key_prefix"].endswith(f".{module}"))
        assert mod["status"] == "supported"
        assert (mod["rank"], mod["native_scale"]) == (spec["rank"], spec["scale"])
        assert mod["base_tensor_id"] == bt[f"demo.layers.0.mlp.{module}.weight"]["id"]
        want = np.array(spec["delta_full"])
        src = {
            "kind": "lora_delta",
            "adapter_id": ad["id"],
            "module_id": mod["module_id"],
            "experiment_multiplier": 1.0,
        }
        got = np.array(roi(api, src, (0, 8, 0, 8))["values"])
        assert np.allclose(got, want, rtol=1e-6, atol=1e-7)  # generator rounded its delta to float32
        half = np.array(roi(api, {**src, "experiment_multiplier": -0.5}, (2, 5, 1, 7))["values"])
        assert np.allclose(half, -0.5 * want[2:5, 1:7], rtol=1e-6, atol=1e-7)
        tr = np.array(roi(api, src, (1, 7, 2, 5), view={**VIEW2D, "transpose": True})["values"])
        assert np.allclose(tr, want.T[1:7, 2:5], rtol=1e-6, atol=1e-7)
        eff = roi(
            api,
            {
                "kind": "virtual_effective",
                "adapter_id": ad["id"],
                "module_id": mod["module_id"],
                "experiment_multiplier": 1.0,
                "research_mode": False,
            },
            (0, 8, 0, 8),
        )
        w = np.array(roi(api, tensor_src(base, bt, f"demo.layers.0.mlp.{module}.weight"), (0, 8, 0, 8))["values"])
        assert np.allclose(np.array(eff["values"]), w + want, rtol=1e-6, atol=1e-7)


def test_unverified_lineage_needs_research_mode(api: Api, project: str, models_dir: Path, tmp_path: Path) -> None:
    base = import_model(api, project, models_dir, "tiny_base.safetensors")
    d = tmp_path / "copy_lora"
    d.mkdir()
    cfg = json.loads((models_dir / "tiny_lora" / "adapter_config.json").read_text())
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    (d / "adapter_model.safetensors").write_bytes(
        (models_dir / "tiny_lora" / "adapter_model.safetensors").read_bytes() + b""
    )
    # Same bytes as the registry adapter, so make it a different adapter by rescaling a factor.
    from weightlab import open_checkpoint

    ck = open_checkpoint(d / "adapter_model.safetensors")
    tensors = {
        m.key: (
            "F32",
            m.shape,
            (ck.read_roi_raw(m.key, 0, m.shape[0], 0, m.shape[1]).values() * 2).astype("<f4").tobytes(),
        )
        for m in ck.metadata()
    }
    ck.close()
    write_safetensors(d / "adapter_model.safetensors", tensors)
    ad = adapter(api, project, base, d)
    assert ad["lineage"]["compatibility_class"] == "L"  # declared only
    mod = ad["modules"][0]
    src = {
        "kind": "virtual_effective",
        "adapter_id": ad["id"],
        "module_id": mod["module_id"],
        "experiment_multiplier": 1.0,
        "research_mode": False,
    }
    assert roi(api, src, (0, 1, 0, 1), expect=409)["error"]["code"] == "LINEAGE_UNVERIFIED"
    ok = roi(api, {**src, "research_mode": True}, (0, 1, 0, 1))
    assert any("research_mode" in w for w in ok["warnings"])


def test_fan_in_fan_out_layout(api: Api, project: str, tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    w = rng.standard_normal((5, 3)).astype(np.float32)  # stored (in=5, out=3), GPT-2 Conv1D
    a = rng.standard_normal((2, 5)).astype(np.float32)  # lora_A (r, in)
    b = rng.standard_normal((3, 2)).astype(np.float32)  # lora_B (out, r)
    root = tmp_path / "m"
    write_safetensors(root / "model.safetensors", {"h.0.attn.c_attn.weight": ("F32", w.shape, w.tobytes())})
    d = tmp_path / "ad"
    write_safetensors(
        d / "adapter_model.safetensors",
        {
            "base_model.model.h.0.attn.c_attn.lora_A.weight": ("F32", a.shape, a.tobytes()),
            "base_model.model.h.0.attn.c_attn.lora_B.weight": ("F32", b.shape, b.tobytes()),
        },
    )
    (d / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none", "fan_in_fan_out": True})
    )
    base = import_model(api, project, root, "model.safetensors")
    ad = adapter(api, project, base, d)
    mod = ad["modules"][0]
    assert mod["transpose"] is True
    want = 2.0 * (b.astype(np.float64) @ a.astype(np.float64)).T  # (in, out), the stored layout
    src = {"kind": "lora_delta", "adapter_id": ad["id"], "module_id": mod["module_id"], "experiment_multiplier": 1.0}
    assert np.allclose(np.array(roi(api, src, (0, 5, 0, 3))["values"]), want, rtol=1e-12)
    assert np.allclose(
        np.array(roi(api, src, (1, 3, 0, 4), view={**VIEW2D, "transpose": True})["values"]),
        want.T[1:3, 0:4],
        rtol=1e-12,
    )


def test_unsupported_adapter_modules(api: Api, project: str, models_dir: Path) -> None:
    base = import_model(api, project, models_dir, "tiny_base.safetensors")
    ad = adapter(api, project, base, models_dir / "tiny_lora_dora_unsupported")
    assert ad["status"] == "unsupported"
    assert all(m["status"] == "unsupported" for m in ad["modules"])
    assert any("use_dora" in (m["reason"] or "") for m in ad["modules"])
    assert any("lora_magnitude_vector" in w for w in ad["warnings"])
    src = {
        "kind": "lora_delta",
        "adapter_id": ad["id"],
        "module_id": ad["modules"][0]["module_id"],
        "experiment_multiplier": 1.0,
    }
    assert roi(api, src, (0, 1, 0, 1), expect=409)["error"]["code"] == "INCOMPATIBLE_TENSOR"
    err = adapter(api, project, base, models_dir / "tiny_base.safetensors", expect=422)
    assert err["error"]["code"] == "VALIDATION_ERROR"  # not a directory


def test_lora_overflow_cells(api: Api, project: str, tmp_path: Path) -> None:
    w = np.array([[1.0, 1.0]])
    root = tmp_path / "m"
    write_safetensors(
        root / "model.safetensors", {"l.0.mlp.up_proj.weight": ("F64", w.shape, w.astype("<f8").tobytes())}
    )
    a, b = np.array([[1e200, 0.0]]), np.array([[1e200]])
    d = tmp_path / "ad"
    write_safetensors(
        d / "adapter_model.safetensors",
        {
            "base_model.model.l.0.mlp.up_proj.lora_A.weight": ("F64", a.shape, a.astype("<f8").tobytes()),
            "base_model.model.l.0.mlp.up_proj.lora_B.weight": ("F64", b.shape, b.astype("<f8").tobytes()),
        },
    )
    (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 1, "lora_alpha": 1, "bias": "none"}))
    base = import_model(api, project, root, "model.safetensors")
    ad = adapter(api, project, base, d)
    src = {
        "kind": "lora_delta",
        "adapter_id": ad["id"],
        "module_id": ad["modules"][0]["module_id"],
        "experiment_multiplier": 1.0,
    }
    got = roi(api, src, (0, 1, 0, 2))
    assert got["numeric_precision"] == "overflowed"
    assert got["values"] == [[None, 0.0]]
    assert got["nonfinite_mask"] == [["overflow", None]]
