"""Lineage evidence paths (ADR-014/018) and LoRA edge cases (doc 04 section 6) on hand-built inputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers import write_safetensors
from tests.integration.test_diff_golden import rows, validate, value
from tests.reference import close, pair_reference
from weightlab.analysis.change_report import Options, run
from weightlab.analysis.lineage import LineageInputs
from weightlab.analysis.lora import gram_frobenius_ssq

RNG = np.random.default_rng(1234)


def f32(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a, dtype="<f4").tobytes()


def model(path: Path, tensors: dict[str, np.ndarray], config: dict[str, Any] | None = None) -> Path:
    write_safetensors(path / "model.safetensors", {k: ("F32", v.shape, f32(v)) for k, v in tensors.items()})
    if config is not None:
        (path / "config.json").write_text(json.dumps(config))
    return path


W = {
    "model.layers.0.mlp.up_proj.weight": RNG.standard_normal((6, 4)).astype(np.float32),
    "model.layers.1.self_attn.q_proj.weight": RNG.standard_normal((4, 4)).astype(np.float32),
}


def adapter(path: Path, mods: dict[str, tuple[np.ndarray, np.ndarray]], **cfg: Any) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for module, (a, b) in mods.items():
        tensors[f"base_model.model.{module}.lora_A.weight"] = ("F32", a.shape, f32(a))
        tensors[f"base_model.model.{module}.lora_B.weight"] = ("F32", b.shape, f32(b))
    write_safetensors(path / "adapter_model.safetensors", tensors)
    conf = {
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 4,
        "use_rslora": False,
        "fan_in_fan_out": False,
        "use_dora": False,
        "bias": "none",
        "rank_pattern": {},
        "alpha_pattern": {},
        "base_model_name_or_path": "org/base",
        **cfg,
    }
    (path / "adapter_config.json").write_text(json.dumps(conf))
    return path


def ab(rank: int, n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    return (RNG.standard_normal((rank, n_in)).astype(np.float32), RNG.standard_normal((n_out, rank)).astype(np.float32))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# -- lineage ---------------------------------------------------------------------------------
def pair(tmp: Path) -> tuple[Path, Path]:
    base = model(tmp / "base", W)
    derived = model(tmp / "derived", {k: v + 0.01 for k, v in W.items()})
    return base, derived


def lineage_of(base: Path, derived: Path, lin: LineageInputs) -> dict[str, Any]:
    r = run(base, derived, lineage=lin)
    validate(r)
    return r["compatibility"]["lineage"]  # type: ignore[no-any-return]


def test_manifest_hash_match_is_class_v_but_not_trusted_ancestry(tmp_path: Path) -> None:
    base, derived = pair(tmp_path)
    manifest = {"schema_version": 1, "base_files": {"model.safetensors": sha(base / "model.safetensors")}}
    lin = lineage_of(base, derived, LineageInputs(manifest=manifest))
    assert (lin["compatibility_class"], lin["base_identity_method"], lin["lineage_trust_level"]) == (
        "V",
        "sha256_recorded",
        "unverified_claim",
    )
    assert lin["transfer_eligibility"] == "T4"  # V with an unverified claim is not T3
    trusted = lineage_of(base, derived, LineageInputs(manifest=manifest, trust_manifest=True))
    assert (trusted["lineage_trust_level"], trusted["transfer_eligibility"]) == ("trusted_statement", "T3")


def test_manifest_hash_mismatch_contradicts_lineage(tmp_path: Path) -> None:
    base, derived = pair(tmp_path)
    manifest = {"schema_version": 1, "base_files": {"model.safetensors": "0" * 64}}
    lin = lineage_of(base, derived, LineageInputs(manifest=manifest))
    assert lin["compatibility_class"] == "S"
    assert lin["base_file_identity_verified"] is False
    assert any("contradicted" in r for r in lin["reasons"])


def test_user_attestation_is_class_l_never_v(tmp_path: Path) -> None:
    base, derived = pair(tmp_path)
    r = run(base, derived, lineage=LineageInputs(user_attestation=True))
    lin = r["compatibility"]["lineage"]
    assert (lin["compatibility_class"], lin["lineage_evidence"], lin["lineage_claim_source"]) == (
        "L",
        "user_attested",
        "user_attestation",
    )
    assert all(f["sha256"] for side in r["provenance"]["inputs"].values() for f in side["files"])  # hashes recorded


def test_no_evidence_is_class_s(tmp_path: Path) -> None:
    base, derived = pair(tmp_path)
    lin = lineage_of(base, derived, LineageInputs())
    assert (lin["compatibility_class"], lin["lineage_evidence"]) == ("S", "none")


def hub_base(tmp: Path, org: str = "org", name: str = "base", corrupt_blob: bool = False) -> tuple[Path, str]:
    repo = tmp / "hub" / f"models--{org}--{name}"
    rev = "a" * 40
    snap = repo / "snapshots" / rev
    staging = model(tmp / "staging", W)
    blob_bytes = (staging / "model.safetensors").read_bytes()
    digest = hashlib.sha256(blob_bytes).hexdigest()
    (repo / "blobs").mkdir(parents=True)
    blob = repo / "blobs" / (("f" * 64) if corrupt_blob else digest)
    blob.write_bytes(blob_bytes)
    snap.mkdir(parents=True)
    try:
        os.symlink(os.path.relpath(blob, snap), snap / "model.safetensors")
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symlinks unavailable: {e}")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(rev)
    return snap, rev


@pytest.mark.parametrize(
    ("revision", "corrupt", "cls", "method"),
    [
        ("main", False, "V", "hf_cache_blob_sha256"),
        ("a" * 40, False, "V", "hf_cache_blob_sha256"),
        (None, False, "L", "none"),
        ("other", False, "L", "none"),
        ("main", True, "L", "none"),
    ],
)
def test_hub_cache_blob_verification(
    tmp_path: Path, revision: str | None, corrupt: bool, cls: str, method: str
) -> None:
    snap, _ = hub_base(tmp_path, corrupt_blob=corrupt)
    ad = adapter(tmp_path / "ad", {"model.layers.0.mlp.up_proj": ab(2, 4, 6)}, revision=revision)
    r = run(snap, adapter=ad)
    validate(r)
    lin = r["compatibility"]["lineage"]
    assert (lin["compatibility_class"], lin["base_identity_method"]) == (cls, method)
    if cls == "V":
        assert (lin["lineage_claim_source"], lin["lineage_trust_level"], lin["transfer_eligibility"]) == (
            "adapter_config",
            "unverified_claim",
            "T2",
        )


def test_hub_cache_declared_other_repo_is_contradiction(tmp_path: Path) -> None:
    snap, _ = hub_base(tmp_path)
    ad = adapter(tmp_path / "ad", {"model.layers.0.mlp.up_proj": ab(2, 4, 6)}, base_model_name_or_path="org/other")
    lin = run(snap, adapter=ad)["compatibility"]["lineage"]
    assert lin["compatibility_class"] == "S"


# -- LoRA edge cases --------------------------------------------------------------------------
def effective_reference(w: np.ndarray, d: np.ndarray) -> dict[str, object]:
    return pair_reference(w.astype(np.float64).ravel().tolist(), (w.astype(np.float64) + d).ravel().tolist())


def test_lora_scale_patterns_and_reference(tmp_path: Path) -> None:
    base = model(tmp_path / "base", W)
    a1, b1 = ab(2, 4, 6)
    a2, b2 = ab(3, 4, 4)
    ad = adapter(
        tmp_path / "ad",
        {"model.layers.0.mlp.up_proj": (a1, b1), "model.layers.1.self_attn.q_proj": (a2, b2)},
        rank_pattern={"q_proj": 3},
        alpha_pattern={"layers.1.self_attn.q_proj": 9},
    )
    r = run(base, adapter=ad)
    validate(r)
    up, q = rows(r)["model.layers.0.mlp.up_proj.weight"], rows(r)["model.layers.1.self_attn.q_proj.weight"]
    assert (up["lora"]["rank"], up["lora"]["scale"]) == (2, 2.0)
    assert (q["lora"]["rank"], q["lora"]["alpha"], q["lora"]["scale"]) == (3, 9.0, 3.0)
    for row, (a, b, s, key) in (
        (up, (a1, b1, 2.0, "model.layers.0.mlp.up_proj.weight")),
        (q, (a2, b2, 3.0, "model.layers.1.self_attn.q_proj.weight")),
    ):
        d = s * (b.astype(np.float64) @ a.astype(np.float64))
        ref = effective_reference(W[key], d)
        for name in ("delta_l2", "relative_l2", "cosine", "max_abs_delta"):
            assert close(value(row, name), ref[name], rtol=1e-12), name  # type: ignore[arg-type]


def test_fan_in_fan_out_conv1d_golden(tmp_path: Path) -> None:
    """GPT-2 Conv1D stores the weight (in, out); delta = s * (B @ A).T in that layout."""
    w = np.arange(12, dtype=np.float32).reshape(4, 3)  # in=4, out=3
    base = model(tmp_path / "base", {"transformer.h.0.attn.c_attn.weight": w})
    a = np.array([[1, 0, 0, 2]], dtype=np.float32)  # r=1, in=4
    b = np.array([[1], [0], [-1]], dtype=np.float32)  # out=3, r=1
    ad = adapter(tmp_path / "ad", {"transformer.h.0.attn.c_attn": (a, b)}, r=1, lora_alpha=2, fan_in_fan_out=True)
    r = run(base, adapter=ad)
    row = rows(r)["transformer.h.0.attn.c_attn.weight"]
    assert row["lora"]["transpose"] is True
    d = 2.0 * (b.astype(np.float64) @ a.astype(np.float64)).T  # hand-checkable: [[2,0,-2],[0,0,0],[0,0,0],[4,0,-4]]
    assert d.tolist() == [[2, 0, -2], [0, 0, 0], [0, 0, 0], [4, 0, -4]]
    ref = effective_reference(w, d)
    assert close(value(row, "delta_l2"), ref["delta_l2"])  # type: ignore[arg-type]
    assert value(row, "max_abs_delta") == 4.0
    assert row["counts"]["changed"] == 4
    plain = adapter(tmp_path / "ad2", {"transformer.h.0.attn.c_attn": (a, b)}, r=1, lora_alpha=2)
    bad = rows(run(base, adapter=plain))["transformer.h.0.attn.c_attn.weight"]
    assert bad["status"] == "unsupported"  # without fan_in_fan_out the shapes do not fit: never silently transposed


@pytest.mark.parametrize(
    ("mods", "cfg", "reason"),
    [
        ({"model.layers.0.mlp.up_proj": ab(3, 4, 6)}, {}, "rank mismatch"),
        ({"model.layers.0.mlp.up_proj": (ab(2, 4, 6)[1], ab(2, 4, 6)[0])}, {}, "rank mismatch"),  # A/B swapped
        ({"model.layers.0.mlp.missing": ab(2, 4, 6)}, {}, "missing source base"),
        ({"model.layers.0.mlp.up_proj": ab(2, 4, 6)}, {"layers_to_transform": [3]}, "outside layers_to_transform"),
        ({"model.layers.0.mlp.up_proj": ab(2, 4, 6)}, {"rank_pattern": {"^model.*(a+)+$": 2}}, "regex syntax"),
        ({"model.layers.0.mlp.up_proj": ab(2, 4, 6)}, {"bias": "all"}, "bias"),
        ({"model.layers.0.mlp.up_proj": ab(2, 4, 6)}, {"modules_to_save": ["lm_head"]}, "modules_to_save"),
        ({"model.layers.0.mlp.up_proj": ab(2, 4, 6)}, {"peft_type": "IA3"}, "not LORA"),
        ({"model.layers.0.mlp.up_proj": ab(2, 4, 6)}, {"lora_bias": True}, "lora_bias"),
    ],
)
def test_unsupported_modules_fail_closed(tmp_path: Path, mods: Any, cfg: dict[str, Any], reason: str) -> None:
    base = model(tmp_path / "base", W)
    r = run(base, adapter=adapter(tmp_path / "ad", mods, **cfg))
    validate(r)
    assert r["complete"] is False
    assert r["completeness"]["modules_unsupported"] >= 1
    assert any(reason in (t["caveat"] or "") for t in r["tensors"]), [t["caveat"] for t in r["tensors"]]


def test_unknown_adapter_keys_are_listed(tmp_path: Path) -> None:
    base = model(tmp_path / "base", W)
    ad = adapter(tmp_path / "ad", {"model.layers.0.mlp.up_proj": ab(2, 4, 6)})
    p = ad / "adapter_model.safetensors"
    a, b = ab(2, 4, 6)
    write_safetensors(
        p,
        {
            "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": ("F32", a.shape, f32(a)),
            "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": ("F32", b.shape, f32(b)),
            "base_model.model.model.embed_tokens.lora_embedding_A": ("F32", (2, 4), f32(np.ones((2, 4)))),
        },
    )
    r = run(base, adapter=ad)
    assert r["complete"] is False
    assert [e["reasons"] for e in r["compatibility"]["not_compared"]] == [["embedding LoRA"]]
    assert rows(r)["model.layers.0.mlp.up_proj.weight"]["status"] == "analyzed"


def test_over_budget_module_gets_gram_norm_only(tmp_path: Path) -> None:
    base = model(tmp_path / "base", W)
    a, b = ab(2, 4, 6)
    ad = adapter(tmp_path / "ad", {"model.layers.0.mlp.up_proj": (a, b)})
    full = rows(run(base, adapter=ad))["model.layers.0.mlp.up_proj.weight"]
    r = run(base, adapter=ad, options=Options(max_lora_gmac=1e-12))
    row = rows(r)["model.layers.0.mlp.up_proj.weight"]
    validate(r)
    assert row["status"] == "over_budget_norm_only"
    assert r["complete"] is False
    assert close(value(row, "delta_l2"), value(full, "delta_l2"), rtol=1e-12)  # type: ignore[arg-type]
    assert close(value(row, "relative_l2"), value(full, "relative_l2"), rtol=1e-12)  # type: ignore[arg-type]
    assert "cosine" not in row["metrics"]


def test_gram_trace_equals_direct_product() -> None:
    for r, n_in, n_out in ((1, 3, 5), (4, 17, 9), (16, 64, 128)):
        a = RNG.standard_normal((r, n_in))
        b = RNG.standard_normal((n_out, r))
        direct = float(np.linalg.norm(1.5 * (b @ a)))
        ssq, issue = gram_frobenius_ssq(a.copy(), b.copy(), 1.5)
        got = ssq.sqrt()
        assert issue is None
        assert got is not None
        assert abs(got - direct) <= 1e-12 * direct
