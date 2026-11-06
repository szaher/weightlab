"""Change report on the provided fixtures (WL-024 golden reports).

Numbers are checked against the exact Fraction reference built from expected_results.json (values
produced by the fixture generator, not by WeightLab); classes against `compat_expected_class`;
report fragments against the frozen contract (Metric, LineageAssessment) and the report schema.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from tests.helpers import ROOT
from tests.reference import close, pair_reference
from weightlab.analysis.change_report import Options, run
from weightlab.analysis.lineage import SYNTHETIC_DERIVATIVES, TINY_BASE

UP = "demo.layers.0.mlp.up_proj.weight"
DOWN = "demo.layers.0.mlp.down_proj.weight"
FIXED_TIME = Options(created_at="2026-11-06T00:00:00Z")


def _contract() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "contracts/openapi.yaml").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def validate(report: dict[str, Any]) -> None:
    schema = json.loads((ROOT / "weightlab/schemas/change_report.schema.json").read_text())
    jsonschema.validate(report, schema)
    comps = _contract()["components"]["schemas"]
    jsonschema.validate(report["compatibility"]["lineage"], comps["LineageAssessment"])
    for t in report["tensors"]:
        for m in t["metrics"].values():
            jsonschema.validate(m, comps["Metric"])
    json.dumps(report, allow_nan=False)


def rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["key"]: t for t in report["tensors"]}


def value(row: dict[str, Any], name: str) -> float | None:
    return row["metrics"][name]["value"]  # type: ignore[no-any-return]


def test_synthetic_registry_matches_generator_hashes(expected: dict[str, Any]) -> None:
    h = expected["file_sha256"]
    assert h["tiny_base.safetensors"] == TINY_BASE
    names = {v[0]: k for k, v in SYNTHETIC_DERIVATIVES.items()}
    assert h["tiny_ft.safetensors"] == names["tiny_ft.safetensors"]
    for d in ("tiny_lora", "tiny_lora_rslora", "tiny_lora_dora_unsupported"):
        assert h[f"{d}/adapter_model.safetensors"] == names[d]


def test_base_vs_ft_against_exact_reference(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    r = run(fixtures_dir / "tiny_base.safetensors", fixtures_dir / "tiny_ft.safetensors", options=FIXED_TIME)
    validate(r)
    lin = r["compatibility"]["lineage"]
    want = expected["compat_expected_class"]["tiny_base vs tiny_ft"]
    for k in ("base_file_identity_verified", "lineage_claim_source", "lineage_trust_level"):
        assert lin[k] == want[k]
    assert lin["compatibility_class"] == want["class"]
    assert r["complete"] is True
    up = rows(r)[UP]
    a = [x for row in expected["base_full"] for x in row]
    b = [x for row in expected["ft_full"] for x in row]
    ref = pair_reference(a, b)
    for name in ("delta_l2", "delta_rms", "relative_l2", "max_abs_delta", "cosine", "norm_a", "norm_b"):
        assert close(value(up, name), ref[name], rtol=1e-12), name  # type: ignore[arg-type]
    assert math.isclose(
        value(up, "delta_rms"), expected["rms_delta_vs_difference_of_rms"]["rms_of_delta"], rel_tol=1e-12
    )  # type: ignore[arg-type]
    assert up["counts"]["changed"] == len(expected["changed_positions"]) == 10
    assert r["tensors"][0]["key"] == UP  # only up_proj changed: it ranks first
    for key, row in rows(r).items():
        if key != UP:
            assert row["counts"]["changed"] == 0
            assert value(row, "delta_l2") == 0.0
    nf = rows(r)["demo.special_nonfinite.weight"]["counts"]
    assert (nf["invalid_pairs"], nf["nan_a"], nf["pos_inf_a"], nf["neg_inf_a"]) == (3, 1, 1, 1)


@pytest.mark.parametrize(
    ("a", "b", "cls", "status"),
    [
        ("tiny_base.safetensors", "compat/different_ancestry.safetensors", "S", None),
        ("tiny_base.safetensors", "compat/permuted_same_function.safetensors", "S", None),
        (
            "tiny_base.safetensors",
            "compat/shape_mismatch.safetensors",
            "X",
            ("demo.layers.0.mlp.up_proj.weight", "shape_mismatch"),
        ),
        ("compat/untied_reference", "compat/tied_embeddings", "S", ("lm_head.weight", "tied")),
        ("compat/untied_reference", "compat/missing_head_no_tie_evidence", "X", ("lm_head.weight", "missing_in_b")),
    ],
)
def test_compatibility_classes(fixtures_dir: Path, a: str, b: str, cls: str, status: tuple[str, str] | None) -> None:
    r = run(fixtures_dir / a, fixtures_dir / b, options=FIXED_TIME)
    validate(r)
    assert r["compatibility"]["lineage"]["compatibility_class"] == cls
    nc = {e["raw_key"]: e for e in r["compatibility"]["not_compared"]}
    if status is None:
        assert not nc
        assert r["complete"] is True
    else:
        key, st = status
        assert nc[key]["status"] == st
        if st == "tied":
            assert nc[key]["tied_to"] == "model.embed_tokens.weight"
            assert nc[key]["tie_evidence"] == "config_and_known_mapping"
            assert r["complete"] is True  # a tie with positive evidence is accounted for
        else:
            assert r["complete"] is False
    if cls in ("S", "X"):
        assert all(t["caveat"] and f"Class {cls}" in t["caveat"] for t in r["tensors"])
    if cls == "X":  # matched subset still inspectable
        assert r["tensors"]
        assert r["compatibility"]["lineage"]["transfer_eligibility"] == "T5"


def test_permuted_same_function_shows_large_raw_delta(fixtures_dir: Path) -> None:
    r = run(fixtures_dir / "tiny_base.safetensors", fixtures_dir / "compat/permuted_same_function.safetensors")
    up = rows(r)[UP]
    assert value(up, "relative_l2") > 0.5  # type: ignore[operator]  # identical function, large coordinate delta


def test_precision_pair_overflow_withheld(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    pr = expected["precision"]
    r = run(fixtures_dir / "tiny_precision_a.safetensors", fixtures_dir / "tiny_precision_b.safetensors")
    validate(r)
    f64 = rows(r)["p.f64"]
    assert f64["counts"]["delta_overflow"] == 1  # index f64_delta_overflow_index
    for name in ("delta_l2", "delta_rms", "relative_l2", "max_abs_delta"):
        assert value(f64, name) is None
        assert f64["metrics"][name]["numeric_precision"] == "overflowed"
    f32 = rows(r)["p.f32_overflow_delta"]
    assert value(f32, "max_abs_delta") == float(pr["f32_overflow_delta_b_minus_a_f64"])
    assert f32["metrics"]["max_abs_delta"]["numeric_precision"] == "rounded"
    assert r["totals"]["delta_l2"] is None  # a withheld member withholds the rollup
    assert r["totals"]["withheld_tensors"] == 1


def _lora_delta(expected: dict[str, Any], adapter: str, module: str) -> list[float]:
    return [x for row in expected["lora"][adapter][module]["delta_full"] for x in row]


@pytest.mark.parametrize("adapter", ["tiny_lora", "tiny_lora_rslora"])
def test_lora_effective_delta_against_reference(fixtures_dir: Path, expected: dict[str, Any], adapter: str) -> None:
    r = run(fixtures_dir / "tiny_base.safetensors", adapter=fixtures_dir / adapter, options=FIXED_TIME)
    validate(r)
    assert r["complete"] is True
    assert r["compatibility"]["lineage"]["compatibility_class"] == "V"
    assert r["compatibility"]["lineage"]["transfer_eligibility"] == "T1"  # V + proven_by_construction
    base = json.loads((fixtures_dir / "expected_results.json").read_text())
    for module, spec in expected["lora"][adapter].items():
        row = rows(r)[f"demo.layers.0.mlp.{module}.weight"]
        assert row["lora"]["rank"] == spec["rank"]
        assert row["lora"]["scale"] == spec["scale"]
        if "rank_from" in spec:
            assert row["lora"]["rank_from"] == spec["rank_from"]
        d = _lora_delta(expected, adapter, module)
        w = [x for rr in base["base_full"] for x in rr] if module == "up_proj" else None
        zeros = [0.0] * len(d)
        ref_d = pair_reference(zeros, d)
        # LoRA products: float64 matmul of float32 factors; the generator rounded its delta to float32.
        assert close(value(row, "delta_l2"), ref_d["delta_l2"], rtol=1e-6), module  # type: ignore[arg-type]
        assert close(value(row, "max_abs_delta"), ref_d["max_abs_delta"], rtol=1e-6)  # type: ignore[arg-type]
        assert row["counts"]["changed"] == ref_d["changed"]
        if w is not None:
            eff = [x + y for x, y in zip(w, d, strict=True)]
            ref = pair_reference(w, eff)
            assert close(value(row, "cosine"), ref["cosine"], rtol=1e-6)  # type: ignore[arg-type]
            assert close(value(row, "relative_l2"), ref["relative_l2"], rtol=1e-6)  # type: ignore[arg-type]
    if adapter == "tiny_lora_rslora":
        assert rows(r)[UP]["lora"]["scale_formula"] == "alpha/sqrt(r) (rsLoRA)"


def test_dora_adapter_fails_closed(fixtures_dir: Path) -> None:
    r = run(fixtures_dir / "tiny_base.safetensors", adapter=fixtures_dir / "tiny_lora_dora_unsupported")
    validate(r)
    assert r["complete"] is False
    assert r["completeness"]["modules_unsupported"] == 1
    assert r["completeness"]["modules_analyzed"] == 0
    assert any("use_dora" in (t["caveat"] or "") for t in r["tensors"])
    assert any(
        e["status"] == "unsupported_key" and "lora_magnitude_vector" in e["raw_key"]
        for e in r["compatibility"]["not_compared"]
    )
    assert r["caveats"][0].startswith("Incomplete")


def _normalize(report: dict[str, Any]) -> Any:
    """Golden comparison: floats to 12 significant digits (BLAS summation order may differ by an ulp
    across platforms), file times dropped."""

    def walk(v: Any) -> Any:
        if isinstance(v, float):
            return float(f"{v:.12g}")
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items() if k != "mtime_ns"}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    return walk(report)


@pytest.mark.parametrize(
    ("name", "a", "b", "adapter"),
    [
        ("base_vs_ft", "tiny_base.safetensors", "tiny_ft.safetensors", None),
        ("tied_vs_untied", "compat/untied_reference", "compat/tied_embeddings", None),
        ("base_plus_tiny_lora", "tiny_base.safetensors", None, "tiny_lora"),
    ],
)
def test_golden_reports(fixtures_dir: Path, name: str, a: str, b: str | None, adapter: str | None) -> None:
    r = run(
        fixtures_dir / a,
        fixtures_dir / b if b else None,
        adapter=fixtures_dir / adapter if adapter else None,
        options=FIXED_TIME,
    )
    r["provenance"]["tool"]["version"] = "<version>"
    golden = ROOT / "tests" / "golden" / f"{name}.json"
    got = _normalize(r)
    if os.environ.get("WEIGHTLAB_UPDATE_GOLDEN") == "1":
        golden.parent.mkdir(exist_ok=True)
        golden.write_text(json.dumps(got, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert got == json.loads(golden.read_text(encoding="utf-8"))
