"""Regression tests for the PR #2 review findings (numerics, resource limits, export safety).

Expected values are derived by hand or with Python integers / Fractions, never from WeightLab."""

from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import weightlab.analysis.lora as lora
import weightlab.analysis.report_io as report_io
from tests.helpers import write_safetensors
from tests.integration.test_diff_golden import rows, validate, value
from weightlab.analysis.change_report import Options, run
from weightlab.analysis.lora import gram_frobenius_ssq, working_set_bytes
from weightlab.cli import main
from weightlab.core.errors import ErrorCode, WeightLabError

UP = "model.layers.0.mlp.up_proj.weight"
MOD = "model.layers.0.mlp.up_proj"


def f64(path: Path, tensors: dict[str, np.ndarray]) -> Path:
    write_safetensors(
        path / "model.safetensors",
        {k: ("F64", v.shape, np.ascontiguousarray(v, "<f8").tobytes()) for k, v in tensors.items()},
    )
    return path


def adapter64(path: Path, a: np.ndarray, b: np.ndarray, **cfg: Any) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    write_safetensors(
        path / "adapter_model.safetensors",
        {
            f"base_model.model.{MOD}.lora_A.weight": ("F64", a.shape, np.ascontiguousarray(a, "<f8").tobytes()),
            f"base_model.model.{MOD}.lora_B.weight": ("F64", b.shape, np.ascontiguousarray(b, "<f8").tobytes()),
        },
    )
    conf = {"peft_type": "LORA", "r": a.shape[0], "lora_alpha": a.shape[0], "bias": "none",
            "base_model_name_or_path": "org/base", **cfg}  # fmt: skip
    (path / "adapter_config.json").write_text(json.dumps(conf))
    return path


def rollup(r: dict[str, Any], kind: str, label: str, v: Any) -> dict[str, Any]:
    return next(x for x in r[kind] if x[label] == v)


# -- 1: rollups use the same epsilon denominator as tensors ----------------------------------
@pytest.mark.parametrize("ref_value", [0.0, 1e-15])  # zero and near-zero reference
def test_rollup_relative_l2_applies_epsilon_policy(tmp_path: Path, ref_value: float) -> None:
    a = np.full((2, 2), ref_value)
    b = a + np.array([[3.0, 0.0], [0.0, 4.0]])  # ||B - A|| = 5 exactly
    r = run(f64(tmp_path / "a", {UP: a}), f64(tmp_path / "b", {UP: b}), options=Options(eps_norm=1e-12))
    validate(r)
    want = 5.0 / 1e-12  # ||A|| = 2 * ref_value < epsilon_norm
    assert math.isclose(value(rows(r)[UP], "relative_l2"), want, rel_tol=1e-12)  # type: ignore[arg-type]
    for row in (rollup(r, "layers", "layer", 0), rollup(r, "roles", "role", "mlp.up"), r["totals"]):
        assert math.isclose(row["relative_l2"], want, rel_tol=1e-12)
        assert row["near_zero_reference"] is True


# -- 2: non-finite or out-of-range limits are rejected ---------------------------------------
@pytest.mark.parametrize(
    "args",
    [["--epsilon-norm", "nan"], ["--epsilon-norm", "inf"], ["--epsilon-norm", "0"], ["--max-lora-gmac", "nan"],
     ["--max-lora-gmac", "inf"], ["--max-lora-gmac", "-1"], ["--chunk-elements", "0"]],
)  # fmt: skip
def test_cli_rejects_non_finite_limits(
    fixtures_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    out = tmp_path / "out"
    code = main(["diff", str(fixtures_dir / "tiny_base.safetensors"), "--adapter", str(fixtures_dir / "tiny_lora"),
                 "--out", str(out), "--quiet", *args])  # fmt: skip
    assert code == 2
    assert "VALIDATION_ERROR" in capsys.readouterr().err
    assert not out.exists()


def test_library_options_are_validated(fixtures_dir: Path) -> None:
    with pytest.raises(WeightLabError) as e:
        run(
            fixtures_dir / "tiny_base.safetensors",
            adapter=fixtures_dir / "tiny_lora",
            options=Options(max_lora_gmac=math.nan),
        )
    assert e.value.code is ErrorCode.VALIDATION_ERROR


# -- 3: 64-bit integer deltas are exact before rounding ---------------------------------------
I64_CASES = [
    ("I64", [2**53, -(2**63), 0, 7], [2**53 + 1, 2**63 - 1, 0, 7]),
    ("U64", [0, 2**53, 2**64 - 1, 5], [2**64 - 1, 2**53 + 1, 2**64 - 2, 5]),
]


@pytest.mark.parametrize(("dtype", "xs", "ys"), I64_CASES)
def test_wide_integer_deltas_are_not_lost(tmp_path: Path, dtype: str, xs: list[int], ys: list[int]) -> None:
    np_dtype = "<i8" if dtype == "I64" else "<u8"
    for side, vals in (("a", xs), ("b", ys)):
        write_safetensors(tmp_path / side / "model.safetensors",
                          {"t": (dtype, (len(vals),), np.array(vals, dtype=np_dtype).tobytes())})  # fmt: skip
    r = run(tmp_path / "a", tmp_path / "b")
    validate(r)
    row = rows(r)["t"]
    d = [y - x for x, y in zip(xs, ys, strict=True)]  # exact Python integers
    assert row["counts"]["changed"] == sum(1 for v in d if v != 0)
    assert value(row, "max_abs_delta") == float(max(abs(v) for v in d))  # correctly rounded exact max
    ssq = sum(Fraction(float(v)) ** 2 for v in d)  # each delta rounded once, then exact sum
    assert math.isclose(value(row, "delta_l2"), math.sqrt(ssq), rel_tol=1e-15)  # type: ignore[arg-type]


def test_random_int64_deltas_match_python_integers(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    a = rng.integers(-(2**63), 2**63 - 1, size=4096, dtype=np.int64)
    b = a.copy()
    b[::3] += rng.integers(-3, 4, size=b[::3].size)  # tiny changes on huge values
    b[::7] = rng.integers(-(2**63), 2**63 - 1, size=b[::7].size)
    for side, v in (("a", a), ("b", b)):
        write_safetensors(tmp_path / side / "model.safetensors", {"t": ("I64", v.shape, v.astype("<i8").tobytes())})
    row = rows(run(tmp_path / "a", tmp_path / "b", options=Options(chunk_elements=1000)))["t"]
    d = [int(y) - int(x) for x, y in zip(a.tolist(), b.tolist(), strict=True)]
    assert row["counts"]["changed"] == sum(1 for v in d if v)
    assert value(row, "max_abs_delta") == float(max(abs(v) for v in d))


# -- 4: Gram fallback is overflow/underflow safe and never a false zero ----------------------
@pytest.mark.parametrize("mag", [1e100, 1e-100, 1e154, 1e-150])
def test_gram_fallback_extreme_factors(mag: float) -> None:
    ssq, issue = gram_frobenius_ssq(np.array([[mag]]), np.array([[mag]]), 1.0)
    assert issue is None
    assert math.isclose(ssq.sqrt(), mag * mag, rel_tol=1e-15)  # type: ignore[arg-type]


def test_gram_fallback_rank1_1e100_through_report(tmp_path: Path) -> None:
    base = f64(tmp_path / "base", {UP: np.array([[2.0]])})
    ad = adapter64(tmp_path / "ad", np.array([[1e100]]), np.array([[1e100]]))
    r = run(base, adapter=ad, options=Options(max_lora_gmac=0.0))
    validate(r)
    row = rows(r)[UP]
    assert row["status"] == "over_budget_norm_only"
    assert math.isclose(value(row, "delta_l2"), 1e200, rel_tol=1e-15)  # type: ignore[arg-type]
    assert math.isclose(value(row, "relative_l2"), 1e200 / 2.0, rel_tol=1e-15)  # type: ignore[arg-type]


def test_gram_fallback_cancellation_and_nonfinite_are_withheld() -> None:
    a = np.array([[1.0, 2.0], [1.0, 2.0]])  # B @ A == 0 exactly, factors nonzero
    b = np.array([[1.0, -1.0]])
    _, issue = gram_frobenius_ssq(a, b, 1.0)
    assert issue is not None
    assert "cancellation" in issue
    _, issue = gram_frobenius_ssq(np.array([[np.nan]]), np.array([[1.0]]), 1.0)
    assert issue == "LoRA factors contain NaN/Inf"


# -- 5: LoRA arithmetic overflow is withheld and makes the report incomplete ------------------
@pytest.mark.parametrize(
    ("w", "fa", "fb", "counter"),
    [(1.0, 1e200, 1e200, "delta_overflow"),  # s B A = 1e400 from finite factors
     (1.5e308, 1.0, 1.5e308, "b_overflow")],  # s B A finite, W + s B A = 3e308
)  # fmt: skip
def test_lora_arithmetic_overflow_is_withheld(tmp_path: Path, w: float, fa: float, fb: float, counter: str) -> None:
    base = f64(tmp_path / "base", {UP: np.array([[w, 1.0]])})
    ad = adapter64(tmp_path / "ad", np.array([[fa, 0.0]]), np.array([[fb]]))
    r = run(base, adapter=ad)
    validate(r)
    row = rows(r)[UP]
    assert row["status"] == "analyzed"
    assert row["counts"][counter] == 1
    assert row["counts"]["invalid_pairs"] == 0  # not disguised as source NaN/Inf
    assert row["counts"]["pos_inf_b"] == 0
    for name in ("delta_l2", "delta_rms", "relative_l2", "max_abs_delta"):
        assert value(row, name) is None
        assert row["metrics"][name]["numeric_precision"] == "overflowed"
    if counter == "b_overflow":
        assert value(row, "cosine") is None
        assert value(row, "norm_b") is None
    assert r["complete"] is False
    assert r["completeness"]["modules_withheld"] == 1
    assert r["totals"]["delta_l2"] is None
    assert r["caveats"][0].startswith("Incomplete")


def test_pair_overflow_also_marks_report_incomplete(fixtures_dir: Path) -> None:
    r = run(fixtures_dir / "tiny_precision_a.safetensors", fixtures_dir / "tiny_precision_b.safetensors")
    assert r["complete"] is False
    assert r["completeness"]["tensors_withheld"] >= 1


# -- 6: hard working-set limit, checked before any allocation --------------------------------
def test_lora_working_set_limit_rejects_before_allocating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rank = 8192  # 1x1 base: tiny factors, but a 3 x 8192^2 float64 Gram scratch (1.5 GiB)
    assert working_set_bytes(1, 1, rank, True, lora.DEFAULT_BLOCK_CELLS) > lora.MAX_WORKING_SET_BYTES
    base = f64(tmp_path / "base", {UP: np.array([[1.0]])})
    ad = adapter64(tmp_path / "ad", np.ones((rank, 1)), np.ones((1, rank)))

    def boom(*_: object) -> None:
        raise AssertionError("factors loaded despite the working-set limit")

    monkeypatch.setattr(lora, "_load_whole", boom)
    r = run(base, adapter=ad, options=Options(max_lora_gmac=0.0))
    row = rows(r)[UP]
    assert row["status"] == "skipped"
    assert "resource limit" in row["caveat"]
    assert r["complete"] is False


# -- 7 and 8: over-budget metrics: epsilon policy, honest finite counts ----------------------
def test_over_budget_zero_base_uses_epsilon(tmp_path: Path) -> None:
    base = f64(tmp_path / "base", {UP: np.zeros((1, 2))})
    ad = adapter64(tmp_path / "ad", np.array([[3.0, 4.0]]), np.array([[1.0]]))  # s B A = [3, 4]
    row = rows(run(base, adapter=ad, options=Options(max_lora_gmac=0.0, eps_norm=1e-6)))[UP]
    assert math.isclose(value(row, "relative_l2"), 5.0 / 1e-6, rel_tol=1e-12)  # type: ignore[arg-type]
    assert any("near_zero_reference" in w for w in row["metrics"]["relative_l2"]["numerical_warnings"])


def test_over_budget_counts_nonfinite_base_and_factors(tmp_path: Path) -> None:
    base = f64(tmp_path / "base", {UP: np.array([[np.nan, 3.0, 4.0]])})
    ad = adapter64(tmp_path / "ad", np.array([[1.0, 0.0, 0.0]]), np.array([[1.0]]))
    r = run(base, adapter=ad, options=Options(max_lora_gmac=0.0))
    validate(r)
    row = rows(r)[UP]
    rel = row["metrics"]["relative_l2"]
    assert rel["finite_count"] == 2
    assert row["counts"]["base_nonfinite"] == 1
    assert math.isclose(rel["value"], 1.0 / 5.0, rel_tol=1e-12)
    assert any("NaN/Inf" in w for w in rel["numerical_warnings"])

    ad2 = adapter64(tmp_path / "ad2", np.array([[np.inf, 0.0, 0.0]]), np.array([[1.0]]))
    r2 = run(base, adapter=ad2, options=Options(max_lora_gmac=0.0))
    validate(r2)
    d = rows(r2)[UP]["metrics"]["delta_l2"]
    assert d["value"] is None
    assert d["numeric_precision"] == "unsupported"
    assert d["finite_count"] == 0
    assert r2["completeness"]["modules_withheld"] == 1


# -- 9: a local path in base_model_name_or_path is not exported without --include-paths --------
@pytest.mark.parametrize("declared", ["/home/alice/models/llama-base", "C:\\Users\\alice\\llama-base", "../llama-base",
                                      "~/llama-base", "a/b/llama-base"])  # fmt: skip
def test_declared_local_base_path_is_redacted(tmp_path: Path, declared: str) -> None:
    base = f64(tmp_path / "base", {UP: np.ones((1, 1))})
    ad = adapter64(tmp_path / "ad", np.ones((1, 1)), np.ones((1, 1)), base_model_name_or_path=declared)
    text = json.dumps(run(base, adapter=ad))
    hidden = declared.rsplit("llama-base", 1)[0].rstrip("/\\")
    assert hidden not in text
    assert "<local path redacted: llama-base>" in text
    shown = json.dumps(run(base, adapter=ad, options=Options(include_paths=True)))
    assert json.dumps(declared)[1:-1] in shown


def test_repo_id_is_not_redacted(tmp_path: Path) -> None:
    base = f64(tmp_path / "base", {UP: np.ones((1, 1))})
    ad = adapter64(tmp_path / "ad", np.ones((1, 1)), np.ones((1, 1)), base_model_name_or_path="Qwen/Qwen2.5-0.5B")
    assert run(base, adapter=ad)["provenance"]["inputs"]["adapter"]["declared_base"] == "Qwen/Qwen2.5-0.5B"


# -- 10: the report files are published as one set ------------------------------------------
def _files(d: Path) -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(d.iterdir())} if d.exists() else {}


def test_report_set_failure_while_staging_publishes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    real = report_io._stage
    calls = [0]

    def flaky(path: Path, text: str) -> Path:
        calls[0] += 1
        if calls[0] == 3:
            raise KeyboardInterrupt  # Ctrl-C in the middle of writing the set
        return real(path, text)

    monkeypatch.setattr(report_io, "_stage", flaky)
    files = {out / f"{n}.txt": n for n in ("a", "b", "c", "d")}
    with pytest.raises(KeyboardInterrupt):
        report_io.write_report_set(files, out)
    assert not out.exists()
    assert [p.name for p in tmp_path.iterdir()] == []  # no temporary directory left behind

    out.mkdir()
    for n in ("a", "b", "c", "d"):
        (out / f"{n}.txt").write_text("old")
    calls[0] = 0
    with pytest.raises(KeyboardInterrupt):
        report_io.write_report_set(files, out)
    assert _files(out) == {f"{n}.txt": "old" for n in ("a", "b", "c", "d")}  # old set intact, no temps


def test_report_set_failure_while_renaming_restores_old_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    for n in ("a", "b", "c"):
        (out / f"{n}.txt").write_text("old")
    real = os.replace
    seen = [0]

    def failing(src: Any, dst: Any) -> None:
        if str(dst) == str(out / "c.txt") and str(src).endswith(".tmp"):
            seen[0] += 1
            raise OSError("disk full")
        real(src, dst)

    monkeypatch.setattr(report_io.os, "replace", failing)
    with pytest.raises(OSError, match="disk full"):
        report_io.write_report_set({out / f"{n}.txt": "new" for n in ("a", "b", "c")}, out)
    assert seen[0] == 1
    assert _files(out) == {f"{n}.txt": "old" for n in ("a", "b", "c")}


# -- 11: absurd JSON numbers in adapter_config are validation errors, not crashes -------------
@pytest.mark.parametrize(
    ("cfg", "kind"),
    [({"lora_alpha": 10**400}, "config"), ({"r": 10**30}, "config"),
     ({"rank_pattern": {"up_proj": 10**400}}, "module"), ({"alpha_pattern": {"up_proj": 10**400}}, "module")],
)  # fmt: skip
def test_huge_json_numbers_fail_cleanly(tmp_path: Path, cfg: dict[str, Any], kind: str) -> None:
    base = f64(tmp_path / "base", {UP: np.ones((1, 1))})
    ad = adapter64(tmp_path / "ad", np.ones((1, 1)), np.ones((1, 1)), **cfg)
    if kind == "config":
        with pytest.raises(WeightLabError) as e:
            run(base, adapter=ad)
        assert e.value.code is ErrorCode.INVALID_FORMAT
        assert main(["diff", str(base), "--adapter", str(ad), "--quiet"]) == 2
    else:
        r = run(base, adapter=ad)
        assert rows(r)[UP]["status"] == "unsupported"
        assert r["complete"] is False
