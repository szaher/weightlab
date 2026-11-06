#!/usr/bin/env python3
"""Fixture SELF-CHECK against an independent reference parser.

Scope (docs/13 section 3): this validates the fixture files and expected_results.json.
It is NOT an application conformance test and is never G0 evidence on its own; the
WeightLab test suite must separately assert its reader's error codes and values.

The reference parser below uses only `struct`, `json` and `numpy.frombuffer`; it does
not use the `safetensors` library, which WeightLab's reader may use. The `safetensors`
library is used only as an accept/reject oracle, with the permitted differences listed
in expected_results.json. A few values are also hard-coded from hand calculation so the
checker does not rely solely on numbers produced by the generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

NP_DTYPES = {
    "F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4", "I16": "<i2",
    "I8": "i1", "U64": "<u8", "U32": "<u4", "U16": "<u2", "U8": "u1", "BOOL": "?",
}  # fmt: skip
MAX_HEADER_BYTES = 100_000_000  # same cap as upstream safetensors
MAX_RANK = 32
MAX_KEY_BYTES = 4096
ENTRY_KEYS = {"dtype", "shape", "data_offsets"}


def is_int(v: object) -> bool:
    """JSON integer that is not a bool (Python bool is a subclass of int)."""
    return type(v) is int


def strict_json(raw: bytes) -> object:
    """Strict JSON: UTF-8 only, no NaN/Infinity tokens, no duplicate keys, bounded nesting."""

    def no_dupes(pairs: list[tuple[str, object]]) -> dict[str, object]:
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(pairs)

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token}")

    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=no_dupes, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as e:
        raise ValueError(f"invalid header JSON: {type(e).__name__}") from e


def read_reference(path: Path) -> dict[str, np.ndarray]:
    """Minimal strict SafeTensors reader for small fixtures. Raises ValueError on any violation."""
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("file shorter than length prefix")
    (n,) = struct.unpack("<Q", raw[:8])
    if n > MAX_HEADER_BYTES or n > len(raw) - 8:
        raise ValueError("header length invalid")
    header = strict_json(raw[8 : 8 + n])
    if not isinstance(header, dict):
        raise ValueError("header is not an object")
    meta = header.pop("__metadata__", {})
    if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
        raise ValueError("__metadata__ must map strings to strings")
    data = raw[8 + n :]
    spans: list[tuple[int, int]] = []
    out: dict[str, np.ndarray] = {}
    for key, spec in header.items():
        if len(key.encode("utf-8")) > MAX_KEY_BYTES:
            raise ValueError("key too long")
        if not isinstance(spec, dict) or set(spec) != ENTRY_KEYS:
            raise ValueError("tensor entry must have exactly dtype, shape, data_offsets")
        dtype, shape, offsets = spec["dtype"], spec["shape"], spec["data_offsets"]
        if not isinstance(dtype, str) or (dtype not in NP_DTYPES and dtype != "BF16"):
            raise ValueError(f"unsupported dtype {dtype!r}")
        if not isinstance(shape, list) or len(shape) > MAX_RANK or not all(is_int(d) and d >= 0 for d in shape):
            raise ValueError("invalid shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(is_int(o) for o in offsets):
            raise ValueError("invalid data_offsets")
        begin, end = offsets
        count = math.prod(shape)  # exact Python integer arithmetic, no overflow
        size = 2 if dtype == "BF16" else np.dtype(NP_DTYPES[dtype]).itemsize
        if not (0 <= begin <= end <= len(data)) or end - begin != count * size:
            raise ValueError("offsets/shape mismatch")
        spans.append((begin, end))
        buf = data[begin:end]
        if dtype == "BF16":
            arr = (np.frombuffer(buf, dtype="<u2").astype("<u4") << 16).view("<f4")
        else:
            arr = np.frombuffer(buf, dtype=NP_DTYPES[dtype])
        out[key] = arr.reshape(shape)
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise ValueError("gap or overlap in data region")
        cursor = end
    if cursor != len(data):
        raise ValueError("trailing bytes in data region")
    return out


def index_problems(model_dir: Path) -> list[str]:
    """Reference validation of model.safetensors.index.json (docs/04 section 3)."""
    idx = strict_json((model_dir / "model.safetensors.index.json").read_bytes())
    problems = []
    for key, shard in idx["weight_map"].items():
        if not isinstance(shard, str):
            problems.append("non_string")
            continue
        if "\\" in shard or PureWindowsPath(shard).drive or PurePosixPath(shard).is_absolute():
            problems.append("not_relative_posix")
        elif ".." in PurePosixPath(shard).parts:
            problems.append("traversal")
        elif not (model_dir / shard).is_file():
            problems.append("missing")
        elif key not in read_reference(model_dir / shard):
            problems.append("key")
    return problems


def upstream_accepts(path: Path) -> bool:
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="np") as f:
            for k in f.keys():
                f.get_slice(k).get_shape()
        return True
    except Exception:
        return False


def run(root: Path) -> None:
    exp = json.loads((root / "expected_results.json").read_text(encoding="utf-8"))
    checks = 0

    def ok(cond: bool, msg: str) -> None:
        nonlocal checks
        assert cond, msg
        checks += 1

    for rel, digest in exp["file_sha256"].items():
        ok(hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, f"hash mismatch {rel}")

    # Hand-calculated constants (independent of the generator's own arithmetic).
    base = read_reference(root / "tiny_base.safetensors")
    ft = read_reference(root / "tiny_ft.safetensors")
    up = "demo.layers.0.mlp.up_proj.weight"
    ok(base[up][3, 5] == np.float32(29) / np.float32(20) - np.float32(1), "ramp[3,5] = 29/20 - 1")
    ok(base[up][0, 0] == np.float32(-1.0), "ramp[0,0] = -1")
    ok(ft[up][7, 7] == base[up][7, 7] + np.float32(1.0), "10th change adds (9+1)/10 = 1.0 at (7,7)")

    key = exp["tensor_key"]
    np.testing.assert_array_equal(base[key], np.array(exp["base_full"], dtype=np.float32))
    np.testing.assert_array_equal(ft[key], np.array(exp["ft_full"], dtype=np.float32))
    delta = ft[key] - base[key]
    np.testing.assert_array_equal(delta, np.array(exp["delta_full"], dtype=np.float32))
    ok(int(np.count_nonzero(delta)) == len(exp["changed_positions"]) == 10, "ten changed cells")
    for other in set(base) - {key}:
        np.testing.assert_array_equal(base[other], ft[other], err_msg=other)
    checks += 4

    r = exp["rms_delta_vs_difference_of_rms"]
    d64 = delta.astype(np.float64)
    ok(abs(math.sqrt(float(np.mean(d64 * d64))) - r["rms_of_delta"]) < 1e-15, "RMS(B-A)")
    ok(abs(r["rms_of_delta"] - r["rms_ft_minus_rms_base"]) > 1e-3, "RMS(B-A) differs from RMS(B)-RMS(A)")

    o = exp["other_tensors"]
    ok(base["demo.scalar"].shape == () and float(base["demo.scalar"]) == o["demo.scalar"]["value"], "scalar")
    ok(base["demo.one_by_one.weight"].tolist() == o["demo.one_by_one.weight"]["value"], "1x1")
    ok(base["demo.cube.weight"][1, 1:3, 2:5].tolist() == o["demo.cube.weight"]["slice_axis0_index1_rows_1_3_cols_2_5"], "3D")
    cs = exp["cube_slices"]
    ok(float(base["demo.cube.weight"][0].mean()) == cs["view_rows_axis1_cols_axis2_fixed_axis0_0_mean"] == 15.5, "slice 0")
    ok(float(base["demo.cube.weight"][1].mean()) == cs["view_rows_axis1_cols_axis2_fixed_axis0_1_mean"] == 47.5, "slice 1")
    sp = base["demo.special_nonfinite.weight"]
    ok(int(np.isnan(sp).sum()) == 1 and int(np.isposinf(sp).sum()) == 1 and int(np.isneginf(sp).sum()) == 1, "nonfinite")

    d = read_reference(root / exp["dtypes"]["file"])
    ref = np.array(exp["dtypes"]["decoded_4x4"], dtype=np.float32)
    np.testing.assert_array_equal(d["demo.bf16.weight"], ref)
    np.testing.assert_array_equal(d["demo.f16.weight"].astype(np.float32), ref)
    bf = d["demo.bf16_specials"]
    ok(bf[0] == 1 and bf[1] == -2 and np.isnan(bf[2]) and np.isposinf(bf[3]) and np.isneginf(bf[4]), "bf16 specials")
    ok(bf[5] == 0 and not np.signbit(bf[5]) and np.signbit(bf[6]) and float(bf[7]) == exp["dtypes"]["bf16_max_finite"], "bf16")
    ok(float(d["demo.f16_specials"][7]) == 65504.0, "f16 max")

    pr = exp["precision"]
    pa, pb = (read_reference(root / f) for f in pr["files"])
    ok([repr(float(v)) for v in pa["p.f64"]] == pr["f64_a"], "f64 a exact")
    ok([repr(float(v)) for v in pb["p.f64"]] == pr["f64_b"], "f64 b exact")
    ok(math.copysign(1.0, float(pa["p.f64"][4])) == -1.0, "signed zero preserved")
    with np.errstate(over="ignore"):
        fd = pb["p.f64"] - pa["p.f64"]
    ok(bool(np.isposinf(fd[pr["f64_delta_overflow_index"]])) and np.isfinite(pa["p.f64"]).all(), "finite inputs, inf delta")
    ok(float(fd[0]) == 2.0**-52, "F64 tiny difference preserved")
    ok([str(int(v)) for v in pa["p.i64"]] == pr["i64"] == [str(v) for v in
       [-(2**63), -1, 0, 2**53 - 1, 2**53, 2**53 + 1, 2**63 - 1]], "i64 exact")  # fmt: skip
    ok([str(int(v)) for v in pa["p.u64"]] == pr["u64"] and int(pa["p.u64"][2]) == 2**64 - 1, "u64 exact")
    ok(float(int(pa["p.i64"][5])) != int(pa["p.i64"][5]), "2^53+1 is not representable as float64")
    f32d = float(pb["p.f32_overflow_delta"][0].astype(np.float64) - pa["p.f32_overflow_delta"][0].astype(np.float64))
    ok(f32d > float(np.finfo(np.float32).max) and repr(f32d) == pr["f32_overflow_delta_b_minus_a_f64"], "f32 delta > f32 max")
    ok(pr["f64_sumsq_overflow"]["sum_sq_finite"] is False, "sum of squares overflows for 1e200")

    e = exp["edge"]
    ed = read_reference(root / e["file"])
    ok(ed["edge.empty_rows"].shape == (0, 4) and ed["edge.empty_cols"].shape == (3, 0), "empty tensors legal")
    ok(e["unicode_key"] in ed and e["unicode_key"].startswith("edge.ü"), "unicode key")
    big = ed[e["big_block"]["key"]].astype(np.float64)
    ok(big[:256, :256].size == e["big_block"]["level8_block_0_0_finite_count"] == 65536 > 65535, "count exceeds uint16")
    ok(float(big[:256, :256].sum()) == e["big_block"]["level8_block_0_0_sum"], "L8 sum")
    ok(float(big.sum()) == e["big_block"]["level9_block_0_0_sum"] and big.size == 131072, "L9 merge sum")
    nb = e["nonfinite_blocks"]
    bits = nb["status_bits"]
    ok(nb["level0_cells"]["3,20"]["status"] == bits["NO_FINITE"] | bits["HAS_NAN"], "L0 NaN keeps its kind")
    ok(nb["level0_cells"]["4,21"]["status"] == bits["NO_FINITE"] | bits["HAS_POS_INF"], "L0 +Inf keeps its kind")
    ok(nb["level0_cells"]["16,1"]["status"] == bits["NO_FINITE"] | bits["HAS_NEG_INF"], "L0 -Inf keeps its kind")
    ok(nb["level0_cells"]["0,0"]["status"] == 0, "finite cell")
    blk = nb["level4_blocks_mean"]
    ok(blk["0,0"]["status"] == 0 and blk["0,0"]["value"] is not None, "finite block")
    ok(blk["0,1"]["status"] == bits["HAS_NAN"] | bits["HAS_POS_INF"] and blk["0,1"]["value"] is not None,
       "mixed block keeps finite aggregate")  # fmt: skip
    ok(blk["1,0"]["status"] == 15 and blk["1,0"]["value"] is None, "all-nonfinite mixed block")
    ok(blk["1,1"]["status"] == bits["NO_FINITE"] | bits["HAS_NAN"] and blk["1,1"]["value"] is None, "all-NaN block")
    nfa = ed[nb["key"]].astype(np.float64)
    f01 = nfa[0:16, 16:32][np.isfinite(nfa[0:16, 16:32])]
    ok(f01.size == 254 and float(f01.mean()) == blk["0,1"]["value"], "mixed block finite mean")

    t = exp["tiles_257x263"]
    odd = base[t["key"]].astype(np.float64)
    ok(list(odd[256:512, 256:512].shape) == t["level0_tile_1_1_shape"], "edge tile shape")
    for lvl in ("level1", "level3"):
        b = t[lvl]["block"]
        r0, c0 = (t[lvl]["output_shape"][0] - 1) * b, (t[lvl]["output_shape"][1] - 1) * b
        last = odd[r0:, c0:]
        ok(last.size == t[lvl]["last_block_count"] and abs(last.mean() - t[lvl]["last_block_mean"]) < 1e-12, lvl)

    def native_scale(cfg: dict, module: str) -> tuple[int, float]:
        r = cfg.get("rank_pattern", {}).get(module, cfg["r"])
        alpha = cfg.get("alpha_pattern", {}).get(module, cfg["lora_alpha"])
        return r, (alpha / r**0.5 if cfg["use_rslora"] else alpha / r)

    for adapter, modules in exp["lora"].items():
        adir = root / adapter
        cfg = json.loads((adir / "adapter_config.json").read_text())
        if cfg.get("use_dora"):
            ok(modules["expected"].startswith("reject"), "dora rejected")
            continue
        w = read_reference(adir / "adapter_model.safetensors")
        for module, m in modules.items():
            prefix = f"base_model.model.demo.layers.0.mlp.{module}"
            A, B = w[f"{prefix}.lora_A.weight"].astype(np.float64), w[f"{prefix}.lora_B.weight"].astype(np.float64)
            r_, s = native_scale(cfg, module)
            ok(A.shape == (r_, 8) and B.shape == (8, r_) and r_ == m["rank"] and s == m["scale"], f"{adapter}/{module}")
            np.testing.assert_allclose(s * (B @ A), np.array(m["delta_full"]), rtol=0, atol=1e-12)
    ok(exp["lora"]["tiny_lora"]["up_proj"]["delta_full"][0][0] == 2.0, "hand: 2*(B[0]·A[:,0]) = 2*(1*1) = 2")

    pm = read_reference(root / "compat" / "permuted_same_function.safetensors")
    dn = "demo.layers.0.mlp.down_proj.weight"
    x = np.linspace(-1, 1, 8)
    y1 = base[dn].astype(np.float64) @ np.maximum(base[up].astype(np.float64) @ x, 0)
    y2 = pm[dn].astype(np.float64) @ np.maximum(pm[up].astype(np.float64) @ x, 0)
    np.testing.assert_allclose(y1, y2, atol=1e-12)
    ok(not np.array_equal(base[up], pm[up]), "permuted weights differ")
    tied_cfg = json.loads((root / "compat" / "tied_embeddings" / "config.json").read_text())
    ok("lm_head.weight" not in read_reference(root / "compat" / "tied_embeddings" / "model.safetensors")
       and tied_cfg["tie_word_embeddings"] is True, "tied: positive config evidence")  # fmt: skip
    nt = root / "compat" / "missing_head_no_tie_evidence"
    ok("lm_head.weight" not in read_reference(nt / "model.safetensors") and not (nt / "config.json").exists(),
       "missing head without tie evidence")  # fmt: skip

    permitted = exp["upstream_permitted_differences"]
    for p in sorted(root.rglob("*.safetensors")):
        rel = p.relative_to(root).as_posix()
        if p.parent.name == "malformed":
            try:
                read_reference(p)
                raise AssertionError(f"reference parser accepted malformed {rel}")
            except ValueError:
                pass
            ok(p.name in exp["malformed_expected"], f"expected code listed for {rel}")
            ok(not upstream_accepts(p) or p.name in permitted, f"upstream accepted {rel} without a permitted difference")
        else:
            read_reference(p)
            ok(upstream_accepts(p), f"upstream rejected valid {rel}")

    for name in [k.rstrip("/") for k in exp["malformed_expected"] if k.endswith("/")]:
        ok(bool(index_problems(root / "malformed" / name)), f"malformed index {name} not detected")
    ok(index_problems(root / "tiny_sharded") == [], "valid index accepted")

    print(f"PASS: {checks} fixture self-checks (not application conformance; see docs/13 section 3).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("examples/fixtures"))
    run(p.parse_args().root)
