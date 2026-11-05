#!/usr/bin/env python3
"""Validate generated synthetic fixtures against an INDEPENDENT reference.

The reference parser below reads SafeTensors with `struct`, `json` and
`numpy.frombuffer` only. It deliberately does not use the `safetensors` library,
which WeightLab's reader may use, so the two code paths can disagree and catch
each other's bugs. The `safetensors` library is used only to confirm that valid
fixtures are accepted and malformed fixtures are rejected by the upstream parser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

NP_DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U64": "<u8",
    "U32": "<u4",
    "U16": "<u2",
    "U8": "u1",
    "BOOL": "?",
}


def read_reference(path: Path) -> dict[str, np.ndarray]:
    """Minimal strict SafeTensors reader for small fixtures. Raises ValueError on any violation."""
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("file shorter than length prefix")
    (n,) = struct.unpack("<Q", raw[:8])
    if n > len(raw) - 8:
        raise ValueError("header length exceeds file")

    def no_dupes(pairs: list[tuple[str, object]]) -> dict[str, object]:
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(pairs)

    try:
        header = json.loads(raw[8 : 8 + n].decode("utf-8"), object_pairs_hook=no_dupes)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"corrupt header: {e}") from e
    header.pop("__metadata__", None)
    data = raw[8 + n :]
    spans: list[tuple[int, int]] = []
    out: dict[str, np.ndarray] = {}
    for key, spec in header.items():
        dtype, shape, (begin, end) = spec["dtype"], spec["shape"], spec["data_offsets"]
        if any((not isinstance(d, int)) or d < 0 for d in shape):
            raise ValueError("invalid shape")
        if dtype not in NP_DTYPES and dtype != "BF16":
            raise ValueError(f"unsupported dtype {dtype}")
        count = 1
        for d in shape:
            count *= d
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


def upstream_accepts(path: Path) -> bool:
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="np") as f:
            for k in f.keys():
                f.get_slice(k).get_shape()
        return True
    except Exception:
        return False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path) -> None:
    exp = json.loads((root / "expected_results.json").read_text())
    key = exp["tensor_key"]
    checks = 0

    for rel, digest in exp["file_sha256"].items():
        assert sha256(root / rel) == digest, f"hash mismatch {rel}"
        checks += 1

    base = read_reference(root / "tiny_base.safetensors")
    ft = read_reference(root / "tiny_ft.safetensors")
    np.testing.assert_array_equal(base[key], np.array(exp["base_full"], dtype=np.float32))  # all 64 cells
    np.testing.assert_array_equal(ft[key], np.array(exp["ft_full"], dtype=np.float32))
    delta = ft[key] - base[key]
    np.testing.assert_array_equal(delta, np.array(exp["delta_full"], dtype=np.float32))
    assert int(np.count_nonzero(delta)) == len(exp["changed_positions"]) == 10
    np.testing.assert_array_equal(-(base[key] - ft[key]), delta)  # antisymmetry
    for other in set(base) - {key}:
        np.testing.assert_array_equal(base[other], ft[other], err_msg=other)
    checks += 5

    o = exp["other_tensors"]
    assert base["demo.scalar"].shape == () and float(base["demo.scalar"]) == o["demo.scalar"]["value"]
    assert base["demo.one_by_one.weight"].tolist() == o["demo.one_by_one.weight"]["value"]
    assert base["demo.cube.weight"][1, 1:3, 2:5].tolist() == o["demo.cube.weight"]["slice_axis0_index1_rows_1_3_cols_2_5"]
    sp = base["demo.special_nonfinite.weight"]
    assert int(np.isnan(sp).sum()) == 1 and int(np.isposinf(sp).sum()) == 1 and int(np.isneginf(sp).sum()) == 1
    assert abs(float(sp[np.isfinite(sp)].astype(np.float64).mean()) - o["demo.special_nonfinite.weight"]["finite_mean"]) < 1e-12
    checks += 5

    d = read_reference(root / exp["dtypes"]["file"])
    ref = np.array(exp["dtypes"]["decoded_4x4"], dtype=np.float32)
    np.testing.assert_array_equal(d["demo.bf16.weight"], ref)
    np.testing.assert_array_equal(d["demo.f16.weight"].astype(np.float32), ref)
    bf = d["demo.bf16_specials"]
    assert bf[0] == 1 and bf[1] == -2 and np.isnan(bf[2]) and np.isposinf(bf[3]) and np.isneginf(bf[4])
    assert bf[5] == 0 and np.signbit(bf[6]) and float(bf[7]) == exp["dtypes"]["bf16_max_finite"]
    assert float(d["demo.f16_specials"][7]) == 65504.0
    checks += 4

    t = exp["tiles_257x263"]
    odd = base[t["key"]].astype(np.float64)
    assert list(odd[256:512, 256:512].shape) == t["level0_tile_1_1_shape"]
    for lvl in ("level1", "level3"):
        b = t[lvl]["block"]
        r0, c0 = (t[lvl]["output_shape"][0] - 1) * b, (t[lvl]["output_shape"][1] - 1) * b
        last = odd[r0:, c0:]
        assert last.size == t[lvl]["last_block_count"]
        assert abs(last.mean() - t[lvl]["last_block_mean"]) < 1e-12
    checks += 3

    def lora_pair(adir: Path, module: str) -> tuple[np.ndarray, np.ndarray]:
        w = read_reference(adir / "adapter_model.safetensors")
        prefix = f"base_model.model.demo.layers.0.mlp.{module}"
        return w[f"{prefix}.lora_A.weight"].astype(np.float64), w[f"{prefix}.lora_B.weight"].astype(np.float64)

    def native_scale(cfg: dict, module: str) -> tuple[int, float]:
        r = cfg.get("rank_pattern", {}).get(module, cfg["r"])
        alpha = cfg.get("alpha_pattern", {}).get(module, cfg["lora_alpha"])
        return r, (alpha / r**0.5 if cfg["use_rslora"] else alpha / r)

    for adapter, modules in exp["lora"].items():
        adir = root / adapter
        cfg = json.loads((adir / "adapter_config.json").read_text())
        if cfg.get("use_dora"):
            assert modules["expected"].startswith("reject")
            continue
        for module, m in modules.items():
            A, B = lora_pair(adir, module)
            r, s = native_scale(cfg, module)
            assert A.shape == (r, 8) and B.shape == (8, r) and r == m["rank"]
            assert s == m["scale"], (adapter, module, s)
            np.testing.assert_allclose(s * (B @ A), np.array(m["delta_full"]), rtol=0, atol=1e-12)
            checks += 1

    perm_m = read_reference(root / "compat" / "permuted_same_function.safetensors")
    up, dn = (
        base["demo.layers.0.mlp.up_proj.weight"].astype(np.float64),
        base["demo.layers.0.mlp.down_proj.weight"].astype(np.float64),
    )
    up2, dn2 = (
        perm_m["demo.layers.0.mlp.up_proj.weight"].astype(np.float64),
        perm_m["demo.layers.0.mlp.down_proj.weight"].astype(np.float64),
    )
    x = np.linspace(-1, 1, 8)
    np.testing.assert_allclose(dn @ np.maximum(up @ x, 0), dn2 @ np.maximum(up2 @ x, 0), atol=1e-12)
    assert not np.array_equal(up, up2)
    tied = read_reference(root / "compat" / "tied_embeddings" / "model.safetensors")
    assert "lm_head.weight" not in tied
    checks += 2

    for p in sorted(root.rglob("*.safetensors")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith("malformed/") and p.parent.name == "malformed":
            try:
                read_reference(p)
                raise AssertionError(f"reference parser accepted malformed {rel}")
            except ValueError:
                pass
            assert not upstream_accepts(p) or rel.endswith("duplicate_key.safetensors"), f"upstream accepted {rel}"
        else:
            read_reference(p)
            assert upstream_accepts(p), f"upstream rejected valid {rel}"
        checks += 1

    for name in ("missing_shard", "path_traversal", "absolute_path", "key_not_in_shard"):
        d_ = root / "malformed" / name
        idx = json.loads((d_ / "model.safetensors.index.json").read_text())
        problems = []
        for k, shard in idx["weight_map"].items():
            sp_ = Path(shard)
            if sp_.is_absolute() or ".." in sp_.parts:
                problems.append("path")
            elif not (d_ / shard).is_file():
                problems.append("missing")
            elif k not in read_reference(d_ / shard):
                problems.append("key")
        assert problems, f"malformed index {name} not detected"
        checks += 1

    print(
        f"PASS: {checks} checks; full 8x8 base/ft/delta, scalar/1x1/3D, BF16/F16 decode, "
        "tile edge blocks, LoRA (rank_pattern, rsLoRA, DoRA reject), compat, malformed rejection."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("examples/fixtures"))
    run(p.parse_args().root)
