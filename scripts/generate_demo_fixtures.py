#!/usr/bin/env python3
"""Generate SMALL, deterministic synthetic WeightLab demo fixtures.

This is supporting test material, NOT a WeightLab application implementation.
Requires numpy and safetensors (`pip install numpy safetensors`).
Does not download any model or connect to the network.

Determinism: every value comes from integer arithmetic followed by IEEE-754
correctly rounded division/conversion (no transcendental functions), and every
pseudo-random value from the SplitMix64 function below, never from NumPy's RNG.
Byte reproducibility is claimed only under the pinned toolchain in
scripts/requirements-spec-tools.txt, because the SafeTensors serializer is a
third-party component; CI compares hashes on Linux, macOS and Windows.
SplitMix64 is also the reference for WeightLab's seeded sampling (docs/05 section 5).

Files that contain BF16/F16 or that are intentionally malformed are written with
the raw writer `write_raw_safetensors`, because the safetensors NumPy backend
cannot represent BF16 and validly refuses to write malformed files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file as _upstream_save_file

GENERATOR_VERSION = 3
MASK_NO_FINITE, MASK_NAN, MASK_POS_INF, MASK_NEG_INF, MASK_NOT_REPRESENTABLE = 1, 2, 4, 8, 16
F32_MAX = float(np.finfo(np.float32).max)
MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------- determinism
def splitmix64(x: int) -> int:
    """SplitMix64 finalizer: maps a 64-bit counter to a 64-bit pseudo-random value."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def uniform_matrix(seed: int, stream: int, shape: tuple[int, ...], quantum: int = 1024) -> np.ndarray:
    """Deterministic values in [-1, 1] as float32, independent of NumPy version.

    Values are rounded to multiples of 1/quantum; quantum=128 keeps them exactly
    representable in BF16 (8 significant bits), quantum=1024 in F16.
    """
    n = int(np.prod(shape))
    vals = [(splitmix64((seed << 32) ^ (stream << 24) ^ i) >> 11) / float(1 << 53) * 2.0 - 1.0 for i in range(n)]
    return (np.round(np.array(vals, dtype=np.float64) * quantum) / quantum).astype(np.float32).reshape(shape)


# ---------------------------------------------------------------- raw writer
DTYPE_SIZE = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}


def f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    """Truncate float32 to bfloat16 bit patterns (inputs chosen to be exactly representable)."""
    bits = np.ascontiguousarray(a, dtype="<f4").view("<u4")
    return (bits >> 16).astype("<u2")


def write_raw_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
    metadata: dict[str, str] | None = None,
    *,
    header_override: bytes | None = None,
    length_override: int | None = None,
    truncate_to: int | None = None,
) -> None:
    """Write a SafeTensors file from raw bytes. Overrides exist only to build malformed fixtures."""
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    payload = b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        payload += data
    hbytes = header_override if header_override is not None else json.dumps(header, separators=(",", ":")).encode()
    hbytes += b" " * (-len(hbytes) % 8)  # spec allows trailing-space padding to 8-byte alignment
    blob = struct.pack("<Q", length_override if length_override is not None else len(hbytes)) + hbytes + payload
    if truncate_to is not None:
        blob = blob[:truncate_to]
    path.write_bytes(blob)


def save_file(tensors: dict[str, np.ndarray], path: str, metadata: dict[str, str] | None = None) -> None:
    """Wrapper around safetensors.numpy.save_file that works around two upstream behaviours
    observed with safetensors 0.8.0:
      * non-C-contiguous arrays (e.g. `a[:, perm]`) are written in memory order, silently
        transposing the data, so every array is made C-contiguous first (np.require, because
        np.ascontiguousarray turns 0-d arrays into 1-d);
      * `__metadata__` key order is not stable between runs, so at most one metadata key
        is allowed to keep file bytes reproducible.
    """
    if metadata is not None and len(metadata) > 1:
        raise ValueError("use a single metadata key for byte-reproducible fixtures")
    _upstream_save_file({k: np.require(v, requirements="C") for k, v in tensors.items()}, path, metadata=metadata)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def block_mean_edges(a: np.ndarray, block: int) -> dict[str, object]:
    """Reference for count-weighted aggregation at partial edge blocks."""
    rows, cols = a.shape
    out_r, out_c = -(-rows // block), -(-cols // block)
    last = a[(out_r - 1) * block :, (out_c - 1) * block :]
    return {
        "block": block,
        "output_shape": [out_r, out_c],
        "last_block_count": int(last.size),
        "last_block_mean": float(last.astype(np.float64).mean()),
        "first_block_mean": float(a[:block, :block].astype(np.float64).mean()),
    }


# ---------------------------------------------------------------- generation
def generate(out: Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source": f"WeightLab synthetic demo v{GENERATOR_VERSION}; NOT a trained model"}

    ramp = (np.arange(64, dtype=np.float32).reshape(8, 8) / np.float32(20) - 1).copy()
    odd = (
        ((((np.arange(257 * 263, dtype=np.int64) * 7919 + 13) % 2001) - 1000).astype(np.float64) / 1000)
        .astype(np.float32)
        .reshape(257, 263)
    )
    special = np.array([[0, 1, np.nan, np.inf], [-np.inf, -1, 2, 3]], dtype=np.float32)
    vector = np.array([1, 2, -3, 0, 0.5, 5], dtype=np.float32)
    cube = np.arange(2 * 4 * 8, dtype=np.float32).reshape(2, 4, 8)
    down = uniform_matrix(seed, 1, (8, 8))
    base = {
        "demo.layers.0.mlp.up_proj.weight": ramp,
        "demo.layers.0.mlp.down_proj.weight": down,
        "demo.layers.0.self_attn.q_proj.weight": odd,
        "demo.special_nonfinite.weight": special,
        "demo.vector.weight": vector,
        "demo.cube.weight": cube,
        "demo.zeros.weight": np.zeros((2, 3), dtype=np.float32),
        "demo.scalar": np.array(3.5, dtype=np.float32),
        "demo.one_by_one.weight": np.array([[-0.25]], dtype=np.float32),
    }
    changed = {key: value.copy() for key, value in base.items()}
    positions = [(0, 0), (0, 1), (1, 1), (2, 4), (3, 5), (4, 4), (5, 2), (6, 7), (7, 0), (7, 7)]
    for index, (r, c) in enumerate(positions):
        changed["demo.layers.0.mlp.up_proj.weight"][r, c] += np.float32((index + 1) / 10)
    save_file(base, str(out / "tiny_base.safetensors"), metadata=meta)
    save_file(changed, str(out / "tiny_ft.safetensors"), metadata=meta)

    # Reduced-precision dtypes. Values are multiples of 1/128 in [-1,1] plus specials,
    # exactly representable in both F16 and BF16. The last special is each dtype's max finite.
    small = uniform_matrix(seed, 2, (4, 4), quantum=128)
    common = [1.0, -2.0, np.nan, np.inf, -np.inf, 0.0, -0.0]
    bf16_specials = np.array(common + [3.3895313892515355e38], dtype=np.float32)  # 0x7F7F
    f16_specials = np.array(common + [65504.0], dtype=np.float32)
    write_raw_safetensors(
        out / "tiny_dtypes.safetensors",
        {
            "demo.bf16.weight": ("BF16", [4, 4], f32_to_bf16_bits(small).tobytes()),
            "demo.bf16_specials": ("BF16", [8], f32_to_bf16_bits(bf16_specials).tobytes()),
            "demo.f16.weight": ("F16", [4, 4], small.astype("<f2").tobytes()),
            "demo.f16_specials": ("F16", [8], f16_specials.astype("<f2").tobytes()),
        },
        meta,
    )

    # Precision edge cases: F64 extremes, 64-bit integers beyond 2^53, signed zero,
    # deltas that overflow although both inputs are finite, sum-of-squares overflow.
    f64_a = [1.0, 1e300, -1e300, 5e-324, -0.0, 1.7976931348623157e308, -1.7976931348623157e308, 2.2250738585072014e-308]
    f64_b = [1.0 + 2.0**-52, 1e300, -1e300, 5e-324, 0.0, 1.7976931348623157e308, 1.7976931348623157e308, 2.2250738585072014e-308]
    i64 = [-(2**63), -1, 0, 2**53 - 1, 2**53, 2**53 + 1, 2**63 - 1]
    u64 = [0, 2**53 + 1, 2**64 - 1]
    precision_common = {
        "p.i64": ("I64", [7], np.array(i64, dtype="<i8").tobytes()),
        "p.u64": ("U64", [3], np.array(u64, dtype="<u8").tobytes()),
        "p.i32": ("I32", [2], np.array([-(2**31), 2**31 - 1], dtype="<i4").tobytes()),
        "p.bool": ("BOOL", [2], np.array([True, False], dtype="?").tobytes()),
        "p.f64_sumsq_overflow": ("F64", [4], np.array([1e200] * 4, dtype="<f8").tobytes()),
    }
    write_raw_safetensors(
        out / "tiny_precision_a.safetensors",
        {
            "p.f64": ("F64", [8], np.array(f64_a, dtype="<f8").tobytes()),
            "p.f32_overflow_delta": ("F32", [2], np.array([-3.4e38, 1.0], dtype="<f4").tobytes()),
            **precision_common,
        },
        meta,
    )
    write_raw_safetensors(
        out / "tiny_precision_b.safetensors",
        {
            "p.f64": ("F64", [8], np.array(f64_b, dtype="<f8").tobytes()),
            "p.f32_overflow_delta": ("F32", [2], np.array([3.4e38, 1.0], dtype="<f4").tobytes()),
            **precision_common,
        },
        meta,
    )

    # Edge shapes, unicode key, >65,535-cell aggregate blocks and non-finite tile blocks.
    nf = ((np.arange(32 * 32, dtype=np.int64) % 97) - 48).astype(np.float64).astype(np.float32).reshape(32, 32) / np.float32(64)
    nf[3, 20] = np.nan  # block (0,1): finite + NaN + +Inf
    nf[4, 21] = np.inf
    nf[16:32, 0:16] = np.nan  # block (1,0): all non-finite, all three kinds
    nf[16, 0] = np.inf
    nf[16, 1] = -np.inf
    nf[16:32, 16:32] = np.nan  # block (1,1): all NaN
    big = ((np.arange(256 * 512, dtype=np.int64) % 17) - 8).astype(np.float32).reshape(256, 512) / np.float32(8)
    write_raw_safetensors(
        out / "tiny_edge.safetensors",
        {
            "edge.empty_rows": ("F32", [0, 4], b""),
            "edge.empty_cols": ("F32", [3, 0], b""),
            "edge.\u00fcn\u00efc\u00f8d\u00e9/\u03ba\u03bb\u03b5\u03b9\u03b4\u03af.weight": (
                "F32",
                [2, 2],
                np.array([1, 2, 3, 4], dtype="<f4").tobytes(),
            ),
            "edge.nonfinite_blocks.weight": ("F32", [32, 32], nf.astype("<f4").tobytes()),
            "edge.block256x512.weight": ("F32", [256, 512], big.astype("<f4").tobytes()),
        },
        meta,
    )

    # Sharded model with a valid index.
    shard_dir = out / "tiny_sharded"
    shard_dir.mkdir(exist_ok=True)
    shard_a, shard_b = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    save_file({"demo.layers.0.mlp.up_proj.weight": ramp}, str(shard_dir / shard_a))
    save_file({"demo.vector.weight": vector}, str(shard_dir / shard_b))
    (shard_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": int(ramp.nbytes + vector.nbytes)},
                "weight_map": {"demo.layers.0.mlp.up_proj.weight": shard_a, "demo.vector.weight": shard_b},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # LoRA adapters. Key layout follows PEFT's saved format: adapter name stripped.
    up_mod = "base_model.model.demo.layers.0.mlp.up_proj"
    down_mod = "base_model.model.demo.layers.0.mlp.down_proj"
    A_up = np.array([[1, 0, 0, 0, 0.5, 0, 0, 0], [0, 0.5, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
    B_up = np.array([[1, 0], [0, 1], [1, 1], [-1, 0], [0, 0.5], [0, 1], [1, -1], [0.2, 0.4]], dtype=np.float32)
    A_down = np.array([[0.5, 0, -1, 0, 0, 0.25, 0, 0]], dtype=np.float32)
    B_down = np.array([[1], [0], [0], [2], [0], [0], [-1], [0]], dtype=np.float32)
    lora_base_cfg = {
        "base_model_name_or_path": "tiny-base-fixture",
        "bias": "none",
        "inference_mode": True,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "fan_in_fan_out": False,
        "use_dora": False,
        "modules_to_save": None,
        "layers_to_transform": None,
    }
    adapter = out / "tiny_lora"
    adapter.mkdir(exist_ok=True)
    save_file(
        {
            f"{up_mod}.lora_A.weight": A_up,
            f"{up_mod}.lora_B.weight": B_up,
            f"{down_mod}.lora_A.weight": A_down,
            f"{down_mod}.lora_B.weight": B_down,
        },
        str(adapter / "adapter_model.safetensors"),
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                **lora_base_cfg,
                "r": 2,
                "lora_alpha": 4,
                "use_rslora": False,
                "rank_pattern": {"down_proj": 1},
                "alpha_pattern": {},
                "target_modules": ["up_proj", "down_proj"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    A_rs = uniform_matrix(seed, 3, (4, 8))
    B_rs = uniform_matrix(seed, 4, (8, 4))
    rs = out / "tiny_lora_rslora"
    rs.mkdir(exist_ok=True)
    save_file({f"{up_mod}.lora_A.weight": A_rs, f"{up_mod}.lora_B.weight": B_rs}, str(rs / "adapter_model.safetensors"))
    (rs / "adapter_config.json").write_text(
        json.dumps(
            {
                **lora_base_cfg,
                "r": 4,
                "lora_alpha": 8,
                "use_rslora": True,
                "rank_pattern": {},
                "alpha_pattern": {},
                "target_modules": ["up_proj"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    dora = out / "tiny_lora_dora_unsupported"
    dora.mkdir(exist_ok=True)
    save_file(
        {
            f"{up_mod}.lora_A.weight": A_up,
            f"{up_mod}.lora_B.weight": B_up,
            f"{up_mod}.lora_magnitude_vector": np.ones(8, dtype=np.float32),
        },
        str(dora / "adapter_model.safetensors"),
    )
    (dora / "adapter_config.json").write_text(
        json.dumps(
            {
                **lora_base_cfg,
                "r": 2,
                "lora_alpha": 4,
                "use_rslora": False,
                "use_dora": True,
                "rank_pattern": {},
                "alpha_pattern": {},
                "target_modules": ["up_proj"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Compatibility fixtures.
    compat = out / "compat"
    compat.mkdir(exist_ok=True)
    other = {k: (uniform_matrix(seed, 10 + i, v.shape) if v.ndim else v.copy()) for i, (k, v) in enumerate(base.items())}
    save_file(other, str(compat / "different_ancestry.safetensors"), metadata=meta)
    mismatch = {k: v.copy() for k, v in base.items()}
    mismatch["demo.layers.0.mlp.up_proj.weight"] = np.zeros((8, 6), dtype=np.float32)
    save_file(mismatch, str(compat / "shape_mismatch.safetensors"), metadata=meta)
    perm = [3, 0, 6, 1, 7, 2, 5, 4]
    permuted = {k: v.copy() for k, v in base.items()}
    permuted["demo.layers.0.mlp.up_proj.weight"] = ramp[perm, :]
    permuted["demo.layers.0.mlp.down_proj.weight"] = down[:, perm]
    save_file(permuted, str(compat / "permuted_same_function.safetensors"), metadata=meta)
    tied = compat / "tied_embeddings"
    tied.mkdir(exist_ok=True)
    embed = uniform_matrix(seed, 30, (16, 8))
    save_file(
        {"model.embed_tokens.weight": embed, "model.norm.weight": np.ones(8, dtype=np.float32)}, str(tied / "model.safetensors")
    )
    (tied / "config.json").write_text(
        json.dumps({"tie_word_embeddings": True, "vocab_size": 16, "hidden_size": 8}, indent=2), encoding="utf-8"
    )
    tied_untied = compat / "untied_reference"
    tied_untied.mkdir(exist_ok=True)
    save_file(
        {"model.embed_tokens.weight": embed, "model.norm.weight": np.ones(8, dtype=np.float32), "lm_head.weight": embed.copy()},
        str(tied_untied / "model.safetensors"),
    )
    (tied_untied / "config.json").write_text(
        json.dumps({"tie_word_embeddings": False, "vocab_size": 16, "hidden_size": 8}, indent=2), encoding="utf-8"
    )

    no_tie = compat / "missing_head_no_tie_evidence"
    no_tie.mkdir(exist_ok=True)
    save_file(
        {"model.embed_tokens.weight": embed, "model.norm.weight": np.ones(8, dtype=np.float32)}, str(no_tie / "model.safetensors")
    )  # deliberately no config.json: absence of lm_head is MISSING, not tied

    # Malformed inputs: every one must be rejected (fail closed).
    mal = out / "malformed"
    mal.mkdir(exist_ok=True)
    for stale in ("missing_shard_index.json", "traversal_index.json"):  # generator v1 layout
        (mal / stale).unlink(missing_ok=True)
    ok16 = np.arange(4, dtype="<f4").tobytes()
    write_raw_safetensors(mal / "truncated_payload.safetensors", {"x.weight": ("F32", [2, 2], ok16)}, truncate_to=-4)
    write_raw_safetensors(
        mal / "header_length_exceeds_file.safetensors", {"x.weight": ("F32", [2, 2], ok16)}, length_override=1 << 40
    )
    write_raw_safetensors(mal / "corrupt_json.safetensors", {}, header_override=b'{"x.weight": {"dtype": "F32",')
    write_raw_safetensors(mal / "unknown_dtype.safetensors", {"x.weight": ("Q4_0", [2, 2], ok16)})
    write_raw_safetensors(mal / "shape_length_mismatch.safetensors", {"x.weight": ("F32", [3, 3], ok16)})
    huge = json.dumps({"x.weight": {"dtype": "F32", "shape": [2**40, 2**40], "data_offsets": [0, 16]}}).encode()
    write_raw_safetensors(mal / "overflow_shape.safetensors", {}, header_override=huge)
    (mal / "overflow_shape.safetensors").write_bytes((mal / "overflow_shape.safetensors").read_bytes() + ok16)
    neg = json.dumps({"x.weight": {"dtype": "F32", "shape": [-1, 4], "data_offsets": [0, 16]}}).encode()
    write_raw_safetensors(mal / "negative_shape.safetensors", {}, header_override=neg)
    (mal / "negative_shape.safetensors").write_bytes((mal / "negative_shape.safetensors").read_bytes() + ok16)
    oob = json.dumps({"x.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [16, 32]}}).encode()
    write_raw_safetensors(mal / "offset_out_of_bounds.safetensors", {}, header_override=oob)
    (mal / "offset_out_of_bounds.safetensors").write_bytes((mal / "offset_out_of_bounds.safetensors").read_bytes() + ok16)
    overlap = json.dumps(
        {
            "a": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
        }
    ).encode()
    write_raw_safetensors(mal / "overlapping_offsets.safetensors", {}, header_override=overlap)
    (mal / "overlapping_offsets.safetensors").write_bytes((mal / "overlapping_offsets.safetensors").read_bytes() + ok16)
    gap = json.dumps(
        {
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "F32", "shape": [1], "data_offsets": [12, 16]},
        }
    ).encode()
    write_raw_safetensors(mal / "gap_in_offsets.safetensors", {}, header_override=gap)
    (mal / "gap_in_offsets.safetensors").write_bytes((mal / "gap_in_offsets.safetensors").read_bytes() + ok16)
    dup = (
        b'{"x.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
        b'"x.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}'
    )
    write_raw_safetensors(mal / "duplicate_key.safetensors", {}, header_override=dup)
    (mal / "duplicate_key.safetensors").write_bytes((mal / "duplicate_key.safetensors").read_bytes() + ok16)

    def raw_header(name: str, header: bytes) -> None:
        write_raw_safetensors(mal / name, {}, header_override=header)
        (mal / name).write_bytes((mal / name).read_bytes() + ok16)

    raw_header("bool_shape.safetensors", b'{"x":{"dtype":"F32","shape":[true,4],"data_offsets":[0,16]}}')
    raw_header("float_offsets.safetensors", b'{"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0.0,16.0]}}')
    raw_header("json_nan_token.safetensors", b'{"x":{"dtype":"F32","shape":[2,NaN],"data_offsets":[0,16]}}')
    raw_header("non_utf8_header.safetensors", b'{"x\xff":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}')
    raw_header(
        "nested_metadata.safetensors", b'{"__metadata__":{"a":{"b":"c"}},"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}'
    )
    raw_header(
        "metadata_non_string.safetensors", b'{"__metadata__":{"a":1},"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}'
    )
    raw_header(
        "nesting_bomb.safetensors",
        b'{"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16],"z":' + b"[" * 5000 + b"]" * 5000 + b"}}",
    )
    raw_header("unknown_entry_field.safetensors", b'{"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16],"extra":1}}')
    raw_header(
        "rank_too_high.safetensors",
        b'{"x":{"dtype":"F32","shape":[' + b",".join([b"1"] * 40) + b'],"data_offsets":[0,4]},'
        b'"y":{"dtype":"F32","shape":[3],"data_offsets":[4,16]}}',
    )

    for name, weight_map in {
        "backslash_path": {"demo.vector.weight": "sub\\model-00001-of-00002.safetensors"},
        "windows_drive_path": {"demo.vector.weight": "C:\\model-00001-of-00002.safetensors"},
        "non_string_shard": {"demo.vector.weight": 5},
        "missing_shard": {
            "demo.vector.weight": "model-00001-of-00002.safetensors",
            "demo.layers.0.mlp.up_proj.weight": "model-00002-of-00002.safetensors",
        },
        "path_traversal": {"demo.vector.weight": "../sensitive.safetensors"},
        "absolute_path": {"demo.vector.weight": "/etc/passwd"},
        "key_not_in_shard": {
            "demo.vector.weight": "model-00001-of-00002.safetensors",
            "demo.not_present.weight": "model-00001-of-00002.safetensors",
        },
    }.items():
        d = mal / name
        d.mkdir(exist_ok=True)
        save_file({"demo.vector.weight": vector}, str(d / "model-00001-of-00002.safetensors"))
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": int(vector.nbytes)}, "weight_map": weight_map}, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------ expectations
    def scale(alpha: float, r: int, rs_lora: bool) -> float:
        return alpha / (r**0.5) if rs_lora else alpha / r

    expected_delta = changed["demo.layers.0.mlp.up_proj.weight"] - ramp
    rms = lambda a: float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))  # noqa: E731

    def status(block: np.ndarray, display: float | None) -> int:
        """Tile status bitfield (docs/09 section 3) for one output cell."""
        bits = 0
        finite = block[np.isfinite(block)]
        if finite.size == 0:
            bits |= MASK_NO_FINITE
        if np.isnan(block).any():
            bits |= MASK_NAN
        if np.isposinf(block).any():
            bits |= MASK_POS_INF
        if np.isneginf(block).any():
            bits |= MASK_NEG_INF
        if finite.size and (display is None or not abs(display) <= F32_MAX):
            bits |= MASK_NOT_REPRESENTABLE
        return bits

    def block_mean(block: np.ndarray) -> float | None:
        f = block[np.isfinite(block)].astype(np.float64)
        return float(f.mean()) if f.size else None

    nf_cells = {"3,20": nf[3:4, 20:21], "4,21": nf[4:5, 21:22], "16,1": nf[16:17, 1:2], "0,0": nf[0:1, 0:1]}
    nf_blocks = {f"{r},{c}": nf[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16] for r in (0, 1) for c in (0, 1)}
    fa, fb = np.array(f64_a), np.array(f64_b)
    with np.errstate(over="ignore", invalid="ignore"):
        f64_delta = fb - fa
        sumsq = float(np.sum(np.square(np.array([1e200] * 4))))
    s_up, s_down, s_rs = scale(4, 2, False), scale(4, 1, False), scale(8, 4, True)
    output = {
        "schema_version": 2,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "purpose": "synthetic numerical fixture; no reasoning improvement or model training implied",
        "tensor_key": "demo.layers.0.mlp.up_proj.weight",
        "shape": [8, 8],
        "base_full": ramp.tolist(),
        "ft_full": changed["demo.layers.0.mlp.up_proj.weight"].tolist(),
        "delta_full": expected_delta.tolist(),
        "changed_positions": [{"row": r, "col": c, "delta": float(expected_delta[r, c])} for r, c in positions],
        "other_tensors": {
            "demo.scalar": {"shape": [], "value": 3.5},
            "demo.one_by_one.weight": {"shape": [1, 1], "value": [[-0.25]]},
            "demo.cube.weight": {"shape": [2, 4, 8], "slice_axis0_index1_rows_1_3_cols_2_5": cube[1, 1:3, 2:5].tolist()},
            "demo.special_nonfinite.weight": {
                "nan": 1,
                "pos_inf": 1,
                "neg_inf": 1,
                "finite": 5,
                "finite_mean": float(special[np.isfinite(special)].astype(np.float64).mean()),
            },
        },
        "dtypes": {
            "file": "tiny_dtypes.safetensors",
            "decoded_4x4": small.tolist(),
            "specials": ["1.0", "-2.0", "nan", "inf", "-inf", "0.0", "-0.0", "max_finite"],
            "bf16_max_finite": float(bf16_specials[-1]),
            "f16_max_finite": 65504.0,
            "bf16_specials_bits": [int(b) for b in f32_to_bf16_bits(bf16_specials)],
            "f16_specials_bits": [int(b) for b in f16_specials.astype("<f2").view("<u2")],
            "note": "safetensors NumPy backend cannot read BF16; WeightLab must decode BF16 bits itself.",
        },
        "rms_delta_vs_difference_of_rms": {
            "key": "demo.layers.0.mlp.up_proj.weight",
            "rms_of_delta": rms(expected_delta),
            "rms_ft_minus_rms_base": rms(changed["demo.layers.0.mlp.up_proj.weight"]) - rms(ramp),
            "note": "Nonlinear reducers of B-A need a paired elementwise pass; they cannot be derived from A and B pyramids.",
        },
        "cube_slices": {
            "key": "demo.cube.weight",
            "view_rows_axis1_cols_axis2_fixed_axis0_0_mean": float(cube[0].astype(np.float64).mean()),
            "view_rows_axis1_cols_axis2_fixed_axis0_1_mean": float(cube[1].astype(np.float64).mean()),
        },
        "precision": {
            "files": ["tiny_precision_a.safetensors", "tiny_precision_b.safetensors"],
            "f64_a": [repr(v) for v in f64_a],
            "f64_b": [repr(v) for v in f64_b],
            "f64_delta_b_minus_a": [repr(float(v)) for v in f64_delta],
            "f64_delta_overflow_index": 6,
            "f32_overflow_delta_b_minus_a_f64": repr(float(np.float64(np.float32(3.4e38)) - np.float64(np.float32(-3.4e38)))),
            "f32_overflow_delta_display_status": MASK_NOT_REPRESENTABLE,
            "i64": [str(v) for v in i64],
            "u64": [str(v) for v in u64],
            "i32": [str(-(2**31)), str(2**31 - 1)],
            "bool": [True, False],
            "f64_sumsq_overflow": {
                "sum": repr(4e200),
                "sum_sq_finite": bool(np.isfinite(sumsq)),
                "expected_numeric_precision_rms": "overflowed",
            },
            "note": "64-bit integers are exact only as decimal strings; "
            "F64 values exact only as float64 JSON numbers or hex bits.",
        },
        "edge": {
            "file": "tiny_edge.safetensors",
            "empty": {"edge.empty_rows": [0, 4], "edge.empty_cols": [3, 0]},
            "unicode_key": "edge.\u00fcn\u00efc\u00f8d\u00e9/\u03ba\u03bb\u03b5\u03b9\u03b4\u03af.weight",
            "big_block": {
                "key": "edge.block256x512.weight",
                "level8_block_0_0_finite_count": 65536,
                "level8_block_0_0_sum": float(big[:256, :256].astype(np.float64).sum()),
                "level9_block_0_0_finite_count": 131072,
                "level9_block_0_0_sum": float(big.astype(np.float64).sum()),
            },
            "nonfinite_blocks": {
                "key": "edge.nonfinite_blocks.weight",
                "status_bits": {
                    "NO_FINITE": MASK_NO_FINITE,
                    "HAS_NAN": MASK_NAN,
                    "HAS_POS_INF": MASK_POS_INF,
                    "HAS_NEG_INF": MASK_NEG_INF,
                    "NOT_REPRESENTABLE": MASK_NOT_REPRESENTABLE,
                },
                "level0_cells": {
                    k: {"status": status(v, None if not np.isfinite(v).any() else float(v[0, 0])), "value": block_mean(v)}
                    for k, v in nf_cells.items()
                },
                "level4_blocks_mean": {
                    k: {"status": status(v, block_mean(v)), "value": block_mean(v)} for k, v in nf_blocks.items()
                },
            },
        },
        "tiles_257x263": {
            "key": "demo.layers.0.self_attn.q_proj.weight",
            "level0_tile_1_1_shape": [1, 7],
            "level1": block_mean_edges(odd, 2),
            "level3": block_mean_edges(odd, 8),
        },
        "lora": {
            "tiny_lora": {
                "up_proj": {"rank": 2, "alpha": 4, "scale": s_up, "delta_full": (s_up * (B_up @ A_up)).tolist()},
                "down_proj": {
                    "rank": 1,
                    "alpha": 4,
                    "scale": s_down,
                    "rank_from": "rank_pattern",
                    "delta_full": (s_down * (B_down @ A_down)).tolist(),
                },
            },
            "tiny_lora_rslora": {
                "up_proj": {
                    "rank": 4,
                    "alpha": 8,
                    "scale": s_rs,
                    "delta_full": (s_rs * (B_rs.astype(np.float64) @ A_rs.astype(np.float64))).tolist(),
                }
            },
            "tiny_lora_dora_unsupported": {"expected": "reject: use_dora"},
        },
        "compat_expected_class": {
            "tiny_base vs tiny_ft": {
                "class": "V",
                "base_file_identity_verified": True,
                "lineage_claim_source": "synthetic_fixture_generator",
                "lineage_trust_level": "proven_by_construction",
                "note": "The generator constructs tiny_ft from tiny_base, so ancestry is known by construction; "
                "a real model whose metadata names a base hash would only get base_file_identity_verified.",
            },
            "tiny_base vs compat/different_ancestry": "S",
            "tiny_base vs compat/shape_mismatch": "X for demo.layers.0.mlp.up_proj.weight; other tensors inspectable",
            "tiny_base vs compat/permuted_same_function": "S; identical function, large raw delta",
            "compat/tied_embeddings vs compat/untied_reference": "lm_head.weight TIED (config tie_word_embeddings=true "
            "and known embed_tokens/lm_head mapping); not MISSING",
            "compat/missing_head_no_tie_evidence vs compat/untied_reference": "lm_head.weight MISSING (no tie evidence)",
        },
        "malformed_expected": {
            "truncated_payload.safetensors": "INVALID_FORMAT",
            "header_length_exceeds_file.safetensors": "INVALID_FORMAT",
            "corrupt_json.safetensors": "INVALID_FORMAT",
            "unknown_dtype.safetensors": "UNSUPPORTED_DTYPE",
            "shape_length_mismatch.safetensors": "INVALID_FORMAT",
            "overflow_shape.safetensors": "INVALID_FORMAT",
            "negative_shape.safetensors": "INVALID_FORMAT",
            "offset_out_of_bounds.safetensors": "INVALID_FORMAT",
            "overlapping_offsets.safetensors": "INVALID_FORMAT",
            "gap_in_offsets.safetensors": "INVALID_FORMAT",
            "duplicate_key.safetensors": "INVALID_FORMAT",
            "bool_shape.safetensors": "INVALID_FORMAT",
            "float_offsets.safetensors": "INVALID_FORMAT",
            "json_nan_token.safetensors": "INVALID_FORMAT",
            "non_utf8_header.safetensors": "INVALID_FORMAT",
            "nested_metadata.safetensors": "INVALID_FORMAT",
            "metadata_non_string.safetensors": "INVALID_FORMAT",
            "nesting_bomb.safetensors": "INVALID_FORMAT",
            "unknown_entry_field.safetensors": "INVALID_FORMAT",
            "rank_too_high.safetensors": "INVALID_FORMAT",
            "backslash_path/": "INVALID_FORMAT (backslash in shard path)",
            "windows_drive_path/": "INVALID_FORMAT (drive-qualified shard path)",
            "non_string_shard/": "INVALID_FORMAT (shard path not a string)",
            "missing_shard/": "MISSING_SHARD",
            "path_traversal/": "INVALID_FORMAT (path outside selected root)",
            "absolute_path/": "INVALID_FORMAT (absolute shard path)",
            "key_not_in_shard/": "INVALID_FORMAT (index key absent from shard)",
        },
        "upstream_permitted_differences": {
            "rank_too_high.safetensors": "WeightLab caps rank at 32; upstream safetensors 0.8.0 accepts",
            "unknown_entry_field.safetensors": "WeightLab rejects unknown tensor-entry fields; upstream ignores them",
            "duplicate_key.safetensors": "both reject; upstream reports an offset error because its JSON map keeps one key",
        },
        "file_sha256": {},
    }
    for p in sorted(out.rglob("*.safetensors")):
        output["file_sha256"][p.relative_to(out).as_posix()] = digest(p)
    (out / "expected_results.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print("Generated", out, "files:", sum(1 for p in out.rglob("*") if p.is_file()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/fixtures"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate(args.output, args.seed)
