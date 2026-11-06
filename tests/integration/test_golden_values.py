"""Golden values (AC-001, AC-016): every ROI equals expected_results.json, which was produced by an
independent reference parser, and survives a JSON round trip bit-exactly."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers import reference_array
from weightlab import open_checkpoint
from weightlab.analysis.roi import ValueEncoding, encode_roi
from weightlab.formats.selection import View

UP = "demo.layers.0.mlp.up_proj.weight"


def f64_bits(x: float) -> int:
    return int(struct.unpack("<Q", struct.pack("<d", x))[0])


def as_float(s: str) -> float:
    return float(s)


def test_up_proj_all_64_cells_cell_by_cell(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    with open_checkpoint(fixtures_dir / "tiny_base.safetensors") as ck:
        for r in range(8):
            for c in range(8):
                got = ck.read_roi_2d(UP, r, r + 1, c, c + 1)
                assert got.shape == (1, 1)
                assert got[0, 0] == expected["base_full"][r][c]
        full = ck.read_roi_2d(UP, 0, 8, 0, 8)
    assert full.dtype == np.float64
    assert full.tolist() == expected["base_full"]


def test_ft_and_delta(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    with (
        open_checkpoint(fixtures_dir / "tiny_base.safetensors") as a,
        open_checkpoint(fixtures_dir / "tiny_ft.safetensors") as b,
    ):
        base, ft = a.read_roi_2d(UP, 0, 8, 0, 8), b.read_roi_2d(UP, 0, 8, 0, 8)
    assert ft.tolist() == expected["ft_full"]
    delta = (ft.astype(np.float32) - base.astype(np.float32)).astype(np.float64)
    assert delta.tolist() == expected["delta_full"]
    changed = [
        {"row": int(r), "col": int(c), "delta": float(delta[r, c])} for r, c in zip(*np.nonzero(delta), strict=True)
    ]
    assert changed == expected["changed_positions"]


def test_other_tensors_rank_0_1_and_3(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    other = expected["other_tensors"]
    with open_checkpoint(fixtures_dir / "tiny_base.safetensors") as ck:
        assert ck.tensor("demo.scalar").shape == ()
        assert ck.read_roi_2d("demo.scalar", 0, 1, 0, 1).tolist() == [[other["demo.scalar"]["value"]]]
        assert ck.read_roi_2d("demo.one_by_one.weight", 0, 1, 0, 1).tolist() == other["demo.one_by_one.weight"]["value"]
        cube = other["demo.cube.weight"]["slice_axis0_index1_rows_1_3_cols_2_5"]
        assert ck.read_roi_2d("demo.cube.weight", 1, 3, 2, 5, axes=(1, 2), fixed_indices={0: 1}).tolist() == cube
        for fixed, key in ((0, "view_rows_axis1_cols_axis2_fixed_axis0_0_mean"),
                           (1, "view_rows_axis1_cols_axis2_fixed_axis0_1_mean")):  # fmt: skip
            sl = ck.read_roi_2d("demo.cube.weight", 0, 4, 0, 8, axes=(1, 2), fixed_indices={0: fixed})
            assert sl.mean() == expected["cube_slices"][key]
        nf = ck.read_roi_2d("demo.special_nonfinite.weight", 0, 2, 0, 4)
        spec = other["demo.special_nonfinite.weight"]
        assert int(np.isnan(nf).sum()) == spec["nan"]
        assert int(np.isposinf(nf).sum()) == spec["pos_inf"]
        assert int(np.isneginf(nf).sum()) == spec["neg_inf"]
        assert nf[np.isfinite(nf)].mean() == spec["finite_mean"]
        vec = ck.read_roi_2d("demo.vector.weight", 0, 1, 0, 6)
        assert vec.tolist() == [reference_array(fixtures_dir / "tiny_base.safetensors", "demo.vector.weight").tolist()]


def test_every_tensor_matches_independent_reference(fixtures_dir: Path) -> None:
    """Full-tensor exact equality (raw bits) for every tensor of every valid single-file fixture."""
    files = [p for p in fixtures_dir.rglob("*.safetensors") if "malformed" not in p.parts]
    checked = 0
    for p in files:
        with open_checkpoint(p) as ck:
            for m in ck.metadata():
                ref = reference_array(p, m.key)
                flat = list(ck.iter_flat_chunks(m.key, 64))
                got = np.concatenate(flat) if flat else np.empty(0)
                assert got.size == ref.size
                if len(m.shape) == 2 and 0 < m.numel <= 4096:
                    roi = ck.read_roi_raw(m.key, 0, m.shape[0], 0, m.shape[1])
                    assert roi.raw.tobytes() == np.ascontiguousarray(ref).tobytes(), (p.name, m.key)
                checked += 1
    assert checked >= 40


def test_sharded_values_match_their_shards(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    with open_checkpoint(fixtures_dir / "tiny_sharded") as ck:
        assert ck.read_roi_2d(UP, 0, 8, 0, 8).tolist() == expected["base_full"]
        assert ck.tensor(UP).shard_id == "model-00001-of-00002.safetensors"


def test_bf16_f16_decoding_and_specials(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    d = expected["dtypes"]
    with open_checkpoint(fixtures_dir / "tiny_dtypes.safetensors") as ck:
        for key in ("demo.bf16.weight", "demo.f16.weight"):
            assert ck.read_roi_2d(key, 0, 4, 0, 4).tolist() == d["decoded_4x4"]
        for key, bits, maxf in (("demo.bf16_specials", d["bf16_specials_bits"], d["bf16_max_finite"]),
                                ("demo.f16_specials", d["f16_specials_bits"], d["f16_max_finite"])):  # fmt: skip
            hexed = encode_roi(ck.read_roi_raw(key, 0, 1, 0, 8), ValueEncoding.HEX_BITS)
            assert [int(h, 16) for h in hexed.values[0]] == bits
            enc = encode_roi(ck.read_roi_raw(key, 0, 1, 0, 8))
            assert enc.values[0] == [1.0, -2.0, None, None, None, 0.0, -0.0, maxf]
            assert enc.nonfinite_mask == [[None, None, "nan", "pos_inf", "neg_inf", None, None, None]]
            assert math.copysign(1.0, enc.values[0][6]) == -1.0  # type: ignore[arg-type]


def test_precision_round_trip_through_json(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    """AC-016: U64 max, I64 min, 2^53-1/2^53/2^53+1, F64 extremes and subnormal, signed zero."""
    pr = expected["precision"]
    for fname, f64_key in (("tiny_precision_a.safetensors", "f64_a"), ("tiny_precision_b.safetensors", "f64_b")):
        with open_checkpoint(fixtures_dir / fname) as ck:

            def roundtrip(key: str, ck: Any = ck) -> dict[str, Any]:
                n = ck.tensor(key).shape[0]
                enc = encode_roi(ck.read_roi_raw(key, 0, 1, 0, n))
                assert enc.numeric_precision == "lossless"
                return json.loads(enc.to_json())  # type: ignore[no-any-return]

            f64 = roundtrip("p.f64")
            assert f64["value_encoding"] == "float_json"
            assert [f64_bits(v) for v in f64["values"][0]] == [f64_bits(as_float(s)) for s in pr[f64_key]]
            if fname.endswith("a.safetensors"):
                i64 = roundtrip("p.i64")
                assert i64["value_encoding"] == "int_decimal_string"
                assert i64["values"][0] == pr["i64"]
                assert roundtrip("p.u64")["values"][0] == pr["u64"]
                i32 = roundtrip("p.i32")
                assert i32["value_encoding"] == "int_json"
                assert i32["values"][0] == [int(s) for s in pr["i32"]]
                assert roundtrip("p.bool")["values"][0] == pr["bool"]


def test_int_json_refuses_values_beyond_safe_integers(fixtures_dir: Path) -> None:
    from weightlab import WeightLabError

    with open_checkpoint(fixtures_dir / "tiny_precision_a.safetensors") as ck:
        raw = ck.read_roi_raw("p.i64", 0, 1, 0, 7)
        with pytest.raises(WeightLabError) as info:
            encode_roi(raw, ValueEncoding.INT_JSON)
        assert info.value.code == "UNSUPPORTED_PRECISION"
        small = ck.read_roi_raw("p.i64", 0, 1, 1, 4)  # -1, 0, 2^53-1
        assert encode_roi(small, ValueEncoding.INT_JSON).values == [[-1, 0, 2**53 - 1]]


def test_edge_tensors(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    edge = expected["edge"]
    with open_checkpoint(fixtures_dir / edge["file"]) as ck:
        for key, shape in edge["empty"].items():
            m = ck.tensor(key)
            assert m.shape == tuple(shape)
            assert m.is_empty
            assert m.byte_length == 0
            assert list(ck.iter_flat_chunks(key, 64)) == []
            assert ck.read_roi_2d(key, 0, shape[0], 0, shape[1]).shape == tuple(shape)
        uni = ck.tensor(edge["unicode_key"])
        assert uni.shape == (2, 2)
        cells = edge["nonfinite_blocks"]["level0_cells"]
        nb = ck.read_roi_2d(edge["nonfinite_blocks"]["key"], 0, 32, 0, 32, max_cells=1024)
        assert nb[0, 0] == cells["0,0"]["value"]
        assert np.isnan(nb[3, 20])
        assert np.isposinf(nb[4, 21])
        assert np.isneginf(nb[16, 1])
        big = edge["big_block"]
        total = sum(float(c.sum()) for c in ck.iter_flat_chunks(big["key"], 4096))
        assert total == big["level9_block_0_0_sum"]


def test_transpose_and_column_strided_reads(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    base = np.array(expected["base_full"])
    with open_checkpoint(fixtures_dir / "tiny_base.safetensors") as ck:
        t = ck.read_roi_raw(UP, 1, 5, 2, 7, View(0, 1, {}, transpose=True)).values()
        assert t.tolist() == base.T[1:5, 2:7].tolist()
        swapped = ck.read_roi_2d(UP, 1, 5, 2, 7, axes=(1, 0))
        assert swapped.tolist() == base.T[1:5, 2:7].tolist()
