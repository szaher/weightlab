"""The exact requests the Phase 1 Task 4A explorer sends (frontend/src/features/roi/view.ts), against values
decoded here straight from the fixture bytes with NumPy (header JSON + np.frombuffer; BF16 by bit shift). Nothing is
compared with WeightLab's own reader or serializer output.

Views: rank 0 and 1 use row_axis = col_axis = null; rank 2 rows = axis 0; rank > 2 the last two axes with the
others fixed; transpose swaps the displayed axes. Windows default to 64 x 64 within roi_max_cells."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from tests.api.conftest import import_model, tensors_by_key
from tests.api.contract import Api

NULL_VIEW = {"row_axis": None, "col_axis": None, "fixed_indices": {}, "transpose": False}
_NP = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "U64": "<u8", "I32": "<i4", "BOOL": "u1"}


def decode(path: Path, key: str) -> npt.NDArray[Any]:
    """The tensor as stored: float dtypes widened to float64 (exact), 64-bit ints as Python ints (object array)."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    entry = json.loads(raw[8 : 8 + n])[key]
    a, b = entry["data_offsets"]
    data = raw[8 + n + a : 8 + n + b]
    dt, shape = entry["dtype"], entry["shape"]
    if dt == "BF16":
        bits = np.frombuffer(data, "<u2").astype(np.uint32) << 16
        return bits.view(np.float32).astype(np.float64).reshape(shape)
    arr = np.frombuffer(data, _NP[dt]).reshape(shape)
    if dt in ("I64", "U64"):
        return np.array([int(v) for v in arr.ravel()], dtype=object).reshape(shape)
    if dt == "BOOL":
        return arr.astype(bool)
    return arr.astype(np.float64)


def ui_request(
    model: dict[str, Any], tensor: dict[str, Any], view: dict[str, Any], roi: tuple[int, int, int, int]
) -> dict[str, Any]:
    return {
        "source": {"kind": "tensor", "model_id": model["id"], "tensor_id": tensor["id"]},
        "view": view,
        "roi": dict(zip(("r0", "r1", "c0", "c1"), roi, strict=True)),
        "value_encoding": "auto",
    }


def as_float(values: list[list[Any]], mask: list[list[Any]] | None) -> npt.NDArray[np.float64]:
    """JSON values back to float64, re-inserting non-finite cells from the mask."""
    special = {"nan": math.nan, "pos_inf": math.inf, "neg_inf": -math.inf}
    out = [
        [special[mask[i][j]] if mask and mask[i][j] else v for j, v in enumerate(row)] for i, row in enumerate(values)
    ]
    return np.array(out, dtype=np.float64)


def same(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> bool:
    """Bitwise float64 equality: -0.0 differs from 0.0, NaN equals NaN."""
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)) or (
        np.array_equal(np.signbit(a), np.signbit(b)) and np.array_equal(a, b, equal_nan=True)))  # fmt: skip


@pytest.fixture
def base(api: Api, project: str, models_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    m = import_model(api, project, models_dir, "tiny_base.safetensors")
    return m, tensors_by_key(api, m["id"]), models_dir / "tiny_base.safetensors"


def test_default_window_of_the_odd_257x263_tensor_and_its_far_corner(api: Api, base: Any) -> None:
    m, t, path = base
    key = "demo.layers.0.self_attn.q_proj.weight"
    want = decode(path, key)
    view = {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}
    got = api.post("/tensors/roi", json_body=ui_request(m, t[key], view, (0, 64, 0, 64)), expect=200)
    assert got["shape"] == [64, 64]
    assert got["coverage"] == "complete"
    assert got["numeric_precision"] == "lossless"
    assert same(as_float(got["values"], got["nonfinite_mask"]), want[0:64, 0:64])
    corner = api.post("/tensors/roi", json_body=ui_request(m, t[key], view, (193, 257, 199, 263)), expect=200)
    assert same(as_float(corner["values"], None), want[193:257, 199:263])  # ragged edges: 257 % 64, 263 % 64
    # Orientation: row r of the answer is tensor row r0 + r (not a column), checked on an asymmetric cell.
    assert corner["values"][63][0] == float(want[256, 199])


def test_transposed_view_swaps_rows_and_columns(api: Api, base: Any) -> None:
    m, t, path = base
    key = "demo.layers.0.self_attn.q_proj.weight"
    want = decode(path, key)
    view = {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": True}
    got = api.post("/tensors/roi", json_body=ui_request(m, t[key], view, (199, 263, 193, 257)), expect=200)
    assert same(as_float(got["values"], None), want[193:257, 199:263].T)


def test_rank0_rank1_and_rank3_views(api: Api, base: Any) -> None:
    m, t, path = base
    scalar = api.post("/tensors/roi", json_body=ui_request(m, t["demo.scalar"], NULL_VIEW, (0, 1, 0, 1)), expect=200)
    assert scalar["values"] == [[float(decode(path, "demo.scalar"))]]
    vec = decode(path, "demo.vector.weight")
    got = api.post(
        "/tensors/roi", json_body=ui_request(m, t["demo.vector.weight"], NULL_VIEW, (0, 1, 0, 6)), expect=200
    )
    assert same(as_float(got["values"], None), vec.reshape(1, 6))
    cube = decode(path, "demo.cube.weight")  # [2, 4, 8]
    default = {"row_axis": 1, "col_axis": 2, "fixed_indices": {"0": 1}, "transpose": False}
    got = api.post("/tensors/roi", json_body=ui_request(m, t["demo.cube.weight"], default, (1, 3, 2, 5)), expect=200)
    assert got["values"] == [[42.0, 43.0, 44.0], [50.0, 51.0, 52.0]]  # expected_results.json cube slice
    assert same(as_float(got["values"], None), cube[1, 1:3, 2:5])
    other = {"row_axis": 0, "col_axis": 2, "fixed_indices": {"1": 3}, "transpose": False}
    got = api.post("/tensors/roi", json_body=ui_request(m, t["demo.cube.weight"], other, (0, 2, 0, 8)), expect=200)
    assert same(as_float(got["values"], None), cube[:, 3, :])


def test_nonfinite_cells_are_masked_not_zero(api: Api, base: Any) -> None:
    m, t, path = base
    key = "demo.special_nonfinite.weight"
    want = decode(path, key)
    view = {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}
    got = api.post("/tensors/roi", json_body=ui_request(m, t[key], view, (0, 2, 0, 4)), expect=200)
    mask = got["nonfinite_mask"]
    for (i, j), v in np.ndenumerate(want):
        if math.isfinite(v):
            assert mask[i][j] is None
            assert got["values"][i][j] == v
        else:
            assert got["values"][i][j] is None
            assert mask[i][j] == ("nan" if math.isnan(v) else "pos_inf" if v > 0 else "neg_inf")


@pytest.mark.parametrize("key", ["demo.bf16_specials", "demo.f16_specials", "demo.bf16.weight", "demo.f16.weight"])
def test_half_precision_dtypes(api: Api, project: str, models_dir: Path, key: str) -> None:
    m = import_model(api, project, models_dir, "tiny_dtypes.safetensors")
    t = tensors_by_key(api, m["id"])[key]
    want = decode(models_dir / "tiny_dtypes.safetensors", key)
    rows, cols = (1, want.size) if want.ndim == 1 else want.shape
    view = NULL_VIEW if want.ndim == 1 else {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}
    got = api.post("/tensors/roi", json_body=ui_request(m, t, view, (0, rows, 0, cols)), expect=200)
    assert same(as_float(got["values"], got["nonfinite_mask"]), want.reshape(rows, cols))


def test_precision_fixture_f64_i64_u64_bool(api: Api, project: str, models_dir: Path) -> None:
    m = import_model(api, project, models_dir, "tiny_precision_a.safetensors")
    t = tensors_by_key(api, m["id"])
    path = models_dir / "tiny_precision_a.safetensors"
    for key, enc in (("p.f64", "float_json"), ("p.i64", "int_decimal_string"), ("p.u64", "int_decimal_string"),
                     ("p.i32", "int_json"), ("p.bool", "bool_json")):  # fmt: skip
        want = decode(path, key)
        got = api.post("/tensors/roi", json_body=ui_request(m, t[key], NULL_VIEW, (0, 1, 0, want.size)), expect=200)
        assert got["value_encoding"] == enc, key
        row = got["values"][0]
        if enc == "int_decimal_string":
            assert row == [str(v) for v in want], key  # full 64-bit range, decimal strings
        elif enc == "float_json":
            assert same(np.array(row, dtype=np.float64), want), key  # includes -0.0 and +/-1.79e308, 5e-324
            assert math.copysign(1.0, row[4]) == -1.0
        else:
            assert row == want.tolist(), key


def test_budget_zero_size_and_bad_bounds(api: Api, project: str, models_dir: Path, base: Any) -> None:
    m, t, _ = base
    key = "demo.layers.0.self_attn.q_proj.weight"
    view = {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}
    over = api.post("/tensors/roi", json_body=ui_request(m, t[key], view, (0, 64, 0, 65)), expect=413)
    assert over["error"]["code"] == "RESOURCE_LIMIT"  # 4,160 cells > roi_max_cells 4,096
    for bad in ((0, 258, 0, 1), (5, 4, 0, 1), (0, 1, 263, 264)):
        assert (
            api.post("/tensors/roi", json_body=ui_request(m, t[key], view, bad), expect=422)["error"]["code"]
            == "INVALID_ROI"
        )
    edge = import_model(api, project, models_dir, "tiny_edge.safetensors")
    empty = tensors_by_key(api, edge["id"])["edge.empty_rows"]  # [0, 4]: the UI never requests it
    got = api.post("/tensors/roi", json_body=ui_request(edge, empty, view, (0, 0, 0, 4)), expect=200)
    assert got["shape"] == [0, 4]
    assert got["values"] == []


def test_appended_bytes_are_file_changed_not_a_format_error(api: Api, base: Any) -> None:
    """A checkpoint edited after import so that it no longer parses must be FILE_CHANGED (re-import), as the UI tells
    the user, never INVALID_FORMAT of the imported model and never stale values."""
    m, t, path = base
    key = "demo.layers.0.mlp.up_proj.weight"
    view = {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}
    api.post("/tensors/roi", json_body=ui_request(m, t[key], view, (0, 8, 0, 8)), expect=200)
    with path.open("ab") as f:
        f.write(b"\0")
    err = api.post("/tensors/roi", json_body=ui_request(m, t[key], view, (0, 8, 0, 8)), expect=409)
    assert err["error"]["code"] == "FILE_CHANGED"
