"""Tile payload codec, metadata and identity (doc 09 section 3, doc 10 section 6, contract TileMetadata / TileJson /
TileStatus). Byte-level expectations are written out by hand with struct, not produced by the encoder."""

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest

from tests.api.contract import component
from tests.helpers import write_safetensors
from weightlab.analysis import tile_codec as tc
from weightlab.analysis.tile import TileBudget, compute_tile, tile_geometry
from weightlab.api.config import DEFAULT_SETTINGS
from weightlab.core.errors import WeightLabError
from weightlab.formats.checkpoint import open_checkpoint
from weightlab.formats.selection import View

NAN = np.float32(np.nan)
SOURCE = {"kind": "tensor", "model_id": "mod_a", "tensor_id": "tns_0009"}


def arrays(values: list[list[float]], status: list[list[int]]) -> tuple[Any, Any]:
    return np.array(values, np.float32), np.array(status, np.uint8)


def edge_tile(fixtures_dir: Path, level: int = 4) -> tuple[Any, dict[str, Any]]:
    with open_checkpoint(fixtures_dir / "tiny_edge.safetensors") as ck:
        t = compute_tile(ck, "edge.nonfinite_blocks.weight", None, level=level, x=0, y=0, aggregation="mean")
        ident = tc.tile_identity(source_kind="tensor", files=ck.fingerprints(["edge.nonfinite_blocks.weight"]),
                                 tensor_keys=["edge.nonfinite_blocks.weight"], view=None, level=level, x=0, y=0,
                                 size=256, aggregation="mean")  # fmt: skip
    return t, tc.metadata(t, cache_key=tc.cache_key(ident), source=SOURCE, view=None)


# --------------------------------------------------------------------------------------------------------- binary
def test_binary_layout_and_byte_order() -> None:
    values, status = arrays([[1.0, -2.5], [NAN, 0.15625]], [[0, 0], [3, 0]])
    body = tc.encode_binary(values, status, level=0)
    assert len(body) == 4 * 5  # rows * cols * 5
    want = struct.pack("<ffff", 1.0, -2.5, float("nan"), 0.15625) + bytes([0, 0, 3, 0])
    assert body[:4] == b"\x00\x00\x80\x3f"  # 1.0f little-endian
    assert body[16:] == bytes([0, 0, 3, 0])  # statuses after all values, row-major
    assert body[:8] == want[:8]
    assert body[12:] == want[12:]
    v, s = tc.decode_binary(body, 2, 2, level=0)
    np.testing.assert_array_equal(v.view(np.uint32), values.view(np.uint32))
    np.testing.assert_array_equal(s, status)


def test_golden_tile_round_trips_binary_and_json_identically(fixtures_dir: Path) -> None:
    t, meta = edge_tile(fixtures_dir)
    component("TileMetadata").validate(meta)
    body = tc.encode_binary(t.values, t.status, t.geometry.level)
    assert len(body) == t.geometry.rows * t.geometry.cols * 5
    doc = tc.to_json(meta, t.values, t.status)
    component("TileJson").validate(doc)
    assert doc["status"] == [[0, 6], [15, 3]]
    assert doc["values"][1] == [None, None]  # null exactly for non-drawable cells
    assert doc["values"][0][0] == float(np.float32(-0.10943603515625))
    wire = json.loads(json.dumps(doc, allow_nan=False))  # JSON never carries NaN/Infinity
    meta2, v2, s2 = tc.from_json(wire)
    vb, sb = tc.decode_binary(body, meta["rows"], meta["cols"], meta["level"])
    np.testing.assert_array_equal(v2.view(np.uint32), vb.view(np.uint32))
    np.testing.assert_array_equal(s2, sb)
    assert meta2 == meta
    h = tc.headers(meta)
    assert h["Content-Type"] == "application/vnd.weightlab.tile.v2"
    assert (h["X-WeightLab-Rows"], h["X-WeightLab-Cols"], h["X-WeightLab-Level"]) == ("2", "2", "4")
    assert (h["X-WeightLab-Row-Start"], h["X-WeightLab-Row-End"]) == ("0", "32")
    assert h["X-WeightLab-Quality"] == "exact"
    assert h["X-WeightLab-Value-Precision"] == "float32_visualization"
    assert h["X-WeightLab-Cache-Key"] == meta["cache_key"]
    assert "X-WeightLab-Sampled-Rows" not in h


def test_level0_json_parity_with_the_real_engine(fixtures_dir: Path) -> None:
    t, meta = edge_tile(fixtures_dir, level=0)
    doc = tc.to_json(meta, t.values, t.status)
    component("TileJson").validate(doc)
    assert doc["status"][3][20] == 3
    assert doc["values"][3][20] is None
    assert doc["status"][4][21] == 5
    assert doc["status"][16][1] == 9
    _, v, s = tc.from_json(doc)
    np.testing.assert_array_equal(s, t.status)
    np.testing.assert_array_equal(v.view(np.uint32), t.values.view(np.uint32))


@pytest.mark.parametrize(
    ("values", "status", "level", "why"),
    [
        ([[1.0]], [[32]], 1, "reserved"),
        ([[1.0]], [[128]], 1, "reserved"),
        ([[NAN]], [[0]], 1, "drawable cell has no finite value"),
        ([[np.inf]], [[0]], 1, "drawable cell has no finite value"),
        ([[1.0]], [[3]], 1, "carries a finite value"),
        ([[1.0]], [[16]], 1, "carries a finite value"),
        ([[NAN]], [[1]], 1, "NO_FINITE without"),
        ([[NAN]], [[19]], 1, "exclusive"),
        ([[1.0]], [[6]], 0, "level-0"),  # a single element cannot be finite and NaN and +Inf
        ([[NAN]], [[15]], 0, "level-0"),
    ],
)
def test_invalid_cells_are_rejected(values: list[list[float]], status: list[list[int]], level: int, why: str) -> None:
    v, s = arrays(values, status)
    with pytest.raises(WeightLabError, match=why):
        tc.encode_binary(v, s, level)
    body = v.astype("<f4").tobytes() + s.tobytes()
    with pytest.raises(WeightLabError, match=why):
        tc.decode_binary(body, 1, 1, level)


def test_block_level_statuses_are_valid_above_level0() -> None:
    v, s = arrays([[0.5, NAN, NAN, NAN]], [[6, 15, 3, 16]])
    tc.decode_binary(tc.encode_binary(v, s, 1), 1, 4, 1)


@pytest.mark.parametrize("size", [0, 4, 19, 21, 4 * 5 + 1])
def test_body_length_must_be_exact(size: int) -> None:
    with pytest.raises(WeightLabError, match="rows\\*cols\\*5"):
        tc.decode_binary(b"\0" * size, 2, 2, 1)


def test_metadata_consistency_is_checked(fixtures_dir: Path) -> None:
    t, meta = edge_tile(fixtures_dir)
    bad: list[dict[str, Any]] = [
        {**meta, "rows": 3},
        {**meta, "row_end": 33},  # 33 rows at L4 would be 3 output rows
        {**meta, "row_start": 1},  # not anchored at a block boundary
        {**meta, "finite_count": meta["finite_count"] + 1},
        {**meta, "sampled_rows": 4},  # exact tiles are never sampled
        {**meta, "schema_version": 1},
    ]
    for m in bad:
        with pytest.raises(WeightLabError):
            tc.to_json(m, t.values, t.status)
    doc = tc.to_json(meta, t.values, t.status)
    doc["values"][1][0] = 0.0  # a value for a non-drawable cell
    with pytest.raises(WeightLabError, match="null exactly"):
        tc.from_json(doc)
    doc = tc.to_json(meta, t.values, t.status)
    doc["values"][0][0] = 0.1  # not a float32
    with pytest.raises(WeightLabError, match="not a float32"):
        tc.from_json(doc)


def test_json_is_limited_to_16384_cells() -> None:
    v = np.zeros((129, 128), np.float32)
    s = np.zeros((129, 128), np.uint8)
    with pytest.raises(WeightLabError) as e:
        tc.to_json({}, v, s)
    assert e.value.code.value == "NOT_ACCEPTABLE"


# -------------------------------------------------------------------------------------------------------- identity
def ident(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "source_kind": "tensor",
        "files": [{"relpath": "m.safetensors", "size": 10, "mtime_ns": 5, "file_id": "1:2"}],
        "tensor_keys": ["demo.cube.weight"],
        "view": View(1, 2, {0: 1}, False),
        "level": 2, "x": 0, "y": 0, "size": 256, "aggregation": "mean",
    }  # fmt: skip
    base.update(over)
    return tc.tile_identity(**base)


def test_cache_keys_are_canonical_and_change_with_every_input() -> None:
    k = tc.cache_key(ident())
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)
    assert tc.cache_key(ident()) == k  # deterministic
    # Key order of mappings does not matter (canonical JSON).
    shuffled = ident(files=[{"file_id": "1:2", "mtime_ns": 5, "size": 10, "relpath": "m.safetensors"}])
    assert tc.cache_key(shuffled) == k
    variants = [
        ident(view=View(1, 2, {0: 0}, False)),  # another slice never shares a tile
        ident(view=View(1, 2, {0: 1}, True)),
        ident(view=View(0, 2, {1: 1}, False)),
        ident(level=3), ident(x=1), ident(y=1), ident(size=128),
        ident(aggregation="rms"),
        ident(files=[{"relpath": "m.safetensors", "size": 10, "mtime_ns": 6, "file_id": "1:2"}]),
        ident(files=[{"relpath": "m.safetensors", "size": 11, "mtime_ns": 5, "file_id": "1:2"}]),
        ident(files=[{"relpath": "m.safetensors", "size": 10, "mtime_ns": 5, "file_id": "1:3"}]),
        ident(tensor_keys=["demo.other"]),
        ident(source_kind="delta"),
        ident(quality_mode="row_sampled", seed=7),
        ident(operation={"transform": "abs_b_minus_a"}),
    ]  # fmt: skip
    keys = {tc.cache_key(v) for v in variants}
    assert k not in keys
    assert len(keys) == len(variants)
    i = ident()
    assert i["numerics_version"] >= 1
    assert i["pyramid_record_format"] == 1
    assert "model_id" not in json.dumps(i)
    assert "tensor_id" not in json.dumps(i)


def test_fingerprint_change_invalidates_the_key(tmp_path: Path, fixtures_dir: Path) -> None:
    path = tmp_path / "tiny_base.safetensors"
    shutil.copy(fixtures_dir / "tiny_base.safetensors", path)

    def key() -> str:
        with open_checkpoint(path, tmp_path) as ck:
            files = ck.fingerprints(["demo.cube.weight"])
        assert files[0]["relpath"] == "tiny_base.safetensors"
        return tc.cache_key(ident(files=files))

    k1 = key()
    assert key() == k1
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # same bytes, new mtime
    assert key() != k1


def test_sharded_fingerprints_include_the_index(fixtures_dir: Path) -> None:
    with open_checkpoint(fixtures_dir / "tiny_sharded", fixtures_dir) as ck:
        key = ck.metadata()[0].key
        files = ck.fingerprints([key])
    assert any(f["relpath"].endswith(".index.json") and "sha256" in f for f in files)
    assert all(not str(f["relpath"]).startswith("/") for f in files)


# ---------------------------------------------------------------------------------------------- geometry, budget
def test_geometry_is_origin_anchored_and_clipped() -> None:
    g = tile_geometry(257, 263, 0, 1, 1)
    assert (g.row_start, g.row_end, g.col_start, g.col_end, g.rows, g.cols) == (256, 257, 256, 263, 1, 7)
    g = tile_geometry(257, 263, 3, 0, 0)
    assert (g.row_end, g.col_end, g.rows, g.cols, g.block) == (257, 263, 33, 33, 8)
    g = tile_geometry(10**6, 10**6, 2, 3, 1, size=100)  # x*size*2^L
    assert (g.row_start, g.col_start, g.rows, g.cols) == (400, 1200, 100, 100)
    g = tile_geometry(1, 6, 1, 0, 0)  # rank 1 as one row
    assert (g.rows, g.cols) == (1, 3)


def test_budget_from_the_frozen_settings() -> None:
    b = TileBudget.from_settings(DEFAULT_SETTINGS)
    assert b.max_tile_cells == 65536
    assert b.direct_tile_max_bytes == 64 * 1024 * 1024
    assert b.max_read_bytes == 4 * 64 * 1024 * 1024
    assert b.memory_bytes == 1024 * 1024 * 1024 // 2  # memory_budget_mib / max_active_tiles
    with pytest.raises(ValueError, match="positive"):
        TileBudget(band_cells=0)


# ------------------------------------------------------------------------------------------ JSON shape limits
def shaped_tile(tmp_path: Path, rows: int, cols: int) -> tuple[Any, dict[str, Any]]:
    a = np.arange(rows * cols, dtype="<f4")
    path = write_safetensors(tmp_path / f"s{rows}x{cols}.safetensors", {"w": ("F32", (rows, cols), a.tobytes())})
    with open_checkpoint(path) as ck:
        t = compute_tile(ck, "w", View(0, 1, {}, False), level=0, x=0, y=0, aggregation="mean")
        key = tc.cache_key(tc.tile_identity(source_kind="tensor", files=ck.fingerprints(["w"]), tensor_keys=["w"],
                                            view=View(0, 1, {}, False), level=0, x=0, y=0, size=256,
                                            aggregation="mean"))  # fmt: skip
    return t, tc.metadata(t, cache_key=key, source=SOURCE, view=View(0, 1, {}, False))


def test_json_128x128_is_valid_against_the_contract(tmp_path: Path) -> None:
    t, meta = shaped_tile(tmp_path, 128, 128)
    doc = tc.to_json(meta, t.values, t.status)
    component("TileJson").validate(doc)
    _, v, s = tc.from_json(json.loads(json.dumps(doc)))
    np.testing.assert_array_equal(v, t.values)
    np.testing.assert_array_equal(s, t.status)


@pytest.mark.parametrize(("rows", "cols"), [(1, 256), (256, 64), (129, 1), (1, 129), (64, 256)])
def test_json_rejects_any_dimension_above_128(tmp_path: Path, rows: int, cols: int) -> None:
    """Each shape is within 16384 cells or not, but has a dimension the frozen TileJson schema forbids."""
    t, meta = shaped_tile(tmp_path, rows, cols)
    with pytest.raises(WeightLabError) as e:
        tc.to_json(meta, t.values, t.status)
    assert e.value.code.value == "NOT_ACCEPTABLE"
    # The decoder rejects such a document too (built by hand, as the encoder refuses to).
    doc = {"metadata": meta, "values": t.values.astype(float).tolist(), "status": t.status.astype(int).tolist()}
    with pytest.raises(jsonschema.ValidationError):
        component("TileJson").validate(doc)  # proof that the shape violates the contract
    with pytest.raises(WeightLabError) as e:
        tc.from_json(doc)
    assert e.value.code.value == "VALIDATION_ERROR"


# -------------------------------------------------------------------------------------- JSON status strictness
def tiny_doc(fixtures_dir: Path) -> dict[str, Any]:
    t, meta = edge_tile(fixtures_dir, level=0)
    return json.loads(json.dumps(tc.to_json(meta, t.values, t.status)))


@pytest.mark.parametrize("bad", ["0", 0.7, 0.0, True, False, None, -1, 32, 255, [0], {"s": 0}])
def test_json_status_must_be_an_integer_within_0_31(fixtures_dir: Path, bad: object) -> None:
    doc = tiny_doc(fixtures_dir)
    i, j = next((i, j) for i, r in enumerate(doc["status"]) for j, s in enumerate(r) if s == 0)
    doc["status"][i][j] = bad
    with pytest.raises(WeightLabError) as e:
        tc.from_json(doc)
    assert e.value.code.value == "VALIDATION_ERROR"
    assert "status" in e.value.detail


@pytest.mark.parametrize("bad", ["1.0", True, [1.0]])
def test_json_values_must_be_numbers_or_null(fixtures_dir: Path, bad: object) -> None:
    doc = tiny_doc(fixtures_dir)
    i, j = next((i, j) for i, r in enumerate(doc["status"]) for j, s in enumerate(r) if s == 0)
    doc["values"][i][j] = bad
    with pytest.raises(WeightLabError) as e:
        tc.from_json(doc)
    assert e.value.code.value == "VALIDATION_ERROR"
