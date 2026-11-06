"""Strict header rules (doc 04 section 2) on hand-built headers."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from tests.helpers import write_raw, write_safetensors
from weightlab import WeightLabError, open_checkpoint
from weightlab.core.limits import Limits
from weightlab.formats.dtypes import decode, lookup
from weightlab.formats.safetensors_reader import parse_header


def code_of(header: bytes | dict[str, object], data_len: int = 0, limits: Limits | None = None) -> str | None:
    raw = header if isinstance(header, bytes) else json.dumps(header).encode()
    try:
        parse_header(raw, data_len, limits or Limits())
        return None
    except WeightLabError as e:
        return e.code.value


def entry(
    dtype: str = "F32", shape: list[object] | None = None, offsets: list[object] | None = None
) -> dict[str, object]:
    return {"dtype": dtype, "shape": [1] if shape is None else shape, "data_offsets": offsets or [0, 4]}


def test_valid_minimal_and_padding() -> None:
    assert code_of(b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}     ', 4) is None
    assert code_of(b"{}", 0) is None


@pytest.mark.parametrize(
    ("header", "data_len", "code"),
    [
        (b' {"a":1}', 0, "INVALID_FORMAT"),  # leading whitespace: upstream InvalidHeaderStart
        (b"[]", 0, "INVALID_FORMAT"),
        (b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{}}', 4, "INVALID_FORMAT"),
        (b'{"__metadata__":{"x":"\\ud800"}}', 0, "INVALID_FORMAT"),  # lone surrogate
        (b'{"x":NaN}', 0, "INVALID_FORMAT"),
        (b'{"x":-Infinity}', 0, "INVALID_FORMAT"),
        (b"{" + b'"a":' * 0 + b'"x":' + b"[" * 100_000 + b"]" * 100_000 + b"}", 0, "INVALID_FORMAT"),
        (b'{"\xff":1}', 0, "INVALID_FORMAT"),
    ],
)
def test_strict_json(header: bytes, data_len: int, code: str) -> None:
    assert code_of(header, data_len) == code


@pytest.mark.parametrize(
    ("e", "data_len", "code"),
    [
        (entry(shape=[True]), 4, "INVALID_FORMAT"),
        (entry(shape=[1.0]), 4, "INVALID_FORMAT"),
        (entry(shape=[-1]), 4, "INVALID_FORMAT"),
        (entry(shape="1"), 4, "INVALID_FORMAT"),  # type: ignore[arg-type]
        (entry(offsets=[0.0, 4]), 4, "INVALID_FORMAT"),
        (entry(offsets=[0, 4, 8]), 4, "INVALID_FORMAT"),
        (entry(offsets=[4, 0]), 4, "INVALID_FORMAT"),
        (entry(offsets=[-4, 0]), 4, "INVALID_FORMAT"),
        (entry(offsets=[False, 4]), 4, "INVALID_FORMAT"),
        (entry(dtype=7), 4, "INVALID_FORMAT"),  # type: ignore[arg-type]
        (entry(dtype="F8_E4M3", offsets=[0, 1]), 1, "UNSUPPORTED_DTYPE"),
        (entry(dtype="C64"), 4, "UNSUPPORTED_DTYPE"),
        (entry(dtype="f32"), 4, "UNSUPPORTED_DTYPE"),
        (entry(shape=[2]), 4, "INVALID_FORMAT"),  # length mismatch
        (entry(shape=[2**62, 2**62]), 4, "INVALID_FORMAT"),  # exact integer arithmetic, no overflow
        ({"dtype": "F32", "shape": [1]}, 4, "INVALID_FORMAT"),  # missing field
        ({**entry(), "extra": 1}, 4, "INVALID_FORMAT"),
        (entry(), 8, "INVALID_FORMAT"),  # trailing bytes
        (entry(), 2, "INVALID_FORMAT"),  # beyond data region
    ],
)
def test_entry_rules(e: dict[str, object], data_len: int, code: str) -> None:
    assert code_of({"t": e}, data_len) == code


def test_rank_and_key_caps() -> None:
    assert code_of({"t": entry(shape=[1] * 32)}, 4) is None
    assert code_of({"t": entry(shape=[1] * 33)}, 4) == "INVALID_FORMAT"
    assert code_of({"k" * 4096: entry()}, 4) is None
    assert code_of({"k" * 4097: entry()}, 4) == "INVALID_FORMAT"
    assert code_of({"κ" * 2048: entry()}, 4) is None  # 4096 UTF-8 bytes
    assert code_of({"κ" * 2049: entry()}, 4) == "INVALID_FORMAT"


def test_metadata_rules() -> None:
    assert code_of({"__metadata__": {"a": "b"}}) is None
    assert code_of({"__metadata__": {"a": 1}}) == "INVALID_FORMAT"
    assert code_of({"__metadata__": {"a": None}}) == "INVALID_FORMAT"
    assert code_of({"__metadata__": {"a": {"b": "c"}}}) == "INVALID_FORMAT"
    assert code_of({"__metadata__": "x"}) == "INVALID_FORMAT"


def test_tiling_overlap_gap_and_empty_tensors() -> None:
    z = entry(shape=[0, 3], offsets=[4, 4])
    assert code_of({"a": entry(), "z": z, "y": entry(shape=[3, 0], offsets=[4, 4])}, 4) is None
    assert code_of({"a": entry(), "b": entry(offsets=[2, 6])}, 6) == "INVALID_FORMAT"  # overlap
    assert code_of({"a": entry(), "b": entry(offsets=[8, 12])}, 12) == "INVALID_FORMAT"  # gap
    assert code_of({"a": entry(), "b": entry(offsets=[0, 4])}, 4) == "INVALID_FORMAT"  # same range twice


def test_header_length_cap_and_bounds(tmp_path: Path) -> None:
    p = tmp_path / "x.safetensors"
    p.write_bytes(struct.pack("<Q", 100_000_001) + b"{}")
    with pytest.raises(WeightLabError, match="cap"):
        open_checkpoint(p)
    p.write_bytes(b"\x01\x02")
    with pytest.raises(WeightLabError) as info:
        open_checkpoint(p)
    assert info.value.code == "INVALID_FORMAT"
    write_raw(p, b"{}", length=50)
    with pytest.raises(WeightLabError, match="exceeds"):
        open_checkpoint(p)


def test_bool_bytes_other_than_0_1_rejected_on_read(tmp_path: Path) -> None:
    p = write_safetensors(tmp_path / "b.safetensors", {"b": ("BOOL", (3,), b"\x00\x01\x02")})
    with open_checkpoint(p) as ck, pytest.raises(WeightLabError) as info:
        ck.read_roi_2d("b", 0, 1, 0, 3)
    assert info.value.code == "INVALID_FORMAT"


def test_bf16_decoding_is_bit_exact_for_all_65536_patterns() -> None:
    import numpy as np

    bits = np.arange(65536, dtype=np.uint16)
    got = decode(bits, lookup("BF16"))
    with np.errstate(invalid="ignore"):
        ref = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        f16_ref = bits.view("<f2").astype(np.float64)
    same = (got == ref) | (np.isnan(got) & np.isnan(ref))
    assert same.all()
    assert (np.signbit(got) == np.signbit(ref)).all()
    f16 = decode(bits.view("<f2"), lookup("F16"))
    assert ((f16 == f16_ref) | (np.isnan(f16) & np.isnan(f16_ref))).all()
    assert np.isnan(f16).sum() == 2 * 1023  # 2 signs x 1023 non-zero mantissas
