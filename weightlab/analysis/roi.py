"""Lossless typed ROI value encodings (doc 09 section 2, ADR-017).

| source dtype                | auto encoding        |
|-----------------------------|----------------------|
| F16, BF16, F32, F64         | float_json           |
| I8..I32, U8..U32            | int_json             |
| I64, U64                    | int_decimal_string   |
| BOOL                        | bool_json            |

`hex_bits` (any dtype) returns raw source bit patterns, preserving NaN payloads. In float_json,
non-finite values are null and `nonfinite_mask` names them; JSON NaN/Infinity literals never appear.
Python's float repr is the shortest round-trip representation and keeps "-0.0".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from weightlab.core.errors import ErrorCode, fail
from weightlab.core.limits import MAX_SAFE_INTEGER
from weightlab.formats.checkpoint import RawRoi
from weightlab.formats.dtypes import bits

SCHEMA_VERSION = 2

JsonValue = float | int | str | bool | None


class ValueEncoding(StrEnum):
    AUTO = "auto"
    FLOAT_JSON = "float_json"
    INT_JSON = "int_json"
    INT_DECIMAL_STRING = "int_decimal_string"
    BOOL_JSON = "bool_json"
    HEX_BITS = "hex_bits"


_ALLOWED = {
    "float": {ValueEncoding.FLOAT_JSON, ValueEncoding.HEX_BITS},
    "int": {ValueEncoding.INT_JSON, ValueEncoding.INT_DECIMAL_STRING, ValueEncoding.HEX_BITS},
    "uint": {ValueEncoding.INT_JSON, ValueEncoding.INT_DECIMAL_STRING, ValueEncoding.HEX_BITS},
    "bool": {ValueEncoding.BOOL_JSON, ValueEncoding.HEX_BITS},
}


@dataclass(frozen=True)
class EncodedRoi:
    shape: tuple[int, int]
    source_dtype: str
    value_encoding: ValueEncoding
    numeric_precision: str  # raw tensor sources are always "lossless"
    values: list[list[JsonValue]]
    nonfinite_mask: list[list[str | None]] | None
    stripe_bytes: int

    def body(self) -> dict[str, Any]:
        """The RoiResponse fields this layer owns (the API adds source, roi and warnings)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "shape": list(self.shape),
            "source_dtype": self.source_dtype,
            "value_encoding": self.value_encoding.value,
            "coverage": "complete",
            "numeric_precision": self.numeric_precision,
            "values": self.values,
            "nonfinite_mask": self.nonfinite_mask,
        }

    def to_json(self) -> str:
        return json.dumps(self.body(), allow_nan=False, separators=(",", ":"))


def auto_encoding(kind: str, size: int) -> ValueEncoding:
    if kind == "float":
        return ValueEncoding.FLOAT_JSON
    if kind == "bool":
        return ValueEncoding.BOOL_JSON
    return ValueEncoding.INT_DECIMAL_STRING if size == 8 else ValueEncoding.INT_JSON


def encode_roi(roi: RawRoi, encoding: ValueEncoding = ValueEncoding.AUTO) -> EncodedRoi:
    dt = roi.dtype
    enc = auto_encoding(dt.kind, dt.size) if encoding is ValueEncoding.AUTO else encoding
    if enc not in _ALLOWED[dt.kind]:
        raise fail(ErrorCode.VALIDATION_ERROR, f"value_encoding {enc.value} does not apply to {dt.name}")
    shape = (int(roi.raw.shape[0]), int(roi.raw.shape[1]))
    mask: list[list[str | None]] | None = None

    if enc is ValueEncoding.HEX_BITS:
        width = dt.size * 2
        values: list[list[JsonValue]] = [[f"0x{int(v):0{width}x}" for v in row] for row in bits(roi.raw, dt)]
    elif enc is ValueEncoding.FLOAT_JSON:
        decoded = roi.values()
        values, mask = [], []
        any_nonfinite = False
        for row in decoded.tolist():
            vrow: list[JsonValue] = []
            mrow: list[str | None] = []
            for x in row:
                if math.isfinite(x):
                    vrow.append(x)
                    mrow.append(None)
                else:
                    any_nonfinite = True
                    vrow.append(None)
                    mrow.append("nan" if math.isnan(x) else "pos_inf" if x > 0 else "neg_inf")
            values.append(vrow)
            mask.append(mrow)
        if not any_nonfinite:
            mask = None
    elif enc is ValueEncoding.INT_JSON:
        decoded = roi.values()
        if decoded.size and (
            int(decoded.max()) > MAX_SAFE_INTEGER or (dt.kind == "int" and int(decoded.min()) < -MAX_SAFE_INTEGER)
        ):
            raise fail(ErrorCode.UNSUPPORTED_PRECISION, "values beyond +/-(2^53-1) need int_decimal_string")
        values = decoded.tolist()
    elif enc is ValueEncoding.INT_DECIMAL_STRING:
        values = [[str(v) for v in row] for row in roi.values().tolist()]
    else:  # BOOL_JSON
        values = roi.values().astype(np.bool_).tolist()
    return EncodedRoi(shape, dt.name, enc, "lossless", values, mask, roi.stripe_bytes)
