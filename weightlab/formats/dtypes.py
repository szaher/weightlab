"""SafeTensors dtypes, storage layout and exact decoding (doc 04 section 2, ADR-011, ADR-017).

Every supported dtype decodes losslessly:
  * F16, BF16, F32 embed exactly in float64; F64 is already float64.
  * Integers decode to int64 / uint64 without conversion through float.
BF16 is decoded by WeightLab (uint16 bits << 16 viewed as float32) because the safetensors
NumPy backend cannot represent it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from weightlab.core.errors import ErrorCode, fail

Kind = Literal["float", "int", "uint", "bool"]


@dataclass(frozen=True)
class DType:
    name: str
    size: int
    kind: Kind
    storage: str  # little-endian NumPy dtype of the raw stored element


DTYPES: dict[str, DType] = {
    d.name: d
    for d in (
        DType("F64", 8, "float", "<f8"),
        DType("F32", 4, "float", "<f4"),
        DType("F16", 2, "float", "<f2"),
        DType("BF16", 2, "float", "<u2"),
        DType("I64", 8, "int", "<i8"),
        DType("I32", 4, "int", "<i4"),
        DType("I16", 2, "int", "<i2"),
        DType("I8", 1, "int", "|i1"),
        DType("U64", 8, "uint", "<u8"),
        DType("U32", 4, "uint", "<u4"),
        DType("U16", 2, "uint", "<u2"),
        DType("U8", 1, "uint", "|u1"),
        DType("BOOL", 1, "bool", "|u1"),
    )
}

# Known to upstream but deliberately unsupported in R1 (no decoder with golden vectors yet).
KNOWN_UNSUPPORTED = frozenset({"F8_E4M3", "F8_E5M2", "F8_E8M0", "F6_E2M3", "F6_E3M2", "F4", "C64", "C128"})

# Raw bit view used for hex_bits output (floats by width, integers as their unsigned bits).
_BITS = {1: "|u1", 2: "<u2", 4: "<u4", 8: "<u8"}


def lookup(name: str) -> DType:
    dt = DTYPES.get(name)
    if dt is None:
        what = "not supported in R1" if name in KNOWN_UNSUPPORTED else "unknown"
        raise fail(ErrorCode.UNSUPPORTED_DTYPE, f"dtype {name!r} is {what}")
    return dt


def decode(raw: npt.NDArray[np.generic], dt: DType) -> npt.NDArray[np.generic]:
    """Decode raw stored elements to an exact, writable array that owns its memory.

    float -> float64, int -> int64, uint -> uint64, bool -> bool. Raises INVALID_FORMAT for a
    BOOL byte other than 0 or 1 (upstream treats those as invalid).
    """
    if dt.kind == "float":
        # Widening is exact; signaling NaNs are quieted (raw bits stay available via `bits`).
        with np.errstate(invalid="ignore"):
            if dt.name == "BF16":
                return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
            return raw.astype(np.float64)
    if dt.kind == "int":
        return raw.astype(np.int64)
    if dt.kind == "uint":
        return raw.astype(np.uint64)
    if raw.size and int(raw.max()) > 1:
        raise fail(ErrorCode.INVALID_FORMAT, "BOOL tensor holds a byte other than 0 or 1")
    return raw.astype(np.bool_)


def bits(raw: npt.NDArray[np.generic], dt: DType) -> npt.NDArray[np.unsignedinteger]:
    """Raw source bit patterns as unsigned integers of the source width."""
    return raw.view(np.dtype(_BITS[dt.size])).copy()
