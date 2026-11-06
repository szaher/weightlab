"""Strict SafeTensors header validation (doc 04 section 2).

WeightLab's own parser: struct + strict JSON + exact integer arithmetic. Upstream safetensors is
used only as a test oracle. Deliberately stricter than upstream 0.8.0 in two recorded places
(rank above 32, unknown tensor-entry fields).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

from weightlab.core.errors import ErrorCode, fail
from weightlab.core.limits import DEFAULT_LIMITS, Limits
from weightlab.formats.base import strict_json
from weightlab.formats.dtypes import DTYPES, KNOWN_UNSUPPORTED, DType
from weightlab.formats.fileio import PositionedFile

ENTRY_FIELDS = frozenset({"dtype", "shape", "data_offsets"})
METADATA_KEY = "__metadata__"


@dataclass(frozen=True)
class Entry:
    key: str
    dtype: DType
    shape: tuple[int, ...]
    begin: int  # relative to the data region
    end: int


@dataclass(frozen=True)
class Header:
    header_len: int  # N, the JSON byte length
    entries: dict[str, Entry]  # in file (JSON) order
    metadata: dict[str, str] | None

    @property
    def data_start(self) -> int:
        return 8 + self.header_len


def _is_json_int(v: object) -> bool:
    return type(v) is int  # bool is an int subclass and must not pass


def parse_header(header: bytes, data_len: int, limits: Limits = DEFAULT_LIMITS) -> Header:
    """Validate the JSON header bytes against a data region of `data_len` bytes."""
    if not header.startswith(b"{"):  # upstream InvalidHeaderStart; only trailing padding is allowed
        raise fail(ErrorCode.INVALID_FORMAT, "header does not start with '{'")
    doc = strict_json(header, "header")
    if not isinstance(doc, dict):
        raise fail(ErrorCode.INVALID_FORMAT, "header is not a JSON object")

    metadata: dict[str, str] | None = None
    raw_entries: list[tuple[str, dict[str, Any]]] = []
    for key, value in doc.items():
        if key == METADATA_KEY:
            if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
                raise fail(ErrorCode.INVALID_FORMAT, "__metadata__ must map strings to strings")
            metadata = value
            continue
        if len(key.encode("utf-8")) > limits.max_key_bytes:
            raise fail(ErrorCode.INVALID_FORMAT, f"tensor key longer than {limits.max_key_bytes} bytes")
        if not isinstance(value, dict) or set(value) != ENTRY_FIELDS:
            raise fail(ErrorCode.INVALID_FORMAT, "tensor entry must have exactly dtype, shape, data_offsets")
        raw_entries.append((key, value))

    # Structure first, so a malformed entry is never masked by a dtype message.
    for _, value in raw_entries:
        shape, offsets, dtype = value["shape"], value["data_offsets"], value["dtype"]
        if not isinstance(dtype, str):
            raise fail(ErrorCode.INVALID_FORMAT, "dtype must be a string")
        if not isinstance(shape, list) or not all(_is_json_int(d) and d >= 0 for d in shape):
            raise fail(ErrorCode.INVALID_FORMAT, "shape must be a list of non-negative integers")
        if len(shape) > limits.max_rank:
            raise fail(ErrorCode.INVALID_FORMAT, f"rank above {limits.max_rank}")
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(_is_json_int(o) for o in offsets)):
            raise fail(ErrorCode.INVALID_FORMAT, "data_offsets must be two integers")
        if not 0 <= offsets[0] <= offsets[1]:
            raise fail(ErrorCode.INVALID_FORMAT, "data_offsets must satisfy 0 <= begin <= end")

    for _, value in raw_entries:
        name = value["dtype"]
        if name not in DTYPES:
            what = "not supported in R1" if name in KNOWN_UNSUPPORTED else "unknown"
            raise fail(ErrorCode.UNSUPPORTED_DTYPE, f"dtype {name!r} is {what}")

    entries: dict[str, Entry] = {}
    for key, value in raw_entries:
        dt = DTYPES[value["dtype"]]
        begin, end = value["data_offsets"]
        if end - begin != math.prod(value["shape"]) * dt.size:
            raise fail(ErrorCode.INVALID_FORMAT, "byte length does not match shape and dtype")
        if end > data_len:
            raise fail(ErrorCode.INVALID_FORMAT, "data_offsets beyond the end of the file")
        entries[key] = Entry(key, dt, tuple(value["shape"]), begin, end)

    cursor = 0
    for e in sorted(entries.values(), key=lambda e: (e.begin, e.end)):
        if e.begin != cursor:
            raise fail(ErrorCode.INVALID_FORMAT, "tensor data ranges overlap" if e.begin < cursor else "gap in data")
        cursor = e.end
    if cursor != data_len:
        raise fail(ErrorCode.INVALID_FORMAT, "trailing bytes after the last tensor")
    return Header(len(header), entries, metadata)


def read_header(f: PositionedFile, limits: Limits = DEFAULT_LIMITS) -> Header:
    size = f.identity.size
    if size < 8:
        raise fail(ErrorCode.INVALID_FORMAT, "file shorter than the 8-byte header length")
    (n,) = struct.unpack("<Q", f.read(0, 8))
    if n > limits.header_max_bytes:
        raise fail(ErrorCode.INVALID_FORMAT, f"header length {n} above the {limits.header_max_bytes} byte cap")
    if 8 + n > size:
        raise fail(ErrorCode.INVALID_FORMAT, "header length exceeds the file size")
    return parse_header(f.read(8, n), size - 8 - n, limits)
