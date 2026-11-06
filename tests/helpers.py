"""Test-only helpers, independent of the reader under test (struct + json + numpy only)."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STORAGE = {
    "F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": "<u2",
    "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "|i1",
    "U64": "<u8", "U32": "<u4", "U16": "<u2", "U8": "|u1", "BOOL": "|u1",
}  # fmt: skip


def header_bytes(entries: dict[str, object]) -> bytes:
    h = json.dumps(entries, separators=(",", ":")).encode()
    return h + b" " * (-len(h) % 8)


def write_safetensors(
    path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]], metadata: dict[str, str] | None = None
) -> Path:
    header: dict[str, object] = {"__metadata__": metadata} if metadata else {}
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    h = header_bytes(header)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(t[2] for t in tensors.values()))
    return path


def write_raw(path: Path, header_obj_or_bytes: object, payload: bytes = b"", length: int | None = None) -> Path:
    h = header_obj_or_bytes if isinstance(header_obj_or_bytes, bytes) else header_bytes(header_obj_or_bytes)  # type: ignore[arg-type]
    path.write_bytes(struct.pack("<Q", len(h) if length is None else length) + h + payload)
    return path


def reference_array(path: Path, key: str) -> np.ndarray:
    """Independent reference: parse with struct/json, view the whole tensor with numpy."""
    blob = path.read_bytes()
    (n,) = struct.unpack("<Q", blob[:8])
    entry = json.loads(blob[8 : 8 + n])[key]
    b, e = entry["data_offsets"]
    return np.frombuffer(blob[8 + n + b : 8 + n + e], dtype=STORAGE[entry["dtype"]]).reshape(entry["shape"])


def upstream_accepts(path: Path) -> bool:
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="np") as f:
            for k in f.keys():  # noqa: SIM118 - safe_open has no __iter__
                f.get_slice(k).get_shape()
        return True
    except Exception:
        return False


def run_python(code: str, *args: str, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run(
        [sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=timeout, env=env, check=False
    )
