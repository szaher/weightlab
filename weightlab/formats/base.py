"""Shared reader types and the strict JSON parser (doc 04 sections 2-4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from weightlab.core.errors import ErrorCode, fail


class Consistency(StrEnum):
    """Change-detection level (doc 03 section 6)."""

    BEST_EFFORT = "best_effort"  # handle + path identity per batch; misses same-size, same-time rewrites
    VERIFIED = "verified"  # also SHA-256 of each input file before and after the job


@dataclass(frozen=True)
class TensorMeta:
    key: str
    shape: tuple[int, ...]
    dtype: str
    shard_id: str  # shard path relative to the checkpoint root, POSIX form
    data_offset: int  # absolute byte offset of the first element in the shard file
    byte_length: int
    role: str | None = None  # heuristic, assigned in Phase 1; never identity

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def is_empty(self) -> bool:
        return self.numel == 0


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-standard JSON token {token}")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate key")
        out[k] = v
    return out


def strict_json(data: bytes, what: str) -> Any:
    """Standard JSON only: UTF-8, no NaN/Infinity tokens, no duplicate keys, bounded nesting,
    no lone surrogates. Any violation is INVALID_FORMAT."""
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_no_duplicates)
        _reject_surrogates(value)
    except RecursionError:
        raise fail(ErrorCode.INVALID_FORMAT, f"{what}: JSON nesting too deep") from None
    except UnicodeEncodeError:
        raise fail(ErrorCode.INVALID_FORMAT, f"{what}: JSON string holds a lone surrogate") from None
    except ValueError as e:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        raise fail(ErrorCode.INVALID_FORMAT, f"{what}: invalid JSON ({type(e).__name__})") from None
    return value


def _reject_surrogates(value: Any) -> None:
    # Iterative walk: an adversarial document is already depth-limited by json.loads.
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            v.encode("utf-8")
        elif isinstance(v, dict):
            for k, item in v.items():
                k.encode("utf-8")
                stack.append(item)
        elif isinstance(v, list):
            stack.extend(v)
