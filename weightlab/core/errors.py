"""Stable structured errors (doc 03 section 9; enum authority: contracts/openapi.yaml)."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN_ORIGIN = "FORBIDDEN_ORIGIN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    PATH_OUTSIDE_ROOT = "PATH_OUTSIDE_ROOT"
    INVALID_FORMAT = "INVALID_FORMAT"
    MISSING_SHARD = "MISSING_SHARD"
    INVALID_ROI = "INVALID_ROI"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    RATE_LIMITED = "RATE_LIMITED"
    UNSUPPORTED_DTYPE = "UNSUPPORTED_DTYPE"
    UNSUPPORTED_PRECISION = "UNSUPPORTED_PRECISION"
    INCOMPATIBLE_TENSOR = "INCOMPATIBLE_TENSOR"
    LINEAGE_UNVERIFIED = "LINEAGE_UNVERIFIED"
    FILE_CHANGED = "FILE_CHANGED"
    NONFINITE_POLICY = "NONFINITE_POLICY"
    OPTIONAL_RUNTIME_MISSING = "OPTIONAL_RUNTIME_MISSING"
    NOT_ACCEPTABLE = "NOT_ACCEPTABLE"


class WeightLabError(Exception):
    """An expected, user-reportable failure. `detail` is safe to show; it never holds file contents."""

    def __init__(self, code: ErrorCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


def fail(code: ErrorCode, detail: str) -> WeightLabError:
    return WeightLabError(code, detail)
