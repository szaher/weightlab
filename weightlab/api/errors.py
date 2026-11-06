"""Error envelope (doc 09 section 7) and the mapping from reader/analysis errors to HTTP statuses.

Payloads never carry stack traces, model bytes, or paths: reader errors already name files by
basename only, messages are length-capped, and `details` is a flat map of scalars.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

from weightlab.core.errors import ErrorCode, WeightLabError

MESSAGE_MAX = 2000

STATUS: dict[ErrorCode, int] = {
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN_ORIGIN: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.PATH_OUTSIDE_ROOT: 403,
    ErrorCode.INVALID_FORMAT: 422,
    ErrorCode.MISSING_SHARD: 422,
    ErrorCode.INVALID_ROI: 422,
    ErrorCode.RESOURCE_LIMIT: 413,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.UNSUPPORTED_DTYPE: 422,
    ErrorCode.UNSUPPORTED_PRECISION: 422,
    ErrorCode.INCOMPATIBLE_TENSOR: 409,
    ErrorCode.LINEAGE_UNVERIFIED: 409,
    ErrorCode.FILE_CHANGED: 409,
    ErrorCode.NONFINITE_POLICY: 422,
    ErrorCode.OPTIONAL_RUNTIME_MISSING: 404,
    ErrorCode.NOT_ACCEPTABLE: 406,
}
RETRYABLE = {ErrorCode.RATE_LIMITED}
ACTIONS: dict[ErrorCode, str] = {
    ErrorCode.FILE_CHANGED: "The checkpoint changed on disk since import; re-import it (or relink if it only moved).",
    ErrorCode.PATH_OUTSIDE_ROOT: "Choose a selected root that contains the model and all of its shards.",
    ErrorCode.RATE_LIMITED: "Retry after the current requests finish.",
    ErrorCode.LINEAGE_UNVERIFIED: "Verify the base model identity, or set research_mode to view it as research.",
}


class ApiError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        status: int | None = None,
        details: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status or STATUS[code]
        self.message = message
        self.details = details or {}


def from_weightlab(e: WeightLabError) -> ApiError:
    # The reader reports an OS permission refusal as VALIDATION_ERROR "permission denied: <name>";
    # doc 09 section 1.1 wants 403 with the OS reason (the contract enum has no dedicated code).
    status = 403 if e.code is ErrorCode.VALIDATION_ERROR and e.detail.startswith("permission denied") else None
    return ApiError(e.code, e.detail, status=status)


def request_id() -> str:
    return f"req_{secrets.token_hex(8)}"


def envelope(err: ApiError, rid: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "code": err.code.value,
        "message": err.message[:MESSAGE_MAX],
        "retryable": err.code in RETRYABLE,
    }
    if err.code in ACTIONS:
        body["recommended_action"] = ACTIONS[err.code]
    if err.details:
        flat = {k[:64]: v for k, v in list(err.details.items())[:32] if isinstance(v, str | int | float | bool | None)}
        body["details"] = {k: (v[:500] if isinstance(v, str) else v) for k, v in flat.items()}
    return {"error": body, "request_id": rid}


def envelope_bytes(err: ApiError, rid: str) -> bytes:
    return json.dumps(envelope(err, rid), allow_nan=False, separators=(",", ":")).encode()
