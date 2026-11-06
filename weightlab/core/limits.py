"""Resource and format limits (doc 04 section 2, doc 09 section 2, doc 11)."""

from __future__ import annotations

from dataclasses import dataclass

MAX_SAFE_INTEGER = 2**53 - 1  # JavaScript safe integer; coordinate cap (doc 09 section 2)


@dataclass(frozen=True)
class Limits:
    header_max_bytes: int = 100_000_000  # upstream safetensors cap
    index_max_bytes: int = 100_000_000
    max_rank: int = 32
    max_key_bytes: int = 4096
    roi_max_cells: int = 4096  # numeric ROI table budget (default)
    roi_hard_max_cells: int = 1_048_576  # hard upper bound for a configured roi_max_cells
    read_chunk_bytes: int = 4 * 1024 * 1024  # largest single positioned read / buffer
    coalesce_gap_bytes: int = 64 * 1024  # merge reads whose gap is at most this
    roi_max_read_bytes: int = 256 * 1024 * 1024  # stripe bytes one ROI may touch

    def __post_init__(self) -> None:
        if not 0 < self.roi_max_cells <= self.roi_hard_max_cells:
            raise ValueError("roi_max_cells must be within (0, roi_hard_max_cells]")
        if self.read_chunk_bytes < 64:
            raise ValueError("read_chunk_bytes too small")


DEFAULT_LIMITS = Limits()
