"""Logical 2-D views of N-D tensors and the positioned-read planner (doc 03 section 3, doc 04 sections 4-5).

Coordinates are zero-based, half-open `[r0, r1) x [c0, c1)` in the view's logical layout.
Rank 0 is a 1x1 view, rank 1 is a 1xN view (row axis absent, column axis 0). Rank >= 2 views name
`row_axis` and `col_axis`; every other axis is fixed by `fixed_indices`. For rank >= 3 the axes are
never guessed. `transpose=True` swaps the two displayed axes and is recorded in provenance.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from weightlab.core.errors import ErrorCode, fail
from weightlab.core.limits import MAX_SAFE_INTEGER, Limits


@dataclass(frozen=True)
class View:
    row_axis: int | None = None
    col_axis: int | None = None
    fixed_indices: Mapping[int, int] = field(default_factory=dict)
    transpose: bool = False


@dataclass(frozen=True)
class ResolvedView:
    row_axis: int | None  # source axis shown as rows (None: single implicit row)
    col_axis: int | None  # source axis shown as columns (None: rank 0)
    fixed: tuple[tuple[int, int], ...]
    rows: int
    cols: int


def _int(v: object) -> bool:
    return type(v) is int


def resolve_view(shape: tuple[int, ...], view: View | None) -> ResolvedView:
    rank = len(shape)
    if rank == 0:
        if view is not None and (view.row_axis, view.col_axis, dict(view.fixed_indices)) != (None, None, {}):
            raise fail(ErrorCode.INVALID_ROI, "a rank-0 tensor has no axes")
        return ResolvedView(None, None, (), 1, 1)
    if rank == 1:
        if view is not None and (
            view.row_axis is not None or view.col_axis not in (None, 0) or dict(view.fixed_indices) or view.transpose
        ):
            raise fail(ErrorCode.INVALID_ROI, "a rank-1 tensor is viewed as one row with col_axis 0")
        return ResolvedView(None, 0, (), 1, shape[0])
    if view is None:
        if rank > 2:
            raise fail(ErrorCode.INVALID_ROI, f"rank-{rank} tensor needs row_axis, col_axis and fixed_indices")
        view = View(0, 1)
    ra, ca = view.row_axis, view.col_axis
    if not (_int(ra) and _int(ca)) or ra is None or ca is None:
        raise fail(ErrorCode.INVALID_ROI, "row_axis and col_axis must be integers")
    if not (0 <= ra < rank and 0 <= ca < rank) or ra == ca:
        raise fail(ErrorCode.INVALID_ROI, "row_axis and col_axis must be distinct axes of the tensor")
    fixed = dict(view.fixed_indices)
    others = set(range(rank)) - {ra, ca}
    if set(fixed) != others or not all(_int(k) for k in fixed):
        raise fail(ErrorCode.INVALID_ROI, f"fixed_indices must name exactly axes {sorted(others)}")
    for axis, index in fixed.items():
        if not _int(index) or not 0 <= index < shape[axis]:
            raise fail(ErrorCode.INVALID_ROI, f"fixed index for axis {axis} out of range")
    if not isinstance(view.transpose, bool):
        raise fail(ErrorCode.INVALID_ROI, "transpose must be a boolean")
    if view.transpose:
        ra, ca = ca, ra
    return ResolvedView(ra, ca, tuple(sorted(fixed.items())), shape[ra], shape[ca])


def validate_roi(rv: ResolvedView, r0: int, r1: int, c0: int, c1: int, max_cells: int) -> None:
    coords = (r0, r1, c0, c1)
    if not all(_int(v) for v in coords) or not all(0 <= v <= MAX_SAFE_INTEGER for v in coords):
        raise fail(ErrorCode.INVALID_ROI, "ROI coordinates must be integers in [0, 2^53-1]")
    if not (r0 <= r1 <= rv.rows and c0 <= c1 <= rv.cols):
        raise fail(ErrorCode.INVALID_ROI, f"ROI outside the {rv.rows}x{rv.cols} view (half-open bounds)")
    if (r1 - r0) * (c1 - c0) > max_cells:
        raise fail(ErrorCode.RESOURCE_LIMIT, f"ROI has more than {max_cells} cells")


def element_indices(
    shape: tuple[int, ...], rv: ResolvedView, r0: int, r1: int, c0: int, c1: int
) -> npt.NDArray[np.int64]:
    """C-order flat element index of every ROI cell, shaped (r1-r0, c1-c0)."""
    strides = [1] * len(shape)
    for k in range(len(shape) - 2, -1, -1):
        strides[k] = strides[k + 1] * shape[k + 1]
    base = sum(index * strides[axis] for axis, index in rv.fixed)
    rs = strides[rv.row_axis] if rv.row_axis is not None else 0
    cs = strides[rv.col_axis] if rv.col_axis is not None else 0
    rows = np.arange(r0, r1, dtype=np.int64) * rs
    cols = np.arange(c0, c1, dtype=np.int64) * cs
    return np.int64(base) + rows[:, None] + cols[None, :]


@dataclass(frozen=True)
class ReadChunk:
    first: int  # first element index read
    stop: int  # one past the last element index read
    lo: int  # slice [lo, hi) of the sorted cell positions this chunk serves
    hi: int


def plan_reads(sorted_idx: npt.NDArray[np.int64], elem_size: int, limits: Limits) -> list[ReadChunk]:
    """Cover sorted, distinct element indices with few positioned reads.

    Neighboring cells are merged when the gap between them is at most `coalesce_gap_bytes`;
    no read exceeds `read_chunk_bytes` (a single element always fits).
    """
    n = len(sorted_idx)
    max_elems = max(1, limits.read_chunk_bytes // elem_size)
    gap_elems = limits.coalesce_gap_bytes // elem_size
    chunks: list[ReadChunk] = []
    i = 0
    while i < n:
        start = int(sorted_idx[i])
        j = int(np.searchsorted(sorted_idx, start + max_elems, side="left"))
        gaps = np.flatnonzero(np.diff(sorted_idx[i:j]) - 1 > gap_elems)
        if gaps.size:
            j = i + int(gaps[0]) + 1
        chunks.append(ReadChunk(start, int(sorted_idx[j - 1]) + 1, i, j))
        i = j
    return chunks


def stripe_bytes(chunks: list[ReadChunk], elem_size: int) -> int:
    return sum((c.stop - c.first) * elem_size for c in chunks)


def flat_ranges(numel: int, elem_size: int, chunk_bytes: int) -> Iterator[tuple[int, int]]:
    """Deterministic contiguous element ranges covering [0, numel) exactly once."""
    step = max(1, chunk_bytes // elem_size)
    for start in range(0, numel, step):
        yield start, min(numel, start + step)
