"""View and ROI validation, read planning (doc 03 section 3, doc 09 section 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import write_safetensors
from weightlab import WeightLabError, open_checkpoint
from weightlab.core.limits import Limits
from weightlab.formats.selection import View, plan_reads, resolve_view, validate_roi


def err(fn: object, *a: object, **k: object) -> str:
    with pytest.raises(WeightLabError) as info:
        fn(*a, **k)  # type: ignore[operator]
    return info.value.code.value


def test_view_defaults_and_rules() -> None:
    assert resolve_view((), None).rows == 1
    assert (resolve_view((5,), None).rows, resolve_view((5,), None).cols) == (1, 5)
    rv = resolve_view((3, 4), None)
    assert (rv.rows, rv.cols) == (3, 4)
    rv = resolve_view((3, 4), View(0, 1, {}, transpose=True))
    assert (rv.row_axis, rv.col_axis, rv.rows, rv.cols) == (1, 0, 4, 3)
    assert err(resolve_view, (2, 3, 4), None) == "INVALID_ROI"  # rank >= 3 never guessed
    assert err(resolve_view, (2, 3, 4), View(1, 1, {0: 0})) == "INVALID_ROI"
    assert err(resolve_view, (2, 3, 4), View(1, 2, {})) == "INVALID_ROI"
    assert err(resolve_view, (2, 3, 4), View(1, 2, {0: 2})) == "INVALID_ROI"
    assert err(resolve_view, (2, 3, 4), View(1, 2, {0: 0, 1: 0})) == "INVALID_ROI"
    assert err(resolve_view, (2, 3, 4), View(True, 2, {0: 0})) == "INVALID_ROI"  # type: ignore[arg-type]
    assert err(resolve_view, (3, 4), View(0, 2)) == "INVALID_ROI"
    assert err(resolve_view, (5,), View(0, 0)) == "INVALID_ROI"
    assert err(resolve_view, (), View(0, 1)) == "INVALID_ROI"


@pytest.mark.parametrize(
    ("coords", "code"),
    [
        ((0, 4, 0, 5), "INVALID_ROI"),  # beyond rows (half-open)
        ((2, 1, 0, 1), "INVALID_ROI"),
        ((-1, 1, 0, 1), "INVALID_ROI"),
        ((True, 1, 0, 1), "INVALID_ROI"),
        ((0.0, 1, 0, 1), "INVALID_ROI"),
        ((0, 2**53, 0, 1), "INVALID_ROI"),
    ],
)
def test_roi_bounds(coords: tuple[int, int, int, int], code: str) -> None:
    rv = resolve_view((3, 4), None)
    assert err(validate_roi, rv, *coords, 4096) == code


def test_roi_cell_budget() -> None:
    rv = resolve_view((100, 100), None)
    validate_roi(rv, 0, 64, 0, 64, 4096)
    assert err(validate_roi, rv, 0, 65, 0, 64, 4096) == "RESOURCE_LIMIT"


def test_plan_reads_coalesces_and_caps() -> None:
    limits = Limits(read_chunk_bytes=64, coalesce_gap_bytes=8)
    idx = np.array([0, 1, 2, 4, 50, 51, 100], dtype=np.int64)  # 4-byte elements
    chunks = plan_reads(idx, 4, limits)
    assert [(c.first, c.stop) for c in chunks] == [(0, 5), (50, 52), (100, 101)]
    dense = np.arange(40, dtype=np.int64)
    chunks = plan_reads(dense, 4, limits)
    assert [(c.first, c.stop) for c in chunks] == [(0, 16), (16, 32), (32, 40)]
    assert sum(c.hi - c.lo for c in chunks) == 40


def test_api_level_errors(tmp_path: Path) -> None:
    data = np.arange(12, dtype="<f4").tobytes()
    p = write_safetensors(tmp_path / "m.safetensors", {"w": ("F32", (3, 4), data)})
    with open_checkpoint(p, limits=Limits(roi_max_read_bytes=16)) as ck:
        assert err(ck.read_roi_2d, "nope", 0, 1, 0, 1) == "NOT_FOUND"
        assert err(ck.read_roi_2d, "w", 0, 3, 0, 4) == "RESOURCE_LIMIT"  # 48 stripe bytes > 16
        assert ck.read_roi_2d("w", 0, 1, 0, 4).tolist() == [[0, 1, 2, 3]]
        assert err(ck.read_roi_2d, "w", 0, 1, 0, 1, max_cells=0) == "RESOURCE_LIMIT"
        assert err(lambda: list(ck.iter_flat_chunks("w", 0))) == "RESOURCE_LIMIT"


def test_returned_arrays_are_independent_of_file_bytes(tmp_path: Path) -> None:
    data = np.arange(4, dtype="<f4").tobytes()
    p = write_safetensors(tmp_path / "m.safetensors", {"w": ("F32", (2, 2), data)})
    with open_checkpoint(p) as ck:
        a = ck.read_roi_2d("w", 0, 2, 0, 2)
        a[0, 0] = 99  # writable, owns its memory
        assert ck.read_roi_2d("w", 0, 1, 0, 1)[0, 0] == 0
        raw = ck.read_roi_raw("w", 0, 2, 0, 2).raw
        assert raw.flags.writeable  # np.frombuffer views of file bytes are read-only
        raw[0, 0] = 7
        assert ck.read_roi_raw("w", 0, 1, 0, 1).raw[0, 0] == 0
