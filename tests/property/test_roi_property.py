"""Property tests: any valid ROI on any N-D view equals an independent NumPy slice, bit for bit,
under any read-planner limits; flat chunks cover every element exactly once."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.helpers import STORAGE, reference_array, write_safetensors
from weightlab import open_checkpoint
from weightlab.core.limits import Limits
from weightlab.formats.selection import View

DTYPES = sorted(STORAGE)


@st.composite
def cases(draw: st.DrawFn) -> tuple[str, tuple[int, ...], bytes, View | None, tuple[int, int, int, int], Limits]:
    dtype = draw(st.sampled_from(DTYPES))
    rank = draw(st.integers(0, 4))
    shape = tuple(draw(st.lists(st.integers(1, 6), min_size=rank, max_size=rank)))
    numel = int(np.prod(shape, dtype=np.int64)) if shape else 1
    size = np.dtype(STORAGE[dtype]).itemsize
    if dtype == "BOOL":
        data = bytes(draw(st.lists(st.integers(0, 1), min_size=numel, max_size=numel)))
    else:
        data = draw(st.binary(min_size=numel * size, max_size=numel * size))
    view: View | None = None
    if rank >= 2:
        ra, ca = draw(st.permutations(range(rank)))[:2]
        fixed = {a: draw(st.integers(0, shape[a] - 1)) for a in range(rank) if a not in (ra, ca)}
        view = View(ra, ca, fixed, draw(st.booleans()))
    if rank == 0:
        rows, cols = 1, 1
    elif rank == 1:
        rows, cols = 1, shape[0]
    else:
        assert view is not None
        r_ax, c_ax = (view.col_axis, view.row_axis) if view.transpose else (view.row_axis, view.col_axis)
        assert r_ax is not None
        assert c_ax is not None
        rows, cols = shape[r_ax], shape[c_ax]
    r0 = draw(st.integers(0, rows))
    r1 = draw(st.integers(r0, rows))
    c0 = draw(st.integers(0, cols))
    c1 = draw(st.integers(c0, cols))
    limits = Limits(read_chunk_bytes=draw(st.sampled_from([64, 96, 4096])), coalesce_gap_bytes=draw(st.integers(0, 64)))
    return dtype, shape, data, view, (r0, r1, c0, c1), limits


def reference_roi(ref: np.ndarray, view: View | None, roi: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = roi
    if ref.ndim == 0:
        m = ref.reshape(1, 1)
    elif ref.ndim == 1:
        m = ref.reshape(1, -1)
    else:
        assert view is not None
        index: list[object] = [slice(None)] * ref.ndim
        for a, i in view.fixed_indices.items():
            index[a] = i
        sub = ref[tuple(index)]
        assert view.row_axis is not None
        assert view.col_axis is not None
        m = sub if view.row_axis < view.col_axis else sub.T
        if view.transpose:
            m = m.T
    return m[r0:r1, c0:c1]


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(cases())
def test_roi_equals_reference_slice(
    case: tuple[str, tuple[int, ...], bytes, View | None, tuple[int, int, int, int], Limits],
) -> None:
    dtype, shape, data, view, roi, limits = case
    with tempfile.TemporaryDirectory() as d:
        p = write_safetensors(Path(d) / "t.safetensors", {"pad": ("U8", (3,), b"abc"), "t": (dtype, shape, data)})
        ref = reference_array(p, "t")
        with open_checkpoint(p, limits=limits) as ck:
            got = ck.read_roi_raw("t", *roi, view, max_cells=4096)
            expected = reference_roi(ref, view, roi)
            assert got.raw.shape == expected.shape
            assert got.raw.tobytes() == np.ascontiguousarray(expected).tobytes()
            chunks = list(ck.iter_flat_chunks("t", limits.read_chunk_bytes))
            assert sum(c.size for c in chunks) == ref.size
            assert ck.bytes_read >= len(data)
