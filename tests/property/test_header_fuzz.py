"""Fail-closed fuzzing: arbitrary corruptions of real fixture bytes either open cleanly (and then read
every tensor) or raise WeightLabError. No other exception type may escape."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.helpers import ROOT, upstream_accepts
from weightlab import WeightLabError, open_checkpoint

SEEDS = [
    ROOT / "examples/fixtures" / n
    for n in ("tiny_base.safetensors", "tiny_dtypes.safetensors", "tiny_edge.safetensors")
]


def exercise(p: Path) -> bool:
    try:
        with open_checkpoint(p) as ck:
            for m in ck.metadata():
                for _ in ck.iter_flat_chunks(m.key, 4096):
                    pass
        return True
    except WeightLabError:
        return False


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.sampled_from(SEEDS),
    edits=st.lists(st.tuples(st.integers(0, 1023), st.integers(0, 255)), min_size=1, max_size=6),
    cut=st.one_of(st.none(), st.integers(0, 2048)),
)
def test_corruption_fails_closed(seed: Path, edits: list[tuple[int, int]], cut: int | None) -> None:
    blob = bytearray(seed.read_bytes())
    for pos, val in edits:
        blob[pos % len(blob)] = val
    if cut is not None:
        del blob[cut:]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "f.safetensors"
        p.write_bytes(bytes(blob))
        ours = exercise(p)
        # WeightLab is never more permissive than upstream (it may be stricter, doc 04 section 2).
        if ours:
            assert upstream_accepts(p)
