"""Bounded caches (doc 10 section 6, SEC-10): memory tile LRU, disk pyramid LRU with atomic publication, session
fallback, validation of stored entries, crash leftovers, purge by project, and never touching foreign files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers import write_safetensors
from weightlab import open_checkpoint
from weightlab.analysis import pyramid as P
from weightlab.analysis import tile_codec as tc
from weightlab.cache import CachedTile, MemoryTileCache, PyramidCache
from weightlab.core.errors import WeightLabError
from weightlab.formats.selection import View

MIB = 1024 * 1024


def test_memory_tile_cache_is_an_lru_bounded_by_bytes() -> None:
    c = MemoryTileCache(3 * (1000 + 2048))
    for k in "abc":
        c.put(k, CachedTile({}, b"x" * 1000))
    assert c.get("a") is not None  # a is now most recent
    c.put("d", CachedTile({}, b"x" * 1000))
    assert c.get("b") is None  # least recently used went first
    assert {k for k in "acd" if c.get(k) is not None} == {"a", "c", "d"}
    assert c.bytes_used <= c.capacity
    c.put("huge", CachedTile({}, b"x" * 10**6))  # larger than the whole cache: not stored, nothing evicted
    assert c.get("huge") is None
    assert len(c) == 3
    c.set_capacity(2048 + 1000)
    assert len(c) == 1
    assert c.hits >= 4
    assert c.evictions >= 3


class Built:
    def __init__(self, tmp: Path, name: str, shape: tuple[int, int]) -> None:
        a = np.arange(shape[0] * shape[1], dtype="<f4").reshape(shape)
        self.path = write_safetensors(tmp / f"{name}.safetensors", {"w": ("F32", shape, a.tobytes())})
        with open_checkpoint(self.path) as ck:
            files = ck.fingerprints(["w"])
        self.identity = tc.pyramid_identity(
            source_kind="tensor", files=files, tensor_keys=["w"], view=View(0, 1, {}, False)
        )
        self.pid = tc.cache_key(self.identity)
        self.nbytes = P.estimate_bytes(*shape)

    def into(self, cache: PyramidCache, owner: str = "prj_a") -> str:
        res = cache.reserve(self.pid, self.nbytes)
        sink: Any = P.FileSink(res.tmp_path) if res.tmp_path is not None else P.MemorySink(res.nbytes)
        with open_checkpoint(self.path) as ck:
            out = P.build_pyramid(ck, "w", View(0, 1, {}, False), self.identity, self.pid, sink)
        cache.publish(res, sink, out.total_bytes, owner)
        return res.where


def make(tmp_path: Path, disk_mib: float = 64, session_mib: float = 64) -> PyramidCache:
    return PyramidCache(tmp_path / "cache", lambda: int(disk_mib * MIB), lambda: int(session_mib * MIB))


def test_publication_is_atomic_and_validated_on_lookup(tmp_path: Path) -> None:
    c = make(tmp_path)
    b = Built(tmp_path, "a", (300, 200))
    res = c.reserve(b.pid, b.nbytes)
    assert res.where == "disk"
    sink = P.FileSink(res.tmp_path)  # type: ignore[arg-type]
    with open_checkpoint(b.path) as ck:
        out = P.build_pyramid(ck, "w", View(0, 1, {}, False), b.identity, b.pid, sink)
    with c.open(b.pid, b.identity) as h:
        assert h is None  # an unpublished build is invisible
    assert not (c.root / f"{b.pid}.wlpyr").exists()
    c.publish(res, sink, out.total_bytes, "prj_a")
    assert not list(c.root.glob("*.tmp-*"))
    with c.open(b.pid, b.identity) as h:
        assert h is not None
        assert h.pyramid.pyramid_id == b.pid
    with c.open(
        b.pid, {**b.identity, "view": {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": True}}
    ) as h:
        assert h is None  # a stored pyramid never serves another identity; the entry is discarded


def test_an_aborted_build_leaves_nothing(tmp_path: Path) -> None:
    c = make(tmp_path)
    b = Built(tmp_path, "a", (64, 64))
    res = c.reserve(b.pid, b.nbytes)
    sink = P.FileSink(res.tmp_path)  # type: ignore[arg-type]
    sink.write_at(0, b"partial")
    c.abort(res, sink)
    assert list(c.root.iterdir()) == []
    assert c.stats().disk_reserved == 0
    with c.open(b.pid, b.identity) as h:
        assert h is None


def test_startup_removes_crash_leftovers_and_keeps_published_entries(tmp_path: Path) -> None:
    c = make(tmp_path)
    b = Built(tmp_path, "a", (64, 64))
    b.into(c)
    leftover = c.root / f"{'e' * 64}.wlpyr.tmp-{'1' * 16}"
    leftover.write_bytes(b"half a pyramid")
    foreign = c.root / "notes.txt"
    foreign.write_text("not ours")
    c2 = make(tmp_path)
    assert not leftover.exists()
    assert foreign.exists()  # only files the cache names are ever deleted
    with c2.open(b.pid, b.identity) as h:
        assert h is not None


def test_disk_cap_is_never_exceeded_lru_evicts_and_reuse_hits(tmp_path: Path) -> None:
    builds = [Built(tmp_path, f"m{i}", (512, 512)) for i in range(5)]
    cap = [1 << 30]
    c = PyramidCache(tmp_path / "cache", lambda: cap[0], lambda: 0)
    for b in builds[:2]:
        b.into(c)
    # two entries plus one more reservation do not fit: exactly one eviction per further build
    cap[0] = c.stats().disk_bytes + builds[0].nbytes - 1
    with c.open(builds[0].pid, builds[0].identity) as h:  # touch m0: m1 becomes least recently used
        assert h is not None
    builds[2].into(c)
    on_disk = sorted(p.name for p in c.root.glob("*.wlpyr"))
    assert on_disk == sorted(f"{b.pid}.wlpyr" for b in (builds[0], builds[2]))
    for b in builds[3:]:
        b.into(c)
        assert sum(p.stat().st_size for p in c.root.glob("*.wlpyr")) <= cap[0]
        assert c.stats().disk_bytes <= cap[0]
    st = c.stats()
    assert st.evictions == 3
    assert st.hits >= 1
    with c.open(builds[4].pid, builds[4].identity) as h:  # reuse after eviction of others
        assert h is not None


def test_a_pyramid_too_large_for_disk_goes_to_the_session_store_or_is_refused(tmp_path: Path) -> None:
    b = Built(tmp_path, "a", (512, 512))
    c = make(tmp_path, disk_mib=0, session_mib=1)
    assert c.placement(b.nbytes) == "memory"
    assert b.into(c) == "memory"
    assert list(c.root.glob("*.wlpyr")) == []  # the disk cap (0) is respected
    with c.open(b.pid, b.identity) as h:
        assert h is not None
        assert h.where == "memory"
    tiny = make(tmp_path / "t", disk_mib=0, session_mib=0)
    assert tiny.placement(b.nbytes) is None
    with pytest.raises(WeightLabError) as e:
        tiny.reserve(b.pid, b.nbytes)
    assert e.value.code.value == "RESOURCE_LIMIT"


def test_corrupt_entries_are_deleted_and_never_served(tmp_path: Path) -> None:
    c = make(tmp_path)
    b = Built(tmp_path, "a", (300, 300))
    b.into(c)
    path = c.root / f"{b.pid}.wlpyr"
    data = bytearray(path.read_bytes())
    data[-10] ^= 0xFF  # footer checksum
    path.write_bytes(bytes(data))
    with c.open(b.pid, b.identity) as h:
        assert h is None
    assert not path.exists()
    assert c.stats().corrupt == 1
    b.into(c)  # rebuilt
    data = bytearray(path.read_bytes())
    data[P.PREAMBLE.size + 30] ^= 0x01  # a record byte: the trailer is intact, a tile read must catch it
    path.write_bytes(bytes(data))
    os.utime(path, ns=(1, 1))
    with c.open(b.pid, b.identity) as h:
        assert h is not None
        with pytest.raises(P.PyramidCorrupt):
            P.pyramid_tile(h.blob, h.pyramid, level=4, x=0, y=0, size=256, aggregation="mean", view_rows=300,
                           view_cols=300, esize=4)  # fmt: skip
    c.report_corrupt(b.pid)
    assert not path.exists()


def test_purge_by_project_removes_only_that_projects_entries(tmp_path: Path) -> None:
    c = make(tmp_path)
    a, b2, shared = Built(tmp_path, "a", (64, 64)), Built(tmp_path, "b", (64, 64)), Built(tmp_path, "s", (64, 64))
    a.into(c, "prj_a")
    b2.into(c, "prj_b")
    shared.into(c, "prj_a")
    c.add_owner(shared.pid, "prj_b")
    sources = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.path, b2.path, shared.path)}
    assert c.purge_owner("prj_a") == 2
    assert not c.contains(a.pid)
    assert not c.contains(shared.pid)  # shared entries go too (recomputed on demand)
    assert c.contains(b2.pid)
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources} == sources  # sources untouched
    c3 = make(tmp_path)
    assert "prj_a" not in str(c3.stats().owners)


def test_reservations_count_against_the_cap(tmp_path: Path) -> None:
    b1, b2 = Built(tmp_path, "a", (512, 512)), Built(tmp_path, "b", (512, 512))
    c = make(tmp_path, disk_mib=1.5 * b1.nbytes / MIB, session_mib=0)
    r1 = c.reserve(b1.pid, b1.nbytes)
    with pytest.raises(WeightLabError):
        c.reserve(b2.pid, b2.nbytes)  # the first build's space is held while it runs
    with pytest.raises(WeightLabError):
        c.reserve(b1.pid, b1.nbytes)  # one build per pyramid
    c.abort(r1, None)
    assert c.reserve(b2.pid, b2.nbytes).where == "disk"


def test_lowering_the_cap_evicts_at_once(tmp_path: Path) -> None:
    cap = [64 * MIB]
    c = PyramidCache(tmp_path / "cache", lambda: cap[0], lambda: 0)
    builds = [Built(tmp_path, f"m{i}", (512, 512)) for i in range(3)]
    for b in builds:
        b.into(c)
    with c.open(builds[0].pid, builds[0].identity) as h:  # m1 is now least recently used
        assert h is not None
    one = max(p.stat().st_size for p in c.root.glob("*.wlpyr"))
    cap[0] = 2 * one  # tile_cache_disk_mib lowered (settings PATCH): the cap holds immediately, not at the next build
    assert c.enforce_caps() == 1
    assert sum(p.stat().st_size for p in c.root.glob("*.wlpyr")) <= cap[0]
    assert not c.contains(builds[1].pid)
    cap[0] = 0
    c.enforce_caps()
    assert list(c.root.glob("*.wlpyr")) == []
