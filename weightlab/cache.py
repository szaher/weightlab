"""Bounded backend caches (doc 10 section 6, doc 11, SEC-10): a memory tile LRU and a disk pyramid LRU, plus a
bounded in-memory home for pyramids that cannot fit the disk cap (doc 09 section 3: "kept in memory for the session
or the job is rejected with RESOURCE_LIMIT").

Keys are the canonical SHA-256 identities of PR #6 (`tile_codec.tile_identity` / `pyramid_identity`): they hash
every input file's quick fingerprint, so a changed source can never hit an old entry, and entries for it are
unreachable until evicted. Nothing else is stored: no token, no checkpoint bytes, no paths outside the cache
directory. Only files the cache itself names (`<64 hex>.wlpyr`, `*.wlpyr.tmp-*`, `owners.json`) inside its own
directory are ever deleted; eviction never touches source checkpoints.

Disk pyramids are written to a temporary file in the cache directory, fsynced and published with an atomic rename
(`os.replace`); a crash leaves at most temporary files, which startup removes. Space is reserved before a build:
LRU entries are evicted until `used + reserved + needed <= tile_cache_disk_mib`, and the filesystem must keep a free
margin; a pyramid that cannot fit goes to the session store (bounded, LRU) or is refused with RESOURCE_LIMIT. Every
lookup validates the stored file (`pyramid.load`); a corrupt or mismatching entry is deleted and reported missing,
so it is rebuilt, never served.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from weightlab.analysis import pyramid as pyr
from weightlab.core.errors import ErrorCode, fail

MIB = 1024 * 1024
ENTRY = re.compile(r"^([0-9a-f]{64})\.wlpyr$")
TMP = re.compile(r"^[0-9a-f]{64}\.wlpyr\.tmp-[0-9a-f]{16}$")
OWNERS = "owners.json"
FREE_MARGIN = 64 * MIB  # never fill the filesystem
TILE_ENTRY_OVERHEAD = 2048  # metadata dict and bookkeeping per cached tile


# ---------------------------------------------------------------------------------------------- memory tiles
@dataclass
class CachedTile:
    meta: dict[str, Any]  # TileMetadata without `source` (the requester's IDs are added per response)
    body: bytes  # the binary tile body (rows*cols*5 bytes)


class MemoryTileCache:
    """LRU of encoded tiles bounded by bytes (`tile_cache_memory_mib`). Thread-safe."""

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity = capacity_bytes
        self._items: OrderedDict[str, CachedTile] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def _size(t: CachedTile) -> int:
        return len(t.body) + TILE_ENTRY_OVERHEAD

    def get(self, key: str) -> CachedTile | None:
        with self._lock:
            t = self._items.get(key)
            if t is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return t

    def put(self, key: str, tile: CachedTile) -> None:
        size = self._size(tile)
        with self._lock:
            if size > self.capacity:
                return
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= self._size(old)
            self._items[key] = tile
            self._bytes += size
            self._shrink()

    def set_capacity(self, capacity_bytes: int) -> None:
        with self._lock:
            self.capacity = capacity_bytes
            self._shrink()

    def _shrink(self) -> None:
        while self._bytes > self.capacity and self._items:
            _, t = self._items.popitem(last=False)
            self._bytes -= self._size(t)
            self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def remove_where(self, pred: Callable[[CachedTile], bool]) -> int:
        with self._lock:
            gone = [k for k, t in self._items.items() if pred(t)]
            for k in gone:
                self._bytes -= self._size(self._items.pop(k))
            return len(gone)

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._items)


class TileMetadataStore:
    """Metadata of tiles served in this session, for GET /tiles/{cache_key}/metadata (bounded LRU by count)."""

    def __init__(self, max_entries: int = 8192) -> None:
        self.max_entries = max_entries
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, key: str, meta: dict[str, Any], project_id: str) -> None:
        with self._lock:
            self._items[key] = {"meta": meta, "project_id": project_id}
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            e = self._items.get(key)
            return None if e is None else dict(e["meta"])

    def remove_project(self, project_id: str) -> None:
        with self._lock:
            for k in [k for k, e in self._items.items() if e["project_id"] == project_id]:
                del self._items[k]


# ------------------------------------------------------------------------------------------------- pyramids
@dataclass
class Reservation:
    pyramid_id: str
    where: Literal["disk", "memory"]
    nbytes: int
    tmp_path: Path | None = None


@dataclass
class _DiskEntry:
    size: int
    stat_key: tuple[int, int, int]  # size, mtime_ns, inode: a validated Pyramid is reused only for the same file
    validated: pyr.Pyramid | None = None


@dataclass
class _SessionEntry:
    data: bytes
    validated: pyr.Pyramid | None = None


@dataclass
class CacheStats:
    disk_bytes: int
    disk_entries: int
    disk_reserved: int
    session_bytes: int
    session_entries: int
    hits: int
    misses: int
    evictions: int
    corrupt: int
    owners: dict[str, list[str]] = field(default_factory=dict)


class PyramidHandle:
    """A validated pyramid with an open blob; use as a context manager (closes a file handle promptly)."""

    def __init__(self, blob: pyr.FileBlob | pyr.BytesBlob, pyramid: pyr.Pyramid, where: str) -> None:
        self.blob, self.pyramid, self.where = blob, pyramid, where

    def __enter__(self) -> PyramidHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.blob.close()


class PyramidCache:
    def __init__(
        self,
        root: Path,
        disk_capacity: Callable[[], int],
        session_capacity: Callable[[], int],
    ) -> None:
        self.root = root
        self.disk_capacity = disk_capacity
        self.session_capacity = session_capacity
        self._lock = threading.RLock()
        self._disk: OrderedDict[str, _DiskEntry] = OrderedDict()
        self._session: OrderedDict[str, _SessionEntry] = OrderedDict()
        self._reserved: dict[str, Reservation] = {}
        self._owners: dict[str, set[str]] = {}
        self._doomed: set[str] = set()  # deletions that failed (Windows: file in use); retried later
        self.hits = self.misses = self.evictions = self.corrupt = 0
        root.mkdir(parents=True, exist_ok=True)
        self._recover()

    # -- startup -----------------------------------------------------------------------------------
    def _recover(self) -> None:
        """Remove leftovers of an unexpected shutdown (temporary files) and index published entries, oldest first."""
        found = []
        for p in self.root.iterdir():
            if TMP.match(p.name):
                with contextlib.suppress(OSError):
                    p.unlink()
            elif m := ENTRY.match(p.name):
                st = p.stat()
                found.append((st.st_mtime_ns, m.group(1), st))
        for _, pid, st in sorted(found):
            self._disk[pid] = _DiskEntry(st.st_size, (st.st_size, st.st_mtime_ns, st.st_ino))
        self._owners = self._read_owners()

    def _read_owners(self) -> dict[str, set[str]]:
        try:
            raw = json.loads((self.root / OWNERS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {k: {p for p in v if isinstance(p, str)} for k, v in raw.items() if isinstance(v, list) and len(k) == 64}

    def _write_owners(self) -> None:
        tmp = self.root / f"{OWNERS}.{secrets.token_hex(8)}.tmp"
        data = {k: sorted(v) for k, v in sorted(self._owners.items()) if v and (k in self._disk or k in self._session)}
        tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.root / OWNERS)

    # -- paths -------------------------------------------------------------------------------------
    def _path(self, pid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", pid):
            raise ValueError("pyramid id must be 64 hex digits")
        return self.root / f"{pid}.wlpyr"

    # -- lookup ------------------------------------------------------------------------------------
    def contains(self, pid: str) -> bool:
        with self._lock:
            return pid in self._disk or pid in self._session

    @contextlib.contextmanager
    def open(self, pid: str, identity: Mapping[str, Any]) -> Iterator[PyramidHandle | None]:
        """A validated pyramid for `pid` with this identity, or None (missing, or corrupt and now deleted)."""
        handle = self._open(pid, identity)
        try:
            yield handle
        finally:
            if handle is not None:
                handle.blob.close()

    def _open(self, pid: str, identity: Mapping[str, Any]) -> PyramidHandle | None:
        with self._lock:
            s = self._session.get(pid)
            if s is not None:
                self._session.move_to_end(pid)
                blob_b = pyr.BytesBlob(s.data)
                try:
                    if s.validated is None:
                        s.validated = pyr.load(blob_b, identity, pid)
                except pyr.PyramidCorrupt:
                    self._discard(pid)
                    self.corrupt += 1
                    self.misses += 1
                    return None
                self.hits += 1
                return PyramidHandle(blob_b, s.validated, "memory")
            e = self._disk.get(pid)
            if e is None:
                self.misses += 1
                return None
            self._disk.move_to_end(pid)
        path = self._path(pid)
        try:
            blob = pyr.FileBlob(path)
        except OSError:
            with self._lock:
                self._disk.pop(pid, None)
            self.misses += 1
            return None
        try:
            st = os.fstat(blob._f.fileno())
            key = (st.st_size, st.st_mtime_ns, st.st_ino)
            with self._lock:
                validated = e.validated if e.stat_key == key else None
            if validated is None:
                validated = pyr.load(blob, identity, pid)
                with self._lock:
                    e.validated, e.stat_key, e.size = validated, key, st.st_size
            elif validated.trailer.get("identity") != json.loads(json.dumps(dict(identity), sort_keys=True)):
                raise pyr.PyramidCorrupt("identity differs")
        except pyr.PyramidCorrupt:
            blob.close()
            with self._lock:
                self._discard(pid)
                self.corrupt += 1
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return PyramidHandle(blob, validated, "disk")

    def report_corrupt(self, pid: str) -> None:
        """A tile read found a corrupt record run: delete the entry so the next request rebuilds it."""
        with self._lock:
            self._discard(pid)
            self.corrupt += 1

    # -- reservation and publication -------------------------------------------------------------
    def disk_used(self) -> int:
        return sum(e.size for e in self._disk.values()) + sum(
            r.nbytes for r in self._reserved.values() if r.where == "disk"
        )

    def session_used(self) -> int:
        return sum(len(e.data) for e in self._session.values()) + sum(
            r.nbytes for r in self._reserved.values() if r.where == "memory"
        )

    def placement(self, nbytes: int) -> Literal["disk", "memory"] | None:
        """Where a pyramid of `nbytes` could go if everything evictable were evicted (no side effects)."""
        if nbytes <= self.disk_capacity():
            return "disk"
        if nbytes <= self.session_capacity():
            return "memory"
        return None

    def reserve(self, pid: str, nbytes: int) -> Reservation:
        """Space for one build, after LRU eviction; RESOURCE_LIMIT when neither the disk cap nor the session store
        can take it. Never exceeds either cap."""
        with self._lock:
            if pid in self._reserved:
                raise fail(ErrorCode.CONFLICT, "a build of this pyramid is already running")
            cap = self.disk_capacity()
            if nbytes <= cap and self._evict_disk(cap - nbytes) and self._free_space_ok(nbytes):
                res = Reservation(pid, "disk", nbytes, self.root / f"{pid}.wlpyr.tmp-{secrets.token_hex(8)}")
            else:
                scap = self.session_capacity()
                if nbytes > scap or not self._evict_session(scap - nbytes):
                    raise fail(
                        ErrorCode.RESOURCE_LIMIT,
                        f"the exact pyramid needs {nbytes} bytes, more than the disk cache "
                        f"(tile_cache_disk_mib, {cap // MIB} MiB) or the session memory store "
                        f"({scap // MIB} MiB) can hold",
                    )
                res = Reservation(pid, "memory", nbytes)
            self._reserved[pid] = res
            return res

    def _free_space_ok(self, nbytes: int) -> bool:
        try:
            return shutil.disk_usage(self.root).free - nbytes >= FREE_MARGIN
        except OSError:
            return False

    def _evict_disk(self, target: int) -> bool:
        """Evict least recently used disk entries until used <= target. False if it cannot get there."""
        self._retry_doomed()
        for pid in list(self._disk):
            if self.disk_used() <= target:
                break
            self._discard(pid)
            self.evictions += 1
        return self.disk_used() <= target

    def _evict_session(self, target: int) -> bool:
        for pid in list(self._session):
            if self.session_used() <= target:
                break
            self._session.pop(pid, None)
            self.evictions += 1
        return self.session_used() <= target

    def _discard(self, pid: str) -> None:
        self._session.pop(pid, None)
        if self._disk.pop(pid, None) is not None or (self.root / f"{pid}.wlpyr").exists():
            try:
                self._path(pid).unlink(missing_ok=True)
            except OSError:  # Windows: still open by a reader; retried on the next eviction pass
                self._doomed.add(pid)

    def _retry_doomed(self) -> None:
        for pid in list(self._doomed):
            try:
                self._path(pid).unlink(missing_ok=True)
                self._doomed.discard(pid)
            except OSError:
                pass

    def publish(self, res: Reservation, sink: pyr.FileSink | pyr.MemorySink, actual_bytes: int, owner: str) -> None:
        """Make a completed build visible: atomic rename into place (disk) or keep the bytes (memory)."""
        with self._lock:
            if self._reserved.get(res.pyramid_id) is not res:
                raise fail(ErrorCode.CONFLICT, "pyramid reservation is no longer valid")
            if actual_bytes > res.nbytes:
                raise fail(ErrorCode.RESOURCE_LIMIT, "pyramid larger than its reservation")
            if res.where == "disk":
                assert isinstance(sink, pyr.FileSink) and res.tmp_path is not None  # noqa: PT018
                sink.close(sync=True)
                final = self._path(res.pyramid_id)
                os.replace(res.tmp_path, final)
                st = final.stat()
                self._disk[res.pyramid_id] = _DiskEntry(st.st_size, (st.st_size, st.st_mtime_ns, st.st_ino))
            else:
                assert isinstance(sink, pyr.MemorySink)
                self._session[res.pyramid_id] = _SessionEntry(bytes(sink.buf[:actual_bytes]))
            del self._reserved[res.pyramid_id]
            self._owners.setdefault(res.pyramid_id, set()).add(owner)
            self._write_owners()

    def abort(self, res: Reservation, sink: pyr.FileSink | pyr.MemorySink | None) -> None:
        """Discard a failed, canceled or interrupted build: nothing becomes visible."""
        with self._lock:
            if isinstance(sink, pyr.FileSink):
                sink.close(sync=False)
            if res.tmp_path is not None:
                with contextlib.suppress(OSError):
                    res.tmp_path.unlink(missing_ok=True)
            if self._reserved.get(res.pyramid_id) is res:
                del self._reserved[res.pyramid_id]

    def enforce_caps(self) -> int:
        """Evict (LRU) until both stores are within their current caps, e.g. after tile_cache_disk_mib was lowered.
        Returns the number of entries removed. Running builds keep their reservations."""
        with self._lock:
            before = self.evictions
            self._evict_disk(max(0, self.disk_capacity()))
            self._evict_session(max(0, self.session_capacity()))
            self._write_owners()
            return self.evictions - before

    def add_owner(self, pid: str, owner: str) -> None:
        with self._lock:
            if pid in self._disk or pid in self._session:
                before = set(self._owners.get(pid, set()))
                self._owners.setdefault(pid, set()).add(owner)
                if before != self._owners[pid]:
                    self._write_owners()

    def purge_owner(self, owner: str) -> int:
        """Remove every pyramid a project used (project deletion with purge_cache). Shared entries are removed too;
        they are recomputed on demand. Source checkpoints are never touched."""
        with self._lock:
            gone = [pid for pid, owners in self._owners.items() if owner in owners]
            for pid in gone:
                self._discard(pid)
                del self._owners[pid]
            for owners in self._owners.values():
                owners.discard(owner)
            self._write_owners()
            return len(gone)

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                disk_bytes=sum(e.size for e in self._disk.values()),
                disk_entries=len(self._disk),
                disk_reserved=sum(r.nbytes for r in self._reserved.values() if r.where == "disk"),
                session_bytes=sum(len(e.data) for e in self._session.values()),
                session_entries=len(self._session),
                hits=self.hits,
                misses=self.misses,
                evictions=self.evictions,
                corrupt=self.corrupt,
                owners={k: sorted(v) for k, v in self._owners.items()},
            )
