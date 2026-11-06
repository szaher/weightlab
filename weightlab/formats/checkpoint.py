"""Read-only checkpoint catalog and exact reads over single or sharded SafeTensors (doc 04 section 4).

    with open_checkpoint("/models/qwen", selected_root="/models") as ck:
        meta = ck.metadata()
        values = ck.read_roi_2d("model.embed_tokens.weight", 0, 4, 0, 8)

Change detection runs before the first read, after every read batch and at the end of every
call (best_effort); `consistency=verified` additionally hashes each touched file before and
after. A detected change raises FILE_CHANGED and no partial result is returned.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt

from weightlab.core.errors import ErrorCode, fail
from weightlab.core.limits import DEFAULT_LIMITS, Limits
from weightlab.formats.base import Consistency, TensorMeta
from weightlab.formats.dtypes import DType, decode
from weightlab.formats.fileio import PositionedFile
from weightlab.formats.safetensors_reader import Header, read_header
from weightlab.formats.selection import (
    View,
    element_indices,
    flat_ranges,
    plan_reads,
    resolve_view,
    stripe_bytes,
    validate_roi,
)
from weightlab.formats.shards import INDEX_SUFFIX, is_within, parse_index, resolve_shard

SINGLE_FILE_NAME = "model.safetensors"
INDEX_FILE_NAME = "model" + INDEX_SUFFIX


@dataclass
class _Shard:
    shard_id: str
    file: PositionedFile
    header: Header
    digest: str | None = None  # recorded at open when hash_files=True


@dataclass(frozen=True)
class RawRoi:
    """Raw stored elements of one ROI, before decoding. `raw` has the storage dtype."""

    meta: TensorMeta
    dtype: DType
    raw: npt.NDArray[np.generic]  # shape (r1-r0, c1-c0), owns its memory
    view_rows: int
    view_cols: int
    transpose: bool
    stripe_bytes: int  # bytes actually read (covering stripes, not only selected cells)

    def values(self) -> npt.NDArray[np.generic]:
        return decode(self.raw, self.dtype)


class Checkpoint:
    def __init__(
        self,
        root: Path,
        shards: dict[str, _Shard],
        tensors: dict[str, TensorMeta],
        warnings: list[str],
        limits: Limits,
    ) -> None:
        self.root = root
        self._shards = shards
        self._tensors = tensors
        self.warnings = warnings
        self.limits = limits

    # -- catalog ---------------------------------------------------------------------------
    def metadata(self) -> list[TensorMeta]:
        return list(self._tensors.values())

    def tensor(self, key: str) -> TensorMeta:
        meta = self._tensors.get(key)
        if meta is None:
            raise fail(ErrorCode.NOT_FOUND, "tensor key not in catalog")
        return meta

    @property
    def shard_ids(self) -> list[str]:
        return list(self._shards)

    def file_metadata(self, shard_id: str) -> dict[str, str] | None:
        return self._shards[shard_id].header.metadata

    @property
    def bytes_read(self) -> int:
        return sum(s.file.bytes_read for s in self._shards.values())

    # -- change detection ------------------------------------------------------------------
    def check(self) -> None:
        for s in self._shards.values():
            s.file.check()

    def _begin(self, shard: _Shard, consistency: Consistency) -> str | None:
        shard.file.check()
        if consistency is not Consistency.VERIFIED:
            return None
        digest = shard.file.sha256(self.limits.read_chunk_bytes)
        if shard.digest is not None and digest != shard.digest:
            raise fail(ErrorCode.FILE_CHANGED, f"{shard.shard_id} content differs from the digest recorded at open")
        return digest

    def _end(self, shard: _Shard, before: str | None) -> None:
        shard.file.check()
        if before is not None and shard.file.sha256(self.limits.read_chunk_bytes) != before:
            raise fail(ErrorCode.FILE_CHANGED, f"{shard.shard_id} content changed during the read")

    # -- exact reads -----------------------------------------------------------------------
    def read_roi_raw(
        self,
        key: str,
        r0: int,
        r1: int,
        c0: int,
        c1: int,
        view: View | None = None,
        *,
        max_cells: int | None = None,
        consistency: Consistency = Consistency.BEST_EFFORT,
    ) -> RawRoi:
        meta = self.tensor(key)
        shard = self._shards[meta.shard_id]
        entry = shard.header.entries[key]
        cap = self.limits.roi_max_cells if max_cells is None else max_cells
        if not 0 < cap <= self.limits.roi_hard_max_cells:
            raise fail(ErrorCode.RESOURCE_LIMIT, "max_cells outside the hard bound")
        rv = resolve_view(meta.shape, view)
        validate_roi(rv, r0, r1, c0, c1, cap)

        idx = element_indices(meta.shape, rv, r0, r1, c0, c1).ravel()
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        esize = entry.dtype.size
        chunks = plan_reads(sorted_idx, esize, self.limits)
        stripes = stripe_bytes(chunks, esize)
        if stripes > self.limits.roi_max_read_bytes:
            raise fail(ErrorCode.RESOURCE_LIMIT, f"ROI needs {stripes} bytes of reads, above roi_max_read_bytes")

        storage = np.dtype(entry.dtype.storage)
        out = np.empty(idx.size, dtype=storage)
        before = self._begin(shard, consistency)
        for c in chunks:
            buf = shard.file.read(meta.data_offset + c.first * esize, (c.stop - c.first) * esize)
            shard.file.check()
            elems = np.frombuffer(buf, dtype=storage)
            out[order[c.lo : c.hi]] = elems[sorted_idx[c.lo : c.hi] - c.first]
        self._end(shard, before)
        transpose = view.transpose if view is not None else False
        return RawRoi(meta, entry.dtype, out.reshape(r1 - r0, c1 - c0), rv.rows, rv.cols, transpose, stripes)

    def read_roi_2d(
        self,
        key: str,
        r0: int,
        r1: int,
        c0: int,
        c1: int,
        axes: tuple[int, int] | None = None,
        fixed_indices: Mapping[int, int] | None = None,
        *,
        transpose: bool = False,
        max_cells: int | None = None,
        consistency: Consistency = Consistency.BEST_EFFORT,
    ) -> npt.NDArray[np.generic]:
        """Exact decoded ROI of shape (r1-r0, c1-c0): float64 / int64 / uint64 / bool, never a view
        of file bytes. Interface of doc 04 section 4."""
        view = None
        if axes is not None or fixed_indices or transpose:
            ra, ca = axes if axes is not None else (0, 1)
            view = View(ra, ca, dict(fixed_indices or {}), transpose)
        raw = self.read_roi_raw(key, r0, r1, c0, c1, view, max_cells=max_cells, consistency=consistency)
        return raw.values()

    def iter_flat_chunks(
        self, key: str, chunk_bytes: int, *, consistency: Consistency = Consistency.BEST_EFFORT
    ) -> Iterator[npt.NDArray[np.generic]]:
        """Every element exactly once, in C order, decoded, in chunks of at most `chunk_bytes`.

        A FILE_CHANGED raised at any point (including after the last chunk) invalidates
        everything already yielded; callers accumulate and commit only on normal exhaustion.
        """
        meta = self.tensor(key)
        shard = self._shards[meta.shard_id]
        entry = shard.header.entries[key]
        if not 0 < chunk_bytes <= self.limits.read_chunk_bytes:
            raise fail(ErrorCode.RESOURCE_LIMIT, "chunk_bytes outside (0, read_chunk_bytes]")
        storage = np.dtype(entry.dtype.storage)
        esize = entry.dtype.size
        before = self._begin(shard, consistency)
        for start, stop in flat_ranges(meta.numel, esize, chunk_bytes):
            buf = shard.file.read(meta.data_offset + start * esize, (stop - start) * esize)
            shard.file.check()
            yield decode(np.frombuffer(buf, dtype=storage), entry.dtype)
        self._end(shard, before)

    # -- lifecycle -------------------------------------------------------------------------
    def close(self) -> None:
        for s in self._shards.values():
            s.file.close()

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def _open_shard(path: Path, shard_id: str, limits: Limits, hash_files: bool) -> _Shard:
    f = PositionedFile(path)
    try:
        header = read_header(f, limits)
        f.check()
        digest = f.sha256(limits.read_chunk_bytes) if hash_files else None
    except BaseException:
        f.close()
        raise
    return _Shard(shard_id, f, header, digest)


def _metas(shard: _Shard, keys: list[str] | None = None) -> dict[str, TensorMeta]:
    entries = shard.header.entries
    out = {}
    for key in keys if keys is not None else list(entries):
        e = entries[key]
        out[key] = TensorMeta(
            key, e.shape, e.dtype.name, shard.shard_id, shard.header.data_start + e.begin, e.end - e.begin
        )
    return out


def open_checkpoint(
    entry_path: str | os.PathLike[str],
    selected_root: str | os.PathLike[str] | None = None,
    *,
    limits: Limits = DEFAULT_LIMITS,
    hash_files: bool = False,
) -> Checkpoint:
    """Open a `.safetensors` file, a `*.safetensors.index.json`, or a directory holding
    `model.safetensors.index.json` or `model.safetensors`. Nothing outside `selected_root`
    (default: the entry's directory) is ever opened."""
    entry = Path(os.path.abspath(entry_path))
    if not entry.exists():
        raise fail(ErrorCode.NOT_FOUND, "entry path does not exist")
    if entry.is_dir():
        if (entry / INDEX_FILE_NAME).is_file():
            entry = entry / INDEX_FILE_NAME
        elif (entry / SINGLE_FILE_NAME).is_file():
            entry = entry / SINGLE_FILE_NAME
        else:
            raise fail(ErrorCode.INVALID_FORMAT, f"directory has neither {INDEX_FILE_NAME} nor {SINGLE_FILE_NAME}")
    root = Path(os.path.abspath(selected_root)) if selected_root is not None else entry.parent
    real_root = os.path.realpath(root)
    if not is_within(os.path.realpath(entry), real_root):
        raise fail(ErrorCode.PATH_OUTSIDE_ROOT, "entry path is outside the selected root")

    if not entry.name.endswith(INDEX_SUFFIX):
        shard = _open_shard(entry, entry.name, limits, hash_files)
        return Checkpoint(root, {shard.shard_id: shard}, _metas(shard), [], limits)

    index_file = PositionedFile(entry)
    try:
        if index_file.identity.size > limits.index_max_bytes:
            raise fail(ErrorCode.INVALID_FORMAT, "index file above the size cap")
        doc = parse_index(index_file.read(0, index_file.identity.size))
        index_file.check()
    finally:
        index_file.close()

    shards: dict[str, _Shard] = {}
    try:
        by_shard: dict[str, list[str]] = {}
        for key, rel in doc.weight_map.items():
            by_shard.setdefault(rel, []).append(key)
        paths = {rel: resolve_shard(entry.parent, rel, real_root) for rel in by_shard}
        for rel, path in paths.items():
            shards[rel] = _open_shard(path, rel, limits, hash_files)
        tensors: dict[str, TensorMeta] = {}
        warnings: list[str] = []
        for rel, keys in by_shard.items():
            entries = shards[rel].header.entries
            missing = [k for k in keys if k not in entries]
            if missing:
                raise fail(ErrorCode.INVALID_FORMAT, f"index key absent from shard {rel} ({len(missing)} keys)")
            extra = [k for k in entries if k not in doc.weight_map or doc.weight_map[k] != rel]
            if extra:
                warnings.append(f"shard {rel} holds {len(extra)} tensor(s) not mapped to it by the index")
            tensors.update(_metas(shards[rel], keys))
        if doc.total_size is not None:
            actual = sum(m.byte_length for m in tensors.values())
            if actual != doc.total_size:
                warnings.append(f"metadata.total_size {doc.total_size} differs from tensor bytes {actual}")
        return Checkpoint(root, shards, {k: tensors[k] for k in doc.weight_map}, warnings, limits)
    except BaseException:
        for s in shards.values():
            s.file.close()
        raise
