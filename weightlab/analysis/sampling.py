"""Row-sampled tile previews (doc 05 section 5 mode 5, doc 03 section 5, ADR-012): only for sources above
`pyramid_exact_max_mib`, never exact, never a source of exact statistics.

For a tile at level L every block-row (2^L view rows, clipped at the edge) contributes k = min(sampled_rows_per_block,
real rows) whole view rows, read across the tile's column range and aggregated per block with the same record
algebra as exact tiles. Selection is deterministic and independent of NumPy:

* `stride` (no seed): rows r0 + floor((2j + 1) * n / (2k)), j = 0..k-1 (centred on k equal strata of the n rows);
* `splitmix64_rows` (seed given): Floyd's algorithm draws k distinct offsets in [0, n) with unbiased bounded draws
  from `splitmix64(seed_mix ^ counter)` (the reference function of scripts/generate_demo_fixtures.py), where
  seed_mix = the first 8 bytes (little-endian) of SHA-256 over canonical JSON {pyramid_id, level, x, y, seed}:
  pyramid_id already hashes the source kind, every input file's fingerprint, the tensor key, view and numerics
  version. counter = (block_row << 20) | draw, with the draw index incremented on each rejection. Pure Python
  integers, so the selection is identical on every OS and NumPy version.

Rows are read whole (positioned reads of the tile's column range of each selected row), sorted, so I/O is
sequential. Labels: quality `row_sampled`, sampled_rows (rows per block), sampling_policy, seed, finite and
non-finite counts of the elements actually read, and a warning with the rows read out of the rows covered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from weightlab.analysis.tile import (
    NOT_REPRESENTABLE,
    Aggregation,
    BlockAggregates,
    Tile,
    _as_float64,
    columns_are_slow,
    precision_warnings,
    tile_geometry,
)
from weightlab.core.errors import ErrorCode, fail
from weightlab.formats.checkpoint import Checkpoint
from weightlab.formats.dtypes import lookup
from weightlab.formats.selection import View, resolve_view

MASK64 = (1 << 64) - 1
DEFAULT_ROWS_PER_BLOCK = 8
Policy = Literal["stride", "splitmix64_rows"]


def splitmix64(x: int) -> int:
    """SplitMix64 finalizer (doc 05 section 5); identical to scripts/generate_demo_fixtures.py `splitmix64`."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def seed_mix(pyramid_id: str, level: int, x: int, y: int, seed: int) -> int:
    doc = {"pyramid_id": pyramid_id, "level": level, "x": x, "y": y, "seed": seed}
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _bounded(mix: int, block_row: int, draw: int, bound: int) -> tuple[int, int]:
    """An unbiased integer in [0, bound) and the next draw index (rejection of the incomplete top range)."""
    limit = (1 << 64) - ((1 << 64) % bound)
    while True:
        v = splitmix64(mix ^ ((block_row << 20) | draw))
        draw += 1
        if v < limit:
            return v % bound, draw


def select_rows(n: int, k: int, *, policy: Policy, mix: int = 0, block_row: int = 0) -> list[int]:
    """k distinct offsets in [0, n), ascending."""
    if not 0 < k <= n:
        raise ValueError("need 0 < k <= n")
    if policy == "stride":
        return [((2 * j + 1) * n) // (2 * k) for j in range(k)]
    chosen: set[int] = set()
    draw = 0
    for j in range(n - k, n):  # Floyd: each j adds one new element, uniform over k-subsets
        t, draw = _bounded(mix, block_row, draw, j + 1)
        chosen.add(j if t in chosen else t)
    return sorted(chosen)


@dataclass(frozen=True)
class SamplePlan:
    rows_per_block: int
    policy: Policy
    seed: int | None
    rows_read: int  # total selected rows
    rows_covered: int
    read_bytes: int
    complete: bool  # the selection is every covered row: the result is exact and must be computed and labelled so


def plan_sample(
    ck: Checkpoint,
    key: str,
    view: View | None,
    *,
    level: int,
    x: int,
    y: int,
    size: int,
    rows_per_block: int | None,
    seed: int | None,
    max_read_bytes: int,
    max_cells: int,
) -> SamplePlan:
    """Choose rows per block (the requested number, or the largest default <= 8 that fits the read budget) and check
    the read and cell budgets before anything is read."""
    meta = ck.tensor(key)
    rv = resolve_view(meta.shape, view)
    g = tile_geometry(rv.rows, rv.cols, level, x, y, size)
    if columns_are_slow(rv, meta.shape):
        raise fail(
            ErrorCode.RESOURCE_LIMIT,
            "row sampling reads whole view rows, which are strided in storage for this (transposed) view; "
            "request the exact pyramid or the untransposed view",
        )
    width = g.col_end - g.col_start
    esize = lookup(meta.dtype).size
    nblocks = g.rows
    real = [min(g.row_end, g.row_start + (i + 1) * g.block) - (g.row_start + i * g.block) for i in range(nblocks)]

    def cost(k: int) -> tuple[int, int]:
        rows = sum(min(k, n) for n in real)
        return rows, rows * width * esize

    if rows_per_block is None:
        k = min(DEFAULT_ROWS_PER_BLOCK, g.block)
        while k > 1 and (cost(k)[1] > max_read_bytes or cost(k)[0] * width > max_cells):
            k //= 2
    else:
        k = rows_per_block
    rows, nbytes = cost(k)
    if nbytes > max_read_bytes or rows * width > max_cells:
        raise fail(
            ErrorCode.RESOURCE_LIMIT,
            f"a row-sampled tile reading {k} row(s) per block needs {nbytes} bytes ({rows} rows of {width} cells), "
            f"above the sampling budget {max_read_bytes}; use fewer sampled rows or a coarser level",
        )
    covered = g.row_end - g.row_start
    policy: Policy = "splitmix64_rows" if seed is not None else "stride"
    return SamplePlan(k, policy, seed, rows, covered, nbytes, rows >= covered)


def sampled_tile(
    ck: Checkpoint,
    key: str,
    view: View | None,
    *,
    level: int,
    x: int,
    y: int,
    size: int,
    aggregation: Aggregation,
    plan: SamplePlan,
    pyramid_id: str,
    band_cells: int = 1 << 18,
) -> Tile:
    if plan.complete:
        raise ValueError("a complete selection is an exact tile; compute it with compute_tile")
    meta = ck.tensor(key)
    rv = resolve_view(meta.shape, view)
    g = tile_geometry(rv.rows, rv.cols, level, x, y, size)
    dt = lookup(meta.dtype)
    mix = seed_mix(pyramid_id, level, x, y, plan.seed) if plan.seed is not None else 0
    acc = BlockAggregates.empty(g.rows, g.cols)
    rounded = False
    bytes_read = 0
    width = g.col_end - g.col_start
    per_row = max(1, min(width, band_cells // max(1, plan.rows_per_block), ck.limits.roi_hard_max_cells))
    for i in range(g.rows):
        r0 = g.row_start + i * g.block
        n = min(g.row_end, r0 + g.block) - r0
        k = min(plan.rows_per_block, n)
        offsets = select_rows(n, k, policy=plan.policy, mix=mix, block_row=(r0 // g.block))
        for c0 in range(g.col_start, g.col_end, per_row):  # column segments; partial blocks merge by addition
            c1 = min(g.col_end, c0 + per_row)
            rows = []
            for off in offsets:
                raw = ck.read_roi_raw(key, r0 + off, r0 + off + 1, c0, c1, view, max_cells=max(c1 - c0, 1))
                bytes_read += raw.stripe_bytes
                vals, rnd = _as_float64(raw.values(), dt)
                rounded |= rnd
                rows.append(vals)
            part = BlockAggregates.of_values(np.vstack(rows), 0, (c0 - g.col_start) % g.block, g.block)
            acc.merge_into(part, i, (c0 - g.col_start) // g.block)
    ck.check()
    values, status = acc.display(aggregation)
    finite = int(acc.finite.sum())
    counted = int(acc.finite.sum() + acc.nan.sum() + acc.pos_inf.sum() + acc.neg_inf.sum())
    not_repr = int(np.count_nonzero(status & NOT_REPRESENTABLE))
    warnings = [
        f"row-sampled preview ({plan.policy}"
        + (f", seed {plan.seed}" if plan.seed is not None else "")
        + f"): {plan.rows_read} of {plan.rows_covered} rows read, at most {plan.rows_per_block} per block; "
        "an exploratory visualization, not exact statistics",
        *precision_warnings(rounded=rounded, not_representable=not_repr),
    ]
    return Tile(
        geometry=g,
        aggregation=aggregation,
        source_dtype=dt.name,
        values=values,
        status=status,
        finite_count=finite,
        nonfinite_count=counted - finite,
        numeric_precision="overflowed" if not_repr else "rounded",
        covered_bytes=g.covered_cells * dt.size,
        bytes_read=bytes_read,
        quality="row_sampled",
        warnings=tuple(warnings),
        sampled_rows=plan.rows_per_block,
        sampling_policy=plan.policy,
        seed=plan.seed,
    )
