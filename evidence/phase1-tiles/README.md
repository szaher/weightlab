# Phase 1 Task 4B foundation: numerical tile engine (evidence)

Branch `phase-1-tile-engine`, based on `main` at cccd5de (merged PR #5). Spec 1.2-final is frozen (ADR-019). No file under `docs/*.md`, `contracts/`, `examples/fixtures/` or the spec scripts was edited.

**Scope.** This PR builds the numerical foundation only:
- exact tiles by direct bounded reads;
- the frozen binary and JSON tile representations;
- canonical tile identities.

**Not in this PR.**
- `/api/v1/tiles` is **not registered**; it stays `deferred`, and `tests/api/test_routes.py` still checks that. The contract's decision table also needs exact pyramids, row-sampled previews and `202` build jobs, which come in PR #7.
- A tile whose covered bytes exceed `direct_tile_max_mib` is refused with `RESOURCE_LIMIT` before any read. It is never sampled or substituted.
- No UI changes.

## What was built

| Module | Contents |
|---|---|
| `weightlab/analysis/tile.py` | `tile_geometry`; `BlockAggregates` (record format v1 fields: finite/NaN/+Inf/−Inf counts, overflow flags, Σx, Σ\|x\|, Σx², min, max; merge by addition and min/max); `display` (`mean`, `mean_abs`, `rms`, `min`, `max` → float32 + status); `TileBudget` (`from_settings`); `compute_tile` |
| `weightlab/analysis/tile_codec.py` | Binary encode/decode (`rows*cols` float32 LE then `rows*cols` uint8), status and metadata validation, `TileMetadata` builder, response headers, `TileJson` to/from (≤ 128 × 128: each dimension ≤ 128 and ≤ 16,384 cells; statuses must be JSON integers 0–31, values numbers or `null`), canonical identity and SHA-256 `cache_key` |
| `weightlab/formats/checkpoint.py` | `roi_read_bytes` (plan a read's stripe bytes without reading), `fingerprints` (quick fingerprint of the files holding given keys, plus the shard index for sharded models) |

### Semantics

- **Geometry.**
  - Tile `(x, y)` at level `L` with `size ≤ 256` covers raw view rows `[y·size·2^L, (y+1)·size·2^L)` and columns `[x·size·2^L, …)`, clipped to the view. Blocks are anchored at the origin, so edge blocks hold only their real cells.
  - Views use the existing `resolve_view`: axes, fixed indices, transpose; rank 0 is 1 × 1 and rank 1 is one row.
  - Errors: a tile starting outside the view is `INVALID_ROI`; coordinates above 2^53−1 are `INVALID_ROI`; a level outside 0..52, a size outside 1..256, a non-integer or a bool is `VALIDATION_ERROR`. All are rejected before any read.
- **Aggregation over finite values only** (doc 05 §6):
  - **Formulas:** mean = Σx/n, mean_abs = Σ|x|/n, rms = √(Σx²/n), min/max over finite elements. Partial records merge by addition, so means are count-weighted and RMS comes from sums of squares. Per-block RMS values are never averaged.
  - **Accumulators:** sums start from −0.0, so a lone −0.0 keeps its sign.
  - **Bounded at every level.** Only the real cells of each block are reduced. When whole-block padding stays within 2× the read rectangle, the rectangle is padded with each reduction's identity and reduced over a `(rows, b, cols, b)` reshape. Otherwise each axis is reduced over the rectangle's own block boundaries (`ufunc.reduceat`). No allocation depends on the block size, so an 8 × 8 tensor at level 52 (blocks of 2^52 × 2^52) costs what the 64 cells cost.
  - **Counts:** NaN, +Inf and −Inf counts give the status bits; a block with no finite element is `NO_FINITE` (1) plus its kinds.
- **Never a fabricated drawable value:**
  - A float64 sum that becomes non-finite although every input was finite sets `SUM_OVERFLOW`, `SUM_ABS_OVERFLOW` or `SUM_SQ_OVERFLOW`, and the flags propagate when records merge.
  - An aggregate that depends on an overflowed field, or whose value exceeds float32 range (\|v\| > `FLT_MAX`, compared in float64 as the fixture generator does), is `NOT_REPRESENTABLE` (16) with a NaN placeholder.
  - The decoder rejects a drawable cell without a finite value and a non-drawable cell with one.
- **Labels.** `quality = exact` is independent of `value_precision = float32_visualization`. The tile's `numeric_precision` describes the float64 aggregate before display:
  - `lossless`: min/max, and L0 mean/mean_abs, which are source values;
  - `rounded`: mean, mean_abs or rms at L ≥ 1; or I64/U64 values beyond ±(2^53−1), which also add a warning;
  - `overflowed`: any cell withheld.
- **Reads.**
  - Only positioned reads of rectangles of at most `band_cells` (default 2^18). Each rectangle runs along the view axis that is contiguous in storage, so transposed and sliced views do not multiply I/O.
  - All reads are planned (`roi_read_bytes`), and the total is checked against the read budget before the first byte.
  - The existing `read_roi_raw` checks the file after every read batch and at the end; `compute_tile` re-checks after the last read. A change at any point is `FILE_CHANGED`, and no partial tile is returned.
- **Budgets** (`TileBudget.from_settings`), all checked before reading:

  | Budget | Default | Source |
  |---|---|---|
  | output cells | ≤ `max_tile_cells` (65,536) | setting |
  | covered source bytes | ≤ `direct_tile_max_mib` (64 MiB) | setting; above this the tile needs the exact pyramid → `RESOURCE_LIMIT` |
  | bytes read, including stripe overhead | ≤ 4 × direct limit | derived |
  | working-set estimate | ≤ `memory_budget_mib` / `max_active_tiles` | settings |

  Levels 0–3 always fit for every dtype; an L3 tile covers at most 2048 × 2048 × 8 B = 32 MiB. Higher levels are computed directly when they fit, for example L4 of a 4096 × 4096 F32 tensor, which is exactly 64 MiB.
- **Payload.**
  - The binary body is exactly `rows*cols*5` bytes.
  - Validation rules:
    - reserved bits 32/64/128 must be 0;
    - `NO_FINITE` needs at least one kind bit and excludes `NOT_REPRESENTABLE`;
    - at L0, a status is one of 0, 3, 5, 9 or 16.
  - Metadata checks: rows and columns match the body and the level's ranges; ranges are block-anchored; finite plus non-finite counts equal the covered elements; an exact tile carries no sampling fields.
  - JSON values are `null` exactly for non-drawable cells and must be float32 values. Headers follow doc 09 §3; `Sampled-Rows` appears only for row-sampled tiles.
- **Identity.** The cache key is the SHA-256 of canonical sorted-key JSON. It covers:
  - the source kind;
  - each input file's quick fingerprint (root-relative path, size, mtime_ns, file ID; the index file's SHA-256 for sharded models);
  - tensor key(s) and the view;
  - operation parameters;
  - `numerics_version`, `pyramid_record_format = 1` and the tile format;
  - level, x, y, size, aggregation, quality mode and seed.

  Opaque database IDs are excluded. Persistent cache management is not implemented.

## Numerical correctness evidence

**Independent reference.** All expected values come from `tests/reference_tiles.py`, which shares no code with `weightlab`:
- it decodes the file bytes itself (header JSON plus `np.frombuffer`; BF16 by bit shift);
- it builds views with plain NumPy indexing;
- it aggregates every block on its own with Python `Fraction` sums (rounded once) and the generator's status rule;
- it also uses the golden values in `examples/fixtures/expected_results.json`.

Float32 display values must match bit for bit at L0 and for min/max. Means and RMS may differ by at most 1 float32 ulp, because float64 summation order differs from the exactly rounded reference. Statuses and counts must match exactly. The property test applies that rule wherever the rigorous float64 recursive-summation bound (`sum_error_bound`, γ(n+1)·Σ\|x\|/n, Higham eq. 4.4) is below 2^-25 of the exact value. In an ill-conditioned block it requires the result to be within that bound, and only the `NOT_REPRESENTABLE` bit may differ (see Findings).

| Requirement | Test(s) | Result |
|---|---|---|
| L0 exact values and coordinates; tile (1,1) of 257 × 263 is 1 × 7 | `test_l0_tiles_are_the_exact_source_cells` | bitwise equal, all four L0 tiles |
| L1–L3 parity, every aggregation | `test_l1_to_l3_match_independent_block_aggregation` (15 cases) | pass |
| Count-weighted partial edges (golden L1/L3 first and last block means) | `test_partial_edge_blocks_are_count_weighted_golden`, `test_edge_tiles_of_the_odd_shape` | pass; the L3 corner is the mean of its 7 real cells |
| RMS from sums of squares, not averaged RMS | `test_rms_comes_from_sums_of_squares_not_averaged_rms` | pass |
| Chunking invariance (37, 263, 1,000 and 262,144-cell bands) | `test_chunking_does_not_change_the_tile` | pass |
| Non-finite golden masks (L0 cells 3/5/9; L4 blocks `[[0,6],[15,3]]`), all-non-finite blocks, L0–L5 × 5 aggregations | `test_nonfinite_golden_status_masks` | pass |
| Counts beyond uint16 (L8 65,536 and L9 131,072 golden) | `test_large_blocks_count_beyond_uint16` | pass |
| F64 extremes, overflow flags, NaN placeholders, −0.0 sign | `test_f64_extremes_…`, `test_overflow_flags_…`, `test_an_overflowed_intermediate_is_withheld_…` | pass; withheld, never Inf |
| I64/U64 labelled `rounded` with a warning; I32 stays `lossless` | `test_64_bit_integers_are_labelled_rounded` | pass |
| BF16/F16 decoding, incl. specials and −0.0 | `test_half_precision_decoding` (4) | pass |
| Rank-3 slices (5 views) and transpose; golden cube slice means 15.5 and 47.5 | `test_rank3_slices_and_transpose`, `test_transposed_view_of_the_odd_tensor` | pass |
| Rank 0, rank 1, BOOL | `test_rank0_rank1_and_bool` | pass |
| Random F32/F64 tensors with NaN/±Inf/−0/1e300/5e-324/3.5e38, random rank, view, level, size, band and aggregation; tiles of a level partition the view | `tests/property/test_tile_property.py` (150 examples, levels 0–8, plus a pinned cancellation case; 3,000-example stress run also passed) | pass |
| Binary layout, byte order (`1.0f` = `00 00 80 3f`), exact length | `test_binary_layout_and_byte_order`, `test_body_length_must_be_exact` | pass |
| JSON parity (golden L4 and L0 tiles), validated against the frozen `TileJson` and `TileMetadata` schemas | `test_golden_tile_round_trips_…`, `test_level0_json_parity_…` | pass |
| Reserved bits, status rules, metadata consistency | `test_invalid_cells_are_rejected` (10), `test_metadata_consistency_is_checked` | pass |
| Source change during reads and after the last read → `FILE_CHANGED` | `test_source_change_during_reads_…`, `test_change_after_the_last_read_…` | pass |
| Oversized coordinates, invalid levels, sizes and aggregations; invalid views; all budgets — rejected before any read | `test_invalid_geometry_is_rejected` (12), `test_invalid_views_are_rejected`, `test_budgets_are_enforced_before_reading`, `test_direct_budget_refuses_tiles_that_need_a_pyramid` | pass; `bytes_read` unchanged |
| Tiny tensors (8 × 8, 3 × 300, 300 × 3) at levels 16, 32 and 52, every aggregation: exact values and a traced allocation peak below 1 MiB | `test_tiny_tensors_at_huge_levels_use_memory_of_the_tensor_not_the_block` (9) | pass; on 1e816f08 levels 32/52 fail (`array is too big`) |
| Blocks larger than the read rectangles (levels 5, 6, 7, 20; bands of 13/64/1000 cells; every phase) | `test_large_blocks_over_partial_rectangles_match_the_reference` | pass; on 1e816f08 level 20 drives the process out of memory (killed) |
| JSON 128 × 128 valid against `TileJson`; 1 × 256, 256 × 64, 129 × 1, 1 × 129, 64 × 256 refused by `to_json` (`NOT_ACCEPTABLE`) and `from_json` (`VALIDATION_ERROR`), and shown invalid by the schema | `test_json_128x128_…`, `test_json_rejects_any_dimension_above_128` (5) | pass; all 5 fail on 1e816f08 |
| JSON statuses `"0"`, 0.7, 0.0, true, false, null, −1, 32, 255, `[0]`, `{}` and values `"1.0"`, true, `[1.0]` rejected | `test_json_status_must_be_an_integer_within_0_31` (11), `test_json_values_must_be_numbers_or_null` (3) | pass; 11 of these fail on 1e816f08 (−1, 32 and `"1.0"` were already rejected) |
| Cache keys change with every identity input; key order irrelevant; a fingerprint change invalidates | `test_cache_keys_…`, `test_fingerprint_change_…`, `test_sharded_fingerprints_include_the_index` | pass |

**Findings while testing** (no change in behaviour was needed):
- For `[[big, big], [−big, −big]]`, whether float64 overflows depends on NumPy's summation order. The engine either returns the exact 0 or withholds the cell, never a wrong number; the tests accept exactly these two outcomes. A deterministic merge order that must overflow is tested on the aggregate records.
- **Float64 cancellation (found by the wider property test).** Record format v1 keeps one float64 Σx per block. In a block that mixes magnitudes more than 2^53 apart, the small terms can vanish: `[3.4e38, 1e300, −0.0, 3.5e38, −0.0, −1e300]` sums to 0 in float64, but the true mean is 1.15e38. The engine then shows 0.0, labelled `numeric_precision = rounded`. This error is inherent to float64 accumulation, and the old property test (levels 0–4) could hit it as well. It is pinned as an `@example` and listed under Known limitations.
- min/max of {+0.0, −0.0} may be either zero (IEEE 754 leaves it open). Above L0, the tests compare those values numerically.

## Memory and I/O (measured)

`scripts/tile_benchmark.py` measures a fresh process on a sparse 4 GiB SafeTensors file holding one 32768 × 32768 F32 tensor, with a known ramp in its last 8 × 8 cells. Results are in `tile_benchmark_macos.json` (macOS 26 arm64; `macos_phys_footprint_peak`). The machine was under unrelated load (VMs), so timings are indicative only.

| Tile | Time | Covered | Read | Peak after (process) |
|---|---|---|---|---|
| — after open | — | — | — | 18.9 MiB |
| L0 origin | 11 ms | 0.25 MiB | 0.25 MiB | 31.0 MiB |
| L0 far corner (value 64 found) | 10 ms | 0.25 MiB | 0.25 MiB | 31.4 MiB |
| L1 | 31 ms | 1 MiB | 1 MiB | 36.1 MiB |
| L2 | 88 ms | 4 MiB | 4 MiB | 38.6 MiB |
| L3 | 314 ms | 16 MiB | 16 MiB | 46.7 MiB |
| L3 far corner (max 64 found) | 267 ms | 16 MiB | 16 MiB | 46.7 MiB |
| L3 transposed | 593 ms | 16 MiB | 16 MiB | 46.8 MiB |
| L4 (64 MiB, the direct limit) | 766 ms | 64 MiB | 64 MiB | 47.0 MiB |
| L5 | refused: `RESOURCE_LIMIT … needs the exact pyramid` | 256 MiB | 0 | — |

Measured after the review fixes. The peak grows by **28 MiB** above the opened state for a 4 GiB file, in line with the engine's working-set estimate (~35 MiB). Reads equal covered bytes, including the transposed view. `tests/perf/test_tile_bounded.py` asserts:
- growth below 128 MiB;
- `bytes_read == covered_bytes`;
- the far-corner values;
- the L5 refusal.

Speed. Replacing two-stage `reduceat` with a padded `(rows, b, cols, b)` reshape reduction roughly halved L4 time (1.8 s → 1.0 s). Read rectangles narrower than the tensor whose row gaps are below `coalesce_gap_bytes` (64 KiB) read whole stripes; the read budget bounds this, and a test covers it.

## Commands run (macOS 26, arm64, Python 3.12)

| Command | Result |
|---|---|
| `uv run ruff format --check . && uv run ruff check .` | clean |
| `uv run mypy --platform {linux,darwin,win32}` | no issues (44 files each) |
| `uv run pytest -q` | **554 passed** (444 before PR #6; 110 new: engine 62, codec 46, property 1, perf 1) |
| `uv run python scripts/phase1_evidence.py --out evidence/phase1` | PASS, 554 passed, 0 failed, 0 skipped |
| `uv run python scripts/tile_benchmark.py --size-gib 4` | table above |
| `check_spec`, `check_contracts`, `check_demo_fixtures`, `manifest --check` | see the PR description |

Linux and Windows were not run locally; the CI matrix covers them, including the sparse-file perf test (marked sparse with `fsutil` on NTFS).

## Known limitations

- **No HTTP route yet.** The internal interface is `compute_tile` plus `tile_codec`. `/tiles` needs the decision table: pyramid, row-sampled preview, 202 + deduplicated job (PR #7).
- **Bounded by the direct budget.** Tiles above `direct_tile_max_mib` are refused. Above L3 that is common for large tensors; this is by design until the pyramid exists.
- **Derived sources.** Only `kind = tensor` sources are computed. The identity function already takes `source_kind`, extra files and `operation` parameters, but delta and LoRA tiles are not implemented.
- **Caching.** There is no persistent or in-memory tile cache, only the key. Reads are single-threaded.
- **Counters.** Counts are int64 per output cell. The uint16 base / uint64 coarse on-disk record layout belongs to the pyramid store, which is not built yet.
- **Overflow is order-dependent.** As documented above, a mathematically representable mean can be withheld when float64 overflows on the way. It is never shown as Inf.
- **Float64 cancellation.** mean is computed from one float64 Σx (record format v1). When a block holds values more than 2^53 apart in magnitude that cancel (for example ±1e300 next to 1e38), the drawable mean can be far from the true mean. Such tiles are labelled `rounded`, and exact values come from `/tensors/roi`. Withholding these cells needs an error-bound rule (for example Σ\|x\| against \|Σx\|). That rule would change the status semantics that the frozen fixture generator defines, so it is left as an open spec decision.
- **Below the float32 range.** Float32 display of values below its range rounds toward 0 (for example 5e-324 → 0); such cells stay drawable, with `value_precision = float32_visualization` stating the loss.
