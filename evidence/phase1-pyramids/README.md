# Phase 1 Task 4B: exact pyramids, row sampling, jobs, caches and the tile API (evidence)

Branch `phase-1-pyramids-jobs-cache`, based on `main` at aaa2eb2 (merged PR #6). Spec 1.2-final is frozen (ADR-019). No file under `docs/*.md`, `contracts/`, `examples/fixtures/` or the spec scripts was edited. No frontend change.

## What was built

| Module | Contents |
|---|---|
| `weightlab/analysis/pyramid.py` | Streaming exact pyramid: L4 base records (56 B, uint16 counters) and coarser levels merged from children (80 B, uint64 counters), frozen record format v1; storage-order streaming (transposed views stored transposed); chunk-invariant base reduction; `.wlpyr` file with per-run CRC32 table, checksummed trailer and footer; validating loader and tile reader |
| `weightlab/analysis/sampling.py` | Row-sampled previews: `stride` and seeded `splitmix64_rows` selection (Floyd's algorithm on the generator's SplitMix64, pure Python integers), budget planning before any read, labelled tiles |
| `weightlab/jobs.py` | Bounded job runner on the existing `jobs` table: state machine, dedup, memory admission, queue cap, cooperative cancellation, wall-time budget, restart handling, joined shutdown |
| `weightlab/cache.py` | Memory tile LRU (`tile_cache_memory_mib`), disk pyramid LRU (`tile_cache_disk_mib`) with reservations and atomic publication, bounded session store, validation, purge by project |
| `weightlab/api/tiles.py`, `api/app.py`, `api/schemas.py` | `POST /tiles`, `GET /tiles/{cache_key}/metadata`, `POST /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel`; strict `TileRequest`, `JobRequest`, `Budget` models (checked against the frozen schema by the generated differential tests) |
| `weightlab/formats/checkpoint.py` | `read_elements` (one positioned read of a contiguous element run), `input_digests` / `input_bytes` (cancellable hashing for `verified` jobs) |
| `weightlab/analysis/tile.py`, `tile_codec.py` | Shared `finish_tile`; `pyramid_identity` (tile identity = pyramid identity + tile fields); row-sampled metadata validation and sampling fields |

### Quality decision table (doc 09 §3), as implemented

| Condition, in this order | `auto` | `exact` | `row_sampled` |
|---|---|---|---|
| covered bytes <= `direct_tile_max_mib`, or level 0-3 | 200 exact, direct read | same | 422 `VALIDATION_ERROR` (not needed) |
| pyramid ready (validated) | 200 exact from the pyramid | same | 422 |
| pyramid not ready, tensor on-disk size <= `pyramid_exact_max_mib` | 202 + deduplicated `pyramid_build` job | 202 + job | 422 |
| pyramid not ready, tensor above the limit | 200 `row_sampled`, labelled | 202 + job | 200 `row_sampled` |

- **Nothing is silently downgraded.** An exact request is never answered with a sampled tile, and no tile is labelled exact unless every covered element was aggregated. A sample whose selection would cover every row is computed as an exact tile and labelled exact.
- **Levels 0–3 are always direct reads.** Doc 03 says "Levels 0-3 always qualify for every supported dtype", and the pyramid starts at L4. This holds even if `direct_tile_max_mib` is set below the 32 MiB that an F64 L3 tile can cover.
- **Cold requests.** A cold request returns 202 with the `Job`. Polling until `completed`, then repeating the request, gives 200 exact. After a cancel, a waiter that repeats the request gets 202 again, never an exact tile.

### Pyramid semantics

- **Records.** Each record holds finite / NaN / +Inf / −Inf counts, overflow flags, Σx, Σ|x|, Σx², min and max.
  - Means are count-weighted; RMS comes from Σx²; flags propagate to parents.
  - A record without finite elements stores −0.0 sums, +Inf / −Inf extrema and no flags.
  - Counter additions are checked: an overflow raises `RESOURCE_LIMIT` instead of wrapping.
  - Base counters that do not fit uint16 are refused.
- **Streaming.**
  - Reads go in storage order, in bands of at most 2^19 cells: plain positioned reads for contiguous views (rank 0–2, contiguous rank-3 slices), bounded `read_roi_raw` rectangles otherwise.
  - Memory held: one band, its records, and one pending row of records per level.
  - The source is checked after every read and after the last one, so a change becomes `FILE_CHANGED`.
  - The control hook after each chunk enforces cancellation and the wall-time budget.
- **Determinism.** Each 16×16 block is reduced as one contiguous 256-value run, so the pyramid bytes are identical whatever the band or column-chunk size (`test_chunking_does_not_change_a_single_byte`).
- **Validation on load and on every tile read.**
  - Footer magic, offsets and the trailer SHA-256.
  - Every trailer field: format, versions, `numerics_version`, `pyramid_record_format`, the pyramid id equal to SHA-256 of the identity, the identity equal to the requested one, and the layout recomputed from the dimensions.
  - The CRC table's SHA-256.
  - Per tile read: the CRC32 of each 256-record run touched, and each record (counts equal to the block's real cell count, flags and padding, finite-less records, flag ⇔ non-finite sum, extrema).
  - A corrupt entry is deleted and rebuilt, never served.

### Sampling semantics

- **Eligibility.** Only sources above `pyramid_exact_max_mib`. Whole view rows across the tile's column range are read, in ascending order.
- **Rows per block.** k = `sampled_rows_per_block`, or by default the largest of 8, 4, 2, 1 that fits the sampling budget.
- **Budget.** Bytes read must stay within `direct_tile_max_mib`; this is checked before any read, otherwise `RESOURCE_LIMIT`.
- **Policy.** `stride` when no seed is given: rows floor((2j+1)·n/2k) of each block's n real rows.
  With a seed, `splitmix64_rows`:
  - Floyd's algorithm picks k distinct rows, with unbiased bounded draws from `splitmix64(seed_mix ^ ((block_row << 20) | draw))`.
  - `seed_mix` is the first 8 bytes (LE) of SHA-256 over {pyramid_id, level, x, y, seed}.
  - The pyramid id covers the source kind, file fingerprints, tensor key, view and numerics version.
  - Pure Python integers make the selection identical on every OS and NumPy version; golden selections are pinned in the tests.
- **Labels.** `quality = row_sampled`, `sampled_rows`, `sampling_policy`, `seed`, and finite and non-finite counts of the elements actually read (always fewer than covered; the codec enforces it). `numeric_precision` is `rounded` or `overflowed`, never lossless. A warning states the rows read and that the result is exploratory, not exact.
- **Non-finite values.** They are counted and reported in the status bits for the rows read.
- **Cache isolation.** Sampled tiles live only in the memory tile cache, under their own identity (quality, seed, rows per block, policy). They never feed a pyramid or a statistic.

### Jobs

- **States.** `created → queued → running → completed | failed | canceled`. Queued jobs can be canceled directly.
- **Shutdown and restart.** A running job becomes `interrupted` at shutdown, or at the next start if the process died. Created and queued jobs are queued again on start.
- **Results and errors.**
  - `result` is written only for `completed`.
  - `failed` carries the doc 09 error object (no tracebacks; internal errors become a generic message).
  - Canceled and interrupted jobs carry neither a result nor a partial one.
- **Dedup and admission.**
  - Dedup is by SHA-256 of {project, kind, pyramid identity, consistency}.
  - `max_heavy_jobs` worker threads (default 1).
  - A queued job starts only when its memory reservation fits: running reservations ≤ `memory_budget_mib`.
  - At most 16 pending jobs; more answers 429.
- **Budgets.**
  - Values above the server caps get 422 `RESOURCE_LIMIT`, with the caps in `details`.
  - A build whose estimated working set exceeds `max_working_set_mib` is refused with 422.
  - Wall time is min(`max_wall_seconds`, `max_job_seconds`); exceeding it fails the job with `RESOURCE_LIMIT`.
- **Progress.** Bytes processed over the estimate (scan, plus two full hashes for `verified`); monotonic, and 1.0 only when completed. `bytes_scanned` is reported.
- **`verified`.** Every input file is hashed before and after the build; any difference is `FILE_CHANGED`.
- **Lifecycle.** Workers are started by the app lifespan and joined at shutdown. Deleting a project cancels its jobs first.

### Caches

- **Memory tile LRU.** Encoded tiles, bounded in bytes by `tile_cache_memory_mib`. Tile metadata for `GET /tiles/{cache_key}/metadata` is kept in a separate bounded LRU (8192 entries) for the session.
- **Disk pyramid LRU.** In `<data_dir>/cache/pyramids/`, bounded by `tile_cache_disk_mib`.
  - Space is reserved before a build: LRU eviction until used + reserved + needed ≤ cap, and a 64 MiB free-space margin.
  - Builds write a temporary file, fsync it and publish with `os.replace`. Startup removes temporary leftovers.
  - Lowering the cap through `PATCH /settings` evicts at once.
  - Only files the cache names (`<64 hex>.wlpyr`, `*.wlpyr.tmp-*`, `owners.json`) are ever deleted.
- **Session store.** A pyramid that cannot fit the disk cap goes to the in-memory session store (`memory_budget_mib / 4`, LRU). If it fits neither, it is refused with `RESOURCE_LIMIT`.
- **Invalidation.** Keys hash every input fingerprint, so a changed file cannot hit an old entry. Such entries are unreachable and age out by LRU.
- **Project deletion.** `purge_cache=true` removes the project's pyramids and tiles. Checkpoints are never opened for writing, moved or deleted.

## Numerical correctness evidence

**Independent reference.** All expected values come from `tests/reference_tiles.py` (its own decoder, NumPy views, exact Fraction aggregation per block) and from the generator's goldens in `expected_results.json`.
- **Sampling reference.** It re-implements the selection on the generator's `splitmix64`.
- **Benchmark reference.** The benchmark compares against a float64 NumPy aggregation of the file bytes.

| Requirement | Test(s) | Result |
|---|---|---|
| Frozen record layout (56 / 80 B, field offsets, uint16 / uint64) | `test_record_sizes_are_the_frozen_format` | pass |
| L4+ pyramid tiles vs reference, all five aggregations, all levels incl. above the top, 257×263 odd edges, partial tiles | `test_odd_257x263_every_level_and_aggregation` (plain and transposed) | pass |
| Rank-3 slices and transposes (5 views) | `test_rank3_slices_and_their_transposes` | pass |
| Non-finite golden blocks (L4 `[[0,6],[15,3]]`), all-non-finite cells | `test_nonfinite_golden_blocks_and_all_nonfinite_cells` | pass |
| Counters > 65,535 (L8 = 65,536, L9 = 131,072 golden) | `test_counts_beyond_uint16_at_level8_and_9` | pass |
| Synthetic counters > 2^32; checked addition; base-field overflow | `test_synthetic_counters_beyond_2_to_32_merge_and_round_trip` | pass |
| Parent = exact merge of children (bitwise) | `test_parents_are_exact_merges_of_children` | pass |
| RMS from merged Σx², not averaged child RMS | `test_rms_comes_from_merged_sums_of_squares` | pass |
| F64 1e300 extrema survive; Σx² overflow flagged and propagated; never Inf | `test_f64_extremes_survive_and_overflow_is_flagged` | pass |
| BF16 / F16 sources incl. specials; I64 / U64 labelled `rounded` | `test_half_precision_sources`, `test_64_bit_integers_are_rounded_in_pyramids` (2) | pass |
| Views never share a pyramid; a stored pyramid serves only its identity | `test_views_never_share_a_pyramid` | pass |
| Chunk-size invariance (bytes) | `test_chunking_does_not_change_a_single_byte` | pass |
| Corruption: record byte (CRC), preamble, CRC table, trailer, footer, truncation, append; record-level validation | `test_corruption_anywhere_is_detected`, `test_record_validation_rejects_inconsistent_records` | pass |
| `FILE_CHANGED` during a build and after its last read; prompt stop | `test_file_change_during_…`, `test_file_change_after_the_last_read_…`, `test_a_raising_control_…` | pass |
| Storage estimate 0.323 B/cell | `test_layout_matches_the_storage_estimate`; benchmark: 195,089,210 B for 603,979,776 cells = **0.3230 B/cell** | pass |
| SplitMix64 = generator function and published vector; selection = independent reference (n up to 2^52, k up to 256, 3 seeds, 3 block rows) | `test_splitmix64_…`, `test_selection_matches_the_independent_reference` (28) | pass |
| Pinned golden selections (reproduced on every CI OS) | `test_golden_selections_are_reproduced_on_every_platform` | pass |
| Fixed seeds repeat; different seeds and sources differ; uniformity (χ², p = 0.001) | `test_fixed_seeds_…`, `test_seeded_selection_is_uniform_enough` | pass |
| Sampled tiles = reference aggregation of the selected rows; labels; metadata validation | `test_sampled_tiles_equal_the_reference_aggregation_…` (3) | pass |
| Sampled metadata never claims exact coverage; complete selections are exact; transposed sampling refused; budgets before reads | `test_sampled_metadata_…`, `test_a_selection_of_every_row_…`, `test_transposed_views_…`, `test_budgets_…` | pass |

## Jobs, cache and HTTP evidence

| Requirement | Test(s) | Result |
|---|---|---|
| Lifecycle, typed result, monotonic progress, `bytes_scanned` | `test_lifecycle_to_completed_…` | pass |
| 12 concurrent identical submissions → one job (11 `deduplicated: true`) | `test_concurrent_identical_requests_share_one_job` | pass |
| Cancel before execution (never runs); idempotent | `test_cancel_before_execution_never_runs_the_job` | pass |
| Cancel during an active scan: < 0.5 s, nothing published | `test_cancel_during_an_active_scan_…` | pass |
| Timeout → failed `RESOURCE_LIMIT`; failures without tracebacks; memory above budget refused; queue cap → 429 | `test_timeout_…`, `test_failures_…`, `test_memory_reservations_…`, `test_queue_capacity_…` | pass |
| Restart: running → interrupted, queued work re-run, interrupted never joined | `test_restart_marks_running_jobs_interrupted_…` | pass |
| Shutdown joins every worker; running → interrupted | `test_shutdown_interrupts_running_jobs_…`, `test_app_shutdown_leaves_no_worker_threads` | pass |
| Memory LRU by bytes; atomic publication; aborted builds leave nothing; crash leftovers removed, foreign files kept | `tests/unit/test_cache.py` (10) | pass |
| Disk cap never exceeded; LRU eviction and reuse; lowering the cap evicts at once; disk cap 0 → session store; neither → `RESOURCE_LIMIT` | same | pass |
| Corrupt entries deleted, never served (footer, record CRC); purge by project; sources untouched | same | pass |
| Every decision-table branch over HTTP; binary body `rows*cols*5`, headers, status bits; JSON = binary, validated against `TileJson` / `TileMetadata` / `Job`; 406 above 128 per dimension and for unsupported `Accept` | `tests/api/test_tiles_api.py` (25) | pass |
| 202 dedup (8 concurrent HTTP requests → 1 job), polling, retry → 200 exact, idempotent cancel, waiters back to 202 | same | pass |
| Sampled 200 with labels, pinned values (row-index pattern), seed reproducibility | same | pass |
| Invalid requests (9 cases), derived sources and unimplemented job kinds → 422; budgets above caps → 422 with caps; unknown job → 404 | same | pass |
| Cross-project IDs → 404 (tiles and jobs) | `test_cross_project_ids_are_not_found` | pass |
| Changed source after a pyramid → 409 `FILE_CHANGED`, never the old tile; corrupt cache → rebuilt | `test_a_changed_source_…`, `test_a_corrupt_cached_pyramid_…` | pass |
| Project deletion purges its cache, deletes its jobs, leaves the checkpoint byte-identical (SHA-256 and mtime) | `test_project_deletion_…` | pass |
| Restart: disk pyramids reused (200 exact); interrupted jobs never claim exact (new 202) | `test_pyramids_survive_a_restart_…` | pass |
| Offline: `socket.connect` / `create_connection` / `getaddrinfo` patched to fail; tiles and jobs work | `test_tiles_and_jobs_work_offline` | pass |
| Token on every new route (401); route table = contract | `tests/api/test_security.py`, `tests/api/test_routes.py` | pass |

## Performance and resources (measured)

`scripts/pyramid_benchmark.py` runs the whole HTTP flow (FastAPI behind the Guard, in-process `TestClient`, no socket) in a fresh child process. Results are in `pyramid_benchmark_macos.json`.
- **Machine:** macOS 26.0, Apple M3 Pro, 36 GiB RAM.
- **Memory metric:** `macos_phys_footprint_peak`.
- **Defaults:** `direct_tile_max_mib` 64, `pyramid_exact_max_mib` 1024, `memory_budget_mib` 1024.
- **Inputs:** sparse files hide physical I/O cost, and the machine is not an 8 GiB laptop. See the limitations.

| Phase | Result | Process peak after |
|---|---|---|
| Start / models imported | — | 47.4 / 47.7 MiB |
| Cold exact pyramid, 8192 × 8192 BF16 (128 MiB) | 202 then completed in **0.91 s** (0.66 s in an earlier run); bytes scanned = 134,217,728 = tensor bytes | 71.7 MiB |
| Tiles from the pyramid, cold / warm p50 / p95 | L5 rms 23.7 / 5.2 / 9.3 ms; L8 6.2 / 5.4 / 6.3 ms; L13 5.7 / 6.3 / 8.1 ms | — |
| Direct tiles, cold / warm p50 / p95 | L0 33.8 / 5.3 / 6.0 ms; L3 96.5 / 3.1 / 3.7 ms; L4 (32 MiB covered, direct) 281 / 2.5 / 4.6 ms | 121.7 MiB |
| Accuracy vs NumPy float64 reference (7 tiles, 328,705 cells) | **0 ulps** max error on every tile, every aggregation | — |
| Row-sampled preview, 36864 × 16384 BF16 (1.125 GiB sparse), L9 | 200 `row_sampled`, 8 rows/block, `stride`; **0.51 s** cold, 3.5 ms warm; 9,437,184 of 603,979,776 elements read (1.56%); values match the known row pattern | 122.1 MiB |
| Cancel an exact build of the 1.125 GiB tensor at 5.5% | **10.2 ms** from POST cancel to `canceled`; result null; no temporary file left; the tile then answers 202 | 122.2 MiB |
| Disk LRU, cap 40 MiB, four 6144² pyramids (12.2 MiB each) | builds 425–494 ms each; at most 36.6 MB on disk (≤ 41.9 MB cap); reuse of the three most recent from disk (2.9–3.7 ms), oldest two rebuilt (≈ 0.58 s): **3 hits, 2 misses**, as LRU predicts | 123.0 MiB |
| Explicit exact build above the auto limit (1.125 GiB) | **7.8 s**, 1,207,959,552 bytes scanned, 195 MB pyramid, tile 200 exact in 11.6 ms; values match the pattern | 123.2 MiB |
| Cache statistics (whole run) | memory tile hit ratio **81%** (78 / 96); pyramid lookups 46 hits / 22 misses; 4 evictions; 0 corrupt | — |

The process peak stays at **123 MiB** through every phase, including the 1.125 GiB scan (growth 75.5 MiB above the imported state). That is far below the 1 GiB backend budget. The CI-scale run (`--small`, `tests/perf/test_pyramid_bounded.py`) asserts:
- growth below 256 MiB;
- bytes scanned equal to the tensor bytes;
- at most 1 ulp against the reference;
- the sampled pattern;
- cancellation within 2 s, with nothing published and a following 202;
- 3 / 2 LRU hits and misses under the cap;
- the exact large build.

## Commands run (macOS 26, arm64, Python 3.12)

| Command | Result |
|---|---|
| `uv run ruff format --check . && uv run ruff check .` | clean |
| `uv run mypy --platform {linux,darwin,win32}` | no issues (49 files each) |
| `uv run pytest -q` | **674 passed** (554 before; 120 new: pyramid 21, sampling 39, cache 10, jobs 11, tiles API 25, perf 1, plus 13 from the extended contract and security parametrizations) |
| `uv run python scripts/phase1_evidence.py --out evidence/phase1` | see `evidence/phase1/PHASE1_FOUNDATION_macos.md` |
| `uv run python scripts/pyramid_benchmark.py --dir … --json …` | table above |
| `check_spec`, `check_contracts`, `check_demo_fixtures`, `manifest --check` | see the PR description |

Linux and Windows were not run locally; the CI matrix covers them, including both perf tests on sparse files (`fsutil` on NTFS).

## Known limitations

- **Derived sources.**
  - `delta`, `lora_delta` and `virtual_effective` tiles answer 422 with an explanation. Their pyramids need paired passes, a later milestone.
  - Job kinds `tensor_stats`, `compare_stats` and `file_hash` answer 422 "not available in this release". The contract enum lists them, so a client can tell them apart from invalid requests by the message.
- **Transposed views and sampling.** Row sampling reads whole view rows. For a view whose rows are strided in storage (a transposed matrix), it is refused with `RESOURCE_LIMIT` rather than reading scattered cells. Exact pyramids of such views are streamed in storage order and are fine.
- **Tile metadata scope.** `GET /tiles/{cache_key}/metadata` answers for tiles served in this session (bounded LRU). After a restart, a tile request is needed first.
- **Threads.** `max_threads` is checked against the server cap, but builds are single-threaded, so it is never exceeded.
- **Scan confirmation.** `max_scan_read_mib_without_confirmation` is not applied to pyramid builds. The frozen decision table makes builds up to `pyramid_exact_max_mib` automatic; there is no confirmation step in the contract.
- **Memory accounting.**
  - Reservations use the working-set estimate (band, records, read buffer), not measured RSS.
  - Pyramids in the session store count against the job that built them while it runs, then against `memory_budget_mib / 4`.
  - The tile memory cache is separate (`tile_cache_memory_mib`).
- **Cache sharing and invalidation.**
  - The cache is content-addressed. Two projects importing the same files with the same selected root share a pyramid, and deleting either project with `purge_cache` removes it for both; it is recomputed on demand.
  - Stale entries (changed sources) are unreachable but occupy space until LRU eviction.
- **Windows.** An entry still open by a reader cannot be deleted. Such deletions are retried at the next eviction, and the entry no longer counts as cached.
- **Float64 cancellation (PR #6).** Unchanged. Sums are one float64 per record (format v1). A block mixing magnitudes more than 2^53 apart can lose small terms; such tiles are labelled `rounded`.
- **Measurement coverage.**
  - Benchmarks use a 36 GiB M3 Pro and sparse files: page-cache effects and real SSD throughput are not measured.
  - The 8 GiB laptop target is not yet verified on such a machine (doc 11 §8 requires that before claiming it).
  - Timings vary run to run, and the machine had unrelated load.
