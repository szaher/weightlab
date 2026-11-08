# Phase 1 foundation evidence (Tasks 1-2): macos (PASS on this OS only)

- Date (UTC): 2026-11-08 00:08
- Platform: macOS-26.0-arm64-arm-64bit / arm64 / Python 3.12.11
- Command: `uv run python scripts/phase1_evidence.py` -> full `pytest` suite (exit 0)
- Totals (whole suite): 674 passed, 0 failed, 0 skipped
- Database schema version: 1; contract: contracts/openapi.yaml 0.3.0-spec-1.2-final

## Results by area

| Area | Tests | Passed | Failed | Skipped |
|---|---|---|---|---|
| Task 1: migrations: fresh/idempotent, newer or edited DB refused unmodified, backup before upgrade, atomic rollback, integrity (`tests/db/test_migrations.py`) | 17 | 17 | 0 | 0 |
| Task 1: repositories: FK/CHECK/UNIQUE constraints, project cascade, 60k batched catalog insert, filters/sort/paging, cross-project look-ups (`tests/db/test_store.py`) | 9 | 9 | 0 | 0 |
| Task 2: request models vs frozen JSON Schema: contract examples + generated differential mutations (`tests/api/test_contract_schemas.py`) | 34 | 34 | 0 | 0 |
| Task 2: route table == contract (implemented + deferred), deferred 404, docs disabled (`tests/api/test_routes.py`) | 4 | 4 | 0 | 0 |
| Task 2: token, Host allowlist, Origin, no CORS, headers, body/concurrency limits, strict bodies, loopback only (`tests/api/test_security.py`) | 58 | 58 | 0 | 0 |
| Task 2: `weightlab start` real process: loopback socket, fresh token per launch, token never in logs (`tests/api/test_server_process.py`) | 5 | 5 | 0 | 0 |
| Task 2: projects, import in place, catalog, relink, stale fingerprints, import errors, no path disclosure (`tests/api/test_models_api.py`) | 21 | 21 | 0 | 0 |
| Task 2: ROI for all four source kinds vs generator values, comparisons, adapters, overflow, lineage gating (`tests/api/test_roi_api.py`) | 13 | 13 | 0 | 0 |
| Task 2: bookmarks with ownership, settings round trip, bounds and persistence (`tests/api/test_bookmarks_settings_api.py`) | 3 | 3 | 0 | 0 |
| PR #3 review: adapter config and shard index changes -> FILE_CHANGED, atomic relink, admission before body, cheap health (`tests/api/test_review_pr3.py`) | 9 | 9 | 0 | 0 |
| Task 3: UI shell served token-free from memory; API auth, Host/Origin intact; traversal and unknown paths 404 (`tests/api/test_ui_static.py`) | 32 | 32 | 0 | 0 |
| Task 4A: the explorer's ROI requests (rank 0/1/2/3, transpose, ragged edges, dtypes, 64-bit ints, -0.0, budget) vs NumPy decode of fixture bytes; edited file -> FILE_CHANGED (`tests/api/test_roi_explorer.py`) | 11 | 11 | 0 | 0 |
| Task 4B foundation: L0-L3+ direct tiles vs independent reference and goldens: aggregations, edges, non-finite, F64 extremes, I64/U64, BF16/F16, slices, transpose, FILE_CHANGED, budgets (`tests/integration/test_tile_engine.py`) | 62 | 62 | 0 | 0 |
| Task 4B foundation: binary layout/byte order, JSON parity vs contract schemas, status and metadata validation, headers, canonical identity and cache keys (`tests/unit/test_tile_codec.py`) | 46 | 46 | 0 | 0 |
| Task 4B foundation: random tensors/views/levels/bands vs the reference; tiles partition the view (`tests/property/test_tile_property.py`) | 1 | 1 | 0 | 0 |
| Task 4B foundation: 4 GiB sparse checkpoint, L0-L4 tiles: flat peak memory, reads = covered bytes, L5 refused (`tests/perf/test_tile_bounded.py`) | 1 | 1 | 0 | 0 |
| Task 4B pyramids: L4+ records vs reference (257x263, transpose, rank-3, non-finite, >uint16, BF16/F16, I64/U64, F64 overflow), merges, chunk invariance, corruption, FILE_CHANGED (`tests/integration/test_pyramid.py`) | 21 | 21 | 0 | 0 |
| Task 4B sampling: stride/SplitMix64 rows vs independent reference, pinned goldens, seeds, labels, budgets (`tests/unit/test_sampling.py`) | 39 | 39 | 0 | 0 |
| Task 4B caches: memory LRU, atomic disk publication, crash leftovers, cap/eviction, session store, corruption, purge (`tests/unit/test_cache.py`) | 10 | 10 | 0 | 0 |
| Task 4B jobs: state machine, dedup, cancel queued/running, timeout, memory, queue cap, restart, shutdown (`tests/integration/test_jobs.py`) | 11 | 11 | 0 | 0 |
| Task 4B API: every decision-table branch, binary/JSON vs schemas, 202 dedup, polling, cancel, isolation, changes, corruption, delete, restart, offline (`tests/api/test_tiles_api.py`) | 25 | 25 | 0 | 0 |
| Task 4B perf (CI scale): bounded memory, exact vs NumPy, sampled rows, cancel latency, LRU under the disk cap (`tests/perf/test_pyramid_bounded.py`) | 1 | 1 | 0 | 0 |
| Reader hardening: FIFOs/devices/directories refused without blocking (`tests/security/test_special_files.py`) | 3 | 3 | 0 | 0 |
| Workstream A: evaluation kit examples reproduce against generator values, checks can fail, docs consistent (`tests/integration/test_evaluation_kit.py`) | 5 | 5 | 0 | 0 |
| Phase 0 / 0.5 suites (regression) | 233 | 233 | 0 | 0 |

## Routes (contract 0.3.0)

| Method | Path | State |
|---|---|---|
| POST | `/comparisons` | implemented |
| GET | `/comparisons/{comparison_id}` | implemented |
| GET | `/comparisons/{comparison_id}/entries` | implemented |
| POST | `/dialogs/select-directory` | deferred: optional capability native_directory_dialog, not advertised |
| POST | `/exports` | deferred: export jobs (a later milestone) |
| GET | `/exports/{export_id}` | deferred: export jobs (a later milestone) |
| GET | `/exports/{export_id}/download` | deferred: export jobs (a later milestone) |
| GET | `/health` | implemented |
| POST | `/jobs` | implemented |
| GET | `/jobs/{job_id}` | implemented |
| POST | `/jobs/{job_id}/cancel` | implemented |
| POST | `/models/{model_id}/relink` | implemented |
| GET | `/models/{model_id}/tensors` | implemented |
| GET | `/models/{model_id}/tensors/{tensor_id}` | implemented |
| GET | `/projects` | implemented |
| POST | `/projects` | implemented |
| DELETE | `/projects/{project_id}` | implemented |
| GET | `/projects/{project_id}` | implemented |
| GET | `/projects/{project_id}/adapters` | implemented |
| POST | `/projects/{project_id}/adapters` | implemented |
| GET | `/projects/{project_id}/bookmarks` | implemented |
| POST | `/projects/{project_id}/bookmarks` | implemented |
| DELETE | `/projects/{project_id}/bookmarks/{bookmark_id}` | implemented |
| GET | `/projects/{project_id}/models` | implemented |
| POST | `/projects/{project_id}/models` | implemented |
| GET | `/settings` | implemented |
| PATCH | `/settings` | implemented |
| POST | `/tensors/roi` | implemented |
| POST | `/tiles` | implemented |
| GET | `/tiles/{cache_key}/metadata` | implemented |

## Contract gaps exercised by the tests

Statuses required by doc 09's behaviour rules but not declared for the route in the frozen contract (bodies still validate against `Error`):

- GET `/comparisons/{comparison_id}/entries` -> 422
- POST `/models/{model_id}/relink` -> 403
- POST `/models/{model_id}/relink` -> 422
- GET `/projects/{project_id}/adapters` -> 404
- POST `/projects/{project_id}/adapters` -> 403
- POST `/projects/{project_id}/adapters` -> 413
- GET `/projects/{project_id}/bookmarks` -> 404
- POST `/projects/{project_id}/bookmarks` -> 404
- DELETE `/projects/{project_id}/bookmarks/{bookmark_id}` -> 404
- GET `/projects/{project_id}/models` -> 404

## Skipped on this OS

- none
