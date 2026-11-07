# Phase 1 foundation evidence (Tasks 1-2): macos (PASS on this OS only)

- Date (UTC): 2026-11-07 02:42
- Platform: macOS-26.0-arm64-arm-64bit / arm64 / Python 3.12.11
- Command: `uv run python scripts/phase1_evidence.py` -> full `pytest` suite (exit 0)
- Totals (whole suite): 398 passed, 0 failed, 0 skipped
- Database schema version: 1; contract: contracts/openapi.yaml 0.3.0-spec-1.2-final

## Results by area

| Area | Tests | Passed | Failed | Skipped |
|---|---|---|---|---|
| Task 1: migrations: fresh/idempotent, newer or edited DB refused unmodified, backup before upgrade, atomic rollback, integrity (`tests/db/test_migrations.py`) | 14 | 14 | 0 | 0 |
| Task 1: repositories: FK/CHECK/UNIQUE constraints, project cascade, 60k batched catalog insert, filters/sort/paging, cross-project look-ups (`tests/db/test_store.py`) | 9 | 9 | 0 | 0 |
| Task 2: request models vs frozen JSON Schema: contract examples + generated differential mutations (`tests/api/test_contract_schemas.py`) | 26 | 26 | 0 | 0 |
| Task 2: route table == contract (implemented + deferred), deferred 404, docs disabled (`tests/api/test_routes.py`) | 4 | 4 | 0 | 0 |
| Task 2: token, Host allowlist, Origin, no CORS, headers, body/concurrency limits, strict bodies, loopback only (`tests/api/test_security.py`) | 53 | 53 | 0 | 0 |
| Task 2: `weightlab start` real process: loopback socket, fresh token per launch, token never in logs (`tests/api/test_server_process.py`) | 5 | 5 | 0 | 0 |
| Task 2: projects, import in place, catalog, relink, stale fingerprints, import errors, no path disclosure (`tests/api/test_models_api.py`) | 21 | 21 | 0 | 0 |
| Task 2: ROI for all four source kinds vs generator values, comparisons, adapters, overflow, lineage gating (`tests/api/test_roi_api.py`) | 13 | 13 | 0 | 0 |
| Task 2: bookmarks with ownership, settings round trip, bounds and persistence (`tests/api/test_bookmarks_settings_api.py`) | 3 | 3 | 0 | 0 |
| PR #3 review: adapter config and shard index changes -> FILE_CHANGED, atomic relink, admission before body, cheap health (`tests/api/test_review_pr3.py`) | 9 | 9 | 0 | 0 |
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
| POST | `/exports` | deferred: needs the job runner (Tasks 5-6) |
| GET | `/exports/{export_id}` | deferred: needs the job runner (Tasks 5-6) |
| GET | `/exports/{export_id}/download` | deferred: needs the job runner (Tasks 5-6) |
| GET | `/health` | implemented |
| POST | `/jobs` | deferred: Phase 1 Tasks 5-6 (stats jobs, cancellation, memory limiter) |
| GET | `/jobs/{job_id}` | deferred: Phase 1 Tasks 5-6 |
| POST | `/jobs/{job_id}/cancel` | deferred: Phase 1 Tasks 5-6 |
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
| POST | `/tiles` | deferred: Phase 1 Task 4 (heatmap tiles) |
| GET | `/tiles/{cache_key}/metadata` | deferred: Phase 1 Task 4 (heatmap tiles) |

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
