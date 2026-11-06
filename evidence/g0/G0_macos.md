# G0 evidence: macos (PASS)

- Date (UTC): 2026-11-06 00:38
- Platform: macOS-26.0-arm64-arm-64bit / arm64 / Python 3.12.11
- Command: `uv run python scripts/g0_evidence.py` -> `python -m pytest -q -rs --junitxml=junit.xml` (exit 0)
- Totals: 131 passed, 0 failed, 0 skipped

## WeightLab test results

| Area | Tests | Passed | Failed | Skipped |
|---|---|---|---|---|
| Malformed fixtures rejected with expected codes; upstream accept/reject agreement (`tests/integration/test_fixture_conformance.py`) | 29 | 29 | 0 | 0 |
| Golden values, lossless BF16/F16/F64/I64/U64 ROI, JSON round trip (`tests/integration/test_golden_values.py`) | 10 | 10 | 0 | 0 |
| Oversized sparse file RSS (AC-002) (`tests/perf/test_large_sparse.py`) | 1 | 1 | 0 | 0 |
| Property: corrupted files fail closed, never more permissive than upstream (`tests/property/test_header_fuzz.py`) | 1 | 1 | 0 | 0 |
| Property: random N-D ROIs equal an independent NumPy slice (`tests/property/test_roi_property.py`) | 1 | 1 | 0 | 0 |
| Mutation/replacement during reads and jobs (AC-014, B11) (`tests/security/test_mutation.py`) | 21 | 21 | 0 | 0 |
| Shard path rules, symlink and root containment (`tests/security/test_paths.py`) | 23 | 23 | 0 | 0 |
| Strict header rules, all 65,536 BF16/F16 bit patterns (`tests/unit/test_header.py`) | 34 | 34 | 0 | 0 |
| View/ROI validation, read planner, budgets (`tests/unit/test_selection.py`) | 11 | 11 | 0 | 0 |

## Mutation and replacement outcomes (AC-014, B11)

| Case | Outcome |
|---|---|
| `test_mutation_yields_file_changed[before_read-truncate]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[before_read-append]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[before_read-same_size_rewrite_new_mtime]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[before_read-atomic_replace]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[before_read-rename_away]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[before_read-rename_and_recreate]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[before_read-symlink_swap]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[mid_scan-truncate]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[mid_scan-append]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[mid_scan-same_size_rewrite_new_mtime]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[mid_scan-atomic_replace]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[mid_scan-rename_away]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[mid_scan-rename_and_recreate]` | outcome=FILE_CHANGED |
| `test_mutation_yields_file_changed[mid_scan-symlink_swap]` | outcome=FILE_CHANGED |
| `test_same_size_rewrite_keeping_mtime[before_read]` | best_effort_outcome=FILE_CHANGED, verified_outcome=FILE_CHANGED |
| `test_same_size_rewrite_keeping_mtime[mid_scan]` | best_effort_outcome=n/a, verified_outcome=FILE_CHANGED |

## Memory: oversized sparse file (AC-002)

- File: 12,884,901,992 bytes on disk (allocated 24576), one BF16 tensor [98304, 65536]
- Peak RSS after import: 30.3 MiB; after catalog: 30.3 MiB; after exact ROIs: 30.6 MiB; after streaming 1.0 GiB: 82.9 MiB
- Catalog 0.17 ms; ROIs 2.08 ms; scan 0.65 s
- ROI values checked against written bit patterns: True; strided column ROI read 128 bytes

## Supported dtypes and format errors

- Lossless exact ROI: F64, F32, F16, BF16 (WeightLab decoder), I64, I32, I16, I8, U64, U32, U16, U8, BOOL.
- Rejected with `UNSUPPORTED_DTYPE`: F8_E4M3, F8_E5M2, F8_E8M0, F6_*, F4, C64, C128 and unknown dtype strings.
- Format errors raised by the reader: `INVALID_FORMAT`, `UNSUPPORTED_DTYPE`, `MISSING_SHARD`, `PATH_OUTSIDE_ROOT`, `FILE_CHANGED`, `INVALID_ROI`, `RESOURCE_LIMIT`, `UNSUPPORTED_PRECISION`, `NOT_FOUND`.

## Skipped on this OS

- none

## Fixture self-check (NOT G0 evidence)

- `scripts/check_demo_fixtures.py`: PASS: 170 fixture self-checks (not application conformance; see docs/13 section 3). (exit 0)
