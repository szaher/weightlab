# 13 — Verification, QA and Test Plan

## 1. Test philosophy

The first milestone is mathematical trust, not a polished UI. All numerical results should be checked against a **small independent reference implementation**, not only against another function using the same code path. Tests are mandatory for file safety, memory bounds, OS portability, UI semantics and claim language.

## 2. Test pyramid

1. **Unit (fast):** Schema/validation, slices, indices, dtype conversion, NaN policy, stats algorithms, aggregation, delta, LoRA, cache-key construction.
2. **Property-based:** Random shapes, tile level, ROI bounds, offsets, finite/nonfinite combinations, chunk invariance and aggregation associativity where mathematically valid.
3. **Integration:** Real SafeTensors reader, file index/shards, DB migrations, API errors, concurrent tasks, cancellation, path restrictions.
4. **End to end:** Browser opens bundled demo, inspect/compare/export/reopen, keyboard navigation, offline behavior.
5. **Performance:** RSS/CPU/disk/browsers under reference and adversarial load.
6. **Research protocol:** Evaluate controlled treatments and invariance across identical settings; no conflation of correlated vs causal results.

## 3. Golden fixture generator

Provided: `scripts/generate_demo_fixtures.py --output tests/fixtures/generated --seed 7` and `scripts/check_demo_fixtures.py --root <dir>`. The checker is a **fixture self-check**: it validates the fixture files and `expected_results.json` with an independent reference parser built on `struct` + `json` + `numpy.frombuffer` (no safetensors library for values), a handful of hand-calculated constants, and upstream safetensors as an accept/reject oracle with the permitted differences recorded in `expected_results.json`. It is **not** application conformance and never counts as G0 evidence: the WeightLab suite must separately assert its own reader's values and the error code listed in `malformed_expected` for every malformed fixture (parameterized test).

**Reproducibility.** All fixture values come from integer arithmetic followed by IEEE-754 correctly rounded division/conversion, and pseudo-random values from SplitMix64; no transcendental functions or NumPy RNG. Byte identity is claimed only under the pinned toolchain in `scripts/requirements-spec-tools.txt`; the spec CI regenerates the fixtures on Linux, macOS and Windows and compares SHA-256 with the committed values. If hashes ever differ on a platform, the difference is diagnosed and recorded, and only numeric parity (not byte identity) is claimed for it. It generates:

- `tiny_base.safetensors`: 8x8 ramp (`up_proj`), 8x8 pseudo-random `down_proj`, 257x263 sine matrix, 3D 2x4x8, 1D vector, rank-0 scalar, 1x1, zeros, non-finite matrix.
- `tiny_ft.safetensors`: *same keys/shapes*, 10 known changed positions in `up_proj`; `expected_results.json` holds the full 8x8 base, fine-tune and delta.
- `tiny_dtypes.safetensors`: BF16 and F16 4x4 matrices plus NaN, +/-Inf, -0.0 and max-finite specials (written by a raw writer, since the safetensors NumPy backend cannot write BF16).
- `tiny_sharded/`: valid `model.safetensors.index.json` referencing two shards.
- `tiny_lora/`: rank-2 `up_proj` (alpha 4, scale 2) and rank-1 `down_proj` via `rank_pattern` (scale 4); `tiny_lora_rslora/`: rank-4, alpha 8, scale `8/sqrt(4)=4`; `tiny_lora_dora_unsupported/`: must be rejected.
- `compat/`: `different_ancestry` (Class S), `shape_mismatch` (Class X for one tensor), `permuted_same_function` (identical function, large raw delta), `tied_embeddings` (config evidence) vs `untied_reference`, and `missing_head_no_tie_evidence` (must be `missing`, not tied).
- `tiny_precision_a/b.safetensors`: F64 extremes (+/-1e300, subnormal, max, min normal), signed zero, a 2^-52 difference, a finite pair whose F64 delta overflows, an F32 pair whose delta exceeds float32, I64 min/max and 2^53-1/2^53/2^53+1, U64 max, I32, BOOL, and F64 values whose sum of squares overflows.
- `tiny_edge.safetensors`: zero-sized tensors `[0,4]` and `[3,0]`, a non-ASCII tensor key, a 256x512 F32 tensor whose L8 block holds 65,536 cells and L9 block 131,072 (beyond uint16), and a 32x32 non-finite golden matrix with single NaN/+Inf/-Inf cells, a mixed finite/non-finite block, an all-non-finite block with all three kinds and an all-NaN block.
- `malformed/`: truncated payload, header length beyond file, corrupt JSON, unknown dtype, shape/length mismatch, overflow and negative shapes, out-of-bounds, overlapping and gapped offsets, duplicate key, boolean shape, float offsets, JSON `NaN` token, non-UTF-8 header, nested metadata, non-string metadata, nesting bomb, unknown entry field, rank above 32; index directories `missing_shard/`, `path_traversal/`, `absolute_path/`, `key_not_in_shard/`, `backslash_path/`, `windows_drive_path/`, `non_string_shard/`. Each has its expected error code in `expected_results.json`.
- Generated per test run, not committed: symlink-escape directory (symlinks are not portable in git checkouts), the >RAM sparse file for B01/AC-002, and files truncated mid-read for AC-014.

## 4. Numerical invariants

- Exact ROI `R[r0:r1,c0:c1]` equals independent NumPy slice for all allowed coordinates.
- Tilings partition the requested exact region with no overlaps/gaps (except intentionally off-viewport loaded tiles).
- Level-zero values equal original source; aggregated full-block mean invariant to chunk boundaries.
- Parent sum/finite_count = sum child; parent RMS derived from sums of squares; edge block weighted correctly.
- Compare `Delta(B,A)=−Delta(A,B)` within specified float error and `Delta(A,A)=0` for all finite pairs.
- `||Delta||₂` agrees with independent dense reference for fixtures; zero reference normalization returns flagged result.
- LoRA `B @ A` ROI equals materialized small fixture; alpha scale and `rsLoRA` verified separately.
- NaN/+Inf/-Inf never serialized as nonstandard JSON literals or drawn as legitimate zero.
- Sampling with identical seed/file/settings is deterministic; different seeds change sample positions on fixture when expected.
- Multi-model comparison refuses impossible shapes and never silently transposes a matrix to make it fit.
- BF16/F16 decoding of `tiny_dtypes.safetensors` is bit-exact including specials; F8 and unknown dtypes are rejected.
- Exact pyramid levels equal an independent block aggregation of the dense fixture, including the L1/L3 edge blocks in `expected_results.json`.
- Tile status bitfields for every golden cell/block equal `expected_results.json` in binary and JSON form; reserved bits are never set.
- Pyramid counters exceed 65,535 (L8 of `edge.block256x512.weight`) and a synthetic merge test exceeds 2^32; F64 1e300 extrema survive in pyramid min/max; sum-of-squares overflow is flagged, never shown as Inf or finite.
- RMS(B-A) from a delta pyramid equals the elementwise reference and differs from RMS(B)-RMS(A) (`rms_delta_vs_difference_of_rms`); two slices of `demo.cube.weight` and two LoRA multipliers never share a pyramid.
- ROI values for `tiny_precision_*` are lossless in `auto` and `hex_bits` encodings; derived values are labeled `rounded`/`overflowed`.
- Strict parsing: every malformed fixture fails with its listed code; WeightLab and upstream agree except for the recorded permitted differences.

## 5. File format and security tests

`SEC-01` through `SEC-12` from security doc map to `tests/security/test_*.py`. Fuzz SafeTensors metadata parser through official library and WeightLab's own parser (they must agree on accept/reject); test DOS with declared giant shape; tests for stale project/cached tile after source change; truncate a file during a read and assert `FILE_CHANGED` with the backend process still alive. Windows path normalization incl drive prefix/UNC and reparse points. Enforce explicit root on selected local model directory.

## 6. API contract tests

- `scripts/check_contracts.py`: OpenAPI valid, references resolve, every positive example in `contracts/examples/` accepted and every negative example rejected, experiment manifest invariants enforced.
- A client generated from `contracts/openapi.yaml` completes every first-run E2E journey using only documented routes (doc 09 section 9).
- 400/401/403/404/409/413/422/429 exact error cases.
- No cross-origin unauthenticated sensitive requests; random token is required by all non-public endpoints.
- JSON responses contain no NaN/Infinity literals. Binary tile body is exactly `rows*cols*5` bytes: `rows*cols` float32 little-endian display values followed by `rows*cols` uint8 status bitfields (bits 1 NO_FINITE, 2 HAS_NAN, 4 HAS_POS_INF, 8 HAS_NEG_INF, 16 NOT_REPRESENTABLE; reserved bits 0); a cell is drawn iff `(status & 17) == 0`; header row/col ranges match the body shape; the JSON form of the same tile carries identical values (null exactly for non-drawable cells) and statuses.
- Job cancel idempotency and state machine transitions; no canceled job marked completed.
- Projects persist and survive process restart, cache recomputes safely after version change.

## 7. Frontend tests

Playwright steps:

1. Block network except localhost, launch app, click Demo, inspect top-level tensor list.
2. Select known 8x8 ramp, verify exact tooltip and displayed dimensions.
3. Set delta mode, verify changed cells and symmetric legend; verify same raw values with different color scale.
4. Pan/zoom and verify selection coordinates map to same underlying cells.
5. Import sharded fixture, locate tensor, open details.
6. Import LoRA fixture and verify native scale vs extra experimental multiplier visually separated.
7. Trigger long scan, cancel and ensure app remains responsive.
8. Export PNG and manifest, assert model bytes absent and paths stripped by default.
9. Keyboard-only navigation through core user journey; automated accessibility scan + manual screen reader check.

## 8. Cross-platform matrix

- OS: macOS latest supported (Apple Silicon/Intel where available), Ubuntu LTS x86_64, Windows 11 x86_64.
- Python: 3.11, 3.12 (additional once tested).
- Browser: Chrome/Edge latest, Firefox recent, Safari recent (macOS).
- RAM profile: 8GB manual proof + 16GB automated reference.
- Hardware GPU: none required. Tests must not automatically initialize GPU frameworks.

## 9. Release gates

**G0 Reader correctness:** WeightLab's own reader (not the fixture self-check) passes: every fixture value and ROI exact, BF16/F16/F64/I64/U64 lossless, every malformed fixture rejected with its code, measured RSS on the oversized sparse file (AC-002), mutation and replacement cases (AC-014, B11) with the backend alive. Evidence is dated command output plus the benchmark report.  
**G1 API/math:** golden stats/delta/tile, error schema and limits, no unbounded read path.  
**G2 UX:** complete offline demo walkthrough, error and accessibility flows.  
**G3 Cross-platform:** clean install and smoke on all supported OS, builds reproduce from lockfiles.  
**G4 Performance:** full benchmark report with 8GB host and browser memory evidence.  
**G5 Release:** security scan, changelog, user guide, signed/versioned artifacts, known limitations and rollback procedure.

Gates cannot be declared green without dated evidence and command output/screenshots. Skipped tests count as limitations, not passes.

## 10. Required local quality commands (proposed)

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy weightlab
uv run pytest tests/unit tests/property tests/integration tests/security -q
npm --prefix frontend ci
npm --prefix frontend run lint
npm --prefix frontend run test
npm --prefix frontend run build
npm --prefix frontend run e2e
uv run python scripts/run_benchmarks.py --profile laptop8g
# specification checks (run in CI on every change, see .github/workflows/spec-checks.yml)
uv run --with-requirements scripts/requirements-spec-tools.txt python scripts/check_demo_fixtures.py --root examples/fixtures
uv run --with-requirements scripts/requirements-spec-tools.txt python scripts/check_contracts.py
python scripts/check_spec.py
python scripts/manifest.py --check
```

Adjust commands only after implementation while keeping equivalent quality coverage.
