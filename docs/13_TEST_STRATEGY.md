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

Provided: `scripts/generate_demo_fixtures.py --output tests/fixtures/generated --seed 7` (byte-reproducible; all pseudo-random values come from SplitMix64, not NumPy's RNG) and `scripts/check_demo_fixtures.py --root <dir>`, an **independent reference** built on `struct` + `json` + `numpy.frombuffer` that does not use the safetensors library. The checker also asserts that upstream safetensors accepts every valid fixture and rejects every malformed one. It generates:

- `tiny_base.safetensors`: 8x8 ramp (`up_proj`), 8x8 pseudo-random `down_proj`, 257x263 sine matrix, 3D 2x4x8, 1D vector, rank-0 scalar, 1x1, zeros, non-finite matrix.
- `tiny_ft.safetensors`: *same keys/shapes*, 10 known changed positions in `up_proj`; `expected_results.json` holds the full 8x8 base, fine-tune and delta.
- `tiny_dtypes.safetensors`: BF16 and F16 4x4 matrices plus NaN, +/-Inf, -0.0 and max-finite specials (written by a raw writer, since the safetensors NumPy backend cannot write BF16).
- `tiny_sharded/`: valid `model.safetensors.index.json` referencing two shards.
- `tiny_lora/`: rank-2 `up_proj` (alpha 4, scale 2) and rank-1 `down_proj` via `rank_pattern` (scale 4); `tiny_lora_rslora/`: rank-4, alpha 8, scale `8/sqrt(4)=4`; `tiny_lora_dora_unsupported/`: must be rejected.
- `compat/`: `different_ancestry` (Class S), `shape_mismatch` (Class X for one tensor), `permuted_same_function` (identical function, large raw delta), `tied_embeddings` vs `untied_reference`.
- `malformed/`: truncated payload, header length beyond file, corrupt JSON, unknown dtype, shape/length mismatch, overflow and negative shapes, out-of-bounds, overlapping and gapped offsets, duplicate key; index directories `missing_shard/`, `path_traversal/`, `absolute_path/`, `key_not_in_shard/`. Each has its expected error code in `expected_results.json`.
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

## 5. File format and security tests

`SEC-01` through `SEC-12` from security doc map to `tests/security/test_*.py`. Fuzz SafeTensors metadata parser through official library and WeightLab's own parser (they must agree on accept/reject); test DOS with declared giant shape; tests for stale project/cached tile after source change; truncate a file during a read and assert `FILE_CHANGED` with the backend process still alive. Windows path normalization incl drive prefix/UNC and reparse points. Enforce explicit root on selected local model directory.

## 6. API contract tests

- OpenAPI schema validation + generated client test where practical.
- 400/401/403/404/409/413/422/429 exact error cases.
- No cross-origin unauthenticated sensitive requests; random token is required by all non-public endpoints.
- JSON response finite and valid; binary tile byte count = `rows*cols*sizeof(dtype)` when no separate mask channel.
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

**G0 Reader correctness:** parse, exact ROI, security, bounded huge-file proof.  
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
```

Adjust commands only after implementation while keeping equivalent quality coverage.
