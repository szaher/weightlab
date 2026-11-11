# Changelog

## Unreleased — Compare dashboard (UI)

- **Compare screen** (side navigation): create a comparison of two imported checkpoints (policy and optional attestation), run the analysis (best effort or verified), cancel and re-run; the job is rediscovered after a reload.
- **Compatibility first**: base/derived, orientation B − A, class V/L/S/X with specific warnings for S and X, lineage evidence, counts, complete/incomplete, consistency, report time, files-changed-since-report warning, and the interpretation caveat.
- **Layer × role grid**: each cell aggregated from exactly that layer's and role's tensors (the CLI rollup arithmetic; unit tests prove per-layer, per-role and total sums reproduce the report's rollups); cosine only for single-tensor cells; unknown and unlayered rows/roles visible; keyboard operable; linear or log colour scale with legend.
- **Rankings**: sortable, filterable tensor, layer and role tables with bars; withheld, undefined and non-aggregable values labelled and sorted last; 100 rows per page.
- **Tensor drill-down**: metrics with method and precision, counts and non-finite values, caveats, and "Inspect in A/B" opening the existing explorer. The delta heatmap is shown as not yet available.
- Evidence: `evidence/alpha2-compare/`.

## Unreleased — contract 0.4.0 (ADR-020)

Additive API extension approved by the project owner (ADR-020 change-control basis); the 1.2-final baseline stays frozen at tag `spec-v1.2-final` (contract 0.3.0).

- **New job kind `comparison_report`** (`POST /jobs` with `comparison_id`): a comparison's change report computed by `change_report.pair_report`, the code behind `weightlab diff`, so values, rankings, rollups, caveats and completeness are identical to the CLI. `JobResult` `{kind, comparison_id, complete, published}`.
- **New route `GET /comparisons/{comparison_id}/report`**: the stored snapshot (paths redacted) with `complete`, `consistency`, `created_at` and `current_model_status`.
- **New route `GET /projects/{project_id}/comparisons`** (newest first); `Comparison` gains `policy`, `report` (summary) and `latest_report_job_id`.
- **Atomic publication**: the report is written in the transaction that marks its job `completed` (new job-runner commit hook); cancellation is linearized on the runner lock; failed, canceled, interrupted and crashed jobs never publish.
- **Ordering**: reports are replaced only by later-submitted jobs (`job_seq`), regardless of finishing order.
- **Cancellable hashing**: `Checkpoint.hash_tick` lets jobs check cancellation and wall time per hashed chunk (lineage, attestation, `verified`, every shard).
- **Limits**: `128 MiB + 16 KiB x entries` reservation, at most 32,768 entries and 128 MiB of report JSON; refused before queueing.
- **Drift and legacy**: comparisons record `record_version`, `user_attestation` and a canonical `structure_sha256`; mismatches and pre-ADR-020 comparisons fail closed (`CONFLICT`).
- **Policy**: `strict` reports Class V/L only (409 `LINEAGE_UNVERIFIED` for S/X); `descriptive_only` acknowledges descriptive use. `Comparison` exposes `policy`.
- **Migration 0003** adds `comparison_reports` and `comparison_report_seq`.
- `POST /jobs` declares 409, which the implementation already returned for changed inputs.
- Contract examples: 2 positive and 4 negative `JobRequest` cases. Tests: `tests/api/test_comparison_report_api.py`, `tests/api/test_report_scale.py`, commit-hook cases in `tests/integration/test_jobs.py`.

## 1.2-final — 2026-11-06 (specification freeze)

Final consolidated amendment pass from the specification audit of revision `f7d6cd3`. The blueprint is frozen (ADR-019): further changes need a demonstrated implementation correctness or security blocker, a versioned ADR, updated contracts/fixtures/tests and an entry here.

### Audit items and resolution

Status: **R** resolved in this revision, **D** deferred with unambiguous R1 behavior, **L** recorded limitation with test consequence.

| Item | Status | Resolution | Where |
|---|---|---|---|
| WL-FINAL-01 non-finite tile status | R | Status byte is a bitfield: 1 NO_FINITE, 2 HAS_NAN, 4 HAS_POS_INF, 8 HAS_NEG_INF, 16 NOT_REPRESENTABLE, reserved bits must be 0; JSON parity; tile media type bumped to `vnd.weightlab.tile.v2`; golden blocks in `tiny_edge.safetensors` | doc 09 §3, `TileStatus`, AC-017 |
| WL-FINAL-02 counters, extrema, storage | R | Versioned 56 B base / 80 B coarse records, uint64 coarse counters, float64 min/max, overflow flags; all-level storage 0.323 B/cell (16.1% of BF16); I64/U64 pyramid values `rounded`; >65,535-cell fixture | doc 03 §5, doc 11, ADR-012 |
| WL-FINAL-03 exact output encodings | R | Lossless typed ROI encodings (`float_json`, `int_json`, `int_decimal_string`, `bool_json`, `hex_bits`); tiles labeled `float32_visualization`; `UNSUPPORTED_PRECISION`; coordinates capped at 2^53−1; precision fixtures | doc 09 §2, ADR-017, AC-016 |
| WL-FINAL-04 experiment lineage schema | R | Schema v2: `lineage_evidence` required, `lineage_verified` and `base_file_identity_verified` derived by conditionals, digest required for cryptographic evidence, class V experiments need verified sources; 13 negative examples | `contracts/experiment.schema.json`, `contracts/examples/` |
| WL-FINAL-05 base identity vs ancestry | R | `base_file_identity_verified`, `lineage_claim_source`, `lineage_trust_level`; class V means base file named, not ancestry proven; T1 needs a trusted statement; fixture is `proven_by_construction` | doc 04 §7, doc 06 §6, ADR-018, AC-018 |
| WL-FINAL-06 mutation detection | R + L | Handle and path identity compared per batch (replace, rename, symlink swap detected); per-job handles on Windows; `consistency=verified` hashing for snapshot guarantees; same-size same-mtime rewrite is a documented best-effort limitation | doc 03 §6, SEC-06, AC-014, B11 |
| WL-FINAL-07 missing R1 endpoints | R | Catalog filters and byte extents, comparison entries and lineage, `PATCH /settings`, relink, project delete, export record and download, optional directory dialog | doc 09 §1, `contracts/openapi.yaml` |
| WL-FINAL-08 free-form sources | R | Discriminated `Source` union (tensor, delta, lora_delta, virtual_effective); closed request schemas with size limits; typed job results, saved views and errors | `contracts/openapi.yaml`, doc 09 §4 |
| WL-FINAL-09 derived pyramids and N-D identity | R | Pyramid identity includes view and operation parameters; paired pass for delta reducers; canonical SHA-256 cache keys | doc 03 §5, doc 05 §6, doc 10 §6 |
| WL-FINAL-10 cold-cache tile behavior | R | Decision table; 202 + deduplicated `pyramid_build` job; no silent sampling under the limit | doc 09 §3 |
| WL-FINAL-11 browser path picking | R | Typed absolute path is mandatory; optional server-side native dialog; no upload; Windows path rules | doc 09 §1.1, doc 08 §5, FR-002 |
| WL-FINAL-12 strict SafeTensors parsing | R | Strict JSON (no NaN tokens, duplicates, nesting bombs), `type(v) is int`, entry field set, caps, empty tensors, index path rules; 9 new malformed binaries, 3 new index fixtures; permitted upstream differences recorded | doc 04 §2–3, fixtures |
| WL-FINAL-13 coverage vs precision | R | `coverage` and `numeric_precision` on every metric; float64 compute; scaled norms; overflow flags | doc 05 §3, `Metric` |
| WL-FINAL-14 percentiles | D | Global percentiles deferred to R2 (FR-021, ADR before R2); R1 has exact ROI percentiles only | FR-019, doc 05 §4 |
| WL-FINAL-15 manifest and ignore rules | R | `scripts/manifest.py` over git-tracked files; full `.gitignore`; `.gitattributes`; CI clean-clone check | doc 18 §8a |
| WL-FINAL-16 cross-OS byte reproducibility | R + L | No transcendental functions (sine fixture replaced by integer pattern); pinned toolchain; 3-OS hash CI. Byte identity on Windows/Linux is unverified until that CI runs | doc 13 §3, `.github/workflows/spec-checks.yml` |
| WL-FINAL-17 self-check vs conformance | R | Checker labeled a fixture self-check; hand-calculated constants added; conformance tests (error codes per fixture) required in Phase 0 | doc 13 §3, §9 |
| WL-FINAL-18 tied embeddings | R | Tied only with config + known mapping or trusted alias metadata; `missing_head_no_tie_evidence` fixture | FR-030, doc 04 §7, AC-015 |
| WL-FINAL-19 evaluation manifest | R | Conditional requirements when `evaluation.enabled`; baseline, treatment and negative-control arms required; disabled evaluation forbids those fields | `contracts/experiment.schema.json` |
| WL-FINAL-20 LoRA change report cost | R | Per-module GMAC estimate and cap, completeness flag, tested-only norm shortcut | doc 11, doc 14 Phase 0.5, WL-024 |
| WL-FINAL-21 ADR-004 wording | R | Positioned reads by default; mmap opt-in in isolated workers | ADR-004 |
| WL-FINAL-22 Phase 0 scope | R | Reader/catalog/ROI only in PRD, roadmap and handoff; G0 requires the application's own evidence | doc 01 R0, doc 14, handoff, doc 13 §9 |
| WL-FINAL-23 security E2E | R | Token entropy and rotation, fragment removal, Host/Origin rules, CSP literal, IPv6 loopback, install vs runtime offline wording; E2E list | doc 12 §4, §8 |
| WL-FINAL-24 reference 12 | R | Citation corrected (NeurIPS 2025 MI Workshop); revision 1.1's "unverified" label was wrong | doc 17 |
| WL-FINAL-25 freeze procedure | R | Version 1.2-final, this changelog, ADR-019, spec CI; tag `spec-v1.2-final` to be created by the maintainer on the freeze commit | README, ADR-019 |

### Other changes

- New fixtures: `tiny_precision_a/b.safetensors`, `tiny_edge.safetensors`, `compat/missing_head_no_tie_evidence/`, malformed `bool_shape`, `float_offsets`, `json_nan_token`, `non_utf8_header`, `nested_metadata`, `metadata_non_string`, `nesting_bomb`, `unknown_entry_field`, `rank_too_high`, and index dirs `backslash_path`, `windows_drive_path`, `non_string_shard`. Generator version 3.
- New scripts: `check_contracts.py`, `check_spec.py`, `manifest.py`; `build_documentation.py --check`.
- The cross-file sweep found truncated doc paths (two-digit prefix plus an ellipsis) in the agent handoff; expanded to real file names.

## 1.1 — 2026-11-05

Review fixes: BF16/F16 mandatory and decoded by WeightLab, positioned-read reader, exact pyramid with row sampling only for very large tensors, R0.5 change report CLI, operational lineage verification, evaluation on the viewer's own weights, URL-fragment token, LoRA edge cases, tied embeddings, resolved contradictions, expanded fixtures; follow-up: `sum_abs`, JSON/binary tile parity, attestation separated from verification, 5-byte tile test. Details in `DELIVERY_NOTES.md`.

## 1.0 — 2026-11-05

Initial planning baseline.
