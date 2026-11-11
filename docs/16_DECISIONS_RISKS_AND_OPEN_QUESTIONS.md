# 16 — Architectural Decisions, Risk Register and Explicit Assumptions

## 1. Architecture Decision Records

### ADR-001: Local-first backend + browser UI
**Decision:** FastAPI bound to localhost, packaged static React UI.  
**Why:** Browser visualization and Python numerical ecosystem without desktop-native overhead.  
**Trade-off:** Local API security and packaging complexity; mitigated via launch token and single wheel.  
**Alternative:** Tauri/Rust all-in-one; consider after validating product and resource envelope.

### ADR-002: SafeTensors primary format
**Decision:** SafeTensors first, sharded indexes supported.  
**Why:** Explicit tensor metadata and non-executable file format.  
**Trade-off:** GGUF user audience initially unsupported; need tested reader later.

### ADR-003: No PyTorch required in core install
**Decision:** NumPy + SafeTensors default, optional Torch/TransformerLens in plugins.  
**Why:** Laptop-friendly startup and install footprint.  
**Trade-off:** BF16 and exotic dtype handling must be explicitly implemented and tested; no arbitrary model execution by default.

### ADR-004: Streaming, tile-first not eager checkpoint load
**Decision:** Read-only, chunked access by bounded positioned reads (ADR-011); mmap only as an opt-in backend inside an isolated worker. Exact ROI and sampled aggregates distinguished.  
**Why:** Avoid RAM proportional to model size.  
**Trade-off:** Exact summary still requires scanning bytes; users choose expensive operations.

### ADR-005: Scientific validity takes precedence
**Decision:** No automatic capability attribution or unverified model merging.  
**Why:** Raw coordinates may be unaligned and correlation is not causation.  
**Trade-off:** Stronger warnings and narrower default functionality than some model-mixing demos.

### ADR-006: Optional inference isolated
**Decision:** Separate process/backend with its own limits.  
**Why:** Model forward passes threaten desktop responsiveness and have heavier dependencies.  
**Trade-off:** Experiments have higher latency and availability varies by machine/model.

### ADR-007: SQLite project metadata, no model byte copies
**Decision:** Keep file references + derived cache only.  
**Why:** Disk efficient and privacy-preserving.  
**Trade-off:** Model moves invalidate references; relink UX necessary.

### ADR-008: Raw-key preservation
**Decision:** Tensor identity is exact source key; semantic role is optional heuristic metadata.  
**Why:** Qwen/Llama/other architecture differences, avoid incorrect auto remapping.  
**Trade-off:** Cross-format comparisons need explicit mapping modules.

### ADR-009: Read-only originals
**Decision:** No changes to imported checkpoint files ever.  
**Why:** User trust, corruption prevention and reproducibility.  
**Trade-off:** Merged export could require additional disk, opt-in.

### ADR-010: R1 is descriptive; R3 is experimental
**Decision:** Strict release boundary and feature flags.  
**Why:** Avoid claiming transfer and mechanistic insights before testing.  
**Trade-off:** Some headline features ship later.

### ADR-011: Positioned-read reader, WeightLab-owned dtype decoding
**Decision:** Default reader uses validated byte ranges with `os.pread`/`readinto` into bounded buffers; WeightLab decodes BF16 and F16; F8 rejected in R1. mmap only as an opt-in backend inside an isolated worker.
**Why:** safetensors 0.8.0's NumPy backend cannot read BF16 (verified 2026-11-05); mmap turns file truncation into SIGBUS and locks files on Windows.
**Trade-off:** We own a small amount of low-level code, so it needs property and security tests and must agree with upstream on accept/reject.

### ADR-012: Exact pyramid first; row sampling only for very large tensors
**Decision:** Direct reads when covered bytes <= `direct_tile_max_mib` (all L0-L3), otherwise an exact one-pass pyramid per source and view; row-sampled previews only above `pyramid_exact_max_mib`; cold requests get a deduplicated 202 job, never a silent sample. Separate budgets for tiles (65,536 cells) and the ROI table (4,096 cells).  
**Why:** Most weight matrices are 30-250 MiB; one sequential pass is cheaper than scattered per-cell sampling, and results are exact.
**Trade-off:** A background build per source and view, and pyramid storage of 0.323 B per cell for all levels (versioned 56 B base / 80 B coarse records with uint64 counters and float64 extrema): ~16.1% of BF16 tensor size.

### ADR-013: Change report CLI before the UI
**Decision:** Ship `weightlab diff` (R0.5) after the reader gate and validate it with practitioners before building the R1 UI.
**Why:** "Which layers changed most?" is the primary job (J03/J04) and needs no UI; feedback should shape the expensive UI work.
**Trade-off:** The first release is terminal-only.

### ADR-014: Operational lineage verification
**Decision:** Class V requires cryptographic file-content evidence: recorded SHA-256, or offline Hugging Face cache blob hashes with a declared revision. User attestation is recorded as `lineage_evidence = user_attested` and keeps the pair in Class L (it satisfies the T2 acknowledgment only). Name-only declarations are Class L with `declared`.  
**Why:** Real adapters rarely record base hashes; without an achievable verification path, V would be unreachable.
**Trade-off:** Hashing large files is a slow, explicit job.

### ADR-015: Evaluation uses the viewer's weights
**Decision:** Default R3 backend is CPU `transformers` + `peft` on the same SafeTensors files; module masks and scales are small derived adapters. llama.cpp is optional and labeled as weight-divergent.
**Why:** The preview must equal the evaluated model; GGUF conversion and quantization break that.
**Trade-off:** Slower inference; practical only for small models on 8 GB machines.

### ADR-016: Launch token in the URL fragment
**Decision:** `weightlab start` prints `http://127.0.0.1:PORT/#token=...`; the UI moves the token to `sessionStorage` and sends it as a bearer header. The HTML shell holds no secret.
**Why:** Fragments are never sent to servers or written to access logs; a proven pattern (Jupyter). Combined with the Host allowlist this covers DNS rebinding.
**Trade-off:** A user who closes the tab needs the printed URL again (or `weightlab start` reprints it).

### ADR-017: Lossless typed exact values; tiles are labeled lossy
**Decision:** The ROI API is lossless for every supported dtype (floats as exact float64 JSON numbers or raw bits, I64/U64 as decimal strings); tiles are float32 visualizations with a status bitfield; every metric carries `coverage` and `numeric_precision`.
**Why:** JSON numbers and float32 cannot carry 64-bit integers or F64 exactly, and "exact coverage" is not "exact arithmetic".
**Trade-off:** Clients must handle typed encodings; tables show rounded digits with full values on demand.

### ADR-018: Base file identity is separate from training lineage
**Decision:** `LineageAssessment` records `base_file_identity_verified`, `lineage_claim_source` and `lineage_trust_level`; Class V means the base file is the one named, and T1 transfer additionally needs a trusted lineage statement.
**Why:** A genuine base hash can be written into a fabricated derivative; hashes prove identity, not ancestry.
**Trade-off:** Real-world pairs rarely reach T1 without the user's own training manifest.

### ADR-019: Specification freeze (1.2-final)
**Decision:** The blueprint is frozen at 1.2-final (tag `spec-v1.2-final`). Further changes require a demonstrated implementation correctness or security blocker, a versioned ADR, updated contracts/fixtures/tests and a changelog entry (section 5). Implementation discoveries are tracked as issues, not silent contract drift.
**Why:** Implementation must start against a stable, testable contract.
**Trade-off:** Some refinements wait for evidence from Phase 0.
**Successor revisions:** additive extensions approved by the project owner are recorded as numbered ADRs with their own contract revision, leaving the frozen baseline and its tag untouched: ADR-020 (contract 0.4.0, comparison reports).

### ADR-020: Comparison change reports over the API (contract 0.4.0)
**Change-control basis (ADR-019):** This is an **additive feature extension**, not a correctness or security fix, so ADR-019's blocker criterion does not authorize it by itself. The project owner (repository maintainer) approved it explicitly on 2026-11-11 as the backend of release 0.1.0a2 ("Visual Checkpoint Comparison", ROADMAP P0.1). The 1.2-final baseline stays frozen and auditable at tag `spec-v1.2-final` and contract `0.3.0-spec-1.2-final`; this ADR defines the successor contract revision `0.4.0-spec-1.2-final-adr-020`. No 0.3.0 field, route or behaviour changes meaning; additions are listed in CHANGELOG.md.

**Decision:**
1. *API.* Job kind `comparison_report` (`POST /jobs` with `comparison_id`; no `source`, `view` or `model_id`) and `GET /comparisons/{comparison_id}/report`. The job runs `change_report.pair_report`, the function behind `weightlab diff`, on the comparison's two imported models, so numbers, rankings, rollups, caveats and `complete` equal the CLI's. Options other than `consistency` stay at CLI defaults. `JobResult` is `{kind, comparison_id, complete, published}`. `GET /projects/{id}/comparisons` and `Comparison.report` / `Comparison.latest_report_job_id` let a client rediscover comparisons, the stored report and a running or failed job after a reload, without client-side state.
2. *Atomic publication.* The report row (table `comparison_reports`, migration 0003) is written by the job's commit hook inside the single transaction that writes the job's `completed` row (weightlab/jobs.py). A crash before COMMIT leaves a `running` row (restart: `interrupted`) and no report. A hook failure (comparison deleted: `NOT_FOUND`; storage error) rolls back and the job is `failed`. Failed, canceled and interrupted jobs never publish; the previous report stays.
3. *Cancellation is linearized on the runner lock.* The runner checks the cancel flag under its lock in the critical section that writes the terminal state; `cancel` sets the flag under the same lock. Whichever takes the lock first wins: a cancel first gives `canceled` (result and hook dropped); completion first gives `completed`, and a later cancel changes nothing.
4. *Ordering.* Each submission gets a strictly increasing `job_seq` (`comparison_report_seq`, AUTOINCREMENT, never reused). A completed job replaces the stored report only if its `job_seq` is greater (enforced by the upsert's WHERE clause in the same transaction), so an earlier-submitted job that finishes later completes with `published: false`. A deduplicated submission joins the pending job and keeps its number.
5. *Bounded hashing.* `Checkpoint.hash_tick` is called after every hashed chunk by `digests()` and `verify_digests()`, whichever layer calls them (lineage, attestation, `verified` before/after, every shard). The job sets it to its context's tick, so cancellation and wall time interrupt hashing between `read_chunk_bytes` (4 MiB) chunks. Hash values and verification are unchanged.
6. *Resource limits.* Reservation `128 MiB + 16 KiB x entries` (128 MiB: G0.5 diff bound; measured ~9 KiB peak per entry for report objects and JSON at 2k and 20k entries, 1.8x margin for the SQLite and response copies). At most `MAX_REPORT_ENTRIES = 32,768` entries (refused with 422 before queueing) and `MAX_REPORT_JSON_BYTES = 128 MiB` of stored JSON (checked before publication). The entry count and byte estimate are SQL aggregates; GET splices the stored JSON text into the response without parsing it. `estimated_bytes` is an upper bound: analyzed tensor bytes plus one hash of every input file (two when `verified`).
7. *Drift.* POST /comparisons records `record_version: 2`, `user_attestation` and `structure_sha256`, the SHA-256 of the canonical structure (every entry's key, status, analyzability, shapes, dtypes and tie relationship, the pair flags, and the model ids and orientation). The job refuses if the reopened models fail their import fingerprints (`FILE_CHANGED`) or the recomputed fingerprint, counts or lineage differ from the record (`CONFLICT`). V/L/S/X semantics are unchanged.
8. *Historical records.* Comparisons created before this ADR (no `record_version`) are refused with `CONFLICT` and the user creates a new comparison. Their attestation input is not reconstructed, so lineage trust is never upgraded by inference.
9. *Policy (FR-035).* `strict` (default) reports Class V and L only; Class S and X are refused with 409 `LINEAGE_UNVERIFIED`. `descriptive_only` is the user's acknowledgement that S/X numbers are descriptive inspection of possibly unaligned coordinates; the report then carries the CLI's class caveats. The comparison view exposes `policy`.
10. *Freshness.* A report is a historical snapshot of the bytes its job read. `best_effort`: quick fingerprints (size, mtime, ctime, file id) at open and per read batch; a same-size rewrite preserving mtime and ctime is undetected (doc 03 section 6). `verified`: SHA-256 of every input file before and after the scan matched and is recorded in the report's provenance. GET reads no tensor bytes; it returns `consistency`, `created_at` and `current_model_status` (stat-only quick fingerprint per model), so clients can show that files changed since the report.
11. *Privacy.* Provenance carries file names only (CLI default). Every string in the stored report, and every job error message, has the models' absolute roots, entry paths (as given and resolved) and the user's home directory replaced with `<redacted path>`.

**Why:** ROADMAP P0.1 needs per-tensor, per-layer and per-role metrics in the UI. The frozen `compare_stats` kind covers one delta source per job; a whole-model dashboard would need one job per tensor and re-implemented rollups. Reusing `pair_report` keeps the API's numbers identical to the CLI's by construction.

**Trade-off:** The report is returned whole (no paging); at the entry cap it is ~90 MiB of JSON, and checkpoints above 32,768 tensors (some MoE models) cannot be reported until paging or streaming exists. `estimated_bytes` can exceed `bytes_scanned` for best-effort jobs. Legacy comparisons must be recreated. `compare_stats` remains unimplemented.

## 2. Risk register

| Risk | Likelihood | Impact | Mitigation / Owner |
|---|---|---|---|
| safetensors NumPy backend cannot read BF16 (confirmed 0.8.0) or copies big data | Confirmed | High | own positioned-read reader + decoders (ADR-011); backend owner |
| Silent transpose (`fan_in_fan_out`, non-C-contiguous arrays passed to `save_file`) | Medium | High | explicit transpose handling or rejection, `np.require(..., 'C')`, golden tests; numerics owner |
| File truncated while being read | Medium | High | positioned reads, `FILE_CHANGED`, no mmap in API process; backend owner |
| Large exact heatmap requires expensive I/O | Medium | High | exact pyramid with bytes estimate; row-sampled preview only above limit (ADR-012); numerics owner |
| Model drift after import produces stale data | Medium | High | file identity and invalidation; storage owner |
| LoRA name/scale mapping incorrect | Medium | High | limited supported variants, golden fixtures, fail closed; numerics owner |
| Weight-delta heatmap overclaims meaning | High | High | interpretation labels, behavioral controls; research owner |
| Different base models have unaligned weights | High | High | V/L/S/X gating, no direct merge; research owner |
| Browser memory leak during pan/zoom | Medium | Medium | capped tile cache, 20min stress; UI owner |
| Windows filesystem differences break handling | Medium | Medium | native Windows CI, path/reparse tests; platform owner |
| Too many optional dependencies | High | Medium | minimal core, extras, reproducible locks; release owner |
| Optional model inference breaks 8GB goal | High | High | separate budget and off by default; experiment owner |
| False scientific gain from contaminated benchmark | Medium | High | held-out sets, controls, documented split; research owner |
| Exports leak private paths or tensor contents | Medium | High | redaction default and explicit consent; security owner |
| Native packaging hard to support | Medium | Medium | start with browser+CLI; release owner |

## 3. Open questions (resolved defaults)

1. **Framework:** Python 3.11 + FastAPI + React/TypeScript Canvas2D. Rationale: fast safe implementation, testable, known ecosystem.
2. **Deployment:** Local-only single process (worker may be separate), no SaaS or account.
3. **Data format:** SafeTensors first; no GGUF R1.
4. **Install:** `uv tool install .` or pipx from wheel with bundled frontend; native application wrapper later.
5. **Model choice:** Synthetic fixtures are canonical. Real-model tutorial suggests small public model with verified license and lineage; no bundled proprietary model.
6. **Default memory:** 1 GiB backend peak cap target, dynamic budgeting + hard request ceilings; measured not assumed.
7. **Default compute:** <=2 workers, no GPU, no inference engine.
8. **Stat quality:** Exact ROI and sampled overview distinguished; precise full stats deferred to explicit scan.
9. **Data export:** JSON+PNG R1, Markdown report R2.
10. **Scientific language:** No claims of copying "math capability" without measured causal/behavioral evidence.
11. **Reader:** positioned reads with own BF16/F16 decoding (ADR-011).
12. **First release:** `weightlab diff` CLI before the UI (ADR-013).
13. **Launch token:** URL fragment (ADR-016).

## 4. Unknowns requiring engineering verification

- Resolved 2026-11-05: safetensors 0.8.0's NumPy backend rejects BF16, so WeightLab owns dtype decoding (ADR-011). Remaining: positioned-read throughput for row stripes vs mmap on Windows and macOS.
- Real 8GB macOS and Windows memory pressure with large mapped files.
- Cross-OS byte identity of generated fixtures (pinned toolchain; verified by the spec CI matrix, not yet run).
- Native directory dialog helpers per OS (optional capability).
- Resolved: launch token via URL fragment (ADR-016). Remaining: verify on Chrome, Firefox and Safari that the fragment is removed from history after the UI reads it.
- Optional inference support for arbitrary adapter intervention without writing full merged model.
- Licensing and source of any real math training/eval dataset.
- Performance thresholds on spinning disk/slow external SSD, if supporting them.

## 5. Change-control

Changing a formula, API field, coordinate convention, cache key semantics, compatibility rule or resource budget requires versioned ADR, fixture updates, OpenAPI revisions, migration when persistent data semantics change, and explicit release note. Avoid hidden breaking behavior.
