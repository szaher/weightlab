# 02 — Functional Requirements and User Acceptance

Conventions: **P0** mandatory R1, **P1** R2/R3, **P2** optional R4. Every requirement has stable ID. IDs appear in backlog and tests. Input sizes and resource ceilings may force the UI to offer sampling rather than exact full scans; this is not a silent degradation.

## A. Project and import

- **FR-001 [P0]:** Create/open local project with project name, timestamp, app/schema version and references to paths, not duplicates of checkpoint bytes.
- **FR-002 [P0]:** Import one `.safetensors` file or a local model directory with `model.safetensors.index.json` and referenced shards, by typed absolute path, read in place. No HTTP upload of model bytes; an optional native folder dialog is a convenience only (doc 09 section 1.1).
- **FR-003 [P0]:** Safely parse tensor metadata without materializing data. List tensor name, shape, dtype, file/shard, byte extent, parameter count and estimated byte count.
- **FR-004 [P0]:** Explicitly reject unsupported formats with an actionable explanation; no eager `torch.load`, pickle or remote code execution.
- **FR-005 [P0]:** Detect non-regular files, symlink escapes (when project policy restricts paths), missing/duplicate shards, incorrect offsets, malformed headers, unsupported dtype, and changed files after import.
- **FR-006 [P0]:** Search/filter by tensor name, shape, dtype, block index and guessed transformer role; role guessing may fail and must be labeled heuristic.
- **FR-007 [P0]:** Compute stable file identity (size, high-resolution mtime, normalized path and optional full SHA-256 on demand); report identity strength. Quick identity is not cryptographic proof.
- **FR-008 [P0]:** Persist and reopen project. File moves trigger relink flow; file mutation invalidates cached summaries.
- **FR-009 [P1]:** Import a multi-checkpoint timeline and parse user-provided training step metadata; missing steps require manual ordering.
- **FR-010 [P1]:** GGUF metadata preview and float tensor inspection with documented tensor layout conversion; packed quantized tensors supported later via verified decoders.

## B. Tensor visualization

- **FR-011 [P0]:** 2D dense weight heatmap with pan/zoom, axis labels, cell coordinates, exact value on hover/click at sufficiently high resolution, and tile error/loading indicators.
- **FR-012 [P0]:** At model-scale zoom, show aggregated downsampled data, not randomly selected pixels posing as a complete matrix.
- **FR-013 [P0]:** Support aggregation modes mean, min, max, mean-absolute and RMS; signed means may cancel, so RMS/mean-absolute are emphasized for magnitude.
- **FR-014 [P0]:** Represent NaN, Inf, missing data, underflow and zero distinctly in legend and tooltips.
- **FR-015 [P0]:** Offer robust color scaling: symmetric diverging for signed delta; fixed range, percentile-clipped (clearly marked) or min/max for raw weights. Default colorblind-safe diverging palette.
- **FR-016 [P0]:** Show row/column coordinate ranges and actual matrix dimensions; origin top-left, row-major logical indexing for all screens. No implicit transpose.
- **FR-017 [P0]:** Display vector as line or histogram with sorted/strided sampling warning, and 3D+ as axis slices selected by user; no semantic assumption for unknown dimensions.
- **FR-018 [P0]:** Exact ROI table within bounded cell count. ROI larger than threshold gives download/export of chunked CSV or asks to narrow request.
- **FR-019 [P0]:** Tensor details: dtype, element/finite/zero/non-finite counts, min, max, mean, population standard deviation, L1/L2/Frobenius norms (when applicable), each with `coverage` (complete/sampled) and `numeric_precision` (lossless/rounded/overflowed), and elapsed scan. Exact percentiles only for an ROI (<= `roi_max_cells`, exact sort, linear interpolation between order statistics, non-finite excluded). Global tensor percentiles are **deferred to R2** (FR-021).
- **FR-020 [P0]:** Disable or cancel expensive full scans. Data rendering must remain interactive during scans.
- **FR-021 [P1]:** Tensor histograms, per-row/per-column profiles, outlier hotspots, sign histograms and numerical precision analysis; global percentiles from a deterministic mergeable sketch whose algorithm, rank-error bound and golden adversarial tests are fixed by an ADR before R2.
- **FR-022 [P1]:** Model-layer treemap or matrix overview driven by catalog and aggregate summaries.
- **FR-023 [P0]:** Ranked change report. For a compatible A/B pair or base + adapter, compute per matched tensor in one bounded streaming pass: exact delta L2, RMS, relative L2 (denominator policy visible), cosine(A,B), max-abs delta, element/finite/non-finite counts; roll up by layer index and role (heuristic, labeled). Delivered as CLI in R0.5 and as the UI model overview in R1; JSON/CSV/Markdown export with provenance. Sorting by a metric never implies causal importance.

## C. Checkpoint comparison

- **FR-030 [P0]:** Choose A and B with explicit model provenance, strict tensor key match and shape/dtype compatibility. Output matched, missing, tied, renamed-suspected and shape-mismatched tensors. A tensor is reported as `tied -> <source key>` only with positive evidence: `tie_word_embeddings: true` in `config.json` together with a known architecture key mapping (e.g. `lm_head.weight` <-> `model.embed_tokens.weight`), or trusted alias metadata. Absence of a key alone is `missing`. The absent tensor is never synthesized.
- **FR-031 [P0]:** Compare same-shaped tensors using `delta=B-A` in explicitly documented promoted computation dtype; no arithmetic on packed quantized representation.
- **FR-032 [P0]:** Modes: A, B, delta, abs(delta), ratio `abs(delta)/(abs(A)+epsilon)` with epsilon visible and configurable, masked epsilon-near-zero values.
- **FR-033 [P0]:** Delta summary: L1, L2, max abs, RMS, relative L2 with denominator policy and dot-product/cosine with explicit degeneracy handling.
- **FR-034 [P0]:** Both raw and delta heatmaps use independently visible value ranges; side-by-side comparison uses optional locked shared scale to avoid deceptive colors.
- **FR-035 [P0]:** Label comparisons `strict lineage verified`, `compatible shape but lineage unverified`, or `incompatible` with exact reasons; unverified compare limited to descriptive inspection and opt-in acknowledgement.
- **FR-036 [P0]:** Idempotent previews: same file versions, tile, method and config -> same result bytes/values, subject to documented floating-point platform variance.
- **FR-037 [P1]:** Layer-level aggregate bar chart built on FR-023 data, select top changed projections, checkpoint timelines and normalized change index.
- **FR-038 [P1]:** Conservative matched tensor mapping for known Qwen/Llama naming conventions; raw keys and explicit confidence always visible.

## D. LoRA and virtual updates

- **FR-040 [P0]:** Load PEFT-style adapter directory with `adapter_model.safetensors` and `adapter_config.json` for supported vanilla linear LoRA modules.
- **FR-041 [P0]:** Validate A/B pairing, shapes `(r,in)` and `(out,r)`, target mapping `(out,in)`, adapter rank and per-module alpha including `rank_pattern` / `alpha_pattern`, and `layers_to_transform` / `layers_pattern`. Honor `fan_in_fan_out` (base weight stored transposed, e.g. GPT-2 `Conv1D`): either apply the documented transpose and record `transpose=true` in provenance, or reject; never guess. Reject explicitly: DoRA (`use_dora` or `lora_magnitude_vector` keys), embedding LoRA (`lora_embedding_A/B`, different layout) in R1, LoHa/LoKr/IA3, prompt tuning, `modules_to_save` effects and non-linear targets.
- **FR-042 [P0]:** Implement `delta=s*(B@A)` for chosen tile or streaming numerical summary only; `s=alpha/r` for standard LoRA and `alpha/sqrt(r)` for rsLoRA (respect module rank/alpha patterns).
- **FR-043 [P0]:** Overlay `effective W = base W + delta` for compatible base/adapter with no permanent merged file. Support `alpha_override` slider only as user-controlled experimental scale separate from adapter's native scaling.
- **FR-044 [P0]:** Strict base-lineage verification through configured base ID, user confirmation and available matching hashes; mismatch produces warning and disables default transfer.
- **FR-045 [P1]:** Compare multiple adapters trained on the same known base; per-module update similarity and overlap; report which modules have no adapter.
- **FR-046 [P1]:** Create virtual partial-update configurations (module toggles, scale) stored as experiment configs; changes to original checkpoint forbidden.

## E. Experiments and interpretation

- **FR-050 [P1]:** Create versioned experiment definition containing source, target, update selector, lambda/alpha sweep, evaluation dataset hash, inference backend, generation settings and hardware details.
- **FR-051 [P1]:** Require valid transfer mode before enabling operations: same-base LoRA, verified common ancestor task delta, or explicit experimental mode with severe warning; refuse shape-incompatible direct addition.
- **FR-052 [P1]:** Virtual-update preview must work with CPU-only weight analysis, independent of inference availability.
- **FR-053 [P1]:** Local evaluation subprocess is opt-in and separate from viewer process; plan memory before launching, expose status/cancel, and stop on timeout or resource cap.
- **FR-054 [P1]:** Paired baseline vs modified run with identical prompts, tokenizer, backend and decoding settings; record exact-match/other metric and per-item outcomes.
- **FR-055 [P1]:** Include negative control (random/shuffled delta with comparable magnitude) and non-math retention/control tasks.
- **FR-056 [P1]:** Export interpretation labels `descriptive`, `correlational`, `interventional evidence`, `unverified`; no claim that a single tensor is a capability.
- **FR-057 [P2]:** Small-model activation recorder with token-level selective hooks; show runtime cost estimate before capture.
- **FR-058 [P2]:** Activation patching and ablation with matched clean/corrupt datasets, careful causal interpretation and multiple-example uncertainty.
- **FR-059 [P2]:** Weight alignment/correspondence research environment with original coordinate systems preserved; never equate two tensors merely because names/shapes match.

## F. Persistence, exports and user trust

- **FR-060 [P0]:** Export image with legend, axes, scale, tensor full key, model fingerprints, zoom, slice, color transform and sampling status.
- **FR-061 [P0]:** JSON export schema with versioned provenance and data quality, no secret local absolute paths unless user confirms.
- **FR-062 [P0]:** Session and bookmark persistence, restore last view, relink missing models.
- **FR-063 [P0]:** Offline demo without any external network; never require login.
- **FR-064 [P0]:** No background uploads, automatic model downloads or analytics collection; explicit network consent required for future features.
- **FR-065 [P0]:** User-facing status for current RAM estimate, active scan, file I/O, progress, cancel, slow disk, and cache size.
- **FR-066 [P0]:** Undo/restore settings; original checkpoint always read-only.
- **FR-067 [P1]:** Portable Markdown/HTML research report with selected images, exact formulas, source provenance, limitations, evaluation metrics and experiment configurations.

## G. Quality and negative acceptance scenarios

- **AC-001:** Open the 2D synthetic 8x8 with known entries; hover all 64 cells and compare with `expected_results.json` `base_full`, produced by an independent reference parser (not the reader under test).
- **AC-002:** A file of 10GB+ on disk can be cataloged and one bounded ROI requested without proportional process RAM use.
- **AC-003:** Deliberately missing shard reports missing filename and no false results.
- **AC-004:** Same-shaped but different-base checkpoints receive `lineage unverified`, not `compatible for transfer`.
- **AC-005:** Values `NaN`, `+Inf`, `-Inf` do not silently color as zero, poison finite-only summaries or crash serialization.
- **AC-006:** Full scan cancel stops worker and frees allocations; UI remains responsive.
- **AC-007:** Two matched SafeTensors checkpoints of same base yield correct delta and row/col coordinates.
- **AC-008:** LoRA tile for rank 2 equals independently computed dense `B@A` fixture to stated tolerance.
- **AC-009:** R1 app installs and runs offline on supported macOS, Windows and Linux CPU-only hosts.
- **AC-010:** Invalid/huge coordinates and malformed JSON return explicit HTTP 4xx, not server crash.
- **AC-011:** Export filenames/manifest never contain auth tokens or source model bytes unintentionally.
- **AC-012:** UI indicates sampled/aggregate tiles and distinguishes them from exact cell values.
- **AC-013:** BF16 and F16 tensors in `tiny_dtypes.safetensors` decode bit-exactly to the expected float32 values, including NaN, +Inf, -Inf, -0.0 and each dtype's max finite value.
- **AC-014:** Truncation, append, size or mtime change, atomic replace/rename and symlink swap of a checkpoint file during a read or job yield `FILE_CHANGED` and the backend process survives. A same-size in-place rewrite that preserves mtime is detected only by `consistency=verified` jobs; this limitation is shown in the UI and docs (doc 03 section 6).
- **AC-015:** `compat/tied_embeddings` vs `compat/untied_reference` reports `lm_head.weight` as tied (config + known mapping), `compat/missing_head_no_tie_evidence` reports it missing; the permuted same-function pair is Class S with a visible caveat although its function is identical.
- **AC-016:** `tiny_precision_a/b` round-trip losslessly through the ROI API: U64 max, I64 min, 2^53-1, 2^53, 2^53+1, F64 extremes, the 2^-52 F64 difference and signed zero. The finite F64 pair whose delta overflows is reported `overflowed`; the F32 delta beyond float32 range is `NOT_REPRESENTABLE` in tiles. Nothing rounded is labeled lossless.
- **AC-017:** The non-finite golden blocks of `tiny_edge.safetensors` produce the status bitfields in `expected_results.json` in both binary and JSON tiles; legend and hover show the same NaN/+Inf/-Inf kinds.
- **AC-018:** A derivative whose metadata names a genuine base SHA-256 but was not built from it is reported with `base_file_identity_verified=true`, `lineage_trust_level=unverified_claim`; it is never shown as proven trained-from-base and is not T1-eligible without a trusted lineage statement.

## H. User stories

1. As a student, I want an offline demo so I can use the app before downloading a large model.
2. As an engineer, I want to import a sharded SafeTensors model so I can list tensors without needing Torch.
3. As an engineer, I want an exact 32x32 ROI so I can validate a surprising heatmap pattern.
4. As an engineer, I want to sort layers by normalized change so I can identify candidate regions to investigate.
5. As a researcher, I want source/target provenance so I do not draw conclusions from unaligned checkpoints.
6. As a researcher, I want a fixed symmetric delta color scale so I can compare signs reliably.
7. As an experimenter, I want to disable individual adapter modules and evaluate behavior without rewriting the base model.
8. As a cautious user, I want strict memory and CPU ceilings so that opening a huge model never makes my laptop unusable.
9. As an educator, I want the UI to explain why a color difference is not proof of a learned capability.
10. As an OSS contributor, I want deterministic fixture-based CI tests so regressions are easy to reproduce.
