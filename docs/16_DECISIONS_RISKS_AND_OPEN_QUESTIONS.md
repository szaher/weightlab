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
**Decision:** Read-only mapped and chunked access, exact ROI and sampled aggregates distinguished.  
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
**Decision:** L0-L3 tiles from direct reads, L>=4 from an exact one-pass pyramid; row-sampled previews only above `pyramid_exact_max_mib`. Separate budgets for tiles (65,536 cells) and the ROI table (4,096 cells).
**Why:** Most weight matrices are 30-250 MiB; one sequential pass is cheaper than scattered per-cell sampling, and results are exact.
**Trade-off:** A background build per tensor and a pyramid cache of ~8% of BF16 tensor size.

### ADR-013: Change report CLI before the UI
**Decision:** Ship `weightlab diff` (R0.5) after the reader gate and validate it with practitioners before building the R1 UI.
**Why:** "Which layers changed most?" is the primary job (J03/J04) and needs no UI; feedback should shape the expensive UI work.
**Trade-off:** The first release is terminal-only.

### ADR-014: Operational lineage verification
**Decision:** Class V requires file-content evidence: recorded SHA-256, offline Hugging Face cache blob hashes with a declared revision, or explicit user attestation (labeled). Name-only declarations are Class L.
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
- Resolved: launch token via URL fragment (ADR-016). Remaining: verify on Chrome, Firefox and Safari that the fragment is removed from history after the UI reads it.
- Optional inference support for arbitrary adapter intervention without writing full merged model.
- Licensing and source of any real math training/eval dataset.
- Performance thresholds on spinning disk/slow external SSD, if supporting them.

## 5. Change-control

Changing a formula, API field, coordinate convention, cache key semantics, compatibility rule or resource budget requires versioned ADR, fixture updates, OpenAPI revisions, migration when persistent data semantics change, and explicit release note. Avoid hidden breaking behavior.
