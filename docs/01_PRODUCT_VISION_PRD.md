# 01 — Product Requirements Document (PRD)

**Product:** WeightLab  
**Document status:** Approved planning baseline (design intent, not delivery evidence)  
**Product philosophy:** Explore weights cheaply; compare scientifically; intervene cautiously.

## 1. Problem statement

LLM checkpoints contain millions or billions of numerical parameters, but existing workflows often require bespoke Python notebooks, heavyweight runtime dependencies, substantial RAM, or manually written scripts to explore them. Developers cannot easily answer: Which parameters changed during fine-tuning? Do changes concentrate in specific transformer components? Is an update sparse, low-rank, or unusually large? How does an adapter modify effective weights? Are two checkpoints really aligned and comparable? What changed in measured behavior when a layer update was removed?

A heatmap is a useful interaction, not a sufficient scientific explanation. WeightLab must combine an honest visual representation with numerical methods, checkpoint provenance, controlled interventions and reproducible evidence.

## 2. Positioning

**For:** Model engineers, fine-tuning researchers, curious ML practitioners, interpretability researchers and educators.  
**Who need:** Fast local understanding of model parameter structure and differences.  
**WeightLab is:** An offline, laptop-first interactive tensor explorer and experimental model-diff laboratory.  
**Unlike:** Large inference platforms or full training observability stacks.  
**It prioritizes:** Minimal dependencies, bounded resource use, accurate numerical comparisons, and reproducibility over always-on model inference or distributed services.

## 3. Jobs to be done

- J01: "Open a checkpoint and locate relevant layers/tensors without writing code."
- J02: "See a 2D matrix as a heatmap and zoom to exact numerical elements."
- J03: "Compare a base checkpoint to a math-fine-tuned checkpoint and quantify the change."
- J04: "Understand which transformer components changed most: Q/K/V/O, MLP, embeddings, norms."
- J05: "Inspect a LoRA adapter without materializing the full merged model."
- J06: "Check whether apparent heatmap anomalies survive alternative normalizations and exact statistics."
- J07: "Export a reproducible, caveated report of what changed."
- J08: "Test whether an identified update is behaviorally useful, using optional evaluation."
- J09: "Experiment with transferring changes to a compatible target under controlled conditions."
- J10: "Perform these tasks without accidentally exhausting RAM, CPU or disk."

## 4. Personas

**P1 — ML engineer:** Knows SafeTensors/PyTorch; limited time; wants before/after model diffs on a development laptop. Success = answer basic change-location question in minutes without loading a full model.  
**P2 — Researcher:** Needs statistically honest plots, exact formulas and reproducible manifests; may enable optional activation/evaluation packages. Success = export a documented, rerunnable analysis.  
**P3 — Student/educator:** Has only a laptop; wants an accurate, interactive visual mental model. Success = can inspect supplied fixtures and learn what weights do *not* prove.

## 5. Product principles

1. **Evidence, not visual storytelling:** No claim of "math neuron" or "reasoning layer" from weight magnitude alone.
2. **Local first:** Checkpoints remain on disk; network is optional and off by default.
3. **Progressive disclosure:** Quick visualization first; advanced statistics/experimental methods on demand.
4. **Hard resource ceilings:** Use small tiles and streaming algorithms; jobs have cancel buttons and budgets.
5. **Transparent uncertainty:** Differentiate exact, estimated, truncated and unsupported analyses.
6. **Reproducible:** Persist file identity, transform chain, software version, seed, eval set and metrics.
7. **Interoperable:** SafeTensors and Hugging Face sharded indices first; GGUF later with format-aware decoding.
8. **Safety:** Never deserialize arbitrary pickle, execute a model repo's custom code, or overwrite original checkpoint.

## 6. Objectives and success measures

| Objective | Metric | Release acceptance |
|---|---|---|
| First value | Import synthetic fixture -> interactive heatmap | First-time walkthrough works offline |
| Weight inspection | ROI value accuracy | Bit-identical for supported integer/finite float reads, except documented float conversion |
| Compare | Delta and norms | Match independent NumPy reference within dtype tolerances |
| Memory | App memory footprint | R1 8 GB laptop gate under documented test profiles; backend idle RSS <= 300 MiB and peak RSS <= 1 GiB during bounded work (doc 11) |
| User autonomy | No cloud credentials needed | Offline E2E green |
| Validity | Misleading comparisons blocked | All incompatible-model and ambiguity fixtures rejected or explicitly labelled inspection-only |
| Reliability | No failure on malformed input | No crash, unhandled exception or arbitrary file reads in fuzz fixtures |
| Reproducibility | Export manifest complete | Full configuration and file fingerprints included |

Thresholds are requirements to verify, **not results already achieved**. Track p50/p95 latencies on reference devices, not a single unverifiable performance promise.

## 7. Scope and staged delivery

### R0 — engineering spike (must be completed before full UI)

- Synthetic SafeTensors fixture generator with known matrices (constant, ramp, pseudo-random, NaNs, infinity, rank 0, 1D, 1x1, 3D), **F16 and BF16 tensors (mandatory: most current LLM checkpoints are BF16)**, at least two shard files, LoRA pairs (rank-1 via `rank_pattern`, rank-2, rsLoRA, unsupported DoRA), compatibility sets and malformed binaries. Provided by `scripts/generate_demo_fixtures.py` and verified by `scripts/check_demo_fixtures.py`.
- Reader proof: tensor catalog, exact rectangular ROI, bytes read evidence, bounded RAM for an oversized sparse synthetic checkpoint, using bounded positioned reads (ADR-011) with WeightLab's own BF16/F16 decoding (the safetensors NumPy backend cannot read BF16).
- Primitive heatmap tile with agreed coordinate conventions; no model inference.
- Scientific notebook or unit test independently verifying every displayed cell.

### R0.5 — change report CLI (validation release)

- `weightlab diff BASE DERIVED` and `weightlab diff BASE --adapter DIR` produce a ranked table of every matched tensor: exact delta L2 and RMS, relative L2, cosine(A,B), max-abs delta, element and non-finite counts, heuristic role/layer; plus per-layer and per-role rollups (FR-023).
- One bounded streaming pass over both checkpoints (no full tensor in RAM), progress, Ctrl-C cancel with no partial result reported as complete.
- Compatibility class (V/L/S/X), unmatched / tied / shape-mismatched tensors printed before any numbers; rows from Class S/X pairs carry the caveat.
- JSON, CSV and Markdown output with the provenance schema from doc 10.
- Exit: used by 5-10 practitioners on real checkpoints; their feedback sets the priority order of R1 UI work (doc 14, Phase 0.5). This answers jobs J03/J04 before any UI exists.

### R1 — minimum useful product (P0)

- Local directory/file import (explicit user path or native picker), format validation, metadata preview, sharded index support.
- Model overview: the R0.5 ranked change table (per tensor, per layer, per role) as the entry point into heatmaps.
- Searchable tensors by name, shape, dtype, transformer component and layer.
- 1D vectors as line plot; 2D matrices as heatmap; 3D+ requires explicit axis/index or reduction selection.
- Pan, zoom, semantic hover tooltip, rectangular selection, arbitrary ROI numerical table, per-tensor statistics.
- Exact/estimated quality badge and signed, symmetric difference colors.
- Side-by-side checkpoint compare for same-shape matching tensors; base + supported LoRA virtual delta.
- Compatibility classification and warnings, unmatched/shape mismatch report, provenance labels.
- Local project/session, export PNG and JSON result/manifest, cancelable long jobs.
- Synthetic offline first-run demo; keyboard navigation and empty/error/loading states.

### R2 — enhanced weight analytics (P1)

- Layer dashboards and charts beyond the R0.5/R1 ranked table, spectral approximations, covariance/correlation where well-defined, time-series of checkpoint changes.
- Quantization-error inspection only when original/dequantized tensor pairs and mapping are available.
- Deterministic sampling strategies and uncertainty displays.
- User-facing notes, comparison bookmarks, reproducible report (HTML/Markdown).

### R3 — experimental transfer/evaluation (P1/P2)

- Virtual update overlay: target + alpha*delta (no permanent output until explicit export).
- LoRA per-module enable/disable and scaling; same-base verified adapter operations first.
- Evaluation subprocess integration with explicit CPU and RAM limits and prior estimated model size; math and control sets; CSV/JSON comparison results.
- Predefined matched-baseline experiments and failure modes; multiple seeds and bootstrap confidence intervals when applicable.
- Model export only if atomic space checks, provenance and audit record pass.

### R4 — mechanistic interpretability / alignment (research)

- Opt-in framework adapters for capture of selected activations and causal interventions on *small supported* models.
- Feature/head/path association views clearly distinguished from weight visualizations.
- Cross-family correspondence proposals are investigatory; direct transfer disabled without validated alignment.

## 8. Out of scope by default

- Hosting arbitrary user checkpoints; SaaS tenancy; automatic Hugging Face model download; weights as human-interpretable concepts; universal reasoning transfer; training/fine-tuning execution; model-wide full-precision SVD on laptops; live billions-of-parameters inference; unbounded prompts/activation caches; automated claims of causal behavior from passive visual differences; arbitrary .pt/.bin pickle import; editing users' original model files.

## 9. Key flows

**Flow A:** Open app -> Demo -> choose `mlp.up_proj.weight` -> inspect tile -> select ROI -> exact values -> export.  
**Flow B:** Import base and derived fine-tune -> lineage check -> ranked change table -> choose a top-ranked tensor -> switch Raw A / Raw B / Delta / Relative Delta -> compare aggregates -> save finding.  
**Flow C:** Base + PEFT LoRA -> parse config and matrix pairs -> verify target names, rank, scale and base lineage -> visual overlay for one ROI -> compare magnitude and sign.  
**Flow D (R3):** Create experiment -> select source delta and compatible target -> alpha grid -> run small held-out eval -> inspect evidence and control metrics -> export no original files modified.

## 10. First run and error states

- No files: show synthetic sample and button `Open local checkpoint`.
- Missing shard: show exact filename, do not silently skip; partial catalog may be visible but compare/exact stats disabled for affected tensors.
- Non-finite tensor: render with special marker, count NaN/+Inf/-Inf, exclude from finite-only statistics and document policy.
- Low disk: reading still allowed if no export/cache required; write operation blocked with estimate.
- Mismatch: red compatibility explanation; never attempt silent name/shape remapping.
- Too large query: offer lower-resolution preview and exact ROI request; never exceed budget and crash.
- Quantized GGUF: initially unsupported with explicit explanation rather than reinterpret packed integers as float weights.

## 11. Release quality gates

Each feature is not "done" until: acceptance story passes, reference numeric tests pass, memory and CPU budgets verified on CI/specified local test devices, cancel handling verified, accessibility checked, clear unsupported state implemented, provenance exported, and negative-path tests pass. Exit criteria are in documents 13-15.

## 12. Business and adoption assumptions

Start as OSS/devtool; validate the R0.5 change report with 5-10 practitioners before building the R1 UI, and again before adding commercial features. Potential later paid features: team-shared research reports, large multi-checkpoint comparisons, enterprise managed provenance or workflow integrations; none are necessary for R1. Instrument success only with local, opt-in analytics; never phone home.
