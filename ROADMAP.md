# WeightLab — Visualization Roadmap

> **Status: proposed, not implemented.** This is the *living feature roadmap* for plots and analysis views beyond WeightLab's existing single-tensor heatmaps. It defines intended user outcomes, sequencing and acceptance criteria; it does **not** promise ship dates or imply features have shipped.
>
> [What works today](README.md#what-works-today) · [Frozen engineering delivery baseline](docs/14_DELIVERY_ROADMAP.md) · [Numerical methods](docs/05_NUMERICAL_METHODS.md) · [Performance budget](docs/11_PERFORMANCE_BUDGET.md)

## Product direction

**From a heatmap viewer to a trustworthy, local-first model-weight analysis workbench.**

Answer three progressively deeper questions:

1. **What is in this checkpoint?** Tensor layout, distribution, zeros, tails and exceptional values.
2. **What changed between checkpoints?** Where changes concentrate, their direction and scale, and how they are distributed.
3. **How do parameters evolve or organize?** Trends across training snapshots and structure in embeddings or matrix spectra.
4. **How does the model architecture relate to those changes?** Map supported Transformer components to checkpoint tensors; optionally investigate residual-stream behaviour using measured activations and controlled experiments.

Existing features: SafeTensors metadata/catalog, whole-tensor **heatmap overview**, exact ROI inspection, resource-bounded tile/pyramid engine, and **CLI/Python** checkpoint and supported LoRA change reports. **A graphical Compare/Adapters workflow does not exist yet.**

## Proposed delivery order

| Order | Visualization | User question | Prerequisite | Status |
| --- | --- | --- | --- | --- |
| **P0 / 1** | **Layer / role / tensor change rankings** (horizontal bars, sortable table) | Which layers or components changed the most? | Existing `weightlab diff` report + new Compare API/UI | **In progress** (API: ADR-020; UI planned) |
| **P0 / 2** | **Weight distribution histograms** (single tensor; optional A/B overlay) | Are weights sparse, heavy-tailed, shifted or anomalous? | New bounded scan / stats service | **Planned** |
| **P1 / 3** | **Delta heatmaps** (B − A; optionally supported LoRA effective delta) | Where in a matrix did values change? | Compatibility and lineage checks; derived tile/ROI source | **Planned** |
| **P1 / 4** | **Before/after scatter or 2-D density** | Is change concentrated around unchanged values? | Matched A/B chunks; bounded points or binned density | **Planned** |
| **P2 / 5** | **Singular-value spectrum / effective-rank diagnostics** | Which matrix directions dominate? | Bounded, optional numerical jobs | **Planned** |
| **P2 / 6** | **Checkpoint evolution timeline** | How did weight changes evolve during training? | Checkpoint series, consistent baseline and provenance | **Planned** |
| **P3 / 7** | **Embedding projections** (PCA first; UMAP optional) | How are selected embedding vectors arranged? | Bounded row selection, labels/tokenizer opt-in, dimensionality-reduction jobs | **Planned** |

**Priorities reflect suggested engineering order**, not a release commitment. The first four items extend the current model/CLI and visualization architecture; the later items need more numerical infrastructure and scientific review. A **parallel architecture and residual-analysis track** appears below; it does not displace these seven visualization commitments or assume an inference runtime is installed.

## Navigation model (proposed)

```mermaid
flowchart LR
    A[Project / checkpoints] --> B[Explore one checkpoint]
    A --> C[Compare compatible checkpoints]
    B --> D[Tensor catalog]
    D --> E[Heatmap + exact ROI / shipped]
    D --> F[Weight histogram / P0]
    C --> G[Layer and role rankings / P0]
    G --> H[Tensor-level delta heatmap / P1]
    G --> I[Before-after density / P1]
    D --> J[Spectral diagnostics / P2]
    C --> K[Checkpoint timeline / P2]
    D --> L[Embedding projection / P3]
```

Target interaction: **model → layer / role → tensor → distribution / visualization → exact values**. A plot click should preserve checkpoint identity, tensor key, coordinates, dtype, coverage and quality flags when linking to the existing Exact ROI explorer.

## Feature specifications and acceptance criteria

### P0.1 — Layer, role and tensor change dashboard

**Value:** A graphical answer to "What changed during fine-tuning?" without reading CSV files.

- Reuse per-tensor / per-layer / per-role delta L2, relative L2, RMS, cosine and counts from the existing CLI analysis rather than inventing new calculations.
- Rank with sortable bars plus a keyboard-accessible table; filter by layer, role and tensor; drill into the selected tensor.
- Show matched, missing, tied, shape/dtype-incompatible and unmeasured/withheld items separately. Surface the **X / V / L / S** compatibility verdict and source lineage caveats before rankings.
- Distinguish whole-layer rollups from averages and from per-tensor results. Label each metric and its direction (**B − A** where applicable).
- **Acceptance:** deterministic fixture pairs produce the same ranking, values, compatibility flags and completeness status as `weightlab diff`; incomplete reports are never shown as complete; runs stay local and within budgets.

**Suggested first implementation PR:** read-only Compare results view over the existing diff engine, with clear eligibility and incomplete-result handling. This is a *new UI feature*, not the already-shipped CLI.

### P0.2 — Streaming weight distribution histogram

**Value:** Reveal range, distribution shape, zero concentration, outliers and non-finite counts in a checkpoint without losing the underlying numeric truth.

- Single-tensor view first, followed by optional base/derived overlay. Support finite-value histogram counts; report zeros, NaN, +Inf and −Inf separately.
- Show bin edges, counts, source dtype, scanned vs total element counts, quality/provenance and whether the values are exact/estimated. Define binning for constant tensors and extreme ranges explicitly.
- Exact whole-tensor mode must be an explicit, bounded, cancellable scan (with bytes/time estimate); a sampled preview must be labeled and reproducible via a seed/policy. Do **not** imply the existing heatmap tiles alone supply an exact value distribution.
- Allow linear/log-count y-axis and a zero-centered view where appropriate, without silently excluding overflow/extreme values.
- **Acceptance:** compare exact histograms with independently computed counts on synthetic BF16/F16/F32/F64 and integer fixtures, including negative zero, non-finite and high-precision edge cases; cancellation yields no misleading complete result.

### P1.1 — Delta heatmap and supported LoRA overlays

**Value:** Go from a ranked changed tensor to the spatial pattern of its update.

- For eligible matched tensors, render **B − A** as a new derived tile source; make sign, colour scale, coordinate axes and coverage labels obvious. Offer exact paired ROI inspection.
- For supported PEFT LoRA, optionally render the **effective** update `scale × (B_lora @ A_lora)`, explicitly distinct from adapter factor weights and from base/derived comparison.
- Preserve compatibility, cost and unsupported-module gates; no silent rebasing of independently trained/unaligned models, no unsupported adapters presented as valid.
- Never call float32 aggregate tile values "lossless"; raw exact paired values/deltas must expose their own numeric precision policy.
- **Acceptance:** golden synthetic fixture deltas and supported LoRA blocks match independent reference values; transpose, slice, NaN/Inf, cache invalidation and cancellation all have tests.

### P1.2 — A/B scatter and density plot

**Value:** See whether most parameters remain close to their original values or deviate together.

- Plot x = base parameter and y = derived parameter with an **unchanged y=x reference line**; show tensor name, matched coverage and compatibility caveats.
- Avoid a browser scatter of millions of points. Use **bounded seeded samples** and/or a streamed 2-D binned density with explicit bin counts, bounds and overflow handling.
- Permit selection/drilldown to individual weights **only** when the plotted sample retains raw coordinates; density bins represent many weights and must not pretend to identify a single cell.
- **Acceptance:** synthetic equal tensors fall on the diagonal; changed fixtures show expected offsets; sampling repeats with the same seed; non-finite and out-of-range counts remain visible and testable.

### P2.1 — Spectral diagnostics

**Value:** Investigate structure and effective rank in dense matrices or updates, not just element magnitudes.

- Display singular-value spectrum, decay and chosen effective-rank metrics for selected 2-D tensors and eligible deltas; explain normalization and metric definitions.
- Start with **small-matrix exact SVD** subject to preflight size limits. For larger matrices, an *opt-in*, bounded, reproducible approximate method may be added with error/coverage disclosure; never label approximation as exact.
- Show compute/IO estimates, progress and cancellation; no GPU required and no full-size dense matrix materialization by default.
- **Acceptance:** exact results match an independent numerical reference within a documented tolerance; approximate runs disclose method, seed and error criteria; oversized requests are refused or require deliberate user opt-in.

### P2.2 — Checkpoint evolution timeline

**Value:** Track model changes throughout fine-tuning rather than only comparing two snapshots.

- Import/associate an **ordered sequence of checkpoints**; choose an explicit baseline, show change metrics by step/layer/role over time, and offer drilldown to a checkpoint pair.
- Timestamps/training steps must come from supplied metadata or user-provided labels—never inferred from filenames silently.
- Require compatible tensor identities and report per-step missing tensors, dtype differences and unverifiable lineage; distinguish delta-to-base from delta-to-previous-step.
- Compute on demand with a bounded cache and reuse the existing diff engine where compatible.
- **Acceptance:** synthetic 3+ snapshot sequence reproduces known trends and correct baselines; missing steps/tensors appear as gaps rather than zeroes; no erroneous causal claims about training improvement.

### P3.1 — Embedding projections (PCA first, UMAP optional)

**Value:** Explore geometry of selected token embedding rows and how their positions may shift.

- Start with **PCA** on an explicitly selected or seeded sampled subset of embedding rows; show explained variance and exact subset/coverage. UMAP may be an optional dependency/compute mode, not part of the core install.
- Add optional vocabulary/tokenizer mapping **only if supplied locally**; no automatic download or implied semantic labels. Preserve row-to-token provenance.
- Label projections as lossy views of high-dimensional vectors. Display sampling/method/seed, and caution that 2-D distances and clusters may be misleading.
- **Acceptance:** deterministic fixture embeddings yield repeatable PCA coordinates up to documented sign conventions; tokenizer alignment and invalid indexes fail closed; UI remains responsive under the configured resource budget.

## Architecture, residual streams and mechanistic analysis (video-inspired track)

> **Status: proposed feature backlog, not implemented or scheduled.** Inspired by the video on ResNets, identity mappings, optimization and residual streams: [video](https://www.youtube.com/watch?v=QgH9sr7G13Q). These ideas motivate potential WeightLab features; they are **not** evidence that inspecting weights reveals model behaviour. The register-token example concerns Vision Transformers, not a universal LLM feature.

### Why a separate track?

The existing visualization roadmap focuses on describing tensor values and checkpoint differences. This track adds an **architecture-aware path** from model components to the same proven numeric views, followed by **opt-in execution-based research**. It extends rather than duplicates the seven visualization items above and the existing Phase 4/5 research intentions in [the delivery baseline](docs/14_DELIVERY_ROADMAP.md).

**Three evidence tiers, never conflated:**

1. **Weights only (A01–A05):** Which checkpoint tensors belong to attention, MLP and normalization components, and where do updates concentrate? Requires known architecture/metadata; **does not** measure residual activations, gradient flow or causal importance.
2. **Measured runtime signals (R01–R06; O01–O04):** What do hidden states, branch outputs, gradients and losses do on specified inputs/training runs? Requires an optional, supported runtime or imported trace data, with provenance.
3. **Controlled interventions (E01–E04):** How does modifying a layer, activation or weight update change measured outcomes? Requires matched controls and repeatable evaluation; results are scoped to the actual intervention and dataset.

Residual skip/identity paths generally have **no trainable weight tensor**. An architecture drawing may depict them, but assigning weight-diff magnitude or a functional contribution to a bare identity path would be incorrect. Transformer block ordering varies (pre/post norm, parallel branches, scaling and model-specific logic): refuse unsupported architectures rather than presenting generic wiring as verified.

### Candidate feature inventory (20 features)

Every ID below is **proposed**, with no implementation claim. The priorities are relative within this track.

| ID | Feature | Outcome / primary view | Data needed | Priority |
| --- | --- | --- | --- | --- |
| **A01** | Interactive Transformer architecture explorer | Navigate blocks, branches, skip paths, and component links | Supported local architecture descriptor and tensor catalog | **Near-term** |
| **A02** | Tensor-to-component mapping | Trace exact checkpoint tensor keys to modules, including unmapped/ambiguous items | Architecture-specific, validated mapping rules | **Near-term** |
| **A03** | Identity, skip and normalization-path view | Show pre/post-norm, residual additions and known component ordering without inventing weights on skips | Verified architecture/configuration metadata | **Near-term** |
| **A04** | Architecture-linked checkpoint deltas | Open A/B weight-diff rankings, tensor heatmaps and exact ROI from diagram components | Existing compatibility-gated diff engine; P1.1 when delta tiles are ready | **Near-term** |
| **A05** | Attention versus MLP update profiles | Compare update concentration across layers, roles and projections | Existing change reports and validated mappings | **Near-term** |
| **R01** | Selective residual-stream activation capture | Record selected hidden states and branch outputs on fixed inputs | Opt-in model execution/hook adapter or trusted imported trace | Later |
| **R02** | Residual branch contribution profiles | Show branch-output norm relative to residual-stream norm and vector alignment | R01 and architecture-aware hook semantics | Later |
| **R03** | Representation evolution | Plot hidden-state norms, directions and similarity across depth | R01, selected positions and bounded traces | Later |
| **R04** | Token-by-layer activation heatmaps | Inspect changes per selected token/position without conflating token and layer axes | R01 and aligned token IDs/positions | Later |
| **R05** | Paired base/fine-tuned activation comparison | Measure activation differences for identical tokenized inputs | R01, compatible architecture/tokenizer, paired execution settings | Later |
| **R06** | High-norm/outlier token diagnostics | Surface and inspect exceptional activations, with honest sample/trace coverage | R01 and non-finite/outlier policies | Later |
| **O01** | Gradient flow by layer | View gradient norms and propagation during a known training step | Opt-in autograd/training trace or imported gradients | Research |
| **O02** | Gradient correlation / shattering diagnostics | Compare gradient directions across inputs, depth and steps | O01, controlled multi-input gradients and explicit statistical definitions | Research |
| **O03** | Depth versus trainability experiments | Compare optimization/validation curves for controlled network depths | External or opt-in training runs, matched setup and logs | Research |
| **O04** | Loss-landscape slices | Inspect limited, reproducible loss evaluations along defined weight-space directions | Opt-in execution, evaluation dataset and parameter perturbation policy | Research |
| **E01** | Layer or branch ablation | Compare matched evaluation outcomes when a supported path is intervened on | Supported runtime, safe hook contract and controls | Research |
| **E02** | Activation patching | Restore/replace chosen activations across controlled runs | R01, same-model/token alignment and intervention engine | Research |
| **E03** | Weight-update scaling | Evaluate bounded alpha-scaled checkpoint or LoRA deltas | Existing Phase 4 virtual-update plan, compatibility checks and evaluation | Research |
| **E04** | Behaviour-linked evaluation reports | Compare task metrics and controls alongside intervention provenance | E01–E03 and reproducible benchmarks | Research |
| **V01** | Vision Transformer register-token analysis | Identify supplied register tokens and examine runtime storage/activation patterns | Separately supported ViT architecture/runtime; not assumed for text LLMs | Optional extension |

### Sequence and dependencies

| Package | Proposed scope | Existing work to reuse | Acceptance gate before implementation is called complete |
| --- | --- | --- | --- |
| **ARCH-1: architecture map** | A01–A03 | Tensor catalog, Explore, supported real-Qwen validation | Explicit allowlist (Qwen2.5 first), tested mappings, truthful residual/norm diagram, unresolved tensors shown, no user-supplied code execution |
| **ARCH-2: weight-change navigation** | A04–A05 | P0.1 Compare, P1.1 delta heatmaps when available, CLI/Python diff metrics | Rankings match verified reports; B − A and lineage displayed; map → tensor → exact ROI navigation works |
| **TRACE-1: optional activation workbench** | R01–R06 | Phase 5 selective capture and existing numeric plotting practices | Same-prompt repeatability, hook/module identity and token positions, bounded samples/storage, offline/opt-in execution, measured norm semantics |
| **OPT-1: training diagnostics** | O01–O04 | Phase 4 research infrastructure and trace import where available | Known training/eval provenance, deterministic tests, no claim that weight deltas are gradients, computational cost and uncertainty reported |
| **INT-1: interventions** | E01–E04 | Phase 4 alpha sweep/controls; Phase 5 activation/weight intervention | Baseline/treatment/negative controls, consistent seeds and prompts, paired metrics, explicit causal limits |
| **VIT-1: vision extension** | V01 | Future architecture plugin infrastructure | Supported ViT/register-token semantics verified; no invented language-model register mapping |

**Suggested order:** complete the already-open Compare dashboard PR #15 and keep P0.2/P1.1 commitments; then add ARCH-1 and ARCH-2 as small, separate PRs. TRACE-1 depends on an explicit optional inference/trace design, and OPT-1 / INT-1 are not alpha-release gates. VIT-1 remains demand-driven. No delivery dates are implied.

```mermaid
flowchart TD
    C["Existing tensor catalog + Explore"] --> A["ARCH-1: A01-A03 architecture mapping"]
    D["Existing diff + Compare dashboard"] --> B["ARCH-2: A04-A05 update navigation"]
    A --> B
    A --> R["TRACE-1: R01-R06 optional activations"]
    R --> E["INT-1: E01-E04 controlled experiments"]
    D --> E
    T["Optional training traces / eval runtime"] --> O["OPT-1: O01-O04 optimization diagnostics"]
    T --> E
    A -. "separate ViT support" .-> V["VIT-1: V01 register tokens"]
```

### Acceptance rules and non-goals

- **No fabricated interpretation:** layer/role weights and delta norms describe parameters, not residual-stream effects or reasoning improvement. An observed activation difference is association; stronger causal claims require controlled intervention and matched evaluation.
- **No invented architecture:** tensor naming heuristics are provisional. Require an explicit supported model family/config revision; clearly distinguish mapped, ambiguous, missing, tied and unsupported components. Represent identity skips as operations, not parameter tensors.
- **No silent runtime dependency:** ARCH-1/2 must remain local-first, read-only, CPU-capable and usable offline without model execution. R/O/E features require a separate opt-in runtime or trace-import design; larger hardware needs must not weaken the base laptop promise.
- **Resource and privacy controls:** preflight memory/compute/storage limits, deterministic bounded capture and sampling, cancellation, no model auto-download or arbitrary remote code execution, and no trace leakage in exports.
- **Scientific rigor:** report inputs, tokenizer/model/checkpoint identity, architecture mapping version, hooks, dtype/precision, coverage, seed, training/eval setup, controls and limitations. For R02, ratios such as norm(branch output) / max(norm(residual stream), epsilon) are **descriptive**, not attribution scores; pre-norm and post-norm interpretations differ.
- **Change control:** new APIs, persistent traces, runtime plugins and numeric methods need ADRs, contracts, reference fixtures, cross-platform evidence and security review under the existing spec process. Only add these requirements when a feature is accepted for implementation.

## Cross-cutting definition of done

Every new visualization must satisfy all of these before being marked **Shipped**:

1. **Truthfulness:** clear source checkpoint(s), tensor, dtype, axes, method and operation; distinguish exact coverage, statistical estimation, sampled preview and numeric rounding.
2. **Correctness:** independent numerical reference with adversarial fixtures (BF16, precision edges, zeros, NaN/Inf, huge tensors, lineage mismatch where applicable). Never suppress withheld or incomplete results.
3. **Resource bounds:** preflight compute/bytes/memory budget, streaming or bounded sampling, cancellable heavy jobs, no unbounded browser points or DOM elements. Reuse cache invalidation when files change.
4. **Privacy/security:** local-first, read-only inputs, no automatic network/model download, no token or path leakage in visual exports.
5. **Usability/accessibility:** keyboard navigation, legends and tables/alternate text, clear selection-to-ROI behavior, screenshots and browser E2E evidence.
6. **Evidence:** publish actual runtime/RSS/accuracy measurements on fixtures and at least one real model where appropriate; do **not** claim the 8 GiB laptop target without testing it.

## Scope and change control

This roadmap records *potential implementation work* and user value, **not new requirements silently added to the frozen 1.2-final specification**. Any new API schema, numeric formula, compatibility rule, persistent data shape or budget must go through the appropriate versioned ADR/contract/fixtures/tests/change-log process before implementation (see [ADR-019](docs/16_DECISIONS_RISKS_AND_OPEN_QUESTIONS.md) and [design change control](docs/16_DECISIONS_RISKS_AND_OPEN_QUESTIONS.md#5-change-control)).

**Next action:** P0.1's API is defined and implemented ([ADR-020](docs/16_DECISIONS_RISKS_AND_OPEN_QUESTIONS.md): `comparison_report` job and `GET /comparisons/{id}/report`, numerically identical to `weightlab diff`). Next, build the Compare screen on it: compatibility verdict first, a layer × role grid coloured by the selected metric, sortable bars and an accessible table, with drill-down to the existing tensor views. Then move to histogram scans.
