# WeightLab — Build-Ready Product & Engineering Blueprint

**Version:** 1.2-final (frozen specification; see `CHANGELOG.md` and ADR-019)  
**Date:** 2026-11-05  
**Status:** Design specification; NOT an implementation and NOT a tested production application  
**Default product:** Local-first, CPU-only, resource-bounded LLM weight exploration, comparison, and eventually scientific experimentation.

## Specification freeze

This blueprint is frozen at **1.2-final**. Changes require a demonstrated implementation correctness or security blocker, a versioned ADR, updated contracts/fixtures/tests and a `CHANGELOG.md` entry (ADR-019). Run the specification checks with the commands in `docs/13_TEST_STRATEGY.md` section 10; CI runs them in `.github/workflows/spec-checks.yml`.

## Implementation status: Phase 0 (reader)

The `weightlab/` package implements Phase 0 only: a strict, read-only SafeTensors catalog with sharded index validation and exact, lossless ROI reads behind a Python interface. No tiles, API or UI yet. Gate G0 evidence lives in `evidence/g0/`.

```bash
uv sync --locked                     # Python 3.12+, pinned toolchain
uv run pytest                        # full suite, includes the 12 GiB sparse-file RSS test
uv run ruff format --check . && uv run ruff check . && uv run mypy
uv run python scripts/g0_evidence.py --out evidence/g0   # writes G0_<os>.md
```

```python
from weightlab import open_checkpoint
from weightlab.analysis.roi import encode_roi

with open_checkpoint("/models/my-model", selected_root="/models") as ck:
    print(ck.metadata()[:3])
    roi = ck.read_roi_raw("model.embed_tokens.weight", 0, 4, 0, 8)
    print(encode_roi(roi).to_json())   # lossless typed values (doc 09 section 2)
```

## Implementation status: Phase 0.5 (change report, WL-024)

`weightlab diff` compares two checkpoints, or a base and a PEFT LoRA adapter, on top of the Phase 0 reader. It provides strict compatibility classes (X/V/L/S), exact streaming metrics (delta L2/RMS, relative L2, cosine, max-abs, non-finite counts), layer and role rankings, and JSON/CSV/Markdown exports with provenance. Memory stays bounded, inputs are read-only, and no model is run. No frontend, inference or transfer yet. The reference is `docs/cli/weightlab-diff.md`; Gate G0.5 evidence, example reports and known limitations are in `evidence/g0.5/` (macOS only; Windows validation is deferred and not passed).

```bash
uv run weightlab diff base/ derived/ --out report/              # report.json, tensors/layers/roles.csv, report.md
uv run weightlab diff base/ --adapter lora/ --json lora.json     # LoRA effective delta vs base
uv run python scripts/g05_evidence.py --out evidence/g0.5        # tests + diff benchmark -> G0.5_<os>.md
```

## Implementation status: Phase 1 (Tasks 1–3) and practitioner evaluation kit

Three pieces are built:

- **Database (`weightlab/db/`):** the SQLite project database with checksummed migrations (Phase 1 task 1).
- **Local API (`weightlab/api/`):** the FastAPI routes of `contracts/openapi.yaml` 0.3.0, with strict request schemas, a loopback-only security boundary and generated contract tests (task 2).
- **Browser UI (`frontend/`, served from `weightlab/ui/static`):** the first usable application (task 3). Create projects, import a local SafeTensors checkpoint by absolute path (read in place, never uploaded), relink moved models, and search, filter, sort and page the tensor catalog on the server, with a metadata inspector.
- **Exact ROI explorer (Phase 1 task 4A, `frontend/src/features/roi/`):** select a tensor to see an exact window of its values (64 × 64 by default, never more than `roi_max_cells`) as a Canvas heatmap and as an accessible table. It supports pan, zoom, jump-to-cell, transpose, row/column axis and fixed-index slicing for rank > 2, colour scales (auto for the visible region, sign-symmetric, fixed) and exact copy. Values come from `POST /tensors/roi` unchanged: −0.0 keeps its sign, I64/U64 stay decimal strings, NaN/±Inf are labelled patterns.

`weightlab start` serves the API and the UI on the same loopback origin. It prints `http://127.0.0.1:<port>/#token=...`, and the page moves the per-launch token into `sessionStorage` and removes it from the address bar. The wheel ships the production build, so Node.js is needed only to rebuild the UI, never to run it.

- **Tile engine foundation (Phase 1 task 4B, `weightlab/analysis/tile.py`, `tile_codec.py`):** exact heatmap tiles by direct bounded reads (origin-anchored 256 x 256 tiles, `block = 2^level`, count-weighted mean / mean_abs / RMS from sums of squares / min / max over finite values, NaN/+Inf/-Inf counts, overflow and float32-range withholding), the frozen binary tile body and its JSON form, and canonical SHA-256 tile identities.
- **Exact pyramids, sampling, jobs and caches (Phase 1 task 4B, PR #7):** `POST /api/v1/tiles` implements the frozen quality decision table for `kind=tensor` sources: direct exact reads (L0-L3 and small tiles), exact tiles from a streaming pyramid (`weightlab/analysis/pyramid.py`, record format v1: 56 B base / 80 B coarse records), `202` + a deduplicated `pyramid_build` job when the pyramid is not ready, and labelled row-sampled previews (`weightlab/analysis/sampling.py`, stride or seeded SplitMix64 rows) only for sources above `pyramid_exact_max_mib`. `/api/v1/jobs` runs bounded, cancellable `pyramid_build` jobs (`weightlab/jobs.py`); `weightlab/cache.py` holds the memory tile LRU and the disk pyramid LRU. Delta / LoRA tiles and other job kinds answer 422, never a substitute. Evidence: `evidence/phase1-pyramids/`.

- **Whole-tensor overview (Phase 1, PR #8, `frontend/src/features/overview/`):** Explore opens every 2D tensor (and every 2D view of a higher-rank one) on an **Overview** tab: a multiresolution Canvas heatmap assembled from binary `POST /tiles` answers, with pan (drag, Shift+arrows), zoom toward the pointer (Ctrl/⌘/Alt + wheel, buttons, +/−), fit/reset, automatic or explicit level, the five aggregations, and block or rectangle selection with raw half-open bounds. Every picture says whether it is **EXACT coverage** or a **ROW-SAMPLED preview** (rows per block, policy, seed), that values are float32 visualizations, and which aggregation and block size it shows. A `202` shows the pyramid build's progress and I/O with a Cancel button; sources above `pyramid_exact_max_mib` get a labelled sampled preview, a seed control, and an explicit, costed "Build the exact overview" action. **Inspect exact values** opens the **Exact ROI** tab on exactly the selected raw bounds and view (regions above `roi_max_cells` are never truncated silently). Evidence: `evidence/phase1-overview/`.

The overview's cells are aggregates; exact numbers only ever come from Exact ROI. Stats jobs, exports, Compare and Adapters are not built yet: export routes answer 404 `deferred`, other job kinds 422, and their navigation entries are marked "planned".

The Phase 0.5 practitioner evaluation (doc 14, Phase 0.5 task 6) has a self-contained kit in `evaluation-kit/`:

- installation instructions;
- three reproducible examples, checked against generator values;
- a 10-minute questionnaire;
- a structured feedback-summary template.

No feedback has been collected yet, and the G0.5 practitioner item stays open.

Further reading:

- review guide, spec gaps and limitations: `evidence/phase1/README.md`;
- UI evidence, screenshots and limitations: `evidence/phase1-ui/README.md`.

```bash
uv run weightlab start                                   # open the printed http://127.0.0.1:<port>/#token=... link
uv run python evaluation-kit/run_examples.py             # the kit's three examples, self-checking
uv run python scripts/build_evaluation_kit.py            # -> dist/weightlab-evaluation-kit-<version>.zip
uv run python scripts/phase1_evidence.py --out evidence/phase1

cd frontend && npm ci                                    # UI development (Node 22+; build-time only)
npm run lint && npm run typecheck && npm test            # ESLint, strict tsc, Vitest
npm run build                                            # -> weightlab/ui/static (committed; CI checks it is current)
npx playwright install chromium && npm run e2e           # browser E2E against a real `weightlab start`
```

## Read this first

This package is designed to hand to an engineer or coding agent without leaving important product, numerical, systems, or delivery decisions implicit. It does **not** guarantee a bug-free first attempt: the acceptance gates and automated tests make failures observable before release.

### Source of truth and reading order

1. `docs/01_PRODUCT_VISION_PRD.md` — users, jobs, outcomes, release boundaries, measurable acceptance.
2. `docs/02_FUNCTIONAL_REQUIREMENTS.md` — exhaustive requirements (IDs, priorities, user stories, negative cases).
3. `docs/03_SYSTEM_ARCHITECTURE.md` — precise technical architecture and decisions.
4. `docs/04_FILE_FORMATS_AND_TENSORS.md` — safe reads, model format support, shard index, dtype and compatibility.
5. `docs/05_NUMERICAL_METHODS.md` — exact maths, visual semantics, statistics, error bounds.
6. `docs/06_TRANSFER_RESEARCH_PROTOCOL.md` — task vectors, LoRA transfer, evidence and controls.
7. `docs/07_ACTIVATIONS_AND_EVALUATION.md` — optional costly experiments and causal interpretation.
8. `docs/08_UX_UI_DESIGN.md` — workflows, screen-level specs, accessibility and sketches.
9. `docs/09_API_CONTRACT.md` + `contracts/openapi.yaml` — REST and job contracts.
10. `docs/10_PERSISTENCE_DATA_MODEL.md` — durable entities, schema, provenance, migrations.
11. `docs/11_PERFORMANCE_BUDGET.md` — memory/CPU/IO constraints and benchmark design.
12. `docs/12_SECURITY_AND_PRIVACY.md` — threat model and secure local operation.
13. `docs/13_TEST_STRATEGY.md` — golden fixtures, correctness, security and E2E.
14. `docs/14_DELIVERY_ROADMAP.md` — staged deliverables, dependencies and release gates.
15. `docs/15_BACKLOG_AND_TRACEABILITY.md` — tickets with definition of done.
16. `docs/16_DECISIONS_RISKS_AND_OPEN_QUESTIONS.md` — architecture decision records and explicit trade-offs.
17. `docs/17_RESEARCH_AND_REFERENCES.md` — evidence and technical citations.
18. `docs/18_OPERATIONS_AND_PACKAGING.md` — install, release, observability and support.
19. `agent/IMPLEMENTATION_AGENT_HANDOFF.md` — directly usable coding-agent instruction sequence.
20. `contracts/*.json` and `contracts/examples/*.json` — experiment schema, sample manifest, positive and negative contract examples (validated by `scripts/check_contracts.py`).
21. `examples/*.md` — user-facing walkthroughs and acceptance scenarios.

### Product scope at a glance

| Release | Core capability | Runtime requirement |
|---|---|---|
| R0, fixture-first | Generate fixture, open SafeTensors model, metadata / tensor list | CPU, no GPU |
| R0.5, change report | `weightlab diff`: ranked per-tensor/per-layer change table, compatibility report, JSON/CSV/Markdown; validate with practitioners | CPU, no UI |
| R1, first useful release | Single tensor heatmap, precise ROI, exact/sample statistics, compare compatible checkpoints and LoRA overlays, local persistence | CPU, 8 GB laptop target |
| R2, analysis | Layer dashboards, training checkpoint timelines, similarity maps, enhanced aggregate plots and portable exports | CPU; bounded jobs |
| R3, research | Task-vector virtual overlays, alpha sweep, optional local evaluation, intervention experiment records | Optional CPU inference; separate budget |
| R4, advanced research | Activation capture, patching and supported cross-family alignment experiments | Optional heavier dependencies; explicit opt-in |

**Do not ship R3/R4 as if they were proven or fully supported in R1.**

### Nonnegotiables

- 100% local operation by default; no background network calls or telemetry without explicit opt-in.
- Model bytes never leave the local machine unless the user explicitly exports them.
- Full model weights are never loaded into process RAM by default.
- Tensor metadata inspection works without PyTorch, transformers, or a GPU.
- All large computations are chunked, interruptible, and governed by explicit budgets.
- Never label a sampled/statistical estimate as exact.
- Cross-checkpoint numerical comparisons are gated by explicit compatibility checks and clear interpretation warnings.
- Raw weights, LoRA deltas, activations and model behavior are distinct scientific objects.
- Ship a reproducible synthetic dataset and demo; no paid cloud account required.

### Suggested implementation order

Read `agent/IMPLEMENTATION_AGENT_HANDOFF.md`, implement one milestone at a time, run tests and benchmarks at each gate, and publish evidence. Never bypass a failed gate by weakening a test or disabling validation.

### License guidance

The documentation is provided as a design artifact. Before publishing an implementation choose an explicit repository license and review dependency/model/dataset licensing. A permissive code license such as Apache-2.0 is an **option**, not selected here.
