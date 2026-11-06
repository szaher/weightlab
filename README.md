# WeightLab — Build-Ready Product & Engineering Blueprint

**Version:** 1.2-final (frozen specification; see `CHANGELOG.md` and ADR-019)  
**Date:** 2026-11-05  
**Status:** Design specification; NOT an implementation and NOT a tested production application  
**Default product:** Local-first, CPU-only, resource-bounded LLM weight exploration, comparison, and eventually scientific experimentation.

## Specification freeze

This blueprint is frozen at **1.2-final**. Changes require a demonstrated implementation correctness or security blocker, a versioned ADR, updated contracts/fixtures/tests and a `CHANGELOG.md` entry (ADR-019). Run the specification checks with the commands in `docs/13_TEST_STRATEGY.md` section 10; CI runs them in `.github/workflows/spec-checks.yml`.

## Implementation status: Phase 0 (reader)

The `weightlab/` package implements Phase 0 only: a strict, read-only SafeTensors catalog with sharded index validation and exact, lossless ROI reads behind a Python interface. No tiles, API, CLI or UI yet. Gate G0 evidence lives in `evidence/g0/`.

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
