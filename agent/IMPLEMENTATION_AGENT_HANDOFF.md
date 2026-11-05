# WeightLab — Master Coding-Agent Handoff

Copy this file into a coding agent's context alongside the whole blueprint package. Treat the docs as product specification and implementation constraints, not as evidence that any implementation exists.

## Assignment

Build **WeightLab**, a cross-platform, local-first, CPU-only LLM checkpoint visualizer. First ship R1: SafeTensors/sharded model inspection, exact ROI, resource-bounded heatmap, truthful aggregate/sampling status, streaming statistics, strict A/B compatibility and checkpoint comparison, supported vanilla PEFT LoRA overlay, SQLite project persistence, local exports, and a polished offline demo. Advanced model inference, activation patching, GGUF quantization and transfer research are separate gated milestones; do not silently implement them as half-working placeholders.

## Instruction hierarchy

1. Security, scientific correctness and explicit resource budget requirements in `README.md`, `docs/03...`, `docs/04...`, `docs/05...`, `docs/11...`, `docs/12...`.
2. R1 acceptance and negative requirements in `docs/01...` and `docs/02...`.
3. `docs/09...` and `contracts/openapi.yaml` for API shapes.
4. `docs/13...` and `docs/14...` for release gates and milestone sequencing.
5. This handoff for working process.

When documents disagree, propose a brief explicit ADR change, do not choose silently. Never mark feature done because tests were skipped or because a heatmap image looks plausible.

## Engineering standards

- Python 3.11+, `uv`, `ruff`, `mypy`, `pytest`, `hypothesis`, FastAPI, NumPy, SafeTensors; React + TypeScript, Vite, test runner, Playwright. Record pinned versions in lockfiles; avoid adding heavy dependencies without measured need.
- Avoid Torch/transformers in core runtime. No CUDA requirement, no model execution, no cloud.
- Use structured typed entities, clear modules and contracts. Support Windows paths and Apple Silicon.
- Implement and verify **offline** behavior, local-only auth, no arbitrary code execution, no changing user model files.
- Build exactly one feature per ticket where feasible, commit after quality gate, provide short evidence summary and next ticket.
- Prefer robust standard libraries and mature upstream format parsers before writing custom parsers; any low-level mmap offset logic requires security/property tests.

## Explicit first 12 actions

1. Create repository tree from `docs/03_SYSTEM_ARCHITECTURE.md`, core Python package and frontend scaffold.
2. Add lockfiles, lint/format/type/test CI, small README and architecture navigation.
3. Adopt the provided deterministic fixtures: generate into `tests/fixtures/generated` with `scripts/generate_demo_fixtures.py`, verify with `scripts/check_demo_fixtures.py`; never hand-edit fixture files.
4. Implement SafeTensors metadata catalog with WeightLab's own header validation and sharded index validation (positioned-read reader, ADR-011), no full model loading.
5. Implement independent direct ROI reads with exact half-open coordinate convention.
6. Implement and test BF16/F16/F32 decoding (safetensors' NumPy backend cannot read BF16), nonfinite and 3D slicing; ensure all invalid/malicious fixtures fail closed with the expected error codes.
7. Implement memory measurement, huge synthetic file ROI proof and mid-read truncation test; publish G0 report.
8. Ship the R0.5 `weightlab diff` change report (FR-023, ADR-013) and collect practitioner feedback; then build local API auth (URL-fragment token, ADR-016), request validation, budget enforcement and project DB.
9. Implement numerics, ROI, tiles (direct L0-L3, exact pyramid, row-sampled above limit, ADR-012), job queue and cancellation.
10. Build Explore UI, offline demo, axes/legend/tooltip/ROI, keyboard and export.
11. Implement strict comparison with provenance gates, then vanilla LoRA ROI overlay; add golden tests.
12. Run all required gates on native macOS/Windows/Linux and an 8GB laptop, fix critical issues, document release limitations.

## Stop-and-fix criteria

- Unexpected RSS proportional to checkpoint size.
- Any path traversal or local API unauthenticated data exposure.
- Incorrect ROI axes/offsets or silent transpose.
- Incorrect LoRA adapter scale/rank mapping.
- `fan_in_fan_out` adapter applied without explicit transpose handling.
- Backend process killed by a file being truncated during a read (SIGBUS), or any BF16 value decoded incorrectly.
- Sampled overview presented as exact.
- Shape compatible but lineage unverified model automatically merged.
- Canceled job reported completed.
- Error causes original file mutation.
- Browser or runtime requiring GPU or network for R1.

## Required response after each milestone

```text
Milestone/Ticket:
Files changed:
What works (demonstrated, not assumed):
Commands/tests run and exact results:
Numerical reference validation:
Peak RSS / memory and CPU evidence:
Security/offline checks:
Known limitations:
Next single task:
```

## Definition of release readiness

Prove `AC-001` through `AC-015`, `SEC-01` through `SEC-12`, benchmark B01-B10 where feasible, all applicable CI and offline browser E2E. Unverified requirements must be listed as release blockers or clearly scoped exclusions. Produce packaged wheel with frontend, a downloadable offline demo and concise user guide.

## First prompt to execute

> Implement Phase 0 of WeightLab only, using the attached blueprint as the source of truth. Create repository skeleton and locked development tools; wire the provided fixtures and `check_demo_fixtures.py` into pytest; implement a safe, read-only tensor catalog and exact ROI reader using positioned reads with WeightLab's own BF16/F16 decoding (ADR-011). Include golden reference tests, the oversized-file memory benchmark and the mid-read truncation test. Do not build the change report CLI, React UI, inference or transfer features yet. Run tests and return a G0 evidence report with measured RSS, supported dtypes, format errors and the next single ticket (WL-024, the change report).
