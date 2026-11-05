# 15 — Implementation Backlog and Requirements Traceability

Tickets ordered for a solo developer/coding agent. Estimates are **relative complexity**, not delivery time. Every ticket must pass the functional acceptance ID and its security/performance dependencies.

| Ticket | Priority | Requirement refs | Deliverable | Complexity | Done when |
|---|---|---|---|---|---|
| WL-001 | P0 | FR-063 | Repo, CI, offline build, lockfiles | M | Clean install without internet after deps installed |
| WL-002 | P0 | FR-003,005 | Validated SafeTensors parser, positioned-read reader, BF16/F16 decode | L | Malformed fixtures fail with expected codes; AC-013, AC-014 |
| WL-003 | P0 | FR-002,005 | Sharded model index support | M | Missing/path traversal shard errors |
| WL-004 | P0 | FR-016,017,018 | ROI reader 0D-4D with explicit axes | L | Exact fixture parity |
| WL-005 | P0 | FR-019,020 | Streaming statistics engine | L | Exact independent ref and cancellation |
| WL-024 | P0 | FR-023,030 | `weightlab diff` change report CLI (R0.5) | M | Golden reports on fixtures; real-checkpoint run in budget; practitioner feedback recorded |
| WL-006 | P0 | FR-011,012,013 | Tile engine: direct L0-L3, exact pyramid, row-sampled preview above limit | XL | Odd-edge aggregations, pyramid parity and resource budget correct |
| WL-007 | P0 | FR-001,008 | SQLite project/import persistence | M | Relink/restart and migrations pass |
| WL-008 | P0 | FR-065,066 | Job queue, budget enforcement | L | Saturation, cancel, resource errors pass |
| WL-009 | P0 | FR-011-020 | Explore UI + canvas axes | XL | Demo screen passes E2E |
| WL-010 | P0 | FR-004,064 | localhost auth + security boundary | L | SEC-01/02/03/04 pass |
| WL-011 | P0 | FR-030-036 | Comparison engine + compatibility | XL | B-A and mismatch fixtures pass |
| WL-012 | P0 | FR-040-044 | PEFT vanilla LoRA ROI/overlay | XL | A/B rank, `rank_pattern`, rsLoRA, `fan_in_fan_out` and DoRA-reject tests pass |
| WL-013 | P0 | FR-060-063 | PNG/JSON export, bookmark | M | Paths redacted and manifest stable |
| WL-014 | P0 | FR-063 | Bundled synthetic demo/tutorial | S | First-run no download |
| WL-015 | P0 | all R1 | Cross-platform & 8GB perf evidence | L | Benchmark and G0-G5 reports |
| WL-016 | P1 | FR-021,022,037 | Layer dashboard, histogram | L | Correct aggregation and legends |
| WL-017 | P1 | FR-009,037 | Checkpoint timeline and trend | L | Valid explicit steps and sorted runs |
| WL-018 | P1 | FR-038,045 | Known architecture mapper / adapter compare | L | Conservative confidence and tests |
| WL-019 | P1 | FR-050-052 | Virtual model update experiment plan | XL | No original writes; compatibility gate |
| WL-020 | P1 | FR-053-055 | Opt-in isolated evaluation | XL | Paired math/control/negative cases |
| WL-021 | P1 | FR-056,067 | Reproducible research report | M | Provenance, warnings, precise evidence |
| WL-022 | P2 | FR-057-059 | Activation capture and patching plugin | XL | Small-model causal study and memory bound |
| WL-023 | P1 | FR-010 | GGUF float reader, then quantized | XL | Golden decoder for each block type |

## One example story broken into acceptance tests

**WL-006 — Heatmap tiles**

- Given the 257x263 fixture, Level 0 tile (0,0) returns exactly first 256x256 cells.
- Given Level 0 tile (1,1), it contains 1x7 cells and correct bounds.
- Given Level 1 tile (0,0), each output is aggregation of exact 2x2 original block; edge blocks count-weighted.
- Given Level >= 4 for a tensor under `pyramid_exact_max_mib`, outputs come from the pyramid and equal an independent dense block aggregation (L3 edge block in `expected_results.json` as anchor).
- Given missing/nonfinite cell, quality mask and per-block counts preserve it.
- Given a tensor above `pyramid_exact_max_mib`, the first tile is a labeled row-sampled preview; an exact result exists only after an explicitly queued pyramid job completes.
- Given repeated request with same seed and file identity, outputs match.
- Given a changed file, stale cache cannot be returned as current.
- Given immediate pan to other tensor, old tile work is canceled and GUI stays responsive.

## R1 traceability minimum

| Requirement family | Primary tests | Evidence |
|---|---|---|
| Import/read | reader unit, malformed and shards integration | pytest log, fixture catalog |
| Change report | ranked-table golden tests (`tiny_ft`: 10 changed cells, all in `up_proj`) | JSON/CSV/Markdown report |
| Heatmap/ROI | coordinate / tile property, Playwright | screenshot, golden values |
| Stats | independent NumPy fixture, nonfinite | reference JSON |
| Compare | delta, shape/lineage matrix | compatibility report |
| LoRA | rank, scale, module mapping | known ROI diff |
| Privacy/security | SEC-01...SEC-12 | security CI log |
| Resource limits | B01...B10 | benchmark report + real 8GB host |
| Offline UX | first-run Playwright | offline E2E report |
| Export | JSON schema, PNG axes/sampling | exported artifact snapshot |

## Agent ticket discipline

Each ticket commits minimal vertical slice, proves its own acceptance and does not merge unrelated refactors. Agent must write implementation notes listing commands run, test outcomes, memory observed, known defects and next actionable ticket. Never claim green on skipped or nonexistent tests.
