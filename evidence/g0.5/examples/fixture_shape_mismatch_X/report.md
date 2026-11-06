# WeightLab change report

- Mode: **checkpoint_pair**; orientation B − A (A = base `tiny_base.safetensors`, B = derived `shape_mismatch.safetensors`)
- Complete: **NO**

## Compatibility

- Class **X**; base file identity verified: False (none)
- Lineage claim source: none; trust: none; evidence: none; transfer eligibility (informational): T5
- no lineage evidence: shape-compatible only; deltas may be meaningless (permutation/alignment)
- structurally incompatible (missing tensors, shape or dtype-family mismatch); matched tensors are shown as unverified inspection only

Counts: matched 8, missing_in_a 0, missing_in_b 0, tied 0, shape_mismatch 1, dtype_mismatch 0

### Not compared (1)

| key | status | reason |
|---|---|---|
| demo.layers.0.mlp.up_proj.weight | shape_mismatch | shape [8, 8] vs [8, 6] |


## Completeness

tensors_analyzed: 8, tensors_withheld: 0, rollups_withheld: 0, entries_not_compared: 1, tied_entries: 0

## Caveats

- Class X: structurally incompatible pair; matched tensors are unverified inspection only.
- Ranking by change magnitude describes how much parameters moved; it does not show that a tensor matters for any behavior (doc 05 section 9).
- Layer and role labels are heuristics from tensor key names, not identity.

## Top 8 of 8 tensors by relative_l2

| # | key | layer | role | shape | rel L2 | Δ L2 | Δ RMS | cos(A,B) | max abs Δ | non-finite pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | demo.cube.weight | — | other | 2×4×8 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 2 | demo.layers.0.mlp.down_proj.weight | 0 | mlp.down | 8×8 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 3 | demo.layers.0.self_attn.q_proj.weight | 0 | attention.q | 257×263 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 4 | demo.one_by_one.weight | — | other | 1×1 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 5 | demo.scalar | — | other |  | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 6 | demo.special_nonfinite.weight | — | other | 2×4 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 3 |
| 7 | demo.vector.weight | — | other | 6 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 8 | demo.zeros.weight | — | other | 2×3 | 0.0 | 0.0 | 0.0 | — | 0.0 | 0 |

Rows with caveats: 8 (see the caveat column in the CSV/JSON).

## By layer (heuristic)

| layer | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| 0 | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| — | 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

## By role (heuristic)

| role | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| attention.q | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| mlp.down | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| other | 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

Whole model (analyzed tensors): rel L2 0.0, Δ L2 0.0, Δ RMS 0.0, 8 tensors, 67741 elements.

## Provenance

- weightlab 0.0.1.dev0 (parser v1, numerics v2), created 2026-11-06T00:00:00Z
- Operation: delta, compute float64, coverage complete, consistency best_effort, epsilon_norm 1e-12
- Files: tiny_base.safetensors 272076 B sha256 not computed; shape_mismatch.safetensors 272012 B sha256 d44de4c6deda2601a1644d0c8bb370dbb9ad82b2e2b3262451e3c59e490819ab
- Paths included: False; input files opened read-only.
