# WeightLab change report

- Mode: **checkpoint_pair**; orientation B − A (A = base `tiny_base.safetensors`, B = derived `tiny_ft.safetensors`)
- Complete: **yes**

## Compatibility

- Class **V**; base file identity verified: True (sha256_recorded)
- Lineage claim source: synthetic_fixture_generator; trust: proven_by_construction; evidence: cryptographically_verified; transfer eligibility (informational): T3
- synthetic fixture pair (tiny_ft.safetensors built from tiny_base by the generator): ancestry known by construction

Counts: matched 9, missing_in_a 0, missing_in_b 0, tied 0, shape_mismatch 0, dtype_mismatch 0

## Completeness

tensors_analyzed: 9, entries_not_compared: 0, tied_entries: 0

## Caveats

- Ranking by change magnitude describes how much parameters moved; it does not show that a tensor matters for any behavior (doc 05 section 9).
- Layer and role labels are heuristics from tensor key names, not identity.

## Top 9 of 9 tensors by relative_l2

| # | key | layer | role | shape | rel L2 | Δ L2 | Δ RMS | cos(A,B) | max abs Δ | non-finite pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | demo.layers.0.mlp.up_proj.weight | 0 | mlp.up | 8×8 | 0.225429 | 1.96214 | 0.245268 | 0.981655 | 1 | 0 |
| 2 | demo.cube.weight | — | other | 2×4×8 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 3 | demo.layers.0.mlp.down_proj.weight | 0 | mlp.down | 8×8 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 4 | demo.layers.0.self_attn.q_proj.weight | 0 | attention.q | 257×263 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 5 | demo.one_by_one.weight | — | other | 1×1 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 6 | demo.scalar | — | other |  | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 7 | demo.special_nonfinite.weight | — | other | 2×4 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 3 |
| 8 | demo.vector.weight | — | other | 6 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 9 | demo.zeros.weight | — | other | 2×3 | 0.0 | 0.0 | 0.0 | — | 0.0 | 0 |

Rows with caveats: 1 (see the caveat column in the CSV/JSON).

## By layer (heuristic)

| layer | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| 0 | 3 | 0.0130371 | 1.96214 | 0.00754006 | 1 | 0 |
| — | 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

## By role (heuristic)

| role | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| attention.q | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| mlp.down | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| mlp.up | 1 | 0.225429 | 1.96214 | 0.245268 | 1 | 0 |
| other | 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

Whole model (analyzed tensors): rel L2 0.00596888, Δ L2 1.96214, Δ RMS 0.00753545, 9 tensors, 67805 elements.

## Provenance

- weightlab 0.0.1.dev0 (parser v1, numerics v1), created 2026-11-06T00:00:00Z
- Operation: delta, compute float64, coverage complete, consistency best_effort, epsilon_norm 1e-12
- Files: tiny_base.safetensors 272076 B sha256 a701e77f066883608beb2a1fc080d52f06ebffca227c1d6fd12de2d484e81133; tiny_ft.safetensors 272076 B sha256 16bbefa9b14f910c751a476ae4a0570100f4589f03f498f242ae3700639c1452
- Paths included: False; input files opened read-only.
