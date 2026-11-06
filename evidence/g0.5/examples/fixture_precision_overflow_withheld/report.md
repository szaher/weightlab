# WeightLab change report

- Mode: **checkpoint_pair**; orientation B − A (A = base `tiny_precision_a.safetensors`, B = derived `tiny_precision_b.safetensors`)
- Complete: **NO**

## Compatibility

- Class **S**; base file identity verified: False (none)
- Lineage claim source: none; trust: none; evidence: none; transfer eligibility (informational): T4
- no lineage evidence: shape-compatible only; deltas may be meaningless (permutation/alignment)

Counts: matched 7, missing_in_a 0, missing_in_b 0, tied 0, shape_mismatch 0, dtype_mismatch 0

## Completeness

tensors_analyzed: 7, tensors_withheld: 1, rollups_withheld: 3, entries_not_compared: 0, tied_entries: 0

## Caveats

- Incomplete: 1 tensor(s)/module(s) and 3 rollup(s) have metrics withheld (float64 overflow or unreliable arithmetic); see 'withheld_metrics'.
- Class S: no lineage evidence; raw deltas may be meaningless (parameter permutation/alignment).
- Ranking by change magnitude describes how much parameters moved; it does not show that a tensor matters for any behavior (doc 05 section 9).
- Layer and role labels are heuristics from tensor key names, not identity.

## Top 7 of 7 tensors by relative_l2

| # | key | layer | role | shape | rel L2 | Δ L2 | Δ RMS | cos(A,B) | max abs Δ | non-finite pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | p.f32_overflow_delta | — | other | 2 | 2 | 6.8e+38 | 4.80833e+38 | -1 | 6.8e+38 | 0 |
| 2 | p.bool | — | other | 2 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 3 | p.f64_sumsq_overflow | — | other | 4 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 4 | p.i32 | — | other | 2 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 5 | p.i64 | — | other | 7 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 6 | p.u64 | — | other | 3 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 7 | p.f64 | — | other | 8 | — | — | — | 0.0 | — | 0 |

Rows with caveats: 7 (see the caveat column in the CSV/JSON).

## By layer (heuristic)

| layer | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| — | 7 | — | — | — | — | 1 |

## By role (heuristic)

| role | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| other | 7 | — | — | — | — | 1 |

Whole model (analyzed tensors): rel L2 —, Δ L2 —, Δ RMS —, 7 tensors, 28 elements.

## Provenance

- weightlab 0.0.1.dev0 (parser v1, numerics v2), created 2026-11-06T00:00:00Z
- Operation: delta, compute float64, coverage complete, consistency best_effort, epsilon_norm 1e-12
- Files: tiny_precision_a.safetensors 738 B sha256 not computed; tiny_precision_b.safetensors 738 B sha256 b75549228b77a639a1aa0c9fa609bde82e5b5cd9d74e37b13822a346e36d88fd
- Paths included: False; input files opened read-only.
