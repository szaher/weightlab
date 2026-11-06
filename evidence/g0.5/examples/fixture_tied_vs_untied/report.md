# WeightLab change report

- Mode: **checkpoint_pair**; orientation B − A (A = base `model.safetensors`, B = derived `model.safetensors`)
- Complete: **yes**

## Compatibility

- Class **S**; base file identity verified: False (none)
- Lineage claim source: none; trust: none; evidence: none; transfer eligibility (informational): T4
- no lineage evidence: shape-compatible only; deltas may be meaningless (permutation/alignment)

Counts: matched 2, missing_in_a 0, missing_in_b 0, tied 1, shape_mismatch 0, dtype_mismatch 0

### Not compared (1)

| key | status | reason |
|---|---|---|
| lm_head.weight | tied | B ties lm_head.weight to model.embed_tokens.weight (config tie_word_embeddings=true); A stores it separately. The tie differs; model.embed_tokens.weight is compared once and lm_head.weight has no delta. |


## Completeness

tensors_analyzed: 2, tensors_withheld: 0, rollups_withheld: 0, entries_not_compared: 1, tied_entries: 1

## Caveats

- Class S: no lineage evidence; raw deltas may be meaningless (parameter permutation/alignment).
- Ranking by change magnitude describes how much parameters moved; it does not show that a tensor matters for any behavior (doc 05 section 9).
- Layer and role labels are heuristics from tensor key names, not identity.

## Top 2 of 2 tensors by relative_l2

| # | key | layer | role | shape | rel L2 | Δ L2 | Δ RMS | cos(A,B) | max abs Δ | non-finite pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | model.embed_tokens.weight | — | embedding | 16×8 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |
| 2 | model.norm.weight | — | norm | 8 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0 |

Rows with caveats: 2 (see the caveat column in the CSV/JSON).

## By layer (heuristic)

| layer | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| — | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

## By role (heuristic)

| role | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| embedding | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| norm | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

Whole model (analyzed tensors): rel L2 0.0, Δ L2 0.0, Δ RMS 0.0, 2 tensors, 136 elements.

## Provenance

- weightlab 0.0.1.dev0 (parser v1, numerics v2), created 2026-11-06T00:00:00Z
- Operation: delta, compute float64, coverage complete, consistency best_effort, epsilon_norm 1e-12
- Files: model.safetensors 1296 B sha256 not computed; model.safetensors 712 B sha256 2e397e9abda513e0c82abd8e0020f7e7861d08b0c4b237ef982134e02b62352a
- Paths included: False; input files opened read-only.
