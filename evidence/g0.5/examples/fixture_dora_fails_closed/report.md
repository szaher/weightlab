# WeightLab change report

- Mode: **adapter**; orientation B − A (A = base `tiny_base.safetensors`, B = base + adapter `tiny_lora_dora_unsupported`)
- Complete: **NO**

## Compatibility

- Class **V**; base file identity verified: True (sha256_recorded)
- Lineage claim source: synthetic_fixture_generator; trust: proven_by_construction; evidence: cryptographically_verified; transfer eligibility (informational): T1
- synthetic fixture pair (tiny_lora_dora_unsupported built from tiny_base by the generator): ancestry known by construction

Counts: modules 1, analyzed 0, over_budget_norm_only 0, unsupported 1, skipped 0

### Not compared (1)

| key | status | reason |
|---|---|---|
| base_model.model.demo.layers.0.mlp.up_proj.lora_magnitude_vector | unsupported_key | DoRA magnitude vector |


## Completeness

modules_analyzed: 0, modules_norm_only: 0, modules_unsupported: 1, modules_skipped: 0, unsupported_adapter_keys: 1, base_tensors_without_adapter: 8

## Caveats

- Incomplete: some adapter modules or keys were not analyzed; this report does not describe the whole model change.
- Ranking by change magnitude describes how much parameters moved; it does not show that a tensor matters for any behavior (doc 05 section 9).
- Layer and role labels are heuristics from tensor key names, not identity.

## Top 1 of 1 modules by relative_l2

| # | key | layer | role | shape | rel L2 | Δ L2 | Δ RMS | cos(A,B) | max abs Δ | non-finite pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | demo.layers.0.mlp.up_proj.weight | 0 | mlp.up | 8×8 | — | — | — | — | — | — |

Rows with caveats: 1 (see the caveat column in the CSV/JSON).

## By layer (heuristic)

| layer | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|

## By role (heuristic)

| role | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|

Whole model (analyzed tensors): rel L2 —, Δ L2 0.0, Δ RMS —, 0 tensors, 0 elements.

## Provenance

- weightlab 0.0.1.dev0 (parser v1, numerics v1), created 2026-11-06T00:00:00Z
- Operation: lora_effective_delta, compute float64, coverage complete, consistency best_effort, epsilon_norm 1e-12
- Files: tiny_base.safetensors 272076 B sha256 a701e77f066883608beb2a1fc080d52f06ebffca227c1d6fd12de2d484e81133; adapter_model.safetensors ? B sha256 abb0af27897cd5b483546c941f4552e287d8011fbb577c1b50df4a636552b879
- Paths included: False; input files opened read-only.
