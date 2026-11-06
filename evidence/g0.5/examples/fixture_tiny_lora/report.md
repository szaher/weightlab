# WeightLab change report

- Mode: **adapter**; orientation B − A (A = base `tiny_base.safetensors`, B = base + adapter `tiny_lora`)
- Complete: **yes**

## Compatibility

- Class **V**; base file identity verified: True (sha256_recorded)
- Lineage claim source: synthetic_fixture_generator; trust: proven_by_construction; evidence: cryptographically_verified; transfer eligibility (informational): T1
- synthetic fixture pair (tiny_lora built from tiny_base by the generator): ancestry known by construction

Counts: modules 2, analyzed 2, over_budget_norm_only 0, unsupported 0, skipped 0

## Completeness

modules_analyzed: 2, modules_withheld: 0, modules_norm_only: 0, modules_unsupported: 0, modules_skipped: 0, unsupported_adapter_keys: 0, base_tensors_without_adapter: 7

## Caveats

- Ranking by change magnitude describes how much parameters moved; it does not show that a tensor matters for any behavior (doc 05 section 9).
- Layer and role labels are heuristics from tensor key names, not identity.

## Top 2 of 2 modules by relative_l2

| # | key | layer | role | shape | rel L2 | Δ L2 | Δ RMS | cos(A,B) | max abs Δ | non-finite pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | demo.layers.0.mlp.down_proj.weight | 0 | mlp.down | 8×8 | 2.43151 | 11.225 | 1.40312 | 0.543957 | 8 | 0 |
| 2 | demo.layers.0.mlp.up_proj.weight | 0 | mlp.up | 8×8 | 0.746781 | 6.5 | 0.8125 | 0.813147 | 2 | 0 |

## By layer (heuristic)

| layer | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| 0 | 2 | 1.31653 | 12.9711 | 1.1465 | 8 | 0 |

## By role (heuristic)

| role | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| mlp.down | 1 | 2.43151 | 11.225 | 1.40312 | 8 | 0 |
| mlp.up | 1 | 0.746781 | 6.5 | 0.8125 | 2 | 0 |

Whole model (analyzed tensors): rel L2 1.31653, Δ L2 12.9711, Δ RMS 1.1465, 2 tensors, 128 elements.

## Provenance

- weightlab 0.0.1.dev0 (parser v1, numerics v2), created 2026-11-06T00:00:00Z
- Operation: lora_effective_delta, compute float64, coverage complete, consistency best_effort, epsilon_norm 1e-12
- Files: tiny_base.safetensors 272076 B sha256 a701e77f066883608beb2a1fc080d52f06ebffca227c1d6fd12de2d484e81133; adapter_model.safetensors ? B sha256 5ea2c555103e23d55a8ea35ae3b862ce5a786d6c6475cf00ae180f764d00a30c
- Paths included: False; input files opened read-only.
