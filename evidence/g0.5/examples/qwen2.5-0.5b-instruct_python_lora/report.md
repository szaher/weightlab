# WeightLab change report

- Mode: **adapter**; orientation B − A (A = base `model.safetensors`, B = base + adapter `5ecafffc2e23deebae61af586617143e596743a1`)
- Complete: **yes**

## Compatibility

- Class **L**; base file identity verified: False (none)
- Lineage claim source: adapter_config; trust: unverified_claim; evidence: declared; transfer eligibility (informational): T2
- adapter declares the base repo but no matching revision; identity not verified
- lineage declared by metadata, not cryptographically verified

Counts: modules 168, analyzed 168, over_budget_norm_only 0, unsupported 0, skipped 0

## Completeness

modules_analyzed: 168, modules_withheld: 0, modules_norm_only: 0, modules_unsupported: 0, modules_skipped: 0, unsupported_adapter_keys: 0, base_tensors_without_adapter: 122

## Caveats

- Class L: lineage declared or attested, not cryptographically verified.
- Ranking by change magnitude describes how much parameters moved; it does not show that a tensor matters for any behavior (doc 05 section 9).
- Layer and role labels are heuristics from tensor key names, not identity.

## Top 25 of 168 modules by relative_l2

| # | key | layer | role | shape | rel L2 | Δ L2 | Δ RMS | cos(A,B) | max abs Δ | non-finite pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | model.layers.1.self_attn.v_proj.weight | 1 | attention.v | 128×896 | 0.00863674 | 0.0337176 | 9.95628e-05 | 0.999963 | 0.000565306 | 0 |
| 2 | model.layers.2.self_attn.v_proj.weight | 2 | attention.v | 128×896 | 0.00835102 | 0.0315823 | 9.32577e-05 | 0.999965 | 0.000672329 | 0 |
| 3 | model.layers.0.self_attn.v_proj.weight | 0 | attention.v | 128×896 | 0.00795679 | 0.0309127 | 9.12806e-05 | 0.999968 | 0.000548969 | 0 |
| 4 | model.layers.0.self_attn.o_proj.weight | 0 | attention.o | 896×896 | 0.00693968 | 0.0816356 | 9.11112e-05 | 0.999976 | 0.000550169 | 0 |
| 5 | model.layers.1.self_attn.o_proj.weight | 1 | attention.o | 896×896 | 0.00636889 | 0.0848383 | 9.46856e-05 | 0.99998 | 0.000624545 | 0 |
| 6 | model.layers.2.self_attn.o_proj.weight | 2 | attention.o | 896×896 | 0.00611064 | 0.0827288 | 9.23312e-05 | 0.999981 | 0.000479659 | 0 |
| 7 | model.layers.8.self_attn.v_proj.weight | 8 | attention.v | 128×896 | 0.00598063 | 0.033055 | 9.76065e-05 | 0.999982 | 0.000550743 | 0 |
| 8 | model.layers.3.self_attn.v_proj.weight | 3 | attention.v | 128×896 | 0.00579359 | 0.0314881 | 9.29797e-05 | 0.999983 | 0.000751474 | 0 |
| 9 | model.layers.3.self_attn.o_proj.weight | 3 | attention.o | 896×896 | 0.00563746 | 0.0817961 | 9.12903e-05 | 0.999984 | 0.000548196 | 0 |
| 10 | model.layers.12.self_attn.o_proj.weight | 12 | attention.o | 896×896 | 0.0056293 | 0.090132 | 0.000100594 | 0.999984 | 0.000604188 | 0 |
| 11 | model.layers.6.self_attn.v_proj.weight | 6 | attention.v | 128×896 | 0.00562291 | 0.030566 | 9.02566e-05 | 0.999984 | 0.000569223 | 0 |
| 12 | model.layers.9.self_attn.o_proj.weight | 9 | attention.o | 896×896 | 0.00560766 | 0.084431 | 9.4231e-05 | 0.999984 | 0.000548902 | 0 |
| 13 | model.layers.13.self_attn.o_proj.weight | 13 | attention.o | 896×896 | 0.0056013 | 0.0873484 | 9.74871e-05 | 0.999984 | 0.000527327 | 0 |
| 14 | model.layers.19.self_attn.o_proj.weight | 19 | attention.o | 896×896 | 0.00557346 | 0.0904991 | 0.000101003 | 0.999984 | 0.000539184 | 0 |
| 15 | model.layers.17.mlp.up_proj.weight | 17 | mlp.up | 4864×896 | 0.00557035 | 0.216397 | 0.000103657 | 0.999984 | 0.000918754 | 0 |
| 16 | model.layers.8.self_attn.o_proj.weight | 8 | attention.o | 896×896 | 0.00554821 | 0.0855112 | 9.54366e-05 | 0.999985 | 0.000602194 | 0 |
| 17 | model.layers.15.self_attn.o_proj.weight | 15 | attention.o | 896×896 | 0.00554529 | 0.0895855 | 9.99839e-05 | 0.999985 | 0.000574219 | 0 |
| 18 | model.layers.12.self_attn.v_proj.weight | 12 | attention.v | 128×896 | 0.00549096 | 0.0347636 | 0.000102652 | 0.999985 | 0.000610611 | 0 |
| 19 | model.layers.16.self_attn.v_proj.weight | 16 | attention.v | 128×896 | 0.00548722 | 0.0339688 | 0.000100305 | 0.999985 | 0.000831237 | 0 |
| 20 | model.layers.14.self_attn.o_proj.weight | 14 | attention.o | 896×896 | 0.00545771 | 0.0895995 | 9.99994e-05 | 0.999985 | 0.000530551 | 0 |
| 21 | model.layers.11.self_attn.o_proj.weight | 11 | attention.o | 896×896 | 0.00545693 | 0.0852938 | 9.5194e-05 | 0.999985 | 0.00052683 | 0 |
| 22 | model.layers.11.self_attn.v_proj.weight | 11 | attention.v | 128×896 | 0.00541142 | 0.03095 | 9.13907e-05 | 0.999985 | 0.000907634 | 0 |
| 23 | model.layers.6.self_attn.o_proj.weight | 6 | attention.o | 896×896 | 0.00540522 | 0.0846028 | 9.44228e-05 | 0.999985 | 0.000581262 | 0 |
| 24 | model.layers.18.mlp.up_proj.weight | 18 | mlp.up | 4864×896 | 0.0053734 | 0.217166 | 0.000104026 | 0.999986 | 0.00081317 | 0 |
| 25 | model.layers.1.mlp.up_proj.weight | 1 | mlp.up | 4864×896 | 0.00537189 | 0.19119 | 9.15827e-05 | 0.999986 | 0.000595784 | 0 |

## By layer (heuristic)

| layer | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| 0 | 7 | 0.00314704 | 0.309604 | 8.01818e-05 | 0.000767339 | 0 |
| 1 | 7 | 0.00402453 | 0.30992 | 8.02636e-05 | 0.000634802 | 0 |
| 2 | 7 | 0.00401814 | 0.308584 | 7.99175e-05 | 0.000672329 | 0 |
| 3 | 7 | 0.0040922 | 0.307116 | 7.95374e-05 | 0.000751474 | 0 |
| 4 | 7 | 0.00411783 | 0.312783 | 8.10051e-05 | 0.000771575 | 0 |
| 5 | 7 | 0.00407068 | 0.311148 | 8.05817e-05 | 0.0006355 | 0 |
| 6 | 7 | 0.004163 | 0.316396 | 8.19408e-05 | 0.000695004 | 0 |
| 7 | 7 | 0.00419471 | 0.317977 | 8.23503e-05 | 0.000727511 | 0 |
| 8 | 7 | 0.00411391 | 0.317417 | 8.22051e-05 | 0.000665525 | 0 |
| 9 | 7 | 0.00426579 | 0.316664 | 8.20102e-05 | 0.000974017 | 0 |
| 10 | 7 | 0.00428013 | 0.319132 | 8.26495e-05 | 0.000756397 | 0 |
| 11 | 7 | 0.00445142 | 0.321698 | 8.33139e-05 | 0.000908491 | 0 |
| 12 | 7 | 0.00451518 | 0.335554 | 8.69025e-05 | 0.000780653 | 0 |
| 13 | 7 | 0.00442195 | 0.328509 | 8.50778e-05 | 0.000749731 | 0 |
| 14 | 7 | 0.00445255 | 0.333662 | 8.64123e-05 | 0.000921389 | 0 |
| 15 | 7 | 0.00439888 | 0.334223 | 8.65577e-05 | 0.000850943 | 0 |
| 16 | 7 | 0.00439454 | 0.337958 | 8.75249e-05 | 0.000910924 | 0 |
| 17 | 7 | 0.0045529 | 0.347703 | 9.00487e-05 | 0.00120826 | 0 |
| 18 | 7 | 0.0045304 | 0.347808 | 9.00759e-05 | 0.000961948 | 0 |
| 19 | 7 | 0.00445352 | 0.345646 | 8.95162e-05 | 0.000790772 | 0 |
| 20 | 7 | 0.00443982 | 0.349774 | 9.0585e-05 | 0.000992226 | 0 |
| 21 | 7 | 0.00423307 | 0.337326 | 8.73613e-05 | 0.000779021 | 0 |
| 22 | 7 | 0.00432441 | 0.341886 | 8.85423e-05 | 0.000764157 | 0 |
| 23 | 7 | 0.00439596 | 0.337307 | 8.73564e-05 | 0.000718503 | 0 |

## By role (heuristic)

| role | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| attention.k | 24 | 0.00309931 | 0.155145 | 9.3513e-05 | 0.000778121 | 0 |
| attention.o | 24 | 0.00537036 | 0.427666 | 9.74296e-05 | 0.000992226 | 0 |
| attention.q | 24 | 0.0036279 | 0.414747 | 9.44864e-05 | 0.000780653 | 0 |
| attention.v | 24 | 0.00488486 | 0.16094 | 9.70061e-05 | 0.000974017 | 0 |
| mlp.down | 24 | 0.00233124 | 0.434248 | 4.24602e-05 | 0.000383512 | 0 |
| mlp.gate | 24 | 0.00454835 | 0.996409 | 9.74275e-05 | 0.00120826 | 0 |
| mlp.up | 24 | 0.00511266 | 0.991607 | 9.69579e-05 | 0.000921389 | 0 |

Whole model (analyzed tensors): rel L2 0.00423442, Δ L2 1.60299, Δ RMS 8.47411e-05, 168 tensors, 357826560 elements.

## Provenance

- weightlab 0.0.1.dev0 (parser v1, numerics v2), created 2026-11-06T00:00:00Z
- Operation: lora_effective_delta, compute float64, coverage complete, consistency best_effort, epsilon_norm 1e-12
- Files: model.safetensors 988097824 B sha256 not computed; adapter_model.safetensors ? B sha256 4fd68c46616906d89f50ebad0e8e08cca579a1438759fba8036bba27cc0cfaac
- Paths included: False; input files opened read-only.
