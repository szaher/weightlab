# WeightLab change report

- Mode: **checkpoint_pair**; orientation B − A (A = base `model.safetensors`, B = derived `model.safetensors`)
- Complete: **yes**

## Compatibility

- Class **S**; base file identity verified: False (none)
- Lineage claim source: none; trust: none; evidence: none; transfer eligibility (informational): T4
- no lineage evidence: shape-compatible only; deltas may be meaningless (permutation/alignment)

Counts: matched 290, missing_in_a 0, missing_in_b 0, tied 0, shape_mismatch 0, dtype_mismatch 0

## Completeness

tensors_analyzed: 290, entries_not_compared: 0, tied_entries: 0

## Caveats

- Class S: no lineage evidence; raw deltas may be meaningless (parameter permutation/alignment).
- Ranking by change magnitude describes how much parameters moved; it does not show that a tensor matters for any behavior (doc 05 section 9).
- Layer and role labels are heuristics from tensor key names, not identity.

## Top 25 of 290 tensors by relative_l2

| # | key | layer | role | shape | rel L2 | Δ L2 | Δ RMS | cos(A,B) | max abs Δ | non-finite pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | model.embed_tokens.weight | — | embedding | 151936×896 | 0.111558 | 20.3218 | 0.00174172 | 0.993906 | 0.019043 | 0 |
| 2 | model.layers.22.self_attn.q_proj.weight | 22 | attention.q | 896×896 | 0.105072 | 2.02251 | 0.00225727 | 0.994501 | 0.0181274 | 0 |
| 3 | model.layers.0.self_attn.o_proj.weight | 0 | attention.o | 896×896 | 0.104117 | 1.25087 | 0.00139606 | 0.994686 | 0.0153809 | 0 |
| 4 | model.layers.23.self_attn.q_proj.weight | 23 | attention.q | 896×896 | 0.10267 | 2.05188 | 0.00229005 | 0.994752 | 0.0219727 | 0 |
| 5 | model.layers.17.self_attn.q_proj.weight | 17 | attention.q | 896×896 | 0.0987019 | 1.96129 | 0.00218894 | 0.995151 | 0.017334 | 0 |
| 6 | model.layers.20.self_attn.q_proj.weight | 20 | attention.q | 896×896 | 0.0956075 | 1.87283 | 0.00209021 | 0.995455 | 0.0172958 | 0 |
| 7 | model.layers.18.self_attn.q_proj.weight | 18 | attention.q | 896×896 | 0.0944961 | 1.81028 | 0.0020204 | 0.995572 | 0.0183105 | 0 |
| 8 | model.layers.19.self_attn.q_proj.weight | 19 | attention.q | 896×896 | 0.0941431 | 1.85662 | 0.00207212 | 0.995631 | 0.0144653 | 0 |
| 9 | model.layers.21.self_attn.q_proj.weight | 21 | attention.q | 896×896 | 0.09374 | 1.812 | 0.00202232 | 0.995643 | 0.0151367 | 0 |
| 10 | model.layers.11.self_attn.q_proj.weight | 11 | attention.q | 896×896 | 0.0936654 | 1.76013 | 0.00196443 | 0.995673 | 0.0273438 | 0 |
| 11 | model.layers.12.mlp.gate_proj.weight | 12 | mlp.gate | 4864×896 | 0.0918542 | 3.90682 | 0.00187142 | 0.995851 | 0.017334 | 0 |
| 12 | model.layers.11.mlp.gate_proj.weight | 11 | mlp.gate | 4864×896 | 0.0912982 | 3.76694 | 0.00180442 | 0.995902 | 0.0136719 | 0 |
| 13 | model.layers.13.mlp.gate_proj.weight | 13 | mlp.gate | 4864×896 | 0.0910225 | 3.73796 | 0.00179054 | 0.99592 | 0.0205078 | 0 |
| 14 | model.layers.23.self_attn.k_proj.weight | 23 | attention.k | 128×896 | 0.0908008 | 0.576046 | 0.00170098 | 0.995894 | 0.0108643 | 0 |
| 15 | model.layers.11.self_attn.o_proj.weight | 11 | attention.o | 896×896 | 0.0896374 | 1.44061 | 0.00160782 | 0.996257 | 0.0344238 | 0 |
| 16 | model.layers.10.mlp.gate_proj.weight | 10 | mlp.gate | 4864×896 | 0.0894457 | 4.03575 | 0.00193318 | 0.996073 | 0.015625 | 0 |
| 17 | model.layers.14.mlp.gate_proj.weight | 14 | mlp.gate | 4864×896 | 0.0893717 | 3.81034 | 0.00182521 | 0.996069 | 0.015625 | 0 |
| 18 | model.layers.9.mlp.gate_proj.weight | 9 | mlp.gate | 4864×896 | 0.0892272 | 3.8825 | 0.00185977 | 0.996083 | 0.0136719 | 0 |
| 19 | model.layers.22.self_attn.k_proj.weight | 22 | attention.k | 128×896 | 0.0891866 | 0.554886 | 0.00163849 | 0.996033 | 0.0158691 | 0 |
| 20 | model.layers.15.mlp.gate_proj.weight | 15 | mlp.gate | 4864×896 | 0.088038 | 3.73862 | 0.00179086 | 0.996177 | 0.0155277 | 0 |
| 21 | model.layers.12.self_attn.q_proj.weight | 12 | attention.q | 896×896 | 0.0880271 | 1.72324 | 0.00192326 | 0.996184 | 0.015625 | 0 |
| 22 | model.layers.0.self_attn.v_proj.weight | 0 | attention.v | 128×896 | 0.0873539 | 0.343225 | 0.00101349 | 0.996205 | 0.00592041 | 0 |
| 23 | model.layers.10.mlp.up_proj.weight | 10 | mlp.up | 4864×896 | 0.0870755 | 3.43307 | 0.00164449 | 0.996322 | 0.0123291 | 0 |
| 24 | model.layers.9.self_attn.q_proj.weight | 9 | attention.q | 896×896 | 0.0870563 | 1.79385 | 0.00200207 | 0.996286 | 0.0253906 | 0 |
| 25 | model.layers.4.self_attn.q_proj.weight | 4 | attention.q | 896×896 | 0.0868842 | 1.82082 | 0.00203216 | 0.996272 | 0.0170898 | 0 |

Rows with caveats: 290 (see the caveat column in the CSV/JSON).

## By layer (heuristic)

| layer | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| 0 | 12 | 0.0149779 | 6.70451 | 0.00173617 | 0.25 | 0 |
| 1 | 12 | 0.0245185 | 5.61544 | 0.00145415 | 0.5 | 0 |
| 2 | 12 | 0.0336105 | 5.551 | 0.00143747 | 0.25 | 0 |
| 3 | 12 | 0.0519793 | 5.63801 | 0.00146 | 0.125 | 0 |
| 4 | 12 | 0.0515536 | 5.83147 | 0.0015101 | 0.148438 | 0 |
| 5 | 12 | 0.0508496 | 5.78536 | 0.00149815 | 0.117188 | 0 |
| 6 | 12 | 0.0511783 | 5.70382 | 0.00147704 | 0.078125 | 0 |
| 7 | 12 | 0.0474641 | 5.89238 | 0.00152587 | 0.0625 | 0 |
| 8 | 12 | 0.0205393 | 6.3222 | 0.00163717 | 0.25 | 0 |
| 9 | 12 | 0.0489505 | 6.38261 | 0.00165282 | 0.109375 | 0 |
| 10 | 12 | 0.0532962 | 6.45254 | 0.00167093 | 0.125 | 0 |
| 11 | 12 | 0.0493498 | 6.41871 | 0.00166216 | 0.078125 | 0 |
| 12 | 12 | 0.0536538 | 6.48026 | 0.00167811 | 0.126953 | 0 |
| 13 | 12 | 0.0526107 | 6.33646 | 0.00164087 | 0.167969 | 0 |
| 14 | 12 | 0.0527064 | 6.34787 | 0.00164382 | 0.238281 | 0 |
| 15 | 12 | 0.0512922 | 6.2827 | 0.00162695 | 0.148438 | 0 |
| 16 | 12 | 0.0521154 | 6.23378 | 0.00161428 | 0.137695 | 0 |
| 17 | 12 | 0.0520182 | 6.26267 | 0.00162176 | 0.109375 | 0 |
| 18 | 12 | 0.04931 | 6.21456 | 0.0016093 | 0.140625 | 0 |
| 19 | 12 | 0.0473856 | 6.03073 | 0.00156169 | 0.125 | 0 |
| 20 | 12 | 0.0500916 | 6.04363 | 0.00156503 | 0.125 | 0 |
| 21 | 12 | 0.0462213 | 6.09884 | 0.00157933 | 0.109375 | 0 |
| 22 | 12 | 0.0494769 | 6.18922 | 0.00160274 | 0.0625 | 0 |
| 23 | 12 | 0.0412265 | 5.77151 | 0.00149457 | 0.109375 | 0 |
| — | 2 | 0.0733433 | 20.5666 | 0.0017627 | 0.125 | 0 |

## By role (heuristic)

| role | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| attention.k | 48 | 0.0062068 | 3.30718 | 0.00199228 | 0.5 | 0 |
| attention.o | 24 | 0.0660357 | 5.37161 | 0.00122374 | 0.0344238 | 0 |
| attention.q | 48 | 0.0251763 | 9.73964 | 0.00221762 | 0.25 | 0 |
| attention.v | 48 | 0.03978 | 1.68702 | 0.00101628 | 0.00976562 | 0 |
| embedding | 1 | 0.111558 | 20.3218 | 0.00174172 | 0.019043 | 0 |
| mlp.down | 24 | 0.0749413 | 14.2305 | 0.00139144 | 0.0219727 | 0 |
| mlp.gate | 24 | 0.0787389 | 17.5155 | 0.00171264 | 0.0273438 | 0 |
| mlp.up | 24 | 0.0797229 | 15.7672 | 0.00154169 | 0.0351562 | 0 |
| norm | 49 | 0.00892495 | 3.50549 | 0.01673 | 0.125 | 0 |

Whole model (analyzed tensors): rel L2 0.0418489, Δ L2 36.341, Δ RMS 0.001635, 290 tensors, 494032768 elements.

## Provenance

- weightlab 0.0.1.dev0 (parser v1, numerics v1), created 2026-11-06T00:00:00Z
- Operation: delta, compute float64, coverage complete, consistency best_effort, epsilon_norm 1e-12
- Files: model.safetensors 988097824 B sha256 not computed; model.safetensors 988097824 B sha256 not computed
- Paths included: False; input files opened read-only.
