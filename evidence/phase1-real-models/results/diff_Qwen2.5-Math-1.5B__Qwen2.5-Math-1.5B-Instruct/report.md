# WeightLab change report

- Mode: **checkpoint_pair**; orientation B − A (A = base `model.safetensors`, B = derived `model.safetensors`)
- Complete: **yes**

## Compatibility

- Class **S**; base file identity verified: False (none)
- Lineage claim source: none; trust: none; evidence: none; transfer eligibility (informational): T4
- no lineage evidence: shape-compatible only; deltas may be meaningless (permutation/alignment)

Counts: matched 338, missing_in_a 0, missing_in_b 0, tied 0, shape_mismatch 0, dtype_mismatch 0

## Completeness

tensors_analyzed: 338, tensors_withheld: 0, rollups_withheld: 0, entries_not_compared: 0, tied_entries: 0

## Caveats

- Class S: no lineage evidence; raw deltas may be meaningless (parameter permutation/alignment).
- Ranking by change magnitude describes how much parameters moved; it does not show that a tensor matters for any behavior (doc 05 section 9).
- Layer and role labels are heuristics from tensor key names, not identity.

## Top 15 of 338 tensors by relative_l2

| # | key | layer | role | shape | rel L2 | Δ L2 | Δ RMS | cos(A,B) | max abs Δ | non-finite pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | model.embed_tokens.weight | — | embedding | 151936×1536 | 0.621213 | 283.771 | 0.0185755 | 0.844223 | 0.258789 | 0 |
| 2 | model.layers.27.mlp.down_proj.weight | 27 | mlp.down | 1536×8960 | 0.182784 | 26.1641 | 0.00705271 | 0.986909 | 0.203125 | 0 |
| 3 | model.layers.27.self_attn.k_proj.weight | 27 | attention.k | 256×1536 | 0.178245 | 3.17654 | 0.00506569 | 0.98717 | 0.0351562 | 0 |
| 4 | model.layers.0.self_attn.v_proj.weight | 0 | attention.v | 256×1536 | 0.171248 | 3.45032 | 0.0055023 | 0.98797 | 0.0400391 | 0 |
| 5 | model.layers.26.mlp.down_proj.weight | 26 | mlp.down | 1536×8960 | 0.169871 | 26.1104 | 0.00703824 | 0.989222 | 0.109375 | 0 |
| 6 | model.layers.27.mlp.gate_proj.weight | 27 | mlp.gate | 8960×1536 | 0.167342 | 25.225 | 0.00679957 | 0.990221 | 0.0722656 | 0 |
| 7 | model.layers.27.mlp.up_proj.weight | 27 | mlp.up | 8960×1536 | 0.167274 | 25.7547 | 0.00694236 | 0.989805 | 0.09375 | 0 |
| 8 | model.layers.26.mlp.gate_proj.weight | 26 | mlp.gate | 8960×1536 | 0.165159 | 24.414 | 0.00658097 | 0.990342 | 0.09375 | 0 |
| 9 | model.layers.27.self_attn.q_proj.weight | 27 | attention.q | 1536×1536 | 0.162478 | 9.15301 | 0.00595899 | 0.990745 | 0.0390625 | 0 |
| 10 | model.layers.25.self_attn.k_proj.weight | 25 | attention.k | 256×1536 | 0.162471 | 3.02346 | 0.00482157 | 0.990361 | 0.0322266 | 0 |
| 11 | model.layers.27.self_attn.o_proj.weight | 27 | attention.o | 1536×1536 | 0.161284 | 11.4439 | 0.00745044 | 0.990646 | 0.0976562 | 0 |
| 12 | model.layers.26.mlp.up_proj.weight | 26 | mlp.up | 8960×1536 | 0.159081 | 25.0034 | 0.00673985 | 0.991252 | 0.0878906 | 0 |
| 13 | model.layers.25.mlp.down_proj.weight | 25 | mlp.down | 1536×8960 | 0.158528 | 24.5209 | 0.00660976 | 0.991272 | 0.113281 | 0 |
| 14 | model.layers.25.mlp.gate_proj.weight | 25 | mlp.gate | 8960×1536 | 0.156386 | 23.2586 | 0.00626951 | 0.991905 | 0.0703125 | 0 |
| 15 | model.layers.26.self_attn.q_proj.weight | 26 | attention.q | 1536×1536 | 0.154202 | 8.79 | 0.00572266 | 0.992074 | 0.0395508 | 0 |

Rows with caveats: 338 (see the caveat column in the CSV/JSON).

## By layer (heuristic)

| layer | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| 0 | 12 | 0.0230912 | 36.4859 | 0.0053335 | 0.113281 | 0 |
| 1 | 12 | 0.117864 | 35.8696 | 0.00524341 | 0.0820312 | 0 |
| 2 | 12 | 0.129535 | 36.3033 | 0.0053068 | 0.0742188 | 0 |
| 3 | 12 | 0.128085 | 36.5595 | 0.00534425 | 0.0742188 | 0 |
| 4 | 12 | 0.129181 | 36.3105 | 0.00530785 | 0.0742188 | 0 |
| 5 | 12 | 0.12848 | 36.5619 | 0.00534461 | 0.125 | 0 |
| 6 | 12 | 0.129033 | 36.3464 | 0.0053131 | 0.0898438 | 0 |
| 7 | 12 | 0.128248 | 36.5515 | 0.0053431 | 0.0820312 | 0 |
| 8 | 12 | 0.130576 | 36.6118 | 0.00535191 | 0.105469 | 0 |
| 9 | 12 | 0.129604 | 36.7509 | 0.00537224 | 0.125 | 0 |
| 10 | 12 | 0.131668 | 36.7631 | 0.00537403 | 0.105469 | 0 |
| 11 | 12 | 0.131206 | 36.8121 | 0.00538119 | 0.125 | 0 |
| 12 | 12 | 0.132404 | 37.0215 | 0.00541179 | 0.125 | 0 |
| 13 | 12 | 0.131505 | 37.2881 | 0.00545076 | 0.101562 | 0 |
| 14 | 12 | 0.134439 | 37.3381 | 0.00545807 | 0.125 | 0 |
| 15 | 12 | 0.119793 | 37.0046 | 0.00540932 | 0.125 | 0 |
| 16 | 12 | 0.133202 | 37.0683 | 0.00541863 | 0.132812 | 0 |
| 17 | 12 | 0.132883 | 37.214 | 0.00543993 | 0.105469 | 0 |
| 18 | 12 | 0.131835 | 37.2219 | 0.00544109 | 0.117188 | 0 |
| 19 | 12 | 0.132677 | 37.6329 | 0.00550116 | 0.113281 | 0 |
| 20 | 12 | 0.133894 | 38.0118 | 0.00555656 | 0.0976562 | 0 |
| 21 | 12 | 0.135175 | 38.5777 | 0.00563928 | 0.109375 | 0 |
| 22 | 12 | 0.135561 | 39.2836 | 0.00574247 | 0.125 | 0 |
| 23 | 12 | 0.139633 | 40.3836 | 0.00590327 | 0.105469 | 0 |
| 24 | 12 | 0.143515 | 41.9066 | 0.00612589 | 0.0976562 | 0 |
| 25 | 12 | 0.149024 | 43.6542 | 0.00638136 | 0.113281 | 0 |
| 26 | 12 | 0.157501 | 46.2655 | 0.00676308 | 0.109375 | 0 |
| 27 | 12 | 0.163247 | 47.271 | 0.00691006 | 0.203125 | 0 |
| — | 2 | 0.608189 | 283.783 | 0.0185763 | 0.258789 | 0 |

## By role (heuristic)

| role | tensors | rel L2 | Δ L2 | Δ RMS | max abs Δ | withheld |
|---|---|---|---|---|---|---|
| attention.k | 56 | 0.0114505 | 17.6951 | 0.00533109 | 0.0527344 | 0 |
| attention.o | 28 | 0.130905 | 43.061 | 0.00529802 | 0.125 | 0 |
| attention.q | 56 | 0.0847391 | 45.4313 | 0.00558784 | 0.113281 | 0 |
| attention.v | 56 | 0.122543 | 18.5465 | 0.0055876 | 0.0605469 | 0 |
| embedding | 1 | 0.621213 | 283.771 | 0.0185755 | 0.258789 | 0 |
| mlp.down | 28 | 0.141853 | 108.775 | 0.00554117 | 0.203125 | 0 |
| mlp.gate | 28 | 0.141526 | 114.677 | 0.0058418 | 0.109375 | 0 |
| mlp.up | 28 | 0.138684 | 107.94 | 0.0054986 | 0.105469 | 0 |
| norm | 57 | 0.0131112 | 3.09289 | 0.0104528 | 0.171875 | 0 |

Whole model (analyzed tensors): rel L2 0.157284, Δ L2 348.919, Δ RMS 0.00888057, 338 tensors, 1543714304 elements.

## Provenance

- weightlab 0.0.1.dev0 (parser v1, numerics v2), created 2026-11-08T20:20:02Z
- Operation: delta, compute float64, coverage complete, consistency best_effort, epsilon_norm 1e-12
- Files: model.safetensors 3087467144 B sha256 not computed; model.safetensors 3087467144 B sha256 not computed
- Paths included: False; input files opened read-only.
