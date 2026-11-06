# `weightlab diff`: change report (Phase 0.5, WL-024)

Implementation notes for the Phase 0.5 command. This file is not part of the frozen 1.2-final specification (`docs/*.md`); where the two differ, the specification wins and the difference is listed in `evidence/g0.5/G0.5_REPORT.md`.

`weightlab diff` compares two SafeTensors checkpoints, or a base checkpoint and a PEFT LoRA adapter, and reports how far each tensor moved. It reads the inputs only, never modifies them, and never runs a model.

```bash
weightlab diff BASE DERIVED [--out DIR]          # checkpoint pair, B = DERIVED, A = BASE
weightlab diff BASE --adapter ADAPTER_DIR [...]  # A = W, B = W + s * (B_lora @ A_lora)
```

`BASE` and `DERIVED` may be a `.safetensors` file, a `*.safetensors.index.json` or a model directory (the Phase 0 reader resolves the entry). Inputs in the offline Hugging Face cache (`.../models--org--name/snapshots/<rev>/`) are contained to the `models--org--name` directory so the cache's blob symlinks resolve; elsewhere each input is contained to its own directory unless `--selected-root` is given.

## Outputs and exit status

| Option | Writes |
|---|---|
| `--out DIR` | `report.json`, `tensors.csv`, `layers.csv`, `roles.csv`, `report.md` |
| `--json F` / `--csv F` / `--md F` | one file each |
| (none) | Markdown summary on stdout unless `--quiet` |

Files are written atomically (temporary file, then rename). WeightLab refuses to write into any input directory. Exit codes: `0` complete report, `3` report written but incomplete (`complete: false`: unmatched tensors, unsupported adapter modules, budget-limited modules), `2` error (nothing written), `130` cancelled with Ctrl-C (nothing written).

The JSON report follows `weightlab/schemas/change_report.schema.json`. Each metric is a contract `Metric` object (`contracts/openapi.yaml`) and `compatibility.lineage` is a contract `LineageAssessment`; the tests validate both against the frozen contract.

## Metrics (per tensor, orientation B − A)

All metrics stream over the tensor in chunks (`--chunk-elements`, default 1,048,576) in float64 and are exact up to float64 rounding. They are independent of the chunk size except for rounding. Squared sums are kept as mantissa × 2^e, so norms of finite inputs never overflow.

| Metric | Definition |
|---|---|
| `delta_l2` | ‖B − A‖₂ over finite pairs |
| `delta_rms` | ‖B − A‖₂ / √(finite pairs) |
| `relative_l2` | ‖B − A‖₂ / max(‖A‖₂, ε), ε = `--epsilon-norm` (default 1e-12); a `near_zero_reference` caveat when ‖A‖₂ < ε |
| `cosine` | ⟨A, B⟩ / (‖A‖₂‖B‖₂), null when either norm is zero |
| `max_abs_delta` | max \|B − A\| |
| `norm_a`, `norm_b` | ‖A‖₂, ‖B‖₂ |

Counts per tensor: `elements`, `finite_pairs`, `invalid_pairs` (either side NaN/±Inf, excluded from every metric), per-side `nan`/`pos_inf`/`neg_inf`, `changed` (B ≠ A) and `delta_overflow`. When B − A is not representable in float64 for some pair, every delta metric of that tensor is withheld (`value: null`, `numeric_precision: overflowed`) rather than reported as Inf. A rollup containing a withheld tensor is withheld too.

Rankings: `--sort` (default `relative_l2`; `cosine` ranks lowest first; nulls last). Rollups by layer index and role are recomputed from the per-tensor sums, not averaged. Layer and role labels are heuristics from tensor key names.

## Compatibility and lineage

Tensors are matched by exact key only. Renamed keys are mentioned in `reasons` as suspected and are never matched.

| Class | Meaning |
|---|---|
| X | Missing tensors, shape mismatch, or a dtype mismatch across numeric families. Matched tensors are still reported. |
| V | Base file identity verified: SHA-256 from a `--lineage-manifest`, the fixture generator registry, or Hugging Face cache blob names matching a fresh SHA-256 for the declared repo and revision. Capped at L if output-head tying or dtypes differ. |
| L | Lineage declared (adapter `base_model_name_or_path`, unverified manifest) or `--attest`ed. Not cryptographically verified. |
| S | No lineage evidence. Shapes match, but raw deltas may be meaningless because of parameter permutation. |

`lm_head.weight` counts as tied only with `tie_word_embeddings: true` in `config.json` **and** a known mapping (`model.embed_tokens.weight`, `transformer.wte.weight`). Otherwise a missing head is `missing_in_b` (Class X). `transfer_eligibility` (T1–T5) is an informational label only: Phase 0.5 performs no transfer.

## LoRA adapters

Supported: PEFT `LORA` with `lora_A`/`lora_B` linear weights, `r`/`lora_alpha`, rsLoRA (`alpha/√r`), `rank_pattern`/`alpha_pattern` with literal suffix keys, `fan_in_fan_out` (Conv1D layout), `layers_to_transform`. The effective delta s·(B_lora@A_lora) is computed in row blocks and compared with the base weight exactly as for a pair. Modules whose dense product exceeds `--max-lora-gmac` (default 50 GMAC) get the exact Frobenius norm from the Gram trace and status `over_budget_norm_only`, and the report becomes incomplete.

Fails closed (module `unsupported`, report incomplete): DoRA, embedding/conv LoRA, `modules_to_save`, trained bias, `lora_bias`, QA-LoRA, `trainable_token_indices`, `target_parameters`, `layer_replication`, `layers_pattern`, regex syntax in patterns, rank mismatch or swapped factors, missing base tensors, non-`LORA` PEFT types, and unknown keys.

## Provenance, privacy and consistency

`provenance` records the tool, parser and numerics versions, the creation time (`WEIGHTLAB_REPORT_TIMESTAMP` overrides it for reproducible reports), and per input: format, entry, file names, sizes, mtimes, SHA-256 when computed, tensor count and bytes. It also records the operation definition, `coverage: complete`, and `read_only: true`. Absolute paths appear only with `--include-paths`. `--consistency verified` hashes every input file before and after the run (two extra full reads) and fails with `FILE_CHANGED` if anything changed. The default `best_effort` relies on the reader's size, mtime and identity checks.

## Library use

```python
from weightlab.analysis.change_report import Options, run
report = run("base/", "derived/", options=Options(sort="delta_l2"))
report = run("base/", adapter="lora_adapter/")
```

`RunControl` (from `weightlab.analysis.diff`) provides a progress callback and a cancel event. Cancelling raises `Cancelled` and returns no partial report.
