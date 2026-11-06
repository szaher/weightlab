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

The files of one run are published as a set. Every file is first written and fsynced under a temporary name; a failure or Ctrl-C at that stage removes the temporaries and publishes nothing. The renames then run with Ctrl-C deferred, and a failed rename restores the files already replaced. A new `--out` directory appears with a single directory rename. WeightLab refuses to write into any input directory.

Exit codes: `0` complete report; `3` report written but incomplete (`complete: false`: unmatched tensors, unsupported, skipped or budget-limited adapter modules, or any tensor/module whose delta metrics are withheld); `2` error, including invalid options (nothing written); `130` cancelled with Ctrl-C (nothing written).

`--epsilon-norm` must be finite and in [1e-300, 1]; `--max-lora-gmac` must be finite and in [0, 1e6]; `--chunk-elements` must be in [1, 2^26]. NaN and Inf are rejected, because they would silently disable a comparison-based limit. The library enforces the same rules (`Options.validate()`).

The JSON report follows `weightlab/schemas/change_report.schema.json`. Each metric is a contract `Metric` object (`contracts/openapi.yaml`) and `compatibility.lineage` is a contract `LineageAssessment`; the tests validate both against the frozen contract.

## Metrics (per tensor, orientation B − A)

I64/U64 deltas are formed exactly (each value split into 32-bit halves) and rounded to float64 once, so a nonzero integer difference such as 2^53 vs 2^53 + 1 is never reported as zero. Norms and cosine of I64/U64 tensors use float64-rounded values (`rounded`, doc 05 §2). Every narrower integer converts to float64 exactly.

All metrics stream over the tensor in chunks (`--chunk-elements`, default 1,048,576) in float64 and are exact up to float64 rounding. They are independent of the chunk size except for rounding. Squared sums are kept as mantissa × 2^e, so norms of finite inputs never overflow.

| Metric | Definition |
|---|---|
| `delta_l2` | ‖B − A‖₂ over finite pairs |
| `delta_rms` | ‖B − A‖₂ / √(finite pairs) |
| `relative_l2` | ‖B − A‖₂ / max(‖A‖₂, ε), ε = `--epsilon-norm` (default 1e-12); a `near_zero_reference` caveat when ‖A‖₂ < ε. The same policy applies to layer, role and total rollups, which carry their own `near_zero_reference` flag |
| `cosine` | ⟨A, B⟩ / (‖A‖₂‖B‖₂), null when either norm is zero |
| `max_abs_delta` | max \|B − A\| |
| `norm_a`, `norm_b` | ‖A‖₂, ‖B‖₂ |

Counts per tensor: `elements`, `finite_pairs`, `invalid_pairs` (either side NaN/±Inf, excluded from every metric), per-side `nan`/`pos_inf`/`neg_inf`, `changed` (B ≠ A) and `delta_overflow`. When B − A is not representable in float64 for some pair, every delta metric of that tensor is withheld (`value: null`, `numeric_precision: overflowed`) rather than reported as Inf. A rollup containing a withheld tensor is withheld too, and the report is marked incomplete (`tensors_withheld` / `modules_withheld`).

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

Supported: PEFT `LORA` with `lora_A`/`lora_B` linear weights, `r`/`lora_alpha`, rsLoRA (`alpha/√r`), `rank_pattern`/`alpha_pattern` with literal suffix keys, `fan_in_fan_out` (Conv1D layout), `layers_to_transform`. The effective delta s·(B_lora@A_lora) is computed in row blocks and compared with the base weight exactly as for a pair. Modules whose dense product exceeds `--max-lora-gmac` (default 50 GMAC) get the Frobenius norm from the Gram trace and status `over_budget_norm_only`, and the report becomes incomplete. The Gram trace scales both factors by exact powers of two first, so huge or tiny factors neither overflow nor underflow to a false zero. When the factors contain NaN/Inf, or the signed Gram sum is within its rounding-error bound of zero (cancellation), the norm is withheld (`unsupported`). Its `relative_l2` uses the same ε policy. `||W||` excludes NaN/Inf base elements, which are counted and reported.

Overflow in WeightLab's own arithmetic, where the inputs are finite but s·B@A or W + s·B@A leaves float64, is counted in `delta_overflow` / `b_overflow`. The affected metrics are withheld (`overflowed`), never masked as source NaN/Inf. Non-finite factors make the affected effective weights genuinely non-finite; they are counted as non-finite B values.

Memory: each module's working set (both factors in float64, plus row-block temporaries or the 3·r² Gram scratch) is estimated from the shapes before anything is loaded. Modules above 384 MiB, or with a factor above 2^24 cells, are `skipped` with a resource-limit reason. `r` and `rank_pattern` values must be integers in [1, 65536], and `lora_alpha` / `alpha_pattern` values must be finite float64 numbers; out-of-range JSON numbers are validation errors, not crashes.

Fails closed (module `unsupported`, report incomplete): DoRA, embedding/conv LoRA, `modules_to_save`, trained bias, `lora_bias`, QA-LoRA, `trainable_token_indices`, `target_parameters`, `layer_replication`, `layers_pattern`, regex syntax in patterns, rank mismatch or swapped factors, missing base tensors, non-`LORA` PEFT types, and unknown keys.

## Provenance, privacy and consistency

`provenance` records the tool, parser and numerics versions, the creation time (`WEIGHTLAB_REPORT_TIMESTAMP` overrides it for reproducible reports), and per input: format, entry, file names, sizes, mtimes, SHA-256 when computed, tensor count and bytes. It also records the operation definition, `coverage: complete`, and `read_only: true`. Absolute paths appear only with `--include-paths`. This includes an adapter's `base_model_name_or_path` when it is a local path rather than a Hugging Face repo id; it is exported as `<local path redacted: NAME>` everywhere, including the lineage reasons. `--consistency verified` hashes every input file before and after the run (two extra full reads) and fails with `FILE_CHANGED` if anything changed. The default `best_effort` relies on the reader's size, mtime and identity checks.

## Library use

```python
from weightlab.analysis.change_report import Options, run
report = run("base/", "derived/", options=Options(sort="delta_l2"))
report = run("base/", adapter="lora_adapter/")
```

`RunControl` (from `weightlab.analysis.diff`) provides a progress callback and a cancel event. Cancelling raises `Cancelled` and returns no partial report.
