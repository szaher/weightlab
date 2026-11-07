# Example 2: LoRA adapter (about 5 minutes)

**Scenario.** You have a base model and a PEFT LoRA adapter. You want to know how large the update is that the adapter applies to each target module, and whether the scale you think it uses is the one it actually uses.

**Inputs.**

- `fixtures/tiny_base.safetensors`: the base W.
- `fixtures/tiny_lora/`: a PEFT adapter with `adapter_config.json` and `adapter_model.safetensors`:

  ```json
  {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "rank_pattern": {"down_proj": 1},
   "target_modules": ["up_proj", "down_proj"], "use_rslora": false}
  ```

- The adapter targets `up_proj` at rank 2 and `down_proj` at rank 1. `rank_pattern` overrides the default rank, so the scales differ: `lora_alpha / r` gives 4/2 = 2.0 and 4/1 = 4.0.

```bash
weightlab diff fixtures/tiny_base.safetensors --adapter fixtures/tiny_lora --out 02_lora_adapter
```

`run_examples.py` already ran this; the report is in `weightlab-eval-<time>/02_lora_adapter/`.

## What it computes

For each target module the effective update is ΔW = s·(B@A), with s from the adapter config (`lora_alpha / r`, or `lora_alpha / √r` with rsLoRA). The report compares W with W + ΔW exactly as it compares a checkpoint pair. The adapter is never merged into a file.

The dense B@A product is built in row blocks within a compute budget (`--max-lora-gmac`). A module above the budget still gets its exact ‖ΔW‖ via a Gram-trace identity and is marked `over_budget_norm_only`. The report never pretends that such a module had the full analysis.

## What to look at in `report.md`

1. **Compatibility.** Class **V** here, by construction.
   - For a real adapter the evidence is usually `base_model_name_or_path` in the config, which gives Class **L**: declared, not verified.
   - It becomes V only when the base files' identity can be verified, for example from a pinned revision in the Hugging Face cache.
2. **Counts.** `modules 2, analyzed 2, over_budget_norm_only 0, unsupported 0, skipped 0`. The 7 base tensors without an adapter are listed as such.
3. **Per-module metrics.** `down_proj` ranks first:

   | Module | Δ L2 | relative L2 |
   |---|---|---|
   | `down_proj` | 11.22 | 2.43 |
   | `up_proj` | 6.5 | 0.75 |

   Both values equal ‖s·B@A‖ computed by the generator. The `lora` block in `report.json` records the rank, alpha, scale and where each came from (`rank_pattern` for `down_proj`).
4. Compare with Example 1: the same base, but this update is much larger relative to W. Notice whether the report helps you judge whether that is plausible.

## While you read, note for the questionnaire

- Did the report make the scale and rank per module clear, including the `rank_pattern` override? (Q7)
- What would you want to see for an adapter that is not here? Examples: singular values or effective rank of ΔW, ΔW per head, comparison of two adapters, the effect of a different scale… (Q8)

## Optional: a real adapter

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
hf download ali4566544/qwen2.5-0.5b-instruct-python-lora --revision 5ecafffc2e23deebae61af586617143e596743a1
weightlab diff <instruct snapshot dir> --adapter <lora snapshot dir> --out qwen_lora --progress
```

In our run:

- Class **L**: the adapter declares its base repo, but not a revision that can be verified.
- 168 modules at r = 16, all analyzed, in 4.4 s.
- `v_proj` and `o_proj` in the early layers changed most.

An adapter you trained yourself is the most informative choice: you know which modules it targets and what it was trained to do.
