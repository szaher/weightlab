# Example 3: incomplete report (about 5 minutes)

**Scenario.** You point the tool at an adapter it cannot fully analyze. Here that is DoRA, which WeightLab does not support yet. The question for you is whether the report makes it obvious **what it does not cover**, so that nobody reads a partial result as the whole picture.

**Inputs.** `fixtures/tiny_base.safetensors` and `fixtures/tiny_lora_dora_unsupported/`.

- This is a PEFT adapter with `use_dora: true` on `up_proj`.
- It has a `lora_magnitude_vector` key.
- DoRA rescales W by learned per-column magnitudes, so s·(B@A) alone would be the wrong update.

```bash
weightlab diff fixtures/tiny_base.safetensors --adapter fixtures/tiny_lora_dora_unsupported --out 03_incomplete_report
echo $?    # 3 = report written but incomplete (Windows PowerShell: $LASTEXITCODE)
```

`run_examples.py` already ran this; the report is in `weightlab-eval-<time>/03_incomplete_report/`.

## What to look at in `report.md`

1. **`Complete: NO`** in the header. The exit status is 3, not 0, so scripts and CI notice it too.
2. **"Not compared"** lists `…up_proj.lora_magnitude_vector` as `unsupported_key` with the reason `DoRA magnitude vector`.
3. **Counts:** `modules 1, analyzed 0, unsupported 1`.
   - The module row is still listed, with every metric shown as `—`.
   - There is no number that could pass for a result.
   - The layer and role tables are empty, and the whole-model line covers 0 tensors.
4. **The first caveat:** "Incomplete: some adapter modules or keys were not analyzed; this report does not describe the whole model change."

The same `complete: false` and exit status 3 appear in every other case where the tool cannot stand behind a full answer:

- tensors present in only one checkpoint, or with mismatched shapes;
- adapter modules skipped because they exceed the memory limit;
- metrics withheld because B − A overflows float64.

## While you read, note for the questionnaire

- Would you have noticed that this report is incomplete if you had only looked at the top of `report.md`? Or only at the CSV? (Q9)
- When a tool cannot analyze part of your model, what do you want it to do: refuse entirely, report the rest as here, or approximate and label it? (Q10)
- Which formats or adapter types that you use would end up here? (Q11)

## Optional: other incomplete or refused cases from the fixtures

The archive ships these fixtures under `fixtures/`. Run the commands from the unpacked kit directory:

```bash
weightlab diff fixtures/tiny_base.safetensors fixtures/compat/shape_mismatch.safetensors          # Class X, matched subset still ranked; exit 3
weightlab diff fixtures/tiny_precision_a.safetensors fixtures/tiny_precision_b.safetensors       # F64 overflow: metrics withheld; exit 3
weightlab diff fixtures/tiny_base.safetensors fixtures/malformed/truncated_payload.safetensors     # refused: INVALID_FORMAT; exit 2
```

In a clone of the repository, run the same commands from the repository root with `uv run` in front and `examples/fixtures/` in place of `fixtures/`.
