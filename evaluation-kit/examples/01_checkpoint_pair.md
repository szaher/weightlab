# Example 1: checkpoint pair (about 5 minutes)

**Scenario.** You have a base checkpoint and a fine-tuned derivative and want to know what moved, where and by how much, before you spend GPU time on evaluations.

**Inputs.** `fixtures/tiny_base.safetensors` (A) and `fixtures/tiny_ft.safetensors` (B). They are synthetic.

- B was built from A by the fixture generator.
- 10 values of `demo.layers.0.mlp.up_proj.weight` (8×8) were changed by +0.1 … +1.0.
- Nothing else changed.
- `demo.special_nonfinite.weight` contains a NaN, a +Inf and a −Inf in both files.

```bash
weightlab diff fixtures/tiny_base.safetensors fixtures/tiny_ft.safetensors --out 01_checkpoint_pair
```

`run_examples.py` already ran this; the report is in `weightlab-eval-<time>/01_checkpoint_pair/`.

## What to look at in `report.md`

1. **Compatibility first.** The report shows Class **V** ("base file identity verified") before any number.
   - The classes run from V (verified ancestry) through L (declared or attested, unverified) and S (shape-compatible only, no evidence) to X (not comparable).
   - Here V holds because the generator recorded the files' SHA-256 digests.
   - A real pair without such evidence gets S, with the caveat that raw deltas may be meaningless.
2. **Completeness.** Look for `Complete: yes` and the counts of matched, missing and tied entries.
3. **The ranking.** `demo.layers.0.mlp.up_proj.weight` is ranked first. Its metrics match the generator:

   | Metric | Value |
   |---|---|
   | relative L2 | 0.2254 |
   | Δ L2 | 1.962 |
   | Δ RMS | 0.2453 |
   | max \|Δ\| | 1 |
   | changed elements | 10 of 64 (in `tensors.csv`) |

   Every other tensor shows zero change.
4. **Non-finite values.** `demo.special_nonfinite.weight` reports 3 non-finite pairs. They are excluded from the metrics and counted, not silently dropped.
5. **Rollups by layer and role.** The labels (`mlp.up`, `attention.q`, …) are guesses from key names, as the caveats say.
6. **Provenance** at the bottom: tool and numerics versions, file sizes and SHA-256, `read-only`, and no absolute paths.

Also open `tensors.csv` (one row per tensor, every counter and metric) and skim `report.json`. Each metric there says how it was computed, its coverage and its precision.

## While you read, note for the questionnaire

- Could you tell **what** changed and **how much**? (Q4)
- What did you want to know next that the report does not tell you? Examples: which rows or columns, a picture of the matrix, a histogram of Δ, a comparison with a third checkpoint… (Q5)
- Was anything confusing or misleading: the compatibility class, a metric, a caveat? (Q6)

## Optional: a real pair

This is the pair we measured: about 1.9 GiB, 4 s, under 100 MiB of memory on an M-series laptop. The download needs network once; `weightlab` itself stays offline.

```bash
hf download Qwen/Qwen2.5-0.5B          --revision 060db6499f32faf8b98477b0a26969ef7d8b9987
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
# each command prints its snapshot directory
weightlab diff <base snapshot dir> <instruct snapshot dir> --out qwen_pair --progress
```

(`hf` comes with `pip install huggingface_hub`; older versions call it `huggingface-cli download`.)

In our run:

- Class **S**: nothing in the files proves that one derives from the other.
- Overall relative L2 was 0.042, and the embedding role changed most (0.112).

Better still is a pair you know from your own work: you can judge whether the ranking matches what you know about the training run.
