# WeightLab `weightlab diff`: questionnaire (about 10 minutes)

Copy this file, fill in the `Answer:` lines and return it to the coordinator. Mark choices with `[x]`.

- Short answers are fine.
- Skip any question you prefer not to answer.
- "Don't know" is a useful answer.
- Please do not paste model weights or confidential tensor names. Describe them instead.

```
Participant ID (from the coordinator): P__
Date:
Kit version (`weightlab --version`):
OS / CPU (first lines printed by run_examples.py):
```

## A. About your work (1 minute)

**Q1. Which best describes your role?** (one)

- [ ] ML/LLM engineer: fine-tuning, serving or MLOps
- [ ] Researcher: interpretability, training dynamics or model merging
- [ ] Researcher: other
- [ ] Other: ______

**Q2. Which model sizes and adapters do you handle?** (any)

- [ ] < 1B
- [ ] 1–10B
- [ ] 10–80B
- [ ] > 80B
- [ ] LoRA / QLoRA adapters
- [ ] other PEFT (DoRA, IA³, prompt tuning …)
- [ ] full fine-tunes
- [ ] merged models

Model families and file formats you mostly use (e.g. Llama/Qwen, SafeTensors/GGUF):

Answer:

**Q3. Today, how do you find out what changed between two checkpoints, or what an adapter does?** (any)

- [ ] I don't; I only run evaluations
- [ ] Ad-hoc scripts (torch/numpy/safetensors)
- [ ] A library or tool: ______
- [ ] Visual inspection (plots, notebooks)
- [ ] Other: ______

## B. The three examples (5 minutes)

Rate 1 = strongly disagree … 5 = strongly agree. Write `n/a` if you did not get that far.

**Q4. Example 1 (checkpoint pair): "From the report I could tell what changed, where, and by how much."**

`1  2  3  4  5  n/a` → Answer:

**Q5. What did you want to know next about the pair that the report did not answer?** List the questions you would have asked. This is the most important question in this form.

Answer:

**Q6. Was anything in the reports confusing, surprising or possibly misleading?** For example: the compatibility class (V/L/S/X), a metric, the ranking, a caveat, the layer/role labels, the file layout.

Answer:

**Q7. Example 2 (LoRA): "The report made each module's effective update s·(B@A), and the rank and scale it used, clear."**

`1  2  3  4  5  n/a` → Answer:

**Q8. What would you want to see for an adapter that the report does not show?**

Answer:

**Q9. Example 3 (incomplete): "I would have noticed this report was incomplete without being told."**

`1  2  3  4  5  n/a` → Answer:

Where did you notice it first?

- [ ] `report.md` header
- [ ] the exit status
- [ ] the counts
- [ ] the caveats
- [ ] the CSV/JSON
- [ ] didn't notice

**Q10. When a tool cannot analyze part of your model, it should …** (one)

- [ ] refuse to produce any report
- [ ] report the analyzable part and mark it incomplete (the current behaviour)
- [ ] approximate the rest and label it as approximate
- [ ] other: ______

## C. Your own models and what comes next (3 minutes)

**Q11. Did you run `weightlab diff` on a checkpoint or adapter of your own (or the optional public pair)?**

- [ ] Yes, own
- [ ] Yes, the public Qwen pair/adapter
- [ ] No

If yes:

- Size:
- Time taken:
- Did it match what you know about that model?
- Was anything unsupported, refused or wrong? Give the error code if one was printed.

If no, what stopped you? For example: format (`.bin`, GGUF, quantized), adapter type, size, time, nothing to compare.

Answer:

**Q12. The graphical version (Phase 1/2) is being planned. Pick the THREE you would use first and number them 1, 2, 3.** (1 = most wanted)

- [ ] a. Searchable tensor list with shapes, dtypes and per-tensor change stats (the report, but interactive)
- [ ] b. Heatmap of one weight matrix, zoomable, with exact values on hover
- [ ] c. Heatmap of the **difference** B − A between two checkpoints, side by side with A and B
- [ ] d. Exact region-of-interest table: select cells, see the exact stored values, export them
- [ ] e. Statistics per tensor: histograms, percentiles, non-finite values
- [ ] f. LoRA overlay: A, B and s·(B@A) per module, with scale experiments
- [ ] g. Layer-by-layer dashboard: change per layer and per role across the whole model
- [ ] h. More than two checkpoints: a timeline over training checkpoints
- [ ] i. Exports with provenance (PNG/CSV/JSON) for papers and reviews
- [ ] j. Something else: ______

Why did you choose your number 1?

Answer:

**Q13. "I would use `weightlab diff` (or its graphical version) in my work."**

`1  2  3  4  5` → Answer:

What single change would raise your answer most?

Answer:

## D. Problems and consent (1 minute)

**Q14. Install or run problems?** Paste the error or the `run_examples.py` output if a check failed.

Answer:

**Q15. Consent** (default: no)

- [ ] You may quote my answers verbatim, without my name, in the project's feedback summary.
- [ ] You may contact me for a 20-minute follow-up. Contact: ______

Anything else:

Answer:
