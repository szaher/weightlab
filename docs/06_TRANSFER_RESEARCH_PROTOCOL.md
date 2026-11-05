# 06 — Research Protocol: Weight Changes, Math Ability and Transfer

## 1. Research questions

**RQ1:** Which weights change when a specific fine-tune improves math accuracy?  
**RQ2:** Are those changes concentrated by component or layer more than a matched control training run?  
**RQ3:** Are changes in weight magnitude/direction related to behavioral improvements?  
**RQ4:** Can selected updates be transferred to a compatible target while preserving general capabilities?  
**RQ5:** Do interventions on an identified update produce changes beyond random matched-norm control updates?  
**RQ6:** What do different models with nominally similar math ability have in common in *representations or function*, not merely raw coordinates?

## 2. Starting experiment (evaluable on a laptop)

Use a small **verified base model** and one math adapter from it. Avoid relying on random unaudited public "math" checkpoints to establish lineage; instead generate fixture adapter and, if possible, train or obtain a documented small math adapter. Suggested family Qwen2.5 0.5B / Qwen3 0.6B **subject to the exact available release, license and device support verified by implementer**. The first test may use synthetic tiny model before real language models. Exact model IDs are configuration, not hardcoded into the viewer.

Dataset: small, license-reviewed held-out arithmetic/word-problem subset. Distinguish tasks from training data; record possible contamination. Include parity, answer extraction and numeric-equality metrics with a fully deterministic answer parser. For a rigorous study, use separate reasoning benchmarks and multiple seeds; small samples are only illustrative.

## 3. Experimental groups

A. Base model (unchanged).  
B. Same base + native math LoRA (positive control).  
C. Same base + math LoRA with selected layer/module disabled, implemented as a small derived adapter file with that module's A/B removed (adapters are megabytes, so no merged checkpoint is written).  
D. Same base + math LoRA scaled by `lambda in {0, 0.25, 0.5, 0.75, 1.0, 1.25}`.  
E. Same base + random/shuffled adapter delta matched in norm and relevant structure (negative control).  
F. Distinct same-base adapter target + sourced math delta (research transfer arm; ensure effective combined scaling convention).  
G. Incompatible/unrelated model: do **not** directly transfer; show compatibility report only.

For cross-checkpoint full-parameter task vectors: A=pretrained checkpoint, B=A fine-tuned; `Delta=B-A`; target C must share sufficiently verified weight coordinate system and intended evaluation configuration. Direct addition to independent B-family checkpoint is disabled until alignment experiment protocol exists.

## 4. Primary and secondary outcomes

Primary: paired held-out accuracy with per-example results (`exact match` after documented answer normalizer, or task-specific numeric checker); improvement relative to base.  
Secondary: non-math task retention, inference latency, token count, sample variation, update magnitude by layer, direction cosine between updates, distribution metrics.  
Negative outcomes: gain only on training-like tasks, control degradation, non-reproducible output, increased hallucination or token verbosity with no accuracy gain.

Do not claim "better at reasoning" from math score alone; it is evidence for measured task performance under stated prompts and generation settings.

## 5. Paired evaluation contract

- Use identical system instructions, prompt templates, tokenizer, chat template, context length, seed, decoding and hardware backend for both arms.
- Deterministic decoding `temperature=0` is preferable for first small experiment, but backend nondeterminism should be recorded. Multiple-seed protocol for stochastic sampling.
- Include per-item `question_id`, prompt hash, reference answer, raw model response, parser result, correctness, token count, response time, error cause and branch ID.
- Calculate paired accuracy difference; show confidence intervals via bootstrap across questions only when enough items exist, with bootstrap seed and method.
- Report raw counts not just percentages; visually label small-N result as exploratory.
- Freeze eval set before inspecting results to reduce cherry-picking; store declared exclusions and reasons.

## 6. Transfer eligibility rules

**T1:** LoRA adapters whose base is Class V (`lineage_evidence = cryptographically_verified`) with compatible target names/scaling -> eligible virtual composition; note potential interference.  
**T2:** LoRA adapter + same model architecture and base ID, lineage `declared` or `user_attested` -> require explicit user acknowledgment (a recorded attestation counts) and block export until cryptographically verified or overridden in recorded research mode.  
**T3:** Full-task delta and second fine-tune descended from exact shared base with matching tensor keys/shapes/dtypes/tokenizer -> permitted research-mode weighted addition, never silently assumed to improve behavior.  
**T4:** Similar model names/architectures but different training base or unverified coordinate system -> descriptive compare only; no automatic direct merge.  
**T5:** Different shapes/families -> no direct addition; an advanced projection/alignment study is separate scope, potentially using activation matching and calibration data, and requires proof of mapping.

## 7. Layer intervention

Module scale vector `m` where `m_l=0` disables adapter update in layer l and `m_l=1` enables it; `W_eff_l=W_base_l + m_l*s*B_l@A_l`. Grouping may use attention.q/k/v/o vs mlp.up/down/gate vs norms. Distinguish *block removal* from zeroing original weights. Realize `m` for evaluation by writing a derived adapter (module removed for `m_l=0`, `B_l` multiplied by `m_l` otherwise) whose hash is persisted; the base checkpoint is never rewritten. Model inference adapter must apply exactly this convention; UI preview must match evaluation's effective model. Persist method and virtual-update hash.

Intervention designs:

- Leave-one-layer-out: full adapter except layer `l` turned off.
- Only-one-layer-in: all off except `l`.
- Select attention/MLP family update selectively.
- Scale selected update while retaining the rest.

These interventions can interact; a simple sum of individual effects need not explain total gain. Present as empirical sensitivities, not independent causal attributions.

## 8. Statistical and scientific safeguards

- Sample variance and small test sizes limit conclusions.
- Training data overlap or benchmark contamination can masquerade as learned general reasoning.
- Different tokenizers/prompt templates make behavioral comparisons ambiguous.
- High raw weight deltas can reflect optimizer dynamics or representational reparameterizations, not greater causal importance.
- Same-shaped tensors can be unaligned due to symmetries/permutations.
- Full LoRA matrices A/B have non-unique factorizations; compare **effective** B@A delta rather than expecting A to align directly across independent trainings.
- Quantization error can confound transfer, especially if virtual update is applied before/after dequantization.
- Public models can use modified training pipelines, tied/untied embeddings, RoPE differences, differing layer counts or hidden base changes.

## 9. Experiment artifact

One experiment directory contains:

```text
experiment-<uuid>/
  manifest.json           # schema v1, file identity, model lineage, config, tool version
  eval_items.jsonl        # input IDs and reference outcomes, only if license permits
  runs/base.jsonl         # model outputs and per-item scores
  runs/treatment.jsonl
  runs/control.jsonl
  metrics.json            # paired aggregate and uncertainty
  interventions.json     # layer/module switches and alpha sweep
  assets/*.png            # plots with all axes/legends
  README.md               # protocol, limitations, conclusions
```

Never write model weights by default. Exports of merged model (later) require a distinct confirmation with disk estimate and license review.

## 10. Exit criteria for a publishable research claim

- Documented shared base lineage and exact weight coordinate system.
- Independent numerical unit tests for applied virtual delta.
- Held-out fixed eval set and at least one non-target control set.
- Negative control update and matched inference settings.
- Repeatability across seeds or deterministic reruns, uncertainty reported.
- Transparent negative results and failure modes.
- Clearly bounded claim: e.g., "on N specified arithmetic examples, toggling LoRA modules in layers X–Y changed accuracy by D points"; not "found math weights".

## 11. Relevant prior research

See `17_RESEARCH_AND_REFERENCES.md`: task arithmetic, TIES merging, Git Re-Basin, reasoning change localization, TransformerLens. WeightLab does not claim these methods as novel. Its distinctive goal is integrating *accurate, local, low-resource* inspection with reproducible experimental comparison.
