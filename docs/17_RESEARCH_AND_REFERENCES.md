# 17 — Literature, Documentation and Research Context

**Evidence checked:** 2026-11-05. Refer to live upstream releases when implementing because library APIs may change. These references support feasibility or scientific caution; they do not imply WeightLab has implemented or validated any feature.

## A. Tooling and formats

1. **SafeTensors documentation**, Hugging Face. Provides selective `safe_open(...).get_slice()` and shape metadata operations. https://huggingface.co/docs/safetensors/index
2. **SafeTensors source & format specification**, Hugging Face. Layout, header, shapes and data offsets. https://github.com/huggingface/safetensors
3. **SafeTensors NumPy Python source**, for actual NumPy backend behavior. https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/numpy.py
4. **Transformers big model documentation**, sharded checkpoint index `weight_map`. https://huggingface.co/docs/transformers/main/big_models
5. **PEFT LoRA API**, scaling and adapter variants; do not assume all variants are vanilla LoRA. https://huggingface.co/docs/peft/main/package_reference/lora
6. **GGUF technical specification** and quantization metadata: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
7. **llama.cpp**, optional CPU inference engine and quantized backend: https://github.com/ggml-org/llama.cpp
8. **TransformerLens patching documentation**, activation patching concepts and limitations: https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.patching.html

## B. Scientific foundations

9. Ilharco et al., **Editing Models with Task Arithmetic**, ICLR 2023. Fine-tuned minus original weights yields a task vector that may steer certain behaviors when applied in aligned models. https://arxiv.org/abs/2212.04089
10. Yadav et al., **TIES-Merging: Resolving Interference When Merging Models**, NeurIPS 2023. Parameter sign conflict and interference complicate merging. https://arxiv.org/abs/2306.01708
11. Ainsworth, Hayase & Srinivasa, **Git Re-Basin: Merging Models modulo Permutation Symmetries**, ICLR 2023. Raw parameter coordinate comparisons may be misleading across independently trained models. https://arxiv.org/abs/2209.04836
12. Max Klabunde and Florian Lemmerich, **Localizing Reasoning Training-Induced Changes in Large Language Models**, Mechanistic Interpretability Workshop at NeurIPS 2025. https://openreview.net/forum?id=WDlhBhceGZ ; author page https://mklabunde.github.io/publications/. Compares weight-space differences with representation similarity (CKA) between Qwen-derived base and reasoning-trained models. Its results are descriptive comparisons, not causal attributions of reasoning to specific weights. (Revision 1.1 wrongly marked this reference unverified; corrected in 1.2.)
13. Hugging Face evaluation guidebook (benchmark examples and cautions). https://github.com/huggingface/evaluation-guidebook

## C. Design implications

- **Selective read is possible**; do not infer every Python/NumPy slice is always zero-copy. Test specific versions and allocation behavior.
- **BF16 is not readable through the safetensors NumPy backend** (verified with 0.8.0); WeightLab decodes it from raw bits (ADR-011).
- **Task arithmetic is real research**; it is not universal capability cloning. Provenance and transfer eligibility are primary UI features, not paperwork.
- **LoRA update factors are non-unique**; compare the effective `B@A` update when reasoning about model deltas.
- **Activation patching is a separate intervention** from plotting trained weight matrices. Do not label attention coefficients as attention projection weights.
- **Large numerical differences are not guaranteed causally important** for improved math/reasoning accuracy.
- **Exact aggregate scans cost I/O** even when the final heatmap is 256x256 pixels.

## D. Planned validation research

Before R3: systematically select multiple adapters of the same verified base with varying fine-tune goals; compare descriptive weight metrics vs out-of-sample behavior. Include matched random control updates and alternate scales. Test reproducibility and confidence intervals. Publish limitations and negative results. Use only licensed data/models and document contamination risks.

## E. Citation discipline for reports

Every generated research report should cite original model identifier/revision, exact source hashes where available, dataset and license, numeric computation formula, inference backend/version, test split, all experiment settings, sampling method and limitations. If base lineage is unverified, front-page warning required.
