# 07 — Optional Activation / Inference Design

## 1. Why optional?

Weights are static checkpoint data. Activations require a forward pass on actual input and can be much more memory-expensive than weight inspection, especially with many layers, tokens and captured positions. R1 must work without inference, Torch or GPU. R3/R4 require separate installation and explicit preflight.

## 2. Capability levels

- **Level 0 (R1):** Weight-only exploration; no model execution; all calculations operate on checkpoint tensors.
- **Level 1 (R3):** Generate/evaluate outputs for a small model on CPU in an isolated subprocess. **Default backend: `transformers` + `peft` on CPU, loading the same SafeTensors files the viewer inspected** (about 2 GiB at FP32 or 1 GiB at BF16 for a 0.5B model), so the evaluated effective weights equal the previewed ones (ADR-015). Module masks and scales are realized as small derived adapter files (doc 06 section 7). `llama.cpp` is an optional faster backend, but it needs GGUF conversion and usually quantization, so its runs are labeled `weights differ from viewer: <quant type>` and cannot satisfy the R3 'preview equals evaluated model' gate. Hardware/model support must be checked before launch.
- **Level 2 (R4):** Selected activation capture for small supported models through optional PyTorch/TransformerLens plugin. Selection by layer, token positions and hook names; no full-network cache default.
- **Level 3 (research):** Activation intervention (zero/restore/patch) and causal metrics. Multiple runs and datasets required; substantial runtime warning.

## 3. Preflight estimator

Estimate: model file memory mapping and working set, runtime KV cache `(layers * heads_or_kv_heads * head_dim * tokens * bytes * K/V factor)` adjusted for architecture and quantization, captured activation `(selected_layers * selected_positions * hidden_dim * bytes * batch)` and output buffers. These are **rough estimates**, not permission to exceed hard OS/process limits. Show available RAM and planned maximum; refuse impossible combinations; enforce subprocess memory restrictions where operating system allows and track observed RSS/commit.

## 4. Backend contract

```python
class EvaluationBackend(Protocol):
    def probe(self) -> BackendCapabilities: ...
    def estimate(self, model, prompt_set, generation_config) -> ResourceEstimate: ...
    def run(self, experiment_spec, cancel_token) -> Iterator[EvaluationItem]: ...
```

Capabilities: `supports_model_format`, `supports_adapter`, `supports_virtual_delta`, `supports_module_mask`, `supports_activation_hooks`, `cpu_only`, `max_context_length`. Never pretend llama.cpp supports arbitrary custom per-layer delta injections; implement an explicitly tested adapter or announce unsupported feature. A fallback that writes a large merged checkpoint defeats laptop goals and is not automatic.

## 5. Activation capture contract

`capture_scope`: named hook points + layers + token positions + batch size + dtype. Default capture selected 1-3 layers and 1-16 token positions on toy models. Return small matrices and metadata. Refuse unbounded `all_layers/all_tokens/full_residual` without explicit high-budget override. Data may contain user prompt content; treat as private. Store only chosen aggregate visualization unless user opts into full capture.

## 6. Interpretation

Attention weights are token-to-token attention coefficients, **not** the trained Q/K/V weight matrices. A screenshot labeled `attention heatmap` must specify which object it depicts. Activation patching is a causal intervention conditional on tested examples and model, not proof of universally localized math/reasoning. It is possible to find many layers involved in a distributed capability.

## 7. Evaluation dataset lifecycle

- R1 demo: purely synthetic numerical fixtures, no model download.
- R3 example: a small disclosed math subset with licensing check, independent control set, answer parser tests and no reliance on network on rerun.
- Each dataset version has SHA-256, `split`, `source`, `license`, `prompt_template`, `normalizer`, and potential contamination warnings.
- User can supply local JSONL with schema validation. Arbitrary Python scoring scripts prohibited by default; scorer plugins are explicit locally trusted code.

## 8. Cancellation and fault isolation

Run optional inference in subprocess with timeout, CPU/thread cap, OS-level containment as available, process group kill on cancel and structured error result. A failed inference must not take down the viewer; background crash reported with traceback only in local diagnostic log (redacted where possible).

## 9. Non-goals

No automatic full fine-tuning, no guarantees of cross-engine output equivalence, no activation capture for all arbitrary architectures, no language-agent execution, no user code execution from model directories, no multi-tenant web service.
