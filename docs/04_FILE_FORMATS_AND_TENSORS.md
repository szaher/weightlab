# 04 — File Formats, Reader Contract and Compatibility

## 1. Format capability matrix

| Input | R1 | Later | Notes |
|---|---|---|---|
| Single SafeTensors | Metadata, ROI, stats, compare | More dtypes | Core format |
| HF sharded SafeTensors + index JSON | Full support | — | Parse `weight_map`, verify referenced files |
| SafeTensors F8 (`F8_E4M3`, `F8_E5M2`, ...) | Explicit unsupported R1 (`UNSUPPORTED_DTYPE`) | R2 | Needs decoder and golden vectors; never reinterpret as int8 |
| PEFT vanilla LoRA adapter SafeTensors | Supported modules/scale | DoRA, IA3, variants | Must validate adapter_config and A/B pair |
| GGUF FP16/F32 | Explicit unsupported R1 | R2 | Names/layout differ; require format reader |
| GGUF Q4/Q5/etc | Explicit unsupported R1 | R2/R3 | Dequantization and block packing format-aware |
| `.bin`, `.pt`, `.pth` | Unsupported | Evaluate safe conversion tool | Never load pickle by default |
| ONNX | Unsupported | Possible reader plugin | Different tensor naming/graphs |
| safetensors from non-LLM network | Raw numerical inspection allowed | Architecture views | Do not imply transformer semantics |

## 2. SafeTensors structural contract

The format stores an 8-byte little-endian header byte length, JSON metadata (may be right-padded with spaces), and a tensor data region. Each tensor definition has dtype, shape and `[begin,end)` byte offsets relative to the start of the tensor data region. Supported core dtypes: F64, F32, F16, BF16, I64, I32, I16, I8, U64, U32, U16, U8, BOOL where a safe numerical conversion exists. **F16 and BF16 are mandatory from R0**; BF16 is decoded by WeightLab because the safetensors NumPy backend cannot represent it (doc 03 section 6). F8 variants, complex and any unknown dtype string are rejected with `UNSUPPORTED_DTYPE` until explicitly tested. Exact inspection is lossless for every listed dtype through the typed ROI encodings of doc 09 section 2 (64-bit integers as decimal strings, floats as exact float64 JSON numbers or raw bits); tiles are float32 visualizations and are labeled as such. 64-bit integers are never converted to JavaScript Number.

Mandatory validation:

1. Header byte length N: 8-byte little-endian prefix, `N <= 100,000,000` (the upstream safetensors cap) and `8 + N <= file size`. The header may be right-padded with spaces.
2. **Strict JSON:** valid UTF-8; standard JSON only, so `NaN`, `Infinity` and `-Infinity` tokens are rejected (Python `json.loads` accepts them by default: pass `parse_constant` that raises); duplicate keys rejected at every level (`object_pairs_hook`); nesting depth bounded (parse failure or `RecursionError` -> `INVALID_FORMAT`). The top level is an object; `__metadata__`, if present, maps strings to strings only (no nesting, numbers or null).
3. **Strict entries:** every other key is a tensor name (UTF-8, <= 4096 bytes, compared byte-for-byte, no Unicode normalization) whose value has exactly the fields `dtype`, `shape`, `data_offsets`. `dtype` is a known string; `shape` is a list of at most 32 non-negative JSON integers (booleans and floats rejected; in Python `bool` is an `int` subclass, so test `type(v) is int`); `data_offsets` is two JSON integers. Element count and byte length use exact integer arithmetic (`math.prod`), and `end - begin == numel * dtype_size`. Zero-sized dimensions are legal (`numel = 0`, `begin == end`): such tensors appear in the catalog with an "empty tensor" state, have no heatmap, and report statistics with `count = 0` and null values.
4. Offsets non-negative and within payload, and the ranges of all tensors, sorted by `begin`, must tile the data region exactly: no overlap, no gap, no trailing bytes. This matches what the upstream safetensors validator enforces (fixtures `overlapping_offsets`, `gap_in_offsets`).
   WeightLab is deliberately stricter than upstream in two places, recorded as permitted differences in `expected_results.json`: rank above 32 and unknown fields in a tensor entry are rejected by WeightLab but accepted by safetensors 0.8.0. For all other fixtures, upstream acceptance is the compatibility oracle.
5. No out-of-bound reads; handle and path identity revalidated before, during and after each job (doc 03 section 6).
6. Detected file changes during analysis fail the job with `FILE_CHANGED`; do not continue against stale offsets. Detection is best-effort unless the job runs with `consistency=verified`.
7. JSON metadata strings are treated as data, not executable code.

Prefer mature official library for parsing and validation; supplemental low-level mmap must use validated ranges only.

## 3. Sharded index contract

Expected `model.safetensors.index.json` with `weight_map`: exact tensor name -> relative `.safetensors` shard, parsed with the same strict JSON rules as headers. Every value must be a string holding a relative POSIX-style path: no backslashes, no drive letter or UNC prefix, not absolute, no `..` component; it is normalized against the selected root and must not escape it through symlinks or reparse points. Keys are compared byte-for-byte. Check every key exists once in its referenced shard and do not treat a catalog from incomplete shards as complete. `metadata.total_size`, if present, must be an integer; a mismatch with the shards is a warning, not an error. Track missing/extra keys and report them. Path traversal is rejected even if the OS would permit it. No automatic downloads.

## 4. Reader interface

```python
@dataclass(frozen=True)
class TensorMeta:
    key: str
    shape: tuple[int, ...]
    dtype: str
    shard_id: str
    data_offset: int
    byte_length: int
    role: str | None

class TensorReader(Protocol):
    def metadata(self) -> list[TensorMeta]: ...
    def read_roi_2d(self, key: str, r0: int, r1: int, c0: int, c1: int,
                    axes: tuple[int,int] | None = None,
                    fixed_indices: dict[int,int] | None = None) -> np.ndarray: ...
    def iter_flat_chunks(self, key: str, chunk_bytes: int) -> Iterator[np.ndarray]: ...
```

Contract: `read_roi_2d` returns exactly `(r1-r0,c1-c0)` in logical layout, within caller budget; never return writable views backed by original bytes; exact value conversion to native float32/f64 may be needed for arithmetic. Every call uses only authorized tensor key and bounded coordinates. `iter_flat_chunks` covers each numeric element exactly once in deterministic order for exact stats. For non-contiguous data, implement stripe planner and I/O cost estimation rather than allocation of full matrix.

## 5. Tensor dimensions and shapes

- Rank 0: scalar numerical metadata, single value.
- Rank 1: render line/histogram/table, not a fake 2D heatmap without labeling.
- Rank 2: raw heatmap `(rows,cols)` exactly as stored.
- Rank >=3: require choice of `row_axis`, `col_axis` and fixed indices for remaining axes; each selected axis must differ; dimension order never guessed silently. Optional reductions for remaining axes are P1 and must state method and cost.
- Rank 0 and 1 comparisons still supported for stats when shapes match.
- Transposition requires explicit UI toggle and appears in provenance as `transpose=true`.

## 6. LoRA mapping contract

For an ordinary linear layer with weight `W` shape `(out_features,in_features)`, `A` shape `(r,in_features)`, `B` shape `(out_features,r)`, `delta = scale * B @ A`. Support only a **declared tested subset**:

- **Key mapping:** saved PEFT keys have the adapter name stripped: `base_model.model.<module path>.lora_A.weight`. Canonical mapping strips the `base_model.model.` prefix and the `.lora_A.weight` / `.lora_B.weight` suffix, then appends `.weight` (or `.base_layer.weight` where the base was saved wrapped). It is a controlled function with tests; no fuzzy name matching without explicit user confirmation.
- **Scale per module:** `r = rank_pattern.get(module, r)`, `alpha = alpha_pattern.get(module, lora_alpha)`; `scale = alpha/r`, or `alpha/sqrt(r)` when `use_rslora`. Pattern keys are matched as PEFT does (suffix/regex), with tests per form.
- **Layer selection:** `layers_to_transform` / `layers_pattern` restrict which layers carry the adapter; modules outside it are `no adapter`, not errors.
- **`fan_in_fan_out=true`** means the base weight is stored `(in,out)` (GPT-2 `Conv1D`): the effective delta is `scale * (B @ A).T` in the stored layout. Apply it only with a dedicated golden test and `transpose=true` in provenance; otherwise reject. This is the classic silent-transpose bug.
- **Unsupported, fail closed:** `use_dora` or `lora_magnitude_vector` keys (fixture `tiny_lora_dora_unsupported`), embedding LoRA (`lora_embedding_A/B`), `modules_to_save`, LoHa/LoKr/IA3, non-linear targets. Disable effective-weight inference for those modules and show which parts cannot be accounted for.

Example precise tile computation:

```python
# W, A, B must already have validated compatible shapes.
# [r0:r1, c0:c1] is desired ROI, and r is adapter rank.
A_sub = A[:, c0:c1].astype(np.float32)
B_sub = B[r0:r1, :].astype(np.float32)
delta_roi = scale * (B_sub @ A_sub)
base_roi = reader.read_roi_2d(weight_key, r0, r1, c0, c1)
effective_roi = base_roi.astype(np.float32) + delta_roi
```

Allow output ROI only if intermediate arrays fit budget. If user chooses a large ROI, tile it or reject with limit error. For native PEFT adapter + base, distinguish adapter's training scale from extra experimental scale `lambda`.

## 7. Model comparability classification

Two questions are kept apart in every compatibility decision (`LineageAssessment` in `contracts/openapi.yaml`):

1. **Is the base file the one the derivative names?** `base_file_identity_verified` is true when the base files are byte-identical to a reference the derivative's record names, established by (a) SHA-256 of every referenced base file equal to a hash recorded in the derivative's provenance (`base_identity_method=sha256_recorded`), or (b) the offline Hugging Face hub cache: the derivative declares the base repo **and revision**, the base snapshot is that revision, and each LFS blob's filename (the hub cache names LFS blobs by the file's SHA-256) matches a fresh hash (`hf_cache_blob_sha256`). No download.
2. **Was the derivative actually trained from that base?** A hash match cannot prove this: anyone can write a genuine base hash into a fabricated derivative's metadata, and model cards are unsigned claims. `lineage_claim_source` records where the ancestry statement comes from (`adapter_config`, `model_card`, `weightlab_manifest`, `synthetic_fixture_generator`, `user_attestation`, `none`) and `lineage_trust_level` how far it is trusted: `proven_by_construction` (only fixtures built by WeightLab's generator), `trusted_statement` (a manifest the user explicitly marks as trusted, e.g. from their own training run), `unverified_claim`, or `none`.

**Class V (base identity verified):** `base_file_identity_verified=true` plus the same architecture, tensor names/shapes and tokenizer/config where behavioral interpretation requires it. Class V means "the base file is the one named", **not** "training ancestry proven". `lineage_evidence` is `cryptographically_verified`.

User attestation is not a route to Class V. When the user states that these exact files were the training base, the hashes are recorded and the pair stays Class L with `lineage_evidence = user_attested`, `lineage_claim_source = user_attestation`; it satisfies the T2 acknowledgment (doc 06 section 6) and nothing more. Real-world adapters usually record only `base_model_name_or_path` without a revision or hash: Class L with `lineage_evidence = declared`. Weight similarity (for example per-tensor cosine(A,B) > 0.99 across all matched tensors) is recorded as supporting evidence for Class L, never as proof.

Transfer eligibility (doc 06 section 6) needs both: T1 requires Class V **and** `lineage_trust_level` in {`proven_by_construction`, `trusted_statement`}; Class V with `unverified_claim` is research-mode only. The generated fixture pair `tiny_base`/`tiny_ft` is Class V with `proven_by_construction` because the generator builds one from the other, and the UI says so ("synthetic: ancestry known by construction").

**Class L (lineage declared or attested, not cryptographically verified):** Metadata references same base ID (`declared`) or the user attested it (`user_attested`), but no cryptographic proof; side-by-side descriptive comparisons allowed; transfer behind acknowledgment, default disabled; the evidence kind is shown wherever the class is shown.  
**Class S (shape-compatible only):** Exact names/shapes but unknown or different origin; difference may be meaningless due to parameter permutation/alignment; display with prominent caveat; transfer disabled.  
**Class X (incompatible):** Missing required tensors, shape mismatch, incompatible dtype/architecture or unsupported packing; no direct numerical combine. Per-tensor matching subsets may still be visualized as **unverified inspection**.

File content hash can establish identity, not functional equivalence or shared optimization trajectory. Naming equal tensors is never proof of semantic alignment.

**Tied parameters.** A missing key is reported as `tied -> <source key>` only with positive evidence: `tie_word_embeddings: true` in `config.json` together with a known architecture mapping (`lm_head.weight` <-> `model.embed_tokens.weight` for Llama/Qwen-style keys, `lm_head.weight` <-> `transformer.wte.weight` for GPT-2-style keys), or trusted alias metadata. Without such evidence the key is `missing` (fixture `compat/missing_head_no_tie_evidence`). A comparison against an untied model compares the source tensor once and states that the tie differs. Missing values are never invented.

## 8. GGUF specific later considerations

GGUF contains metadata and tensors with documented block formats; quantized values are packed groups, not ordinary fp16 floats. Require validated parser/decoder, test vectors for each quantization type, and documented layout conversion from GGUF naming/transpose conventions. Compare dequantized numeric values only with disclosed precision loss. Inference via llama.cpp does not imply correctness of arbitrary GGUF heatmap slicing. Do not use packed block bytes as heatmap values.

## 9. Data integrity / provenance

Persist canonical path/selected root, type, file size, high-resolution mtime and optional SHA-256; loaded tensor key/shape/dtype; API parser version; program version; color/aggregation/sampling settings; source role maps; adapter scale config. Never assert a hash from only path+mtime is collision-resistant.

## 10. Required format tests

- 1x1, empty permitted tensor shapes, 2D odd dimensions, 3D, huge metadata, shard index, BF16 bit decode.
- Truncated header, corrupt JSON, negative/overflow shape, offset overlap, missing shard, key missing in shard, path traversal, symlink escape.
- LoRA A/B swapped, rank mismatch, `rank_pattern`/`alpha_pattern`, rsLoRA factor, `fan_in_fan_out`, embedding LoRA rejected, nonstandard module, missing source base, DoRA present.
- BF16/F16 bit decode incl. specials; F8 and unknown dtype rejected; duplicate key; gap and overlap in offsets; tied embeddings.
- Change file mid-read -> fail and invalidate cache.
