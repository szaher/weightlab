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

The format stores an 8-byte little-endian header byte length, JSON metadata (may be right-padded with spaces), and a tensor data region. Each tensor definition has dtype, shape and `[begin,end)` byte offsets relative to the start of the tensor data region. Supported core dtypes: F64, F32, F16, BF16, I64, I32, I16, I8, U64, U32, U16, U8, BOOL where a safe numerical conversion exists. **F16 and BF16 are mandatory from R0**; BF16 is decoded by WeightLab because the safetensors NumPy backend cannot represent it (doc 03 section 6). F8 variants, complex and any unknown dtype string are rejected with `UNSUPPORTED_DTYPE` until explicitly tested. Dtypes with limited browser precision require output conversion disclosure; do not convert 64-bit integer values to JavaScript Number silently.

Mandatory validation:

1. Header byte length sane and accepted by format library limits; full file must accommodate header.
2. Header JSON well-formed and no duplicate tensor keys. Python's `json` silently keeps the last duplicate, so WeightLab parses with `object_pairs_hook` and rejects duplicates itself.
3. Shapes non-negative integers, product checked for integer overflow, and byte length exactly `product(shape)*dtype_size` for fixed-width tensors.
4. Offsets non-negative and within payload, and the ranges of all tensors, sorted by `begin`, must tile the data region exactly: no overlap, no gap, no trailing bytes. This matches what the upstream safetensors validator enforces (fixtures `overlapping_offsets`, `gap_in_offsets`).
5. No out-of-bound reads; file size and stat identity revalidated before each job.
6. Any file mutations during analysis cause job to fail with `FILE_CHANGED`; do not continue against stale offsets.
7. JSON metadata strings are treated as data, not executable code.

Prefer mature official library for parsing and validation; supplemental low-level mmap must use validated ranges only.

## 3. Sharded index contract

Expected `model.safetensors.index.json` with `weight_map`: exact tensor name -> relative `.safetensors` shard. Validate all referenced shard names are ordinary relative paths, normalize against selected root, disallow `..` or absolute paths and symlink escapes, check every key exists once in referenced shard, and do not treat catalog from incomplete shards as complete. Track missing/extra keys and report them. Path traversal rejected even if underlying OS would permit it. No automatic downloads.

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

**Class V (verified same base):** WeightLab can establish *cryptographically* that A's files are byte-identical to the base the derivative declares (`lineage_evidence = cryptographically_verified`), by one of:
1. SHA-256 of every referenced base file equals a hash recorded in the derivative's provenance (a WeightLab manifest, a model card hash list, or `expected_results.json` for fixtures).
2. Offline Hugging Face hub cache: both repos resolve in the local cache, the derivative declares the base repo **and revision**, the base snapshot is that revision, and each LFS blob's filename (the hub cache names LFS blobs by the file's SHA-256) matches a fresh hash of the file. No download.

Plus the same architecture, tensor names/shapes and tokenizer/config where behavioral interpretation requires it. Real-world adapters usually record only `base_model_name_or_path` without a revision or hash; those are Class L, not V. Weight similarity (for example per-tensor cosine(A,B) > 0.99 across all matched tensors) is recorded as supporting evidence for Class L, never as proof of V.  

User attestation is **not** a route to Class V. When the user states that these exact files were the training base, the hashes are recorded and the pair stays Class L with `lineage_evidence = user_attested`; that attestation satisfies the T2 acknowledgment (doc 06 section 6) but never unlocks behavior reserved for cryptographically verified lineage. Every compatibility decision records `lineage_evidence` as one of `cryptographically_verified`, `user_attested`, `declared` (name/ID only) or `none`.

**Class L (lineage declared or attested, not cryptographically verified):** Metadata references same base ID (`declared`) or the user attested it (`user_attested`), but no cryptographic proof; side-by-side descriptive comparisons allowed; transfer behind acknowledgment, default disabled; the evidence kind is shown wherever the class is shown.  
**Class S (shape-compatible only):** Exact names/shapes but unknown or different origin; difference may be meaningless due to parameter permutation/alignment; display with prominent caveat; transfer disabled.  
**Class X (incompatible):** Missing required tensors, shape mismatch, incompatible dtype/architecture or unsupported packing; no direct numerical combine. Per-tensor matching subsets may still be visualized as **unverified inspection**.

File content hash can establish identity, not functional equivalence or shared optimization trajectory. Naming equal tensors is never proof of semantic alignment.

**Tied parameters.** When `config.json` declares `tie_word_embeddings: true`, or a SafeTensors save dropped an alias of a shared tensor, the absent key (typically `lm_head.weight`) is reported as `tied -> model.embed_tokens.weight` and does not make the pair Class X. A comparison against an untied model compares the source tensor once and states that the tie differs.

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
