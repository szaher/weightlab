# 05 — Numerical Semantics, Statistics and Visualization Math

## 1. Prime rule

Numerical methods are products, not implementation details. Every metric includes formula, dtype, missing-value policy, approximation status, and enough metadata for reproduction. A heatmap pixel may represent one true parameter, hundreds of parameters, or a sample: the UI must disclose which.

## 2. Scalar representations

- Raw `W[i,j]`: input data in source dtype or displayed promoted float; tooltip includes original dtype.
- `Delta[i,j] = W_B[i,j] - W_A[i,j]`: explicit orientation B minus A.
- `Absolute delta = abs(Delta)`.
- `Relative delta = abs(Delta)/(abs(W_A)+epsilon)`; default epsilon `1e-8` for float32 comparisons, user adjustable; not a statistical percent and can be unstable near zero, hence mask/flag entries with `abs(W_A) < epsilon`.
- `LoRA delta = s (B @ A)` with native adapter scale and optional separate experiment multiplier.
- `Virtual W(alpha) = W_target + alpha*Delta_source` with alignment/lineage eligibility checks.
- `Gradient`: never call a weight difference a gradient. Delta is an accumulated result of optimization, not an instantaneous derivative.

## 3. Numerical promotion / accumulation

Default arithmetic conversion: F16/BF16 -> float32, F32 -> float32 for tiles and float64 accumulator for reductions; F64 -> float64. Integer comparisons promote to float64 only when exact range representable; otherwise require Python int or separate exact integer math to avoid silent precision loss. Arithmetic overflow/underflow safeguards; return `nonfinite_count`. Stable online Welford for mean/variance and compensated summation for sums when feasible. Use `np.isfinite` policy by default: finite-only stats with separate nonfinite counters. Norms on very large magnitude use scaled sum-of-squares algorithm to avoid overflow.

**R1 tolerance:** Tests state per-dtype tolerances, e.g. for float32 direct deltas `atol=1e-6, rtol=1e-5`; LoRA matmul float32 `atol=2e-5,rtol=1e-4` for fixture shapes; refine based on BLAS cross-platform numerical behavior. Do not set blanket tolerance for every tensor scale/dtype.

## 4. Exact streaming statistics

For finite elements `x1...xn`:

- count `n`, zeros, negatives, positives; NaN/+Inf/-Inf separate.
- mean `mu = sum(x)/n`, standard deviation **population** `sigma=sqrt(sum((x-mu)^2)/n)` unless `sample=true` explicitly.
- L1 `sum(abs(x))`, L2/Frobenius `sqrt(sum(x^2))` for matrix, RMS `sqrt(sum(x^2)/n)`.
- maxAbs, min, max.
- `relative_L2 = ||B-A||_2 / max(||A||_2, epsilon_norm)`; expose denominator and a `near_zero_reference` flag.
- cosine similarity `dot(A,B)/(||A||*||B||)`; null for zero norms or no overlapping finite pairs.
- Pearson correlation requires enough nonconstant finite pairs; null for zero variance or `n<2`.
- Percentiles exact requires memory-heavy sorting or external algorithm; R1 use bounded deterministic approximate quantile sketch/mergeable histogram with stated error policy, or offer offline exact selected ROI only. Never label global percentiles exact without proof.

For deltas, pairs with non-finite in either input excluded from finite pair metrics but counted in `invalid_pair_count`. All metrics declare total tensor elements vs analyzed finite elements.

## 5. Sampling

Supported sampling modes:

1. **Exact ROI:** Values for every element in small rectangle; no sampling.
2. **Deterministic stride preview:** Select predictable grid representatives; visually diagnostic but biased on periodic structures; label `stride preview`.
3. **Seeded pseudorandom sample:** For distribution summaries, sample positions by reproducible seed and report n and seed, avoiding duplicate indices where relevant. Positions are derived from SplitMix64 (`splitmix64(seed_mix ^ counter)`, reference implementation in `scripts/generate_demo_fixtures.py`), where `seed_mix` hashes (file fingerprint, tensor key, level, tile, user seed). Never use NumPy RNG streams, whose output is not guaranteed stable across NumPy releases.
4. **Exact pyramid / aggregate scan:** Stream all elements once in row stripes and build exact block aggregates for every level (doc 03 section 5). Default for tensors <= `pyramid_exact_max_mib`; an explicit queued job above it.
5. **Row-sampled tile:** Only for tensors above `pyramid_exact_max_mib`. Read a declared number of whole rows (deterministic stride or seeded selection) sequentially and aggregate them per block; report `sampled_rows`, total rows and policy. Per-cell scattered sampling is not used because its random I/O can exceed an exact scan.

Do not use ordinary sampled estimates to infer tiny differences in models without confidence/variance reporting. Where uncertainty is not estimated, mark `exploratory` not statistically validated. Sampled L2 norms must be called *estimates* with explicit scaling and error caveat.

## 6. Aggregation definitions

For a block `S` of original tensor elements:

- mean: `sum_{i in finite S} x_i / count_finite`
- mean-absolute: `sum |x_i| / count_finite`
- RMS: `sqrt(sum x_i^2 / count_finite)`
- min / max: min or max over finite values
- non-finite mask: show if any NaN/Inf; optionally show fraction overlay
- empty/all-nonfinite: return null and status, never fabricate zero

For a *difference heatmap* compute elementwise difference then aggregate; subtracting independently aggregated RMS or absolute values is **not** the same operation. For signed mean, `mean(B-A)=mean(B)-mean(A)` with identical finite pair support only. For a LoRA overlay, compute each needed block/ROI without allocating the full effective weight.

## 7. Color/legend

- Default signed difference: colorblind-safe **diverging** centered at exactly zero; symmetric `[−M,+M]`, where M is visible fixed or robust percentile threshold; out-of-range clipped samples indicated.
- Default raw: sequential palette for magnitude with optional diverging option for signed raw, user-controlled.
- Shared-scale comparison between A/B; raw side-by-side autoscaling prohibited unless visibly unlocked and per-panel legends shown.
- Missing/non-finite transparent checkerboard or patterned neutral texture; separate markers for NaN, Inf and unavailable.
- Quantization and relative delta views use separate legend with method text.
- Keep real numeric scale and data min/max distinct; changing scale must not change underlying values.

## 8. Relationship between weights

The UI may show:

- Row/column cosine similarities, but computing all pairwise similarities is O(n^2) in number of rows and costly. Permit selected rows or bounded samples first.
- Norm profiles across rows/columns, activation-independent descriptive geometry.
- Sign agreement, delta direction cosine and overlap of large-magnitude changes.
- Singular value spectra estimated via randomized SVD on bounded tensors only, labeled approximations; full decomposition may require large RAM.
- Cross-layer comparison with different shapes only via **summaries**, never elementwise subtraction.
- Attention/MLP role aggregation helps navigation but does not prove a semantic function.

## 9. Causal interpretation guardrails

**Observed update:** `||Delta||` high means parameters changed in magnitude, not that they were responsible for reasoning improvement.  
**Association:** A change correlates with improved benchmark accuracy; it may be confounded by other updates.  
**Intervention:** Disable/restore/scale update and measure output change with matched settings; supports more causal claims about that specific intervention, subject to distribution shift and interactions.  
**Mechanistic proof:** Requires deeper circuit and activation evidence beyond weight heatmaps.

## 10. Numerics acceptance fixtures

Fixture family:

- Linear ramp 8x8 with known cell values; parity tests for x/y axes, transpose and ROI.
- All zeros (relative denominators and zero cosine).
- Constant 1 (zero variance Pearson).
- Mixed signs with exactly known mean/RMS.
- BF16/F16 known values, NaN, +/-Inf, -0.0 and max finite (`tiny_dtypes.safetensors`).
- Nan/Inf/−Inf mask.
- Delta introduced at two known coordinates.
- LoRA rank-1 (via `rank_pattern`) and rank-2 hand-computable examples; rsLoRA rank-4; DoRA rejection.
- Odd-size 257x263 sine matrix for tile boundary and weighted parent aggregation (edge block expectations at L1 and L3).
- Large sparse file fixture to test bounded RAM and cancellation.
- Same-shape permuted representation with identical function (`compat/permuted_same_function`) to show why simple coordinate alignment is misleading.

## 11. Exported metric schema

Every metric object includes `name`, `value`, `units`, `method`, `source_dtype`, `compute_dtype`, `element_count`, `finite_count`, `sample_count`, `exact`, `seed` (if any), `epsilon` (if any), `numerical_warnings`, `tool_version`, and model fingerprint references.
