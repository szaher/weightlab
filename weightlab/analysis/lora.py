"""PEFT LoRA adapter analysis for the change report (doc 04 section 6, FR-023, doc 11).

Declared tested subset only:
  * keys `base_model.model.<module>.lora_A.weight` / `.lora_B.weight` (adapter name stripped) map to
    base `<module>.weight` or `<module>.base_layer.weight`; no fuzzy matching;
  * per-module rank/alpha from `rank_pattern` / `alpha_pattern` (literal suffix patterns only;
    patterns with other regex syntax fail closed), scale alpha/r or alpha/sqrt(r) with `use_rslora`;
  * `fan_in_fan_out=true`: base stored (in, out), delta = scale * (B @ A).T (transpose=true);
  * `layers_to_transform` consistency checked against the module's layer index.
Unsupported, fail closed and listed: DoRA (`use_dora`, `lora_magnitude_vector`), embedding LoRA,
`modules_to_save`, non-LORA peft types, bias training, unknown adapter keys, regex patterns.
Effective delta per module is computed in float64 row blocks, never materializing B @ A for the
whole matrix; modules above `max_lora_scan_gmac` get only the Frobenius norm via
||s B A||_F^2 = s^2 * sum((B^T B) * (A A^T)) (tested against the direct product).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from weightlab.analysis.accumulate import ExpSum, PairStats
from weightlab.analysis.diff import RunControl
from weightlab.analysis.roles import layer_index
from weightlab.core.errors import ErrorCode, fail
from weightlab.formats.base import Consistency
from weightlab.formats.checkpoint import Checkpoint, open_checkpoint, read_json_object

PREFIX = "base_model.model."
_LITERAL = re.compile(r"^[A-Za-z0-9_.]{1,256}$")
_AB = re.compile(r"^(?P<module>.+)\.lora_(?P<which>A|B)\.weight$")
DEFAULT_MAX_GMAC = 50.0
DEFAULT_BLOCK_CELLS = 1 << 18
MAX_FACTOR_CELLS = 1 << 24  # A or B loaded whole: 128 MiB of float64 at most
MAX_RANK = 1 << 16  # larger r / rank_pattern values are rejected as invalid configuration
# Hard cap on the module working set, estimated from shapes before anything is allocated:
# both factors in float64, plus either the row-block temporaries or the r x r Gram scratch.
MAX_WORKING_SET_BYTES = 384 * 1024 * 1024
_BLOCK_TEMPS = 4  # w, d, eff and a transposed slice per row block
_GRAM_TEMPS = 3  # A A^T, B^T B (reused for the product) and one matmul temporary


@dataclass
class AdapterConfig:
    raw: dict[str, Any]
    r: int
    alpha: float
    use_rslora: bool
    fan_in_fan_out: bool
    rank_pattern: dict[str, Any]
    alpha_pattern: dict[str, Any]
    layers_to_transform: list[int] | None
    global_unsupported: list[str]


@dataclass
class Module:
    module: str  # module path without prefix/suffix
    base_key: str | None
    status: str  # analyzed | over_budget_norm_only | unsupported | skipped
    rank: int | None = None
    alpha: float | None = None
    scale: float | None = None
    scale_formula: str | None = None
    rank_from: str | None = None
    transpose: bool = False
    gmac: float | None = None
    stats: PairStats | None = None
    norm_only_l2: float | None = None
    norm_only_ssq: ExpSum | None = None
    norm_only_issue: str | None = None  # why the norm-only value is withheld (None = valid)
    base_ssq: ExpSum | None = None
    base_norm_l2: float | None = None
    base_finite: int = 0
    base_nonfinite: int = 0
    reasons: list[str] = field(default_factory=list)


@dataclass
class AdapterResult:
    config: AdapterConfig
    modules: list[Module]
    unsupported_keys: list[tuple[str, str]]  # (adapter key, reason)
    base_tensors_without_adapter: int
    adapter_digest: str

    @property
    def complete(self) -> bool:
        return not self.unsupported_keys and all(m.status == "analyzed" for m in self.modules)


def _num(v: object) -> bool:
    """A finite JSON number that converts to float64 (huge JSON integers do not)."""
    if type(v) not in (int, float):
        return False
    try:
        return math.isfinite(float(v))  # type: ignore[arg-type]
    except OverflowError:
        return False


def _rank_ok(v: object) -> bool:
    return type(v) is int and 0 < v <= MAX_RANK


def load_config(adapter_dir: Path, root: Path | None = None) -> AdapterConfig:
    path = adapter_dir / "adapter_config.json"
    if not path.is_file():
        raise fail(ErrorCode.NOT_FOUND, "adapter_config.json not found in the adapter directory")
    doc: dict[str, Any] = read_json_object(path, root or adapter_dir, "adapter_config.json")
    unsupported: list[str] = []
    if doc.get("peft_type") != "LORA":
        unsupported.append(f"peft_type {doc.get('peft_type')!r} is not LORA")
    if doc.get("use_dora") is True:
        unsupported.append("use_dora=true (DoRA magnitude vectors are not supported)")
    if doc.get("modules_to_save"):
        unsupported.append("modules_to_save replaces whole modules; those weights are not analyzed")
    if doc.get("bias") not in (None, "none"):
        unsupported.append(f"bias={doc.get('bias')!r}: trained biases are not analyzed")
    for name, why in (
        ("lora_bias", "lora_bias adds a trained bias term"),
        ("use_qalora", "QALoRA"),
        ("trainable_token_indices", "trainable token embeddings"),
        ("target_parameters", "LoRA on raw parameters"),
        ("layer_replication", "layer replication"),
    ):
        if doc.get(name) not in (None, False, [], {}, ""):
            unsupported.append(f"{name}: {why} is not supported")
    if doc.get("layers_pattern") not in (None, [], ""):
        unsupported.append("layers_pattern is not supported in R0.5")
    r, alpha = doc.get("r"), doc.get("lora_alpha")
    if not _rank_ok(r) or not _num(alpha):
        raise fail(
            ErrorCode.INVALID_FORMAT,
            f"adapter_config needs an integer r in [1, {MAX_RANK}] and a finite numeric lora_alpha",
        )
    for name in ("rank_pattern", "alpha_pattern"):
        if not isinstance(doc.get(name) or {}, dict):
            raise fail(ErrorCode.INVALID_FORMAT, f"{name} must be an object")
    ltt = doc.get("layers_to_transform")
    layers: list[int] | None
    if ltt is None:
        layers = None
    elif type(ltt) is int:
        layers = [ltt]
    elif isinstance(ltt, list) and all(type(v) is int for v in ltt):
        layers = list(ltt)
    else:
        raise fail(ErrorCode.INVALID_FORMAT, "layers_to_transform must be an integer or a list of integers")
    for flag in ("use_rslora", "fan_in_fan_out"):
        if doc.get(flag) not in (None, True, False):
            raise fail(ErrorCode.INVALID_FORMAT, f"{flag} must be a boolean")
    return AdapterConfig(
        doc,
        r,  # type: ignore[arg-type]  # _rank_ok() checked above
        float(alpha),  # type: ignore[arg-type]  # _num() checked above
        doc.get("use_rslora") is True,
        doc.get("fan_in_fan_out") is True,
        dict(doc.get("rank_pattern") or {}),
        dict(doc.get("alpha_pattern") or {}),
        layers,
        unsupported,
    )


def _pattern_value(patterns: dict[str, Any], module: str) -> tuple[str | None, Any, str | None]:
    """(matched pattern, value, error). PEFT semantics: a pattern matches the module path or a
    dot-delimited suffix of it; first match in insertion order wins."""
    for pat, value in patterns.items():
        if not _LITERAL.match(pat):
            return None, None, f"pattern {pat[:40]!r} uses regex syntax (not supported; fail closed)"
        if module == pat or module.endswith("." + pat):
            return pat, value, None
    return None, None, None


def resolve_scale(cfg: AdapterConfig, module: str) -> tuple[int, float, float, str, str] | str:
    """(rank, alpha, scale, formula, rank source) or an error string."""
    rp, rv, err = _pattern_value(cfg.rank_pattern, module)
    if err:
        return err
    ap, av, err = _pattern_value(cfg.alpha_pattern, module)
    if err:
        return err
    rank = cfg.r if rp is None else rv
    alpha = cfg.alpha if ap is None else av
    if not _rank_ok(rank) or not _num(alpha):
        return f"rank/alpha pattern value is not an integer in [1, {MAX_RANK}] / a finite number"
    alpha = float(alpha)
    if cfg.use_rslora:
        return rank, alpha, alpha / math.sqrt(rank), "alpha/sqrt(r) (rsLoRA)", "rank_pattern" if rp else "r"
    return rank, alpha, alpha / rank, "alpha/r", "rank_pattern" if rp else "r"


def gram_frobenius_ssq(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], scale: float
) -> tuple[ExpSum, str | None]:
    """||scale * B @ A||_F^2 = scale^2 * sum((B^T B) * (A A^T)), O((in+out) r^2) time, 3 r^2 scratch.

    Both factors are first scaled by exact powers of two so their largest magnitude is in [0.5, 1);
    the scaled Gram sum is then far from float64 limits and the true exponent is carried separately
    (an ExpSum), so huge or tiny factors neither overflow nor underflow to a false zero.
    Returns (sum, issue): `issue` is set, and the sum must not be reported, when the factors are not
    finite or when floating-point cancellation leaves no trustworthy non-negative value.
    `a` and `b` are scaled in place (callers pass private copies)."""
    out = ExpSum()
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return out, "LoRA factors contain NaN/Inf"
    pa = float(np.max(np.abs(a))) if a.size else 0.0
    pb = float(np.max(np.abs(b))) if b.size else 0.0
    if pa == 0.0 or pb == 0.0 or scale == 0.0:
        return out, None  # exactly zero update
    ka, kb = math.frexp(pa)[1], math.frexp(pb)[1]
    np.ldexp(a, -ka, out=a)
    np.ldexp(b, -kb, out=b)
    ga = a @ a.T
    gb = b.T @ b
    np.multiply(gb, ga, out=gb)
    total = float(np.sum(gb))
    # Rounding error of the signed total is at most gamma * sum((|B|^T |B|) * (|A| |A|^T)) with
    # gamma ~ (in + out + r^2) eps (Higham, dot products and sums); a total within twice that of zero
    # (or negative) cannot be told apart from cancellation noise. Same buffers, no new scratch.
    np.abs(a, out=a)
    np.abs(b, out=b)
    np.matmul(a, a.T, out=ga)
    np.matmul(b.T, b, out=gb)
    np.multiply(gb, ga, out=gb)
    gamma = 2 * (a.shape[1] + b.shape[0] + a.shape[0] ** 2 + 2) * float(np.finfo(np.float64).eps)
    bound = float(np.sum(gb)) * gamma
    if not math.isfinite(total) or total <= bound:
        return out, "Gram-trace sum lost to cancellation (B @ A nearly zero relative to its factors)"
    m, e = math.frexp(total)
    sm, se = math.frexp(scale)
    out.add(m * sm * sm, e + 2 * (ka + kb + se))
    return out, None


def working_set_bytes(n_out: int, n_in: int, rank: int, norm_only: bool, block_cells: int) -> int:
    """Upper estimate of the float64 bytes one module needs, computed before allocating."""
    factors = 8 * rank * (n_in + n_out)
    scratch = _GRAM_TEMPS * rank * rank if norm_only else _BLOCK_TEMPS * max(block_cells, n_in, n_out)
    return factors + 8 * scratch


def _load_whole(ck: Checkpoint, key: str) -> npt.NDArray[np.float64]:
    meta = ck.tensor(key)
    if meta.numel > MAX_FACTOR_CELLS:
        raise fail(ErrorCode.RESOURCE_LIMIT, f"{key}: LoRA factor above {MAX_FACTOR_CELLS} cells")
    parts = [np.asarray(c, dtype=np.float64) for c in ck.iter_flat_chunks(key, ck.limits.read_chunk_bytes)]
    flat = np.concatenate(parts) if parts else np.empty(0)
    return flat.reshape(meta.shape)


def analyze_adapter(
    base: Checkpoint,
    adapter_dir: Path,
    control: RunControl,
    *,
    max_gmac: float = DEFAULT_MAX_GMAC,
    block_cells: int = DEFAULT_BLOCK_CELLS,
    consistency: Consistency = Consistency.BEST_EFFORT,
    root: Path | None = None,
) -> AdapterResult:
    cfg = load_config(adapter_dir, root)
    weights = adapter_dir / "adapter_model.safetensors"
    if not weights.is_file():
        raise fail(ErrorCode.NOT_FOUND, "adapter_model.safetensors not found (adapter_model.bin is never loaded)")
    with open_checkpoint(weights, selected_root=root or adapter_dir, hash_files=True) as ad:
        digest = next(iter(ad.digests().values()))
        pairs: dict[str, dict[str, str]] = {}
        unsupported_keys: list[tuple[str, str]] = []
        for key in (m.key for m in ad.metadata()):
            m = _AB.match(key[len(PREFIX) :]) if key.startswith(PREFIX) else None
            if m is None:
                why = (
                    "DoRA magnitude vector"
                    if "lora_magnitude_vector" in key
                    else "embedding LoRA"
                    if "lora_embedding_" in key
                    else "modules_to_save weight"
                    if "modules_to_save" in key
                    else "not a supported LoRA key"
                )
                unsupported_keys.append((key, why))
                continue
            pairs.setdefault(m["module"], {})[m["which"]] = key
        modules: list[Module] = []
        base_keys = {t.key for t in base.metadata()}
        for module, ab in pairs.items():
            modules.append(_analyze_module(base, ad, cfg, module, ab, base_keys, control, max_gmac, block_cells))
        if consistency is Consistency.VERIFIED:
            ad.verify_digests()
    analyzed_bases = {m.base_key for m in modules if m.base_key}
    without = sum(1 for k in base_keys if k not in analyzed_bases)
    return AdapterResult(cfg, modules, unsupported_keys, without, digest)


def _analyze_module(
    base: Checkpoint,
    ad: Checkpoint,
    cfg: AdapterConfig,
    module: str,
    ab: dict[str, str],
    base_keys: set[str],
    control: RunControl,
    max_gmac: float,
    block_cells: int,
) -> Module:
    base_key = next((k for k in (f"{module}.weight", f"{module}.base_layer.weight") if k in base_keys), None)
    mod = Module(module, base_key, "unsupported", transpose=cfg.fan_in_fan_out)
    if cfg.global_unsupported:
        mod.reasons += cfg.global_unsupported
        return mod
    if set(ab) != {"A", "B"}:
        mod.reasons.append(f"lora_{'B' if 'A' in ab else 'A'} missing for this module")
        return mod
    if base_key is None:
        mod.reasons.append("missing source base tensor (no .weight or .base_layer.weight in the base)")
        return mod
    if cfg.layers_to_transform is not None:
        li = layer_index(module)
        if li is None or li not in cfg.layers_to_transform:
            mod.reasons.append(f"module layer {li} is outside layers_to_transform {cfg.layers_to_transform}")
            return mod
    scale = resolve_scale(cfg, module)
    if isinstance(scale, str):
        mod.reasons.append(scale)
        return mod
    rank, alpha, s, formula, rank_from = scale
    mod.rank, mod.alpha, mod.scale, mod.scale_formula, mod.rank_from = rank, alpha, s, formula, rank_from
    ma, mb, mw = ad.tensor(ab["A"]), ad.tensor(ab["B"]), base.tensor(base_key)
    if len(ma.shape) != 2 or len(mb.shape) != 2 or len(mw.shape) != 2:
        mod.reasons.append("only rank-2 linear weights are supported")
        return mod
    (ra, n_in), (n_out, rb) = ma.shape, mb.shape
    if ra != rank or rb != rank:
        swapped = ma.shape == (n_in, rank) or mb.shape == (rank, n_out)
        mod.reasons.append(
            f"rank mismatch: config r={rank}, lora_A {list(ma.shape)}, lora_B {list(mb.shape)}"
            + (" (A/B swapped?)" if swapped else "")
        )
        return mod
    expected = (n_in, n_out) if cfg.fan_in_fan_out else (n_out, n_in)
    if mw.shape != expected:
        mod.reasons.append(
            f"base {list(mw.shape)} does not match B@A {[n_out, n_in]}"
            + (" with fan_in_fan_out" if cfg.fan_in_fan_out else "")
        )
        return mod
    if mw.dtype not in ("F64", "F32", "F16", "BF16"):
        mod.reasons.append(f"base dtype {mw.dtype} is not a float")
        return mod
    if ma.dtype not in ("F64", "F32", "F16", "BF16") or mb.dtype not in ("F64", "F32", "F16", "BF16"):
        mod.reasons.append("LoRA factors must be float tensors")
        return mod
    gmac = n_out * n_in * rank / 1e9
    norm_only = gmac > max_gmac
    need = working_set_bytes(n_out, n_in, rank, norm_only, block_cells)
    if ma.numel > MAX_FACTOR_CELLS or mb.numel > MAX_FACTOR_CELLS or need > MAX_WORKING_SET_BYTES:
        mod.status = "skipped"
        mod.reasons.append(
            f"resource limit: estimated working set {need / 2**20:.0f} MiB (limit {MAX_WORKING_SET_BYTES >> 20} MiB, "
            f"factors at most {MAX_FACTOR_CELLS} cells); nothing allocated"
        )
        return mod
    mod.gmac = gmac
    a = _load_whole(ad, ab["A"])
    b = _load_whole(ad, ab["B"])
    esize = mw.byte_length // max(mw.numel, 1)
    if norm_only:
        mod.status = "over_budget_norm_only"
        ssq, issue = gram_frobenius_ssq(a, b, s)  # scales a and b in place; not used afterwards
        mod.norm_only_ssq, mod.norm_only_issue = ssq, issue
        mod.norm_only_l2 = None if issue else ssq.sqrt()
        ssq_w = ExpSum()
        for chunk in base.iter_flat_chunks(base_key, base.limits.read_chunk_bytes):
            w = np.asarray(chunk, dtype=np.float64)
            fin = np.isfinite(w)
            n_fin = int(fin.sum())
            mod.base_finite += n_fin
            mod.base_nonfinite += w.size - n_fin
            ssq_w.add_squares(w if n_fin == w.size else w[fin])
            control.tick(w.size * esize, base_key)
        mod.base_ssq = ssq_w
        mod.base_norm_l2 = ssq_w.sqrt()
        mod.reasons.append(
            f"{gmac:.2f} GMAC above max_lora_scan_gmac={max_gmac}: Frobenius norm only "
            "(Gram trace); max-abs, cosine and non-finite counts of the update not computed"
        )
        if issue:
            mod.reasons.append(f"{issue}: delta norm withheld")
        if mod.base_nonfinite:
            mod.reasons.append(f"base has {mod.base_nonfinite} NaN/Inf element(s), excluded from ||W||")
        return mod

    factors_finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    stats = PairStats()
    rows, cols = mw.shape
    step = max(1, block_cells // max(cols, 1))
    bt = b.T  # (r, out), used for fan_in_fan_out
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        w = base.read_roi_raw(base_key, r0, r1, 0, cols, max_cells=(r1 - r0) * cols).values().astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            d = s * (a[:, r0:r1].T @ bt) if cfg.fan_in_fan_out else s * (b[r0:r1] @ a)
            eff = w + d
        w, d, eff = w.ravel(), d.ravel(), eff.ravel()
        if factors_finite:
            # Finite W and finite factors: a non-finite delta or effective weight is float64
            # overflow of WeightLab's own arithmetic, not a source NaN/Inf; withhold, never mask.
            fw = np.isfinite(w)
            fd = np.isfinite(d)
            bad_d = fw & ~fd
            bad_b = fw & fd & ~np.isfinite(eff)
            n_d, n_b = int(bad_d.sum()), int(bad_b.sum())
            if n_d or n_b:
                keep = ~(bad_d | bad_b)
                stats.add_overflow(n_d, n_b, n_d + n_b)
                w, d, eff = w[keep], d[keep], eff[keep]
        stats.update(w, eff, d)
        control.tick((r1 - r0) * cols * esize, base_key)
    mod.stats = stats
    mod.status = "analyzed"
    if not factors_finite:
        mod.reasons.append(
            "LoRA factors contain NaN/Inf: affected effective weights are counted as non-finite B values"
        )
    if stats.delta_withheld:
        mod.reasons.append(
            f"float64 overflow in s B A ({stats.delta_overflow}) or W + s B A ({stats.b_overflow}) although inputs "
            "are finite: delta metrics withheld"
        )
    if cfg.fan_in_fan_out:
        mod.reasons.append("fan_in_fan_out: delta = scale * (B @ A).T in the stored (in, out) layout (transpose=true)")
    return mod
