"""Build the change report (FR-023, ADR-013): compatibility first, then ranked tables, with provenance.

The report is a plain JSON-ready dict (schema: weightlab/schemas/change_report.schema.json).
Metric objects follow `Metric` in contracts/openapi.yaml; the lineage block follows
`LineageAssessment`. Ranking by magnitude never implies causal importance (doc 05 section 9).
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weightlab import PARSER_VERSION, __version__
from weightlab.analysis.accumulate import PairStats, Rollup, near_zero, relative_l2
from weightlab.analysis.compat import Structure, compare_structure
from weightlab.analysis.diff import DEFAULT_CHUNK_ELEMENTS, DEFAULT_EPS_NORM, NUMERICS_VERSION, RunControl, pair_stats
from weightlab.analysis.lineage import Assessment, LineageInputs, assess
from weightlab.analysis.lora import DEFAULT_MAX_GMAC, AdapterResult, Module, analyze_adapter
from weightlab.analysis.roles import layer_index, role
from weightlab.core.errors import ErrorCode, fail
from weightlab.core.limits import MAX_SAFE_INTEGER
from weightlab.formats.base import Consistency
from weightlab.formats.checkpoint import Checkpoint, hub_cache_root

REPORT_SCHEMA_VERSION = 1
MIN_EPS_NORM = 1e-300
MAX_EPS_NORM = 1.0
MAX_CHUNK_ELEMENTS = 1 << 26  # 64 Mi elements: ~0.5 GiB of float64 per side, the most a chunk may cost
MAX_LORA_GMAC = 1e6
SORT_KEYS = ("relative_l2", "delta_l2", "delta_rms", "max_abs_delta", "cosine")
CAUSAL_CAVEAT = (
    "Ranking by change magnitude describes how much parameters moved; it does not show that a "
    "tensor matters for any behavior (doc 05 section 9)."
)
ROLE_CAVEAT = "Layer and role labels are heuristics from tensor key names, not identity."


@dataclass
class Options:
    sort: str = "relative_l2"
    eps_norm: float = DEFAULT_EPS_NORM
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS
    consistency: Consistency = Consistency.BEST_EFFORT
    include_paths: bool = False
    max_lora_gmac: float = DEFAULT_MAX_GMAC
    created_at: str | None = None  # override for reproducible output (else now, UTC)

    def validate(self) -> None:
        """Reject values that would disable a limit or a policy (NaN/Inf compare false)."""
        problems = []
        if self.sort not in SORT_KEYS:
            problems.append(f"sort must be one of {', '.join(SORT_KEYS)}")
        if not (math.isfinite(self.eps_norm) and MIN_EPS_NORM <= self.eps_norm <= MAX_EPS_NORM):
            problems.append(f"epsilon_norm must be finite and in [{MIN_EPS_NORM}, {MAX_EPS_NORM}]")
        if type(self.chunk_elements) is not int or not 1 <= self.chunk_elements <= MAX_CHUNK_ELEMENTS:
            problems.append(f"chunk_elements must be an integer in [1, {MAX_CHUNK_ELEMENTS}]")
        if not (math.isfinite(self.max_lora_gmac) and 0 <= self.max_lora_gmac <= MAX_LORA_GMAC):
            problems.append(f"max_lora_gmac must be finite and in [0, {MAX_LORA_GMAC}]")
        if problems:
            raise fail(ErrorCode.VALIDATION_ERROR, "; ".join(problems))


def _int(n: int) -> int | str:
    return n if n <= MAX_SAFE_INTEGER else str(n)


def metric(
    name: str,
    value: float | None,
    method: str,
    source_dtype: str,
    elements: int,
    finite: int,
    precision: str,
    *,
    epsilon: float | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "units": None,
        "method": method,
        "source_dtype": source_dtype,
        "compute_dtype": "float64",
        "element_count": _int(elements),
        "finite_count": _int(finite),
        "sample_count": None,
        "coverage": "complete",
        "numeric_precision": precision,
        "seed": None,
        "epsilon": epsilon,
        "numerical_warnings": warnings or [],
    }


def _precision(value: float | None, withheld: bool, defined: bool) -> str:
    if withheld:
        return "overflowed"
    if not defined:
        return "unsupported"
    return "overflowed" if value is None else "rounded"


def pair_metrics(s: PairStats, dtype: str, eps_norm: float) -> dict[str, dict[str, Any]]:
    n, f = s.elements, s.finite_pairs
    w = s.delta_withheld
    warn_over = (
        [f"{s.delta_overflow} finite pair(s) overflow float64 in B - A; delta metrics withheld"]
        if s.delta_overflow
        else []
    ) + (
        [f"{s.b_overflow} computed B value(s) overflow float64; delta, cosine and norm_b withheld"]
        if s.b_overflow
        else []
    )
    warn_none = ["no finite pairs"] if f == 0 else []
    out: dict[str, dict[str, Any]] = {}
    v = s.delta_l2()
    out["delta_l2"] = metric(
        "l2",
        v,
        "||B-A||_2 over finite pairs (power-of-two scaled sum of squares)",
        dtype,
        n,
        f,
        _precision(v, w, True),
        warnings=warn_over,
    )
    v = s.delta_rms()
    out["delta_rms"] = metric(
        "rms",
        v,
        "sqrt(sum((B-A)^2)/finite_pairs)",
        dtype,
        n,
        f,
        _precision(v, w, f > 0),
        warnings=warn_over + warn_none,
    )
    v = s.relative_l2(eps_norm)
    near = s.near_zero_reference(eps_norm)
    out["relative_l2"] = metric(
        "relative_l2",
        v,
        "||B-A||_2 / max(||A||_2, epsilon_norm) over finite pairs",
        dtype,
        n,
        f,
        _precision(v, w, f > 0),
        epsilon=eps_norm,
        warnings=warn_over + warn_none + (["near_zero_reference: ||A|| < epsilon_norm"] if near else []),
    )
    v = s.cosine()
    out["cosine"] = metric(
        "cosine",
        v,
        "dot(A,B)/(||A|| ||B||) over finite pairs",
        dtype,
        n,
        f,
        _precision(v, s.b_withheld, f > 0 and v is not None),
        warnings=warn_over if s.b_withheld else [] if v is not None else ["undefined: zero norm or no finite pairs"],
    )
    v = s.max_abs_delta()
    out["max_abs_delta"] = metric(
        "max_abs",
        v,
        "max |B-A| over finite pairs",
        dtype,
        n,
        f,
        _precision(v, w, f > 0),
        warnings=warn_over + warn_none,
    )
    for side, ssq in (("a", s.ssq_a), ("b", s.ssq_b)):
        held = side == "b" and s.b_withheld
        v = None if held else ssq.sqrt()
        out[f"norm_{side}"] = metric(
            "l2", v, f"||{side.upper()}||_2 over finite pairs", dtype, n, f, _precision(v, held, True)
        )
    return out


def counts(s: PairStats) -> dict[str, int | str]:
    keys = (
        "elements",
        "finite_pairs",
        "invalid_pairs",
        "nan_a",
        "pos_inf_a",
        "neg_inf_a",
        "nan_b",
        "pos_inf_b",
        "neg_inf_b",
        "delta_overflow",
        "b_overflow",
        "changed",
    )
    return {k: _int(getattr(s, k)) for k in keys}


def withheld_metrics(metrics: dict[str, dict[str, Any]], extra: tuple[str, ...] = ()) -> list[str]:
    """Metrics whose true value exists but could not be reported: `overflowed` precision (the value
    left float64 range) plus `extra` (e.g. a Gram norm lost to cancellation). Mathematically undefined
    values (cosine with a zero vector, RMS of no finite pairs) are `unsupported` and not listed."""
    return sorted({k for k, m in metrics.items() if m["numeric_precision"] == "overflowed"} | set(extra))


def _rollup_row(label: str, value: Any, r: Rollup, eps: float) -> dict[str, Any]:
    vals = {
        "delta_l2": r.delta_l2(),
        "delta_rms": r.delta_rms(),
        "relative_l2": r.relative_l2(eps),
        "max_abs_delta": None if r.withheld or r.finite_pairs == 0 else r.max_abs,
    }
    # With finite pairs every rollup metric is defined, so a null is a withheld member or overflow.
    held = sorted(k for k, v in vals.items() if v is None and r.finite_pairs > 0)
    return {
        label: value,
        "tensors": r.tensors,
        "elements": _int(r.elements),
        "finite_pairs": _int(r.finite_pairs),
        "invalid_pairs": _int(r.invalid_pairs),
        "withheld_tensors": r.withheld,
        **vals,
        "near_zero_reference": r.near_zero_reference(eps),
        "withheld_metrics": held,
    }


def _model_info(ck: Checkpoint, include_paths: bool) -> dict[str, Any]:
    files = ck.file_info()
    hashed = all(f.sha256 for f in files)
    return {
        "format": "safetensors_sharded" if ck.entry.name.endswith(".index.json") else "safetensors",
        "entry": str(ck.entry) if include_paths else ck.entry.name,
        "files": [{"name": f.shard_id, "size": f.size, "mtime_ns": f.mtime_ns, "sha256": f.sha256} for f in files],
        "fingerprint_kind": "sha256" if hashed else "size_mtime",
        "strong_hash_verified": hashed,
        "tensor_count": len(ck.metadata()),
        "tensor_bytes": sum(m.byte_length for m in ck.metadata()),
    }


def _now(opts: Options) -> str:
    return opts.created_at or dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sort_rows(rows: list[dict[str, Any]], key: str) -> None:
    def k(row: dict[str, Any]) -> tuple[int, float, str]:
        m = row["metrics"].get(key)
        v = None if m is None else m["value"]
        if v is None:
            return (1, 0.0, row["key"])
        return (0, v if key == "cosine" else -v, row["key"])  # cosine: least similar first

    rows.sort(key=k)
    for i, row in enumerate(rows, 1):
        row["rank"] = i


def _rollups(
    items: list[tuple[str, PairStats]], eps: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    layers: dict[int | None, Rollup] = {}
    roles: dict[str, Rollup] = {}
    total = Rollup()
    for key, s in items:
        layers.setdefault(layer_index(key), Rollup()).add(s)
        roles.setdefault(role(key), Rollup()).add(s)
        total.add(s)
    layer_rows = [
        _rollup_row("layer", li, r, eps) for li, r in sorted(layers.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
    ]
    role_rows = [_rollup_row("role", name, r, eps) for name, r in sorted(roles.items())]
    return layer_rows, role_rows, _rollup_row("scope", "all_analyzed_tensors", total, eps)


def _class_caveat(cls: str) -> str | None:
    return {
        "S": "Class S: no lineage evidence; raw deltas may be meaningless (parameter permutation/alignment).",
        "X": "Class X: structurally incompatible pair; matched tensors are unverified inspection only.",
        "L": "Class L: lineage declared or attested, not cryptographically verified.",
    }.get(cls)


def _entry_dict(e: Any) -> dict[str, Any]:
    return {
        "raw_key": e.key,
        "status": e.status,
        "shape_a": list(e.a.shape) if e.a else None,
        "shape_b": list(e.b.shape) if e.b else None,
        "dtype_a": e.a.dtype if e.a else None,
        "dtype_b": e.b.dtype if e.b else None,
        "tied_to": e.tied_to,
        "tie_evidence": e.tie_evidence,
        "reasons": e.reasons,
    }


def _base_report(
    mode: str, opts: Options, lineage: Assessment, inputs: dict[str, Any], op: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": "weightlab_change_report",
        "mode": mode,
        "orientation": "B_minus_A",
        "compatibility": {"lineage": lineage.as_dict()},
        "provenance": {
            "schema_version": 1,
            "tool": {
                "name": "weightlab",
                "version": __version__,
                "parser_version": PARSER_VERSION,
                "numerics_version": NUMERICS_VERSION,
            },
            "created_at": _now(opts),
            "inputs": inputs,
            "operation": op,
            "quality": {"coverage": "complete", "sampling_method": None},
            "privacy": {"paths_included": opts.include_paths},
            "read_only": True,
        },
        "caveats": [c for c in (_class_caveat(lineage.compatibility_class), CAUSAL_CAVEAT, ROLE_CAVEAT) if c],
        "settings": {
            "sort": opts.sort,
            "epsilon_norm": opts.eps_norm,
            "chunk_elements": opts.chunk_elements,
            "consistency": opts.consistency.value,
        },
    }


def assess_pair(a: Checkpoint, b: Checkpoint, lin: LineageInputs) -> tuple[Structure, Assessment]:
    """Structural comparison and lineage assessment of base A and derived B (shared with the API)."""
    structure: Structure = compare_structure(a, b)
    caps = (["the tie of the output head differs"] if structure.tie_differs else []) + (
        ["tensor dtypes differ"] if structure.dtype_differs else []
    )
    derived_digest = next(iter(b.digests().values())) if len(b.shard_ids) == 1 and _small(b) else None
    lineage = assess(
        a, derived_digest, lin, structure_blocking=bool(structure.blocking), structure_caps_at_l=caps, adapter=False
    )
    return structure, lineage


def pair_report(a: Checkpoint, b: Checkpoint, lin: LineageInputs, opts: Options, control: RunControl) -> dict[str, Any]:
    """Base (A) vs derived (B)."""
    if opts.consistency is Consistency.VERIFIED or lin.user_attestation or lin.manifest is not None:
        a.digests()
        b.digests()
    structure, lineage = assess_pair(a, b, lin)

    analyzable = [e for e in structure.entries if e.analyzable]
    control.total_bytes = sum(e.a.byte_length + e.b.byte_length for e in analyzable if e.a and e.b)
    rows: list[dict[str, Any]] = []
    items: list[tuple[str, PairStats]] = []
    caveat = _class_caveat(lineage.compatibility_class) if lineage.compatibility_class in ("S", "X") else None
    for e in analyzable:
        assert e.a is not None
        assert e.b is not None
        s = pair_stats(a, b, e, control, opts.chunk_elements)
        items.append((e.key, s))
        dtype = e.a.dtype if e.a.dtype == e.b.dtype else f"{e.a.dtype}/{e.b.dtype}"
        rows.append(
            {
                "rank": 0,
                "key": e.key,
                "layer": layer_index(e.key),
                "role": role(e.key),
                "status": e.status,
                "shape": list(e.a.shape),
                "dtype_a": e.a.dtype,
                "dtype_b": e.b.dtype,
                "counts": counts(s),
                "metrics": (met := pair_metrics(s, dtype, opts.eps_norm)),
                "withheld_metrics": withheld_metrics(met),
                "caveat": "; ".join(
                    x
                    for x in (
                        caveat,
                        *(e.reasons if e.status == "dtype_mismatch" else []),
                        "near_zero_reference: ||A|| < epsilon_norm, relative_l2 divides by epsilon_norm"
                        if s.near_zero_reference(opts.eps_norm)
                        else None,
                    )
                    if x
                )
                or None,
            }
        )
    if opts.consistency is Consistency.VERIFIED:
        a.verify_digests()
        b.verify_digests()
    a.check()
    b.check()
    _sort_rows(rows, opts.sort)
    layers, roles, totals = _rollups(items, opts.eps_norm)
    not_compared = [e for e in structure.entries if not e.analyzable]
    accounted = all(e.status == "tied" for e in not_compared)
    report = _base_report(
        "checkpoint_pair",
        opts,
        lineage,
        {"base": _model_info(a, opts.include_paths), "derived": _model_info(b, opts.include_paths)},
        {"kind": "delta", "orientation": "B_minus_A", "compute_dtype": "float64"},
    )
    report["compatibility"]["counts"] = structure.counts()
    report["compatibility"]["not_compared"] = [_entry_dict(e) for e in not_compared]
    withheld, rollups_withheld = _withheld_counts(rows, layers, roles, totals)
    report["complete"] = accounted and withheld == 0 and rollups_withheld == 0
    report["completeness"] = {
        "tensors_analyzed": len(rows),
        "tensors_withheld": withheld,
        "rollups_withheld": rollups_withheld,
        "entries_not_compared": len(not_compared),
        "tied_entries": sum(1 for e in not_compared if e.status == "tied"),
    }
    if withheld or rollups_withheld:
        report["caveats"].insert(0, WITHHELD_CAVEAT.format(n=withheld, m=rollups_withheld))
    report.update(tensors=rows, layers=layers, roles=roles, totals=totals)
    report["warnings"] = [*a.warnings, *b.warnings]
    return report


def _withheld_counts(rows: list[dict[str, Any]], layers: list[dict[str, Any]], roles: list[dict[str, Any]],
                     totals: dict[str, Any]) -> tuple[int, int]:  # fmt: skip
    return (sum(1 for r in rows if r["withheld_metrics"]),
            sum(1 for r in [*layers, *roles, totals] if r["withheld_metrics"]))  # fmt: skip


WITHHELD_CAVEAT = (
    "Incomplete: {n} tensor(s)/module(s) and {m} rollup(s) have metrics withheld (float64 overflow or "
    "unreliable arithmetic); see `withheld_metrics`."
)


_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def looks_like_local_path(value: str) -> bool:
    """True unless `value` is a plain Hugging Face repo id ("name" or "org/name"). Anything with
    a drive, root, home, relative or backslash prefix, or more than one '/', is treated as a path."""
    return not _REPO_ID.match(value) or value.startswith((".", "~"))


def redact_path(value: str) -> str:
    name = re.split(r"[\\/]", value.rstrip("/\\"))[-1] or "?"
    return f"<local path redacted: {name}>"


def _small(ck: Checkpoint) -> bool:
    """Only tiny files can be generator fixtures; avoids hashing real models just for the registry."""
    return sum(f.size for f in ck.file_info()) <= 1 << 20


def _module_row(m: Module, base: Checkpoint, eps: float) -> dict[str, Any]:
    meta = base.tensor(m.base_key) if m.base_key else None
    dtype = meta.dtype if meta else ""
    row: dict[str, Any] = {
        "rank": 0,
        "key": m.base_key or m.module,
        "module": m.module,
        "layer": layer_index(m.module),
        "role": role(m.base_key or m.module),
        "status": m.status,
        "shape": list(meta.shape) if meta else None,
        "lora": {
            "rank": m.rank,
            "alpha": m.alpha,
            "scale": m.scale,
            "scale_formula": m.scale_formula,
            "rank_from": m.rank_from,
            "transpose": m.transpose,
            "gmac": m.gmac,
        },
        "counts": counts(m.stats) if m.stats else None,
        "metrics": {},
        "caveat": "; ".join(m.reasons) or None,
    }
    if m.stats is not None:
        met = pair_metrics(m.stats, dtype, eps)
        met["norm_a"]["method"] = "||W||_2 (base) over finite elements"
        met["norm_b"]["method"] = "||W + s B A||_2 (effective) over finite elements"
        met["cosine"]["method"] = "cos(W, W + s B A)"
        for k in ("delta_l2", "delta_rms", "relative_l2", "max_abs_delta"):
            met[k]["method"] = met[k]["method"].replace("B-A", "s B A")
        row["metrics"] = met
    elif m.status == "over_budget_norm_only" and meta is not None:
        n = meta.numel
        issue = m.norm_only_issue
        row["counts"] = {
            "elements": _int(n),
            "base_finite": _int(m.base_finite),
            "base_nonfinite": _int(m.base_nonfinite),
        }
        issue_warn = [issue] if issue else []
        row["metrics"]["delta_l2"] = metric(
            "l2",
            m.norm_only_l2,
            "||s B A||_F via s^2 sum((B^T B)*(A A^T)), power-of-two scaled; covers every element",
            dtype,
            n,
            0 if issue else n,  # the update itself is finite everywhere when its norm is
            _precision(m.norm_only_l2, False, issue is None),
            warnings=issue_warn,
        )
        rel = None
        near = False
        if issue is None and m.norm_only_ssq is not None and m.base_ssq is not None:
            rel = relative_l2(m.norm_only_ssq, m.base_ssq, eps)
            near = near_zero(m.base_ssq, eps)
        rel_warn = issue_warn + (["near_zero_reference: ||W|| < epsilon_norm"] if near else [])
        if m.base_nonfinite:
            rel_warn.append(f"||W|| excludes {m.base_nonfinite} NaN/Inf base element(s); ||s B A|| covers all")
        row["metrics"]["relative_l2"] = metric(
            "relative_l2",
            rel,
            "||s B A||_F / max(||W||_2, epsilon_norm)",
            dtype,
            n,
            m.base_finite,
            _precision(rel, False, issue is None),
            epsilon=eps,
            warnings=rel_warn,
        )
    row["withheld_metrics"] = withheld_metrics(row["metrics"], ("delta_l2", "relative_l2") if m.norm_only_issue else ())
    return row


def adapter_report(
    base: Checkpoint,
    adapter_dir: Path,
    lin: LineageInputs,
    opts: Options,
    control: RunControl,
    selected_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    from weightlab.analysis.lora import load_config

    root = Path(os.path.abspath(selected_root)) if selected_root else hub_cache_root(adapter_dir) or adapter_dir
    cfg = load_config(adapter_dir, root)
    declared = cfg.raw.get("base_model_name_or_path")
    revision = cfg.raw.get("revision")
    declared = declared if isinstance(declared, str) and declared else None
    if declared is not None and not opts.include_paths and looks_like_local_path(declared):
        declared = redact_path(declared)  # also keeps it out of the lineage reasons
    lin.declared_base = declared
    lin.declared_revision = revision if isinstance(revision, str) and revision else None
    control.total_bytes = sum(m.byte_length for m in base.metadata())
    if opts.consistency is Consistency.VERIFIED:
        base.digests()
    result: AdapterResult = analyze_adapter(
        base, adapter_dir, control, root=root, max_gmac=opts.max_lora_gmac, consistency=opts.consistency
    )
    if opts.consistency is Consistency.VERIFIED:
        base.verify_digests()
    base.check()
    missing_bases = [m for m in result.modules if m.base_key is None]
    lineage = assess(
        base, result.adapter_digest, lin, structure_blocking=bool(missing_bases), structure_caps_at_l=[], adapter=True
    )
    rows = [_module_row(m, base, opts.eps_norm) for m in result.modules]
    _sort_rows(rows, opts.sort)
    items = [(m.base_key or m.module, m.stats) for m in result.modules if m.stats is not None]
    layers, roles, totals = _rollups(items, opts.eps_norm)
    adapter_info = {
        "format": "peft_lora",
        "entry": str(adapter_dir) if opts.include_paths else adapter_dir.name,
        "files": [{"name": "adapter_model.safetensors", "sha256": result.adapter_digest}],
        "declared_base": lin.declared_base,
        "declared_revision": lin.declared_revision,
        "config": {
            k: cfg.raw.get(k)
            for k in (
                "peft_type",
                "r",
                "lora_alpha",
                "use_rslora",
                "use_dora",
                "fan_in_fan_out",
                "rank_pattern",
                "alpha_pattern",
                "layers_to_transform",
                "target_modules",
                "bias",
            )
        },
    }
    report = _base_report(
        "adapter",
        opts,
        lineage,
        {"base": _model_info(base, opts.include_paths), "adapter": adapter_info},
        {
            "kind": "lora_effective_delta",
            "orientation": "B_minus_A",
            "definition": "A = base W, B = W + s * (B_lora @ A_lora); delta computed directly",
            "compute_dtype": "float64",
            "max_lora_scan_gmac": opts.max_lora_gmac,
        },
    )
    status = {
        s: sum(1 for m in result.modules if m.status == s)
        for s in ("analyzed", "over_budget_norm_only", "unsupported", "skipped")
    }
    report["compatibility"]["counts"] = {"modules": len(result.modules), **status}
    report["compatibility"]["not_compared"] = [
        {"raw_key": k, "status": "unsupported_key", "reasons": [why]} for k, why in result.unsupported_keys
    ]
    withheld, rollups_withheld = _withheld_counts(rows, layers, roles, totals)
    report["complete"] = result.complete and withheld == 0 and rollups_withheld == 0
    report["completeness"] = {
        "modules_analyzed": status["analyzed"],
        "modules_withheld": withheld,
        "rollups_withheld": rollups_withheld,
        "modules_norm_only": status["over_budget_norm_only"],
        "modules_unsupported": status["unsupported"],
        "modules_skipped": status["skipped"],
        "unsupported_adapter_keys": len(result.unsupported_keys),
        "base_tensors_without_adapter": result.base_tensors_without_adapter,
    }
    if withheld or rollups_withheld:
        report["caveats"].insert(0, WITHHELD_CAVEAT.format(n=withheld, m=rollups_withheld))
    if not result.complete:
        report["caveats"].insert(
            0,
            "Incomplete: some adapter modules or keys were not analyzed; this report does "
            "not describe the whole model change.",
        )
    report.update(tensors=rows, layers=layers, roles=roles, totals=totals)
    report["warnings"] = list(base.warnings)
    return report


def run(
    base_path: str | os.PathLike[str],
    derived_path: str | os.PathLike[str] | None = None,
    *,
    adapter: str | os.PathLike[str] | None = None,
    selected_root: str | os.PathLike[str] | None = None,
    lineage: LineageInputs | None = None,
    options: Options | None = None,
    control: RunControl | None = None,
    opener: Callable[..., Checkpoint] | None = None,
) -> dict[str, Any]:
    """Library entry point used by the CLI."""
    from weightlab.formats.checkpoint import open_checkpoint

    open_ = opener or open_checkpoint
    lin = lineage or LineageInputs()
    opts = options or Options()
    opts.validate()
    ctl = control or RunControl()
    verified = opts.consistency is Consistency.VERIFIED
    with open_(base_path, selected_root or hub_cache_root(base_path), hash_files=verified) as a:
        if adapter is not None:
            return adapter_report(a, Path(os.path.abspath(adapter)), lin, opts, ctl, selected_root)
        assert derived_path is not None
        with open_(derived_path, selected_root or hub_cache_root(derived_path), hash_files=verified) as b:
            return pair_report(a, b, lin, opts, ctl)
