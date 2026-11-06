"""POST /tensors/roi for all four source kinds (doc 09 sections 2 and 4).

* tensor: Phase 0 positioned read + lossless typed encoding (`encode_roi`), unchanged.
* delta: elementwise B - A of one analyzable comparison entry, both sides read for the same cells
  (a paired pass, never from aggregates); `abs_b_minus_a` = |B - A|; `relative_abs` = |B - A| / max(|A|, eps)
  (the denominator rule of relative_l2, doc 05; the spec names this transform without defining it).
* lora_delta: multiplier * native_scale * (B @ A) for the requested cells only: rows of B and columns of A
  are read for the ROI, never the full product. Honors `transpose` (fan_in_fan_out).
* virtual_effective: base W + lora_delta; needs base lineage class V or research_mode (else 409).
Derived values are float64 (`numeric_precision: rounded`). A value whose inputs are finite but whose
computation overflows is null with mask `overflow` and the response is `overflowed`; source NaN/Inf
propagate as null with their own mask kind. JSON NaN/Infinity never appear.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any

import numpy as np
import numpy.typing as npt

from weightlab.analysis.roi import ValueEncoding, encode_roi
from weightlab.api.catalog import open_model
from weightlab.api.config import DEFAULT_SETTINGS
from weightlab.api.errors import ApiError
from weightlab.api.relations import adapter_module, open_adapter_checked
from weightlab.api.schemas import DeltaSource, LoraDeltaSource, RoiRequest, TensorSource, View, VirtualEffectiveSource
from weightlab.core.errors import ErrorCode
from weightlab.db import store
from weightlab.formats import selection
from weightlab.formats.checkpoint import Checkpoint

F64 = npt.NDArray[np.float64]


def _view(v: View) -> selection.View:
    return selection.View(v.row_axis, v.col_axis, {int(k): i for k, i in v.fixed_indices.items()}, v.transpose)


def _roi_args(req: RoiRequest) -> tuple[int, int, int, int]:
    b = req.roi
    return b.r0, b.r1, b.c0, b.c1


def compute(conn: sqlite3.Connection, req: RoiRequest) -> dict[str, Any]:
    settings = store.get_settings(conn, DEFAULT_SETTINGS)
    cap = int(settings["roi_max_cells"])
    src = req.source
    if req.value_encoding == "hex_bits" and not isinstance(src, TensorSource):
        raise ApiError(ErrorCode.VALIDATION_ERROR, "hex_bits is only available for kind=tensor")
    if isinstance(src, TensorSource):
        body = _tensor(conn, src, req, cap)
    elif isinstance(src, DeltaSource):
        body = _delta(conn, src, req, cap)
    else:
        body = _lora(conn, src, req, cap)
    body["source"] = src.model_dump(exclude_unset=True)
    body["roi"] = req.roi.model_dump()
    return body


def _tensor(conn: sqlite3.Connection, src: TensorSource, req: RoiRequest, cap: int) -> dict[str, Any]:
    if store.get_model(conn, src.model_id) is None:
        raise ApiError(ErrorCode.NOT_FOUND, "model not found")
    t = store.get_tensor(conn, src.tensor_id, src.model_id)
    if t is None:
        raise ApiError(ErrorCode.NOT_FOUND, "tensor not found in this model")
    with open_model(conn, src.model_id) as ck:
        raw = ck.read_roi_raw(t["raw_key"], *_roi_args(req), _view(req.view), max_cells=cap)
        ck.check()
    enc = encode_roi(raw, ValueEncoding(req.value_encoding))
    return {**enc.body(), "warnings": []}


def _derived(values: F64, overflow: npt.NDArray[np.bool_], source_dtype: str, warnings: list[str]) -> dict[str, Any]:
    out: list[list[float | None]] = []
    mask: list[list[str | None]] = []
    any_mask = False
    for vrow, orow in zip(values.tolist(), overflow.tolist(), strict=True):
        o_vals: list[float | None] = []
        o_mask: list[str | None] = []
        for x, over in zip(vrow, orow, strict=True):
            if over:
                o_vals.append(None)
                o_mask.append("overflow")
                any_mask = True
            elif math.isfinite(x):
                o_vals.append(x)
                o_mask.append(None)
            else:
                o_vals.append(None)
                o_mask.append("nan" if math.isnan(x) else "pos_inf" if x > 0 else "neg_inf")
                any_mask = True
        out.append(o_vals)
        mask.append(o_mask)
    return {
        "schema_version": 2,
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "source_dtype": source_dtype,
        "value_encoding": "float_json",
        "coverage": "complete",
        "numeric_precision": "overflowed" if bool(overflow.any()) else "rounded",
        "values": out,
        "nonfinite_mask": mask if any_mask else None,
        "warnings": [w[:500] for w in warnings[:50]],
    }


def _delta(conn: sqlite3.Connection, src: DeltaSource, req: RoiRequest, cap: int) -> dict[str, Any]:
    cmp_row = store.get_comparison(conn, src.comparison_id)
    if cmp_row is None:
        raise ApiError(ErrorCode.NOT_FOUND, "comparison not found")
    entry = store.get_entry(conn, src.comparison_id, src.entry_id)
    if entry is None:
        raise ApiError(ErrorCode.NOT_FOUND, "entry not found in this comparison")
    if entry["status"] not in ("matched", "dtype_mismatch") or not entry["detail_json"].get("analyzable"):
        raise ApiError(ErrorCode.INCOMPATIBLE_TENSOR, f"entry is {entry['status']}: no elementwise B - A exists")
    key, view = entry["raw_key"], _view(req.view)
    with open_model(conn, cmp_row["model_a_id"]) as a, open_model(conn, cmp_row["model_b_id"]) as b:
        ra = a.read_roi_raw(key, *_roi_args(req), view, max_cells=cap)
        rb = b.read_roi_raw(key, *_roi_args(req), view, max_cells=cap)
        a.check()
        b.check()
    va, vb = ra.values().astype(np.float64), rb.values().astype(np.float64)
    finite_in = np.isfinite(va) & np.isfinite(vb)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        d = vb - va
        if src.transform == "abs_b_minus_a":
            d = np.abs(d)
        elif src.transform == "relative_abs":
            d = np.abs(d) / np.maximum(np.abs(va), src.epsilon)
    overflow = finite_in & ~np.isfinite(d)
    lineage = cmp_row["compatibility_json"]["lineage"]
    warnings = []
    if lineage["compatibility_class"] in ("S", "X"):
        warnings.append(f"Class {lineage['compatibility_class']}: raw deltas may be meaningless (no verified lineage)")
    if ra.dtype.name != rb.dtype.name:
        warnings.append(f"dtypes differ: A {ra.dtype.name}, B {rb.dtype.name}; both promoted to float64")
    if src.transform == "relative_abs":
        warnings.append(f"relative_abs = |B - A| / max(|A|, {src.epsilon!r})")
    return _derived(d, overflow, ra.dtype.name, warnings)


def _read(ck: Checkpoint, key: str, r0: int, r1: int, c0: int, c1: int) -> F64:
    cells = (r1 - r0) * (c1 - c0)
    hard = ck.limits.roi_hard_max_cells
    if cells > hard:
        raise ApiError(ErrorCode.RESOURCE_LIMIT, f"LoRA factor slice of {cells} cells is above {hard}")
    if cells == 0:
        return np.zeros((r1 - r0, c1 - c0))
    return ck.read_roi_raw(key, r0, r1, c0, c1, max_cells=cells).values().astype(np.float64)


def _lora(
    conn: sqlite3.Connection, src: LoraDeltaSource | VirtualEffectiveSource, req: RoiRequest, cap: int
) -> dict[str, Any]:
    row = store.get_adapter(conn, src.adapter_id)
    if row is None:
        raise ApiError(ErrorCode.NOT_FOUND, "adapter not found")
    mod = adapter_module(row, src.module_id)
    if mod is None:
        raise ApiError(ErrorCode.NOT_FOUND, "module not found in this adapter")
    if mod["status"] != "supported":
        raise ApiError(ErrorCode.INCOMPATIBLE_TENSOR, f"module is {mod['status']}: {mod['reason'] or 'not supported'}")
    lineage_class = row["lineage_json"]["compatibility_class"]
    virtual = isinstance(src, VirtualEffectiveSource)
    if isinstance(src, VirtualEffectiveSource) and lineage_class != "V" and not src.research_mode:
        raise ApiError(ErrorCode.LINEAGE_UNVERIFIED, f"base lineage is class {lineage_class}, not V")
    r0, r1, c0, c1 = _roi_args(req)
    with open_model(conn, row["base_model_id"]) as base, open_adapter_checked(row) as ad:
        meta = base.tensor(mod["_base_key"])
        rv = selection.resolve_view(meta.shape, _view(req.view))
        selection.validate_roi(rv, r0, r1, c0, c1, cap)
        # stored (row, col) ranges of the base layout that the displayed ROI covers
        swap = rv.row_axis == 1
        (sr0, sr1), (sc0, sc1) = ((c0, c1), (r0, r1)) if swap else ((r0, r1), (c0, c1))
        rank = int(mod["rank"])
        if not mod["transpose"]:  # stored D = s * B @ A, shape (out, in)
            bs = _read(ad, mod["_b_key"], sr0, sr1, 0, rank)
            as_ = _read(ad, mod["_a_key"], 0, rank, sc0, sc1)
        else:  # fan_in_fan_out: stored D = (s * B @ A)^T, shape (in, out): D[i, j] = B[j] . A[:, i]
            bs = _read(ad, mod["_b_key"], sc0, sc1, 0, rank)
            as_ = _read(ad, mod["_a_key"], 0, rank, sr0, sr1)
        w = (
            base.read_roi_raw(mod["_base_key"], r0, r1, c0, c1, _view(req.view), max_cells=cap).values()
            if virtual
            else None
        )
        base.check()
        ad.check()
    scale = float(mod["native_scale"]) * float(src.experiment_multiplier)
    factors_finite = bool(np.isfinite(bs).all() and np.isfinite(as_).all())
    with np.errstate(over="ignore", invalid="ignore"):
        d = scale * (bs @ as_)
        if mod["transpose"]:
            d = d.T
        if swap:
            d = d.T
        overflow = ~np.isfinite(d) if factors_finite else np.zeros(d.shape, dtype=bool)
        warnings = [
            f"multiplier {src.experiment_multiplier!r} x native scale {mod['native_scale']!r} (alpha/r rules of PEFT)"
        ]
        if virtual:
            assert w is not None
            wv = w.astype(np.float64)
            eff = wv + d
            overflow = overflow | (np.isfinite(wv) & np.isfinite(d) & ~np.isfinite(eff))
            values = eff
            if lineage_class != "V":
                warnings.append(
                    f"research_mode: base lineage class {lineage_class} is not verified; visualization only"
                )
            dtype = meta.dtype
        else:
            values = d
            dtype = meta.dtype
    if not factors_finite:
        warnings.append("LoRA factor slices contain NaN/Inf; affected cells are non-finite")
    return _derived(values, overflow, dtype, warnings)
