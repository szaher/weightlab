"""Comparisons (V/L/S/X + entries) and PEFT adapter imports, reusing Phase 0.5 analysis unchanged.

* Comparisons: `assess_pair` (compare_structure + lineage assess) on the two imported models; entries are
  stored so they can be paged and addressed by `entry_id` from a delta source.
* Adapters: `adapter_pairs` + `plan_module` validate every module from metadata only (no tensor data
  read); the lineage is assessed exactly as the CLI report does, local base paths redacted.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from weightlab.analysis.change_report import assess_pair, looks_like_local_path, redact_path
from weightlab.analysis.lineage import Assessment, LineageInputs, assess
from weightlab.analysis.lora import AdapterConfig, adapter_pairs, load_config, plan_module
from weightlab.api.catalog import open_model, tensor_id, validate_paths
from weightlab.api.errors import ApiError
from weightlab.api.schemas import AdapterImport, ComparisonRequest
from weightlab.core.errors import ErrorCode, WeightLabError
from weightlab.db import store
from weightlab.formats.checkpoint import Checkpoint, fingerprint_side_file, hub_cache_root, open_checkpoint

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
API_MAX_RANK = 4096  # AdapterModule.rank maximum in the frozen contract


def _cap(items: list[str], n: int, width: int = 500) -> list[str]:
    return [x[:width] for x in items[:n]]


def lineage_view(a: Assessment) -> dict[str, Any]:
    d = a.as_dict()
    d["reasons"] = _cap(list(d["reasons"]), 100)
    return d


# -- comparisons -------------------------------------------------------------------------------
def create_comparison(conn: sqlite3.Connection, req: ComparisonRequest) -> dict[str, Any]:
    if store.get_project(conn, req.project_id) is None:
        raise ApiError(ErrorCode.NOT_FOUND, "project not found")
    for mid in (req.model_a_id, req.model_b_id):
        if store.get_model(conn, mid, req.project_id) is None:
            raise ApiError(ErrorCode.NOT_FOUND, "model not found")
    a = open_model(conn, req.model_a_id, req.project_id)
    try:
        b = open_model(conn, req.model_b_id, req.project_id)
        try:
            lin = LineageInputs(user_attestation=req.user_attestation)
            if req.user_attestation:  # attestation names the exact bytes: hash both sides (as the CLI does)
                a.digests()
                b.digests()
            structure, lineage = assess_pair(a, b, lin)
            a.check()
            b.check()
        finally:
            b.close()
    finally:
        a.close()
    entries = []
    for e in structure.entries:
        entries.append(
            {
                "entry_id": store.new_id("ent"),
                "raw_key": e.key,
                "status": e.status,
                "tensor_id_a": tensor_id(req.model_a_id, e.key) if e.a else None,
                "tensor_id_b": tensor_id(req.model_b_id, e.key) if e.b else None,
                "shape_a": list(e.a.shape) if e.a else None,
                "shape_b": list(e.b.shape) if e.b else None,
                "dtype_a": e.a.dtype if e.a else None,
                "dtype_b": e.b.dtype if e.b else None,
                "tied_to": e.tied_to,
                "tie_evidence": e.tie_evidence,
                "reasons": _cap(list(e.reasons), 20),
                "analyzable": e.analyzable,
            }
        )
    compat = {"lineage": lineage_view(lineage), "counts": structure.counts(), "policy": req.policy}
    row = store.insert_comparison(
        conn,
        project_id=req.project_id,
        model_a_id=req.model_a_id,
        model_b_id=req.model_b_id,
        compatibility=compat,
        entries=entries,
    )
    return comparison_view(row)


def comparison_view(row: dict[str, Any]) -> dict[str, Any]:
    c = row["compatibility_json"]
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "model_a_id": row["model_a_id"],
        "model_b_id": row["model_b_id"],
        "lineage": c["lineage"],
        "counts": c["counts"],
        "created_at": row["created_at"],
    }


def entry_view(row: dict[str, Any]) -> dict[str, Any]:
    d = row["detail_json"]
    return {
        "entry_id": row["id"],
        "raw_key": row["raw_key"],
        "status": row["status"],
        "tensor_id_a": row["tensor_id_a"],
        "tensor_id_b": row["tensor_id_b"],
        **{k: d.get(k) for k in ("shape_a", "shape_b", "dtype_a", "dtype_b", "tied_to", "tie_evidence")},
        "reasons": d.get("reasons", []),
    }


# -- adapters ----------------------------------------------------------------------------------
def module_id(adapter_id: str, module: str) -> str:
    return "amd_" + hashlib.sha256(f"{adapter_id}\0{module}".encode()).hexdigest()[:40]


def _adapter_dir(path: str) -> Path:
    d, _, _ = validate_paths(path, path)
    if not d.is_dir():
        raise ApiError(ErrorCode.VALIDATION_ERROR, "adapter_directory must be a directory")
    for name in ADAPTER_FILES:
        if not (d / name).is_file():
            raise ApiError(
                ErrorCode.INVALID_FORMAT, f"adapter directory has no {name} (adapter_model.bin is never loaded)"
            )
    return d


def _adapter_root(d: Path) -> Path:
    return hub_cache_root(d) or d


def open_adapter_weights(d: Path) -> Checkpoint:
    return open_checkpoint(d / "adapter_model.safetensors", _adapter_root(d), hash_files=True)


def config_fingerprint(d: Path) -> dict[str, Any]:
    """adapter_config.json fingerprint: size, mtime and SHA-256 of its bytes. The config holds r, lora_alpha,
    rank_pattern, ..., from which the stored native scales were derived, so an edit to it alone must be
    detected (PR #3 review); hashing a small capped file on every adapter read is cheap."""
    f = fingerprint_side_file(d / "adapter_config.json", _adapter_root(d), "adapter_config.json")
    return {"size": f.identity.size, "mtime_ns": f.identity.mtime_ns, "sha256": f.sha256}


def adapter_files(d: Path, ck: Checkpoint, config: dict[str, Any]) -> dict[str, Any]:
    ident = next(iter(ck.file_identities().values()))
    return {
        "adapter_model.safetensors": {
            "size": ident.size,
            "mtime_ns": ident.mtime_ns,
            "ctime_ns": ident.ctime_ns,  # doc 03: a same-size rewrite with mtime_ns restored still moves ctime_ns
            "file_id": f"{ident.dev}:{ident.ino}",
            "sha256": next(iter(ck.digests().values())),
        },
        "adapter_config.json": config,
    }


def import_adapter(conn: sqlite3.Connection, project_id: str, req: AdapterImport) -> dict[str, Any]:
    if store.get_model(conn, req.base_model_id, project_id) is None:
        raise ApiError(ErrorCode.NOT_FOUND, "base model not found")
    d = _adapter_dir(req.adapter_directory)
    config_before = config_fingerprint(d)
    cfg: AdapterConfig = load_config(d, _adapter_root(d))
    adapter_id = store.new_id("adp")
    base = open_model(conn, req.base_model_id, project_id)
    try:
        with open_adapter_weights(d) as ad:
            pairs, unsupported_keys = adapter_pairs(ad)
            base_keys = {t.key for t in base.metadata()}
            plans = [plan_module(base, ad, cfg, module, ab, base_keys) for module, ab in pairs.items()]
            declared = cfg.raw.get("base_model_name_or_path")
            declared = declared if isinstance(declared, str) and declared else None
            if declared is not None and looks_like_local_path(declared):
                declared = redact_path(declared)
            revision = cfg.raw.get("revision")
            lin = LineageInputs(
                declared_base=declared, declared_revision=revision if isinstance(revision, str) and revision else None
            )
            digest = next(iter(ad.digests().values()))
            missing = [p for p in plans if p.module.base_key is None]
            lineage = assess(base, digest, lin, structure_blocking=bool(missing), structure_caps_at_l=[], adapter=True)
            if config_fingerprint(d) != config_before:  # edited while the import read it
                raise ApiError(ErrorCode.FILE_CHANGED, "adapter_config.json changed during import")
            files = adapter_files(d, ad, config_before)
            ad.check()
        base.check()
    finally:
        base.close()
    modules = []
    for p in plans:
        mod = p.module
        rank = mod.rank if mod.rank is not None else cfg.r
        if rank > API_MAX_RANK:
            raise ApiError(ErrorCode.RESOURCE_LIMIT, f"adapter rank {rank} is above the API limit {API_MAX_RANK}")
        alpha = mod.alpha if mod.alpha is not None else cfg.alpha
        scale = mod.scale if mod.scale is not None else (alpha / rank**0.5 if cfg.use_rslora else alpha / rank)
        status = "supported" if p.ok else "no_base_tensor" if mod.base_key is None else "unsupported"
        modules.append(
            {
                "module_id": module_id(adapter_id, mod.module),
                "adapter_key_prefix": f"base_model.model.{mod.module}"[:4096],
                "base_tensor_id": tensor_id(req.base_model_id, mod.base_key) if mod.base_key else None,
                "rank": rank,
                "alpha": alpha,
                "native_scale": scale,
                "transpose": bool(mod.transpose and p.ok),
                "status": status,
                "reason": "; ".join(mod.reasons)[:500] or None,
                # internal (not part of the Adapter response): needed to compute lora_delta ROIs
                "_module": mod.module,
                "_base_key": mod.base_key,
                "_a_key": p.a_key,
                "_b_key": p.b_key,
            }
        )
    supported = sum(1 for m in modules if m["status"] == "supported")
    warnings = [f"unsupported adapter key {k[:200]!r}: {why}" for k, why in unsupported_keys]
    warnings += [f"adapter: {w}" for w in cfg.global_unsupported]
    status = (
        "ready"
        if supported == len(modules) and modules and not warnings
        else "unsupported"
        if supported == 0
        else "partial"
    )
    raw_cfg = {
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
    }
    raw_cfg["declared_base"] = declared
    store_id = store.insert_adapter(
        conn,
        project_id=project_id,
        base_model_id=req.base_model_id,
        adapter_path=str(d),
        config=raw_cfg,
        modules=modules,
        status=status,
        lineage=lineage_view(lineage),
        files=files,
        warnings=_cap(warnings, 100),
        adapter_id=adapter_id,
    )
    row = store.get_adapter(conn, store_id)
    assert row is not None
    return adapter_view(row)


def adapter_view(row: dict[str, Any]) -> dict[str, Any]:
    mapping = row["mapping_json"]
    out: dict[str, Any] = {
        "id": row["id"],
        "project_id": row["project_id"],
        "base_model_id": row["base_model_id"],
        "status": row["status"],
        "lineage": row["lineage_json"],
        "modules": [{k: v for k, v in m.items() if not k.startswith("_")} for m in mapping["modules"]],
    }
    if mapping.get("warnings"):
        out["warnings"] = mapping["warnings"]
    return out


def adapter_module(row: dict[str, Any], mid: str) -> dict[str, Any] | None:
    return next((m for m in row["mapping_json"]["modules"] if m["module_id"] == mid), None)


def open_adapter_checked(row: dict[str, Any]) -> Checkpoint:
    """Reopen an imported adapter's weights; FILE_CHANGED unless both adapter files are the imported ones:
    the weights by quick fingerprint, adapter_config.json (the source of rank, alpha and scale) by content."""
    d = Path(row["adapter_path"])
    want = row["files_json"]["adapter_model.safetensors"]
    want_cfg = row["files_json"]["adapter_config.json"]
    try:
        cfg_now = config_fingerprint(d)
    except WeightLabError as e:
        if e.code is ErrorCode.NOT_FOUND:
            raise ApiError(ErrorCode.FILE_CHANGED, "adapter files are missing (moved or deleted)") from None
        raise ApiError(ErrorCode.FILE_CHANGED, "adapter_config.json changed since import; re-import it") from None
    if cfg_now["sha256"] != want_cfg.get("sha256") or cfg_now["size"] != want_cfg["size"]:
        raise ApiError(ErrorCode.FILE_CHANGED, "adapter_config.json changed since import; re-import it")
    try:
        ck = open_checkpoint(d / "adapter_model.safetensors", _adapter_root(d))
    except WeightLabError as e:
        if e.code is ErrorCode.NOT_FOUND:
            raise ApiError(ErrorCode.FILE_CHANGED, "adapter files are missing (moved or deleted)") from None
        raise
    ident = next(iter(ck.file_identities().values()))
    if (ident.size, ident.mtime_ns, f"{ident.dev}:{ident.ino}") != (
        want["size"],
        want["mtime_ns"],
        want["file_id"],
    ) or (
        want.get("ctime_ns") is not None and ident.ctime_ns != want["ctime_ns"]  # absent for earlier imports
    ):
        ck.close()
        raise ApiError(ErrorCode.FILE_CHANGED, "adapter_model.safetensors changed since import")
    return ck
