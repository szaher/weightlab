"""Base identity and lineage assessment (doc 04 section 7, ADR-014, ADR-018, doc 06 section 6).

Two questions stay separate:
1. Is the base file the one the derivative names? (`base_file_identity_verified`) Only from file
   content: SHA-256 recorded in a manifest, the offline Hugging Face cache blob hashes for a declared
   repo *and* revision, or WeightLab's own synthetic fixture generator.
2. Was the derivative trained from it? (`lineage_trust_level`) A hash match never proves this.

Class V = structure compatible + base identity verified. L = declared or user-attested lineage.
S = shape-compatible only. X = structurally incompatible. Weight similarity is never evidence.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from weightlab.core.errors import ErrorCode, fail
from weightlab.formats.base import strict_json
from weightlab.formats.checkpoint import Checkpoint

# Files produced by scripts/generate_demo_fixtures.py (generator v3, seed 7): derivative digest ->
# required base digest. tests/integration/test_diff_golden.py keeps this in sync with
# examples/fixtures/expected_results.json.
TINY_BASE = "a701e77f066883608beb2a1fc080d52f06ebffca227c1d6fd12de2d484e81133"
SYNTHETIC_DERIVATIVES: dict[str, tuple[str, str]] = {
    "16bbefa9b14f910c751a476ae4a0570100f4589f03f498f242ae3700639c1452": ("tiny_ft.safetensors", TINY_BASE),
    "5ea2c555103e23d55a8ea35ae3b862ce5a786d6c6475cf00ae180f764d00a30c": ("tiny_lora", TINY_BASE),
    "284d457788ddcb1a3e937a74bdcd08ea28db164fce5006d60c2bfa8676d670da": ("tiny_lora_rslora", TINY_BASE),
    "abb0af27897cd5b483546c941f4552e287d8011fbb577c1b50df4a636552b879": ("tiny_lora_dora_unsupported", TINY_BASE),
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_HF_SNAPSHOT = re.compile(r"models--(?P<org>[^/\\]+?)--(?P<name>[^/\\]+)[/\\]snapshots[/\\](?P<rev>[0-9a-f]{40})$")


@dataclass
class LineageInputs:
    manifest: dict[str, Any] | None = None  # parsed --lineage-manifest
    trust_manifest: bool = False
    user_attestation: bool = False
    declared_base: str | None = None  # adapter_config base_model_name_or_path
    declared_revision: str | None = None  # adapter_config revision


@dataclass
class Assessment:
    compatibility_class: str
    base_file_identity_verified: bool
    base_identity_method: str
    lineage_claim_source: str
    lineage_trust_level: str
    lineage_evidence: str
    transfer_eligibility: str
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


MANIFEST_MAX_BYTES = 16 * 1024 * 1024


def load_manifest(path: Path) -> dict[str, Any]:
    """Lineage manifest: {"schema_version": 1, "base_files": {relpath: sha256}, "base_id": str?}."""
    with path.open("rb") as f:
        raw = f.read(MANIFEST_MAX_BYTES + 1)
    if len(raw) > MANIFEST_MAX_BYTES:
        raise fail(ErrorCode.RESOURCE_LIMIT, f"lineage manifest exceeds {MANIFEST_MAX_BYTES} bytes")
    doc = strict_json(raw, "lineage manifest")
    if not isinstance(doc, dict) or doc.get("schema_version") != 1:
        raise fail(ErrorCode.VALIDATION_ERROR, "lineage manifest must be an object with schema_version 1")
    files = doc.get("base_files")
    if files is not None and (
        not isinstance(files, dict) or not all(isinstance(v, str) and _SHA.match(v) for v in files.values())
    ):
        raise fail(ErrorCode.VALIDATION_ERROR, "base_files must map file names to lowercase SHA-256 hex digests")
    if "base_id" in doc and not isinstance(doc["base_id"], str):
        raise fail(ErrorCode.VALIDATION_ERROR, "base_id must be a string")
    return doc


def hf_snapshot(ck: Checkpoint) -> tuple[str, str, Path] | None:
    """(repo_id, commit, snapshot dir) when the entry sits in an HF hub cache snapshot."""
    m = _HF_SNAPSHOT.search(os.path.abspath(ck.entry_dir))
    if not m:
        return None
    return f"{m['org']}/{m['name']}", m["rev"], ck.entry_dir


def _hf_revision_matches(snapshot: Path, commit: str, declared: str) -> bool:
    if declared == commit:
        return True
    ref = snapshot.parent.parent / "refs" / declared
    if re.fullmatch(r"[A-Za-z0-9._-]{1,100}", declared) and ref.is_file():
        return ref.read_text(encoding="utf-8").strip() == commit
    return False


def _hf_blob_check(ck: Checkpoint) -> list[str]:
    """Problems with the cache-blob proof; empty list = every shard blob name equals its fresh SHA-256."""
    problems = []
    digests = ck.digests()
    for shard_id in ck.shard_ids:
        real = Path(os.path.realpath(ck.shard_path(shard_id)))
        if real.parent.name != "blobs" or not _SHA.match(real.name):
            problems.append(f"{shard_id} is not a hub-cache LFS blob")
        elif real.name != digests[shard_id]:
            problems.append(f"{shard_id} blob name differs from its content SHA-256")
    return problems


def assess(
    base: Checkpoint,
    derived_digest: str | None,
    inputs: LineageInputs,
    *,
    structure_blocking: bool,
    structure_caps_at_l: list[str],
    adapter: bool,
) -> Assessment:
    reasons: list[str] = []
    verified, method, claim, trust = False, "none", "none", "none"
    base_digests: dict[str, str] | None = None

    def digests() -> dict[str, str]:
        nonlocal base_digests
        if base_digests is None:
            base_digests = base.digests()
        return base_digests

    # 1. WeightLab's own generator: ancestry known by construction.
    synthetic = SYNTHETIC_DERIVATIVES.get(derived_digest or "")
    if synthetic and len(base.shard_ids) == 1 and next(iter(digests().values())) == synthetic[1]:
        verified, method, claim, trust = (
            True,
            "sha256_recorded",
            "synthetic_fixture_generator",
            "proven_by_construction",
        )
        reasons.append(
            f"synthetic fixture pair ({synthetic[0]} built from tiny_base by the generator): "
            "ancestry known by construction"
        )
    # 2. Manifest with recorded base hashes.
    if not verified and inputs.manifest is not None:
        files = inputs.manifest.get("base_files")
        claim = "weightlab_manifest"
        trust = "trusted_statement" if inputs.trust_manifest else "unverified_claim"
        if files:
            mismatched = sorted(set(files) ^ set(digests())) + [
                k for k in files if digests().get(k) not in (None, files[k])
            ]
            if not mismatched:
                verified, method = True, "sha256_recorded"
                reasons.append("every base file matches the SHA-256 recorded in the lineage manifest")
            else:
                reasons.append(
                    f"lineage manifest hashes do not match the base files ({len(mismatched)} differ or are missing)"
                )
                return _finish(
                    "S" if not structure_blocking else "X",
                    False,
                    "none",
                    claim,
                    trust,
                    "none",
                    adapter,
                    [*reasons, "the named base is not this base: lineage claim contradicted"],
                )
        else:
            reasons.append("lineage manifest declares a base without file hashes")
    # 3. Offline HF cache: declared repo + revision, blob names equal fresh hashes.
    if not verified and adapter and inputs.declared_base:
        claim, trust = "adapter_config", "unverified_claim"
        snap = hf_snapshot(base)
        if snap is not None:
            repo, commit, path = snap
            if inputs.declared_base != repo:
                reasons.append(f"adapter declares base {inputs.declared_base!r} but the base is {repo!r}")
                return _finish(
                    "S" if not structure_blocking else "X",
                    False,
                    "none",
                    claim,
                    trust,
                    "none",
                    adapter,
                    [*reasons, "lineage claim contradicted by the hub-cache identity"],
                )
            if inputs.declared_revision and _hf_revision_matches(path, commit, inputs.declared_revision):
                problems = _hf_blob_check(base)
                if not problems:
                    verified, method = True, "hf_cache_blob_sha256"
                    reasons.append(
                        f"hub cache: declared {repo}@{inputs.declared_revision} is snapshot {commit} and "
                        "every blob name equals its SHA-256"
                    )
                else:
                    reasons += problems
            else:
                reasons.append("adapter declares the base repo but no matching revision; identity not verified")
        else:
            reasons.append(f"adapter declares base {inputs.declared_base!r}; base location is not a hub-cache snapshot")

    if verified:
        evidence = "cryptographically_verified"
        cls = "V"
        if structure_caps_at_l:
            cls = "L"
            reasons += [f"capped at Class L: {why}" for why in structure_caps_at_l]
    elif inputs.user_attestation:
        digests()
        claim, trust, evidence, cls = "user_attestation", "unverified_claim", "user_attested", "L"
        reasons.append("user attested these exact files (hashes recorded); not cryptographic lineage evidence")
    elif claim in ("adapter_config", "weightlab_manifest"):
        evidence, cls = "declared", "L"
        reasons.append("lineage declared by metadata, not cryptographically verified")
    else:
        evidence, cls = "none", "S"
        reasons.append("no lineage evidence: shape-compatible only; deltas may be meaningless (permutation/alignment)")
    if structure_blocking:
        cls = "X"
        reasons.append(
            "structurally incompatible (missing tensors, shape or dtype-family mismatch); matched "
            "tensors are shown as unverified inspection only"
        )
    return _finish(cls, verified, method, claim, trust, evidence, adapter, reasons)


def _finish(
    cls: str, verified: bool, method: str, claim: str, trust: str, evidence: str, adapter: bool, reasons: list[str]
) -> Assessment:
    trusted = trust in ("proven_by_construction", "trusted_statement")
    if cls == "X":
        transfer = "T5"
    elif cls == "S":
        transfer = "T4"
    elif adapter:
        transfer = "T1" if cls == "V" and trusted else "T2"
    else:
        transfer = "T3" if cls == "V" and trusted else "T4"
    return Assessment(cls, verified, method, claim, trust, evidence, transfer, reasons)
