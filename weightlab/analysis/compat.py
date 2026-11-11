"""Strict structural comparison of two checkpoints (FR-030, doc 04 section 7).

Keys are compared byte-for-byte. Statuses follow `ComparisonEntry` in contracts/openapi.yaml:
matched, missing_in_a, missing_in_b, tied, shape_mismatch, dtype_mismatch. A missing key is
`tied -> <source>` only with positive evidence (config.json `tie_word_embeddings: true` plus a
known key mapping); a missing tensor is never synthesized. Renamed keys are only *suspected*
(same shape and dtype, unique candidate) and are never matched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from weightlab.formats.base import TensorMeta
from weightlab.formats.checkpoint import Checkpoint
from weightlab.formats.dtypes import DTYPES

# Known architecture mappings for tied output heads: missing key -> candidate source keys.
TIE_MAPPINGS: dict[str, tuple[str, ...]] = {
    "lm_head.weight": ("model.embed_tokens.weight", "transformer.wte.weight"),
}


@dataclass
class Entry:
    key: str
    status: str
    a: TensorMeta | None
    b: TensorMeta | None
    analyzable: bool = False
    tied_to: str | None = None
    tie_evidence: str | None = None
    tied_side: str | None = None  # "a" or "b": the side that ties the key
    reasons: list[str] = field(default_factory=list)


@dataclass
class Structure:
    entries: list[Entry]
    tie_differs: bool
    dtype_differs: bool

    def counts(self) -> dict[str, int]:
        keys = ("matched", "missing_in_a", "missing_in_b", "tied", "shape_mismatch", "dtype_mismatch")
        out = dict.fromkeys(keys, 0)
        for e in self.entries:
            out[e.status] += 1
        return out

    @property
    def blocking(self) -> list[Entry]:
        """Entries that make the pair Class X."""
        return [
            e
            for e in self.entries
            if e.status in ("missing_in_a", "missing_in_b", "shape_mismatch")
            or (e.status == "dtype_mismatch" and not e.analyzable)
        ]


def family(dtype: str) -> str:
    kind = DTYPES[dtype].kind
    return "int" if kind in ("int", "uint") else kind


def tie_word_embeddings(config: dict[str, object] | None) -> bool:
    return config is not None and config.get("tie_word_embeddings") is True


def _tie_source(key: str, side: dict[str, TensorMeta], tied: bool) -> str | None:
    if not tied:
        return None
    for source in TIE_MAPPINGS.get(key, ()):
        if source in side:
            return source
    return None


def compare_structure(a: Checkpoint, b: Checkpoint) -> Structure:
    ma = {m.key: m for m in a.metadata()}
    mb = {m.key: m for m in b.metadata()}
    tied_a, tied_b = tie_word_embeddings(a.config()), tie_word_embeddings(b.config())
    entries: list[Entry] = []
    tie_differs = dtype_differs = False
    for key in [*ma, *(k for k in mb if k not in ma)]:
        ta, tb = ma.get(key), mb.get(key)
        if ta is not None and tb is not None:
            if ta.shape != tb.shape:
                entries.append(
                    Entry(key, "shape_mismatch", ta, tb, reasons=[f"shape {list(ta.shape)} vs {list(tb.shape)}"])
                )
            elif ta.dtype != tb.dtype:
                dtype_differs = True
                same = family(ta.dtype) == family(tb.dtype)
                why = (
                    f"dtype {ta.dtype} vs {tb.dtype}; compared as float64 values"
                    if same
                    else f"dtype {ta.dtype} vs {tb.dtype}: different numeric families, not compared"
                )
                entries.append(Entry(key, "dtype_mismatch", ta, tb, analyzable=same, reasons=[why]))
            else:
                entries.append(Entry(key, "matched", ta, tb, analyzable=True))
            continue
        present, missing_side = (ta, "b") if ta is not None else (tb, "a")
        assert present is not None
        missing_metas, tied_flag = (mb, tied_b) if missing_side == "b" else (ma, tied_a)
        source = _tie_source(key, missing_metas, tied_flag)
        if source is not None and missing_metas[source].shape == present.shape:
            tie_differs = True
            other = "A" if missing_side == "b" else "B"
            entries.append(
                Entry(
                    key,
                    "tied",
                    ta,
                    tb,
                    tied_to=source,
                    tie_evidence="config_and_known_mapping",
                    tied_side=missing_side,
                    reasons=[
                        f"{missing_side.upper()} ties {key} to {source} (config tie_word_embeddings=true); "
                        f"{other} stores it separately. The tie differs; {source} is compared once and "
                        f"{key} has no delta."
                    ],
                )
            )
            continue
        why = f"absent from {missing_side.upper()}"
        if tied_flag and key in TIE_MAPPINGS:
            why += "; tie_word_embeddings is set but no known source key with a matching shape exists"
        elif key in TIE_MAPPINGS:
            why += "; no tie evidence (config.json tie_word_embeddings is not true)"
        entries.append(Entry(key, f"missing_in_{missing_side}", ta, tb, reasons=[why]))
    _suspect_renames(entries)
    return Structure(entries, tie_differs, dtype_differs)


def _suspect_renames(entries: list[Entry]) -> None:
    only_a = [e for e in entries if e.status == "missing_in_b"]
    only_b = [e for e in entries if e.status == "missing_in_a"]
    for e in only_a:
        assert e.a is not None
        cands = [o for o in only_b if o.b is not None and o.b.shape == e.a.shape and o.b.dtype == e.a.dtype]
        if len(cands) == 1 and cands[0].key.split(".")[-1] == e.key.split(".")[-1]:
            e.reasons.append(f"renamed-suspected: {cands[0].key} in B has the same shape and dtype (not matched)")
            cands[0].reasons.append(f"renamed-suspected: {e.key} in A has the same shape and dtype (not matched)")


STRUCTURE_FINGERPRINT_VERSION = 1


def structure_fingerprint(s: Structure, identity: dict[str, Any] | None = None) -> str:
    """SHA-256 of the canonical structure: every entry's key, status, analyzability, shapes, dtypes and tie
    relationship (sorted by key), the pair-level flags, and `identity` (e.g. the comparison's model ids). Two
    structures with equal aggregate counts but any differing key, shape, dtype, status or tie differ here."""
    rows = sorted(
        (
            [e.key, e.status, e.analyzable,
             list(e.a.shape) if e.a else None, list(e.b.shape) if e.b else None,
             e.a.dtype if e.a else None, e.b.dtype if e.b else None,
             e.tied_to, e.tie_evidence, e.tied_side]
            for e in s.entries
        ),
        key=lambda r: str(r[0]),
    )  # fmt: skip
    payload = {
        "version": STRUCTURE_FINGERPRINT_VERSION,
        "identity": identity or {},
        "tie_differs": s.tie_differs,
        "dtype_differs": s.dtype_differs,
        "entries": rows,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
