"""Report writers: JSON, CSV (tensor/module table, layer and role rollups) and Markdown.

Tensor keys and metadata strings are untrusted data: Markdown cells are escaped and CSV cells that
a spreadsheet would evaluate as a formula (leading = + - @ tab CR) are prefixed with a quote.
The files of one run are published as a set (`write_report_set`): a cancelled or failed export
rolls back to the previous state, so it leaves neither a partial file nor a mix of old and new
files behind (not atomic for concurrent readers while the final renames run).
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import shutil
import signal
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

METRIC_COLUMNS = ("delta_l2", "delta_rms", "relative_l2", "cosine", "max_abs_delta", "norm_a", "norm_b")
COUNT_COLUMNS = (
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
ROLLUP_COLUMNS = (
    "tensors",
    "elements",
    "finite_pairs",
    "invalid_pairs",
    "withheld_tensors",
    "delta_l2",
    "delta_rms",
    "relative_l2",
    "max_abs_delta",
    "near_zero_reference",
    "withheld_metrics",
)


def to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):
        return "x".join(str(x) for x in v)
    s = str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def _csv(header: list[str], rows: list[list[Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([_cell(v) for v in r])
    return buf.getvalue()


def tensors_csv(report: dict[str, Any]) -> str:
    adapter = report["mode"] == "adapter"
    head = ["rank", "key", "layer", "role", "status", "shape"] + ([] if adapter else ["dtype_a", "dtype_b"])
    head += (["lora_rank", "lora_scale", "gmac"] if adapter else []) + list(COUNT_COLUMNS) + list(METRIC_COLUMNS)
    head += ["numeric_precision", "withheld_metrics", "caveat"]
    rows = []
    for t in report["tensors"]:
        c = t.get("counts") or {}
        m = t.get("metrics") or {}
        precisions = sorted({m[k]["numeric_precision"] for k in m})
        row = [t["rank"], t["key"], t["layer"], t["role"], t["status"], t["shape"]]
        row += [] if adapter else [t["dtype_a"], t["dtype_b"]]
        row += [t["lora"]["rank"], t["lora"]["scale"], t["lora"]["gmac"]] if adapter else []
        row += [c.get(k) for k in COUNT_COLUMNS] + [m[k]["value"] if k in m else None for k in METRIC_COLUMNS]
        row += ["/".join(precisions), "/".join(t.get("withheld_metrics") or []), t["caveat"]]
        rows.append(row)
    return _csv(head, rows)


def rollup_csv(report: dict[str, Any], kind: str) -> str:
    label = "layer" if kind == "layers" else "role"

    def cell(v: Any) -> Any:
        return "/".join(v) if isinstance(v, list) else v

    return _csv([label, *ROLLUP_COLUMNS], [[r[label], *(cell(r[k]) for k in ROLLUP_COLUMNS)] for r in report[kind]])


def _md(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v == 0.0 or not math.isfinite(v):
            return repr(v)
        return f"{v:.6g}"
    if isinstance(v, list):
        return "×".join(str(x) for x in v)
    s = str(v).replace("\\", "\\\\").replace("|", "\\|").replace("`", "'").replace("\n", " ").replace("\r", " ")
    return s.replace("<", "&lt;").replace(">", "&gt;")


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_md(v) for v in r) + " |" for r in rows]
    return out


def to_markdown(report: dict[str, Any], top: int = 25) -> str:
    lin = report["compatibility"]["lineage"]
    inputs = report["provenance"]["inputs"]
    adapter = report["mode"] == "adapter"
    second = inputs["adapter"] if adapter else inputs["derived"]
    lines = [
        "# WeightLab change report",
        "",
        f"- Mode: **{report['mode']}**; orientation B − A (A = base `{_md(inputs['base']['entry'])}`, "
        f"B = {'base + adapter' if adapter else 'derived'} `{_md(second['entry'])}`)",
        f"- Complete: **{'yes' if report['complete'] else 'NO'}**",
        "",
        "## Compatibility",
        "",
        f"- Class **{lin['compatibility_class']}**; base file identity verified: {lin['base_file_identity_verified']} "
        f"({lin['base_identity_method']})",
        f"- Lineage claim source: {lin['lineage_claim_source']}; trust: {lin['lineage_trust_level']}; "
        f"evidence: {lin['lineage_evidence']}; transfer eligibility (informational): {lin['transfer_eligibility']}",
        *[f"- {_md(r)}" for r in lin["reasons"]],
        "",
        "Counts: " + ", ".join(f"{k} {v}" for k, v in report["compatibility"]["counts"].items()),
        "",
    ]
    nc = report["compatibility"]["not_compared"]
    if nc:
        lines += [f"### Not compared ({len(nc)})", ""]
        lines += _table(
            ["key", "status", "reason"], [[e["raw_key"], e["status"], "; ".join(e["reasons"])] for e in nc[:200]]
        )
        lines += [f"\n… {len(nc) - 200} more in the JSON report" if len(nc) > 200 else "", ""]
    lines += [
        "## Completeness",
        "",
        ", ".join(f"{k}: {v}" for k, v in report["completeness"].items()),
        "",
        "## Caveats",
        "",
        *[f"- {_md(c)}" for c in report["caveats"]],
        "",
    ]
    sort = report["settings"]["sort"]
    rows = report["tensors"][:top]
    head = [
        "#",
        "key",
        "layer",
        "role",
        "shape",
        "rel L2",
        "Δ L2",
        "Δ RMS",
        "cos(A,B)",
        "max abs Δ",
        "non-finite pairs",
    ]
    body = []
    for t in rows:
        m, c = t["metrics"], t.get("counts") or {}
        body.append(
            [
                t["rank"],
                t["key"],
                t["layer"],
                t["role"],
                t["shape"],
                *(
                    m[k]["value"] if k in m else None
                    for k in ("relative_l2", "delta_l2", "delta_rms", "cosine", "max_abs_delta")
                ),
                c.get("invalid_pairs"),
            ]
        )
    lines += [f"## Top {len(rows)} of {len(report['tensors'])} {'modules' if adapter else 'tensors'} by {sort}", ""]
    lines += [*_table(head, body), ""]
    flagged = [t for t in report["tensors"] if t.get("caveat")]
    if flagged:
        lines += [f"Rows with caveats: {len(flagged)} (see the caveat column in the CSV/JSON).", ""]
    for kind, label in (("layers", "layer"), ("roles", "role")):
        lines += [f"## By {label} (heuristic)", ""]
        lines += [
            *_table(
                [label, "tensors", "rel L2", "Δ L2", "Δ RMS", "max abs Δ", "withheld"],
                [
                    [
                        r[label],
                        r["tensors"],
                        r["relative_l2"],
                        r["delta_l2"],
                        r["delta_rms"],
                        r["max_abs_delta"],
                        r["withheld_tensors"],
                    ]
                    for r in report[kind]
                ],
            ),
            "",
        ]
    tot = report["totals"]
    lines += [
        f"Whole model (analyzed tensors): rel L2 {_md(tot['relative_l2'])}, Δ L2 {_md(tot['delta_l2'])}, "
        f"Δ RMS {_md(tot['delta_rms'])}, {tot['tensors']} tensors, {tot['elements']} elements.",
        "",
    ]
    prov = report["provenance"]
    files = "; ".join(
        f"{_md(f['name'])} {f.get('size', '?')} B sha256 {f['sha256'] or 'not computed'}"
        for side in inputs.values()
        for f in side["files"]
    )
    lines += [
        "## Provenance",
        "",
        f"- weightlab {prov['tool']['version']} (parser v{prov['tool']['parser_version']}, numerics "
        f"v{prov['tool']['numerics_version']}), created {prov['created_at']}",
        f"- Operation: {prov['operation']['kind']}, compute float64, coverage complete, "
        f"consistency {report['settings']['consistency']}, epsilon_norm {report['settings']['epsilon_norm']}",
        f"- Files: {files}",
        f"- Paths included: {prov['privacy']['paths_included']}; input files opened read-only.",
        *(f"- Warning: {_md(w)}" for w in report["warnings"]),
        "",
    ]
    return "\n".join(lines)


def write_atomic(path: Path, text: str) -> None:
    """One file: temporary file in the same directory, fsync, rename."""
    write_report_set({path: text})


def _stage(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return Path(tmp)


@contextlib.contextmanager
def _sigint_deferred() -> Iterator[None]:
    """Keep Ctrl-C from splitting the (short) rename phase; a deferred Ctrl-C is dropped because
    the complete set has been published by then."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, lambda *_: None)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def write_report_set(files: dict[Path, str], bundle_dir: Path | None = None) -> None:
    """Publish several report files as one set: on any failure, no new file stays published.

    1. Stage: every file is written and fsynced under a temporary name next to its target (or in a
       temporary sibling of a new `bundle_dir`). A failure or Ctrl-C here removes the temporaries
       and publishes nothing.
    2. Commit, with Ctrl-C deferred: a new `bundle_dir` (`--out`) appears with one directory rename;
       every other target is renamed into place, previous versions kept as backups. If any step
       fails, everything already published in this call is rolled back: replaced files are restored
       from their backups, new files are removed and a newly published `bundle_dir` is removed.

    Guarantee: a failed call leaves the previous state (best effort if the rollback itself fails,
    e.g. a vanished disk). It is not atomic for concurrent readers: while the renames run, a reader
    can see some targets new and others old. Only a new `bundle_dir` is published atomically.
    """
    files = {Path(os.path.abspath(p)): t for p, t in files.items()}
    tmp_dir: Path | None = None
    bundle: dict[Path, str] = {}
    if bundle_dir is not None:
        bundle_dir = Path(os.path.abspath(bundle_dir))
        if not bundle_dir.exists():
            bundle = {p: t for p, t in files.items() if p.parent == bundle_dir}
    rest = {p: t for p, t in files.items() if p not in bundle}
    staged: list[tuple[Path, Path]] = []
    try:
        if bundle:
            assert bundle_dir is not None
            bundle_dir.parent.mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(prefix=f".{bundle_dir.name}.", suffix=".tmp", dir=bundle_dir.parent))
            for p, t in bundle.items():
                os.replace(_stage(tmp_dir / p.name, t), tmp_dir / p.name)
        staged = _stage_all(rest)
        with _sigint_deferred():
            published_dir = False
            try:
                if tmp_dir is not None:
                    assert bundle_dir is not None
                    os.rename(tmp_dir, bundle_dir)  # fails if bundle_dir appeared meanwhile
                    published_dir = True
                _commit(staged)  # restores its own targets on failure
            except BaseException:
                if published_dir:
                    assert bundle_dir is not None
                    shutil.rmtree(bundle_dir, ignore_errors=True)
                raise
    except BaseException:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        for _, tmp in staged:
            tmp.unlink(missing_ok=True)
        raise


def _stage_all(files: dict[Path, str]) -> list[tuple[Path, Path]]:
    staged: list[tuple[Path, Path]] = []
    try:
        for p, t in files.items():
            staged.append((p, _stage(p, t)))
    except BaseException:
        for _, tmp in staged:
            tmp.unlink(missing_ok=True)
        raise
    return staged


def _commit(staged: list[tuple[Path, Path]]) -> None:
    done: list[tuple[Path, Path | None]] = []  # (target, backup of its previous version)
    try:
        for target, tmp in staged:
            backup = None
            if target.exists():
                fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".bak", dir=target.parent)
                os.close(fd)
                backup = Path(name)
                os.replace(target, backup)  # over the reserved empty file
            done.append((target, backup))
            os.replace(tmp, target)
    except BaseException:
        for target, backup in reversed(done):
            if backup is not None:
                os.replace(backup, target)
            else:
                target.unlink(missing_ok=True)
        for _, tmp in staged:
            tmp.unlink(missing_ok=True)
        raise
    for _, backup in done:
        if backup is not None:
            backup.unlink(missing_ok=True)
