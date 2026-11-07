#!/usr/bin/env python3
"""Run the test suite and write Phase 1 evidence (Tasks 1-2: database and local API; kit; Tasks 3-4A server side; Task 4B tile engine foundation).

    uv run python scripts/phase1_evidence.py --out evidence/phase1

Writes <out>/PHASE1_FOUNDATION_<os>.md and <out>/junit.xml (ignored by git). The route table and the
contract-gap list are read from the code (weightlab.api.app, tests.api.contract), not retyped.
Exit status is pytest's.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AREAS = [
    (
        "tests/db/test_migrations.py",
        "Task 1: migrations: fresh/idempotent, newer or edited DB refused unmodified, backup before upgrade, atomic rollback, integrity",
    ),
    (
        "tests/db/test_store.py",
        "Task 1: repositories: FK/CHECK/UNIQUE constraints, project cascade, 60k batched catalog insert, filters/sort/paging, cross-project look-ups",
    ),
    (
        "tests/api/test_contract_schemas.py",
        "Task 2: request models vs frozen JSON Schema: contract examples + generated differential mutations",
    ),
    (
        "tests/api/test_routes.py",
        "Task 2: route table == contract (implemented + deferred), deferred 404, docs disabled",
    ),
    (
        "tests/api/test_security.py",
        "Task 2: token, Host allowlist, Origin, no CORS, headers, body/concurrency limits, strict bodies, loopback only",
    ),
    (
        "tests/api/test_server_process.py",
        "Task 2: `weightlab start` real process: loopback socket, fresh token per launch, token never in logs",
    ),
    (
        "tests/api/test_models_api.py",
        "Task 2: projects, import in place, catalog, relink, stale fingerprints, import errors, no path disclosure",
    ),
    (
        "tests/api/test_roi_api.py",
        "Task 2: ROI for all four source kinds vs generator values, comparisons, adapters, overflow, lineage gating",
    ),
    (
        "tests/api/test_bookmarks_settings_api.py",
        "Task 2: bookmarks with ownership, settings round trip, bounds and persistence",
    ),
    (
        "tests/api/test_review_pr3.py",
        "PR #3 review: adapter config and shard index changes -> FILE_CHANGED, atomic relink, admission before body, cheap health",
    ),
    (
        "tests/api/test_ui_static.py",
        "Task 3: UI shell served token-free from memory; API auth, Host/Origin intact; traversal and unknown paths 404",
    ),
    (
        "tests/api/test_roi_explorer.py",
        "Task 4A: the explorer's ROI requests (rank 0/1/2/3, transpose, ragged edges, dtypes, 64-bit ints, -0.0, budget) vs NumPy decode of fixture bytes; edited file -> FILE_CHANGED",
    ),
    (
        "tests/integration/test_tile_engine.py",
        "Task 4B foundation: L0-L3+ direct tiles vs independent reference and goldens: aggregations, edges, non-finite, F64 extremes, I64/U64, BF16/F16, slices, transpose, FILE_CHANGED, budgets",
    ),
    (
        "tests/unit/test_tile_codec.py",
        "Task 4B foundation: binary layout/byte order, JSON parity vs contract schemas, status and metadata validation, headers, canonical identity and cache keys",
    ),
    (
        "tests/property/test_tile_property.py",
        "Task 4B foundation: random tensors/views/levels/bands vs the reference; tiles partition the view",
    ),
    (
        "tests/perf/test_tile_bounded.py",
        "Task 4B foundation: 4 GiB sparse checkpoint, L0-L4 tiles: flat peak memory, reads = covered bytes, L5 refused",
    ),
    ("tests/security/test_special_files.py", "Reader hardening: FIFOs/devices/directories refused without blocking"),
    (
        "tests/integration/test_evaluation_kit.py",
        "Workstream A: evaluation kit examples reproduce against generator values, checks can fail, docs consistent",
    ),
]


def os_tag() -> str:
    return {"darwin": "macos", "win32": "windows"}.get(sys.platform, sys.platform)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "evidence" / "phase1")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    junit = out / "junit.xml"
    started = dt.datetime.now(dt.UTC)
    cmd = [sys.executable, "-m", "pytest", "-q", "-rs", f"--junitxml={junit}", "-o", "junit_family=xunit1"]
    test = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)

    per_file: dict[str, Counter[str]] = {}
    skipped: list[str] = []
    for c in ET.parse(junit).getroot().iter("testcase"):  # noqa: S314 - our own pytest output
        f = c.get("file") or c.get("classname", "").replace(".", "/") + ".py"
        status = "failed" if c.find("failure") is not None or c.find("error") is not None else (
            "skipped" if c.find("skipped") is not None else "passed")  # fmt: skip
        per_file.setdefault(f, Counter())[status] += 1
        if status == "skipped":
            msg = c.find("skipped")
            skipped.append(f"{c.get('name')}: {msg.get('message', '') if msg is not None else ''}")
    totals = sum(per_file.values(), Counter())

    from tests.api.contract import CONTRACT_GAPS
    from weightlab.api.app import DEFERRED, IMPLEMENTED
    from weightlab.db.schema import latest_version

    verdict = "PASS" if test.returncode == 0 and totals["failed"] == 0 else "FAIL"
    lines = [
        f"# Phase 1 foundation evidence (Tasks 1-2): {os_tag()} ({verdict} on this OS only)",
        "",
        f"- Date (UTC): {started:%Y-%m-%d %H:%M}",
        f"- Platform: {platform.platform()} / {platform.machine()} / Python {platform.python_version()}",
        f"- Command: `uv run python scripts/phase1_evidence.py` -> full `pytest` suite (exit {test.returncode})",
        f"- Totals (whole suite): {totals['passed']} passed, {totals['failed']} failed, {totals['skipped']} skipped",
        f"- Database schema version: {latest_version()}; contract: contracts/openapi.yaml 0.3.0-spec-1.2-final",
        "",
        "## Results by area",
        "",
        "| Area | Tests | Passed | Failed | Skipped |",
        "|---|---|---|---|---|",
    ]
    listed = set()
    for f, label in AREAS:
        cnt = per_file.get(f, Counter())
        listed.add(f)
        lines.append(
            f"| {label} (`{f}`) | {sum(cnt.values())} | {cnt['passed']} | {cnt['failed']} | {cnt['skipped']} |"
        )
    rest = sum((c for f, c in per_file.items() if f not in listed), Counter())
    lines.append(
        f"| Phase 0 / 0.5 suites (regression) | {sum(rest.values())} | {rest['passed']} | {rest['failed']} | {rest['skipped']} |"
    )
    lines += ["", "## Routes (contract 0.3.0)", "", "| Method | Path | State |", "|---|---|---|"]
    for method, path in sorted(IMPLEMENTED | set(DEFERRED), key=lambda x: (x[1], x[0])):
        state = "implemented" if (method, path) in IMPLEMENTED else f"deferred: {DEFERRED[(method, path)]}"
        lines.append(f"| {method.upper()} | `{path}` | {state} |")
    lines += ["", "## Contract gaps exercised by the tests", "",
              "Statuses required by doc 09's behaviour rules but not declared for the route in the frozen contract "
              "(bodies still validate against `Error`):", ""]  # fmt: skip
    lines += [f"- {m.upper()} `{p}` -> {s}" for m, p, s in sorted(CONTRACT_GAPS, key=lambda x: (x[1], x[0], x[2]))]
    lines += ["", "## Skipped on this OS", "", *([f"- {s}" for s in skipped] or ["- none"]), ""]
    report = out / f"PHASE1_FOUNDATION_{os_tag()}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(test.stdout[-1500:])
    print(f"wrote {report}: {verdict}")
    return test.returncode


if __name__ == "__main__":
    sys.exit(main())
