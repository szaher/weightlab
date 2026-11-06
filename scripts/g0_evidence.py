#!/usr/bin/env python3
"""Run WeightLab's own test suite and write the G0 evidence report (doc 13 section 9).

    uv run python scripts/g0_evidence.py --out evidence/g0

Writes <out>/G0_<os>.md, <out>/junit.xml and <out>/benchmark.json. Exit status is pytest's.
The fixture self-check (scripts/check_demo_fixtures.py) runs too but is listed separately: it is
not G0 evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUPS = {
    "tests/integration/test_fixture_conformance.py": "Malformed fixtures rejected with expected codes; upstream accept/reject agreement",
    "tests/integration/test_golden_values.py": "Golden values, lossless BF16/F16/F64/I64/U64 ROI, JSON round trip",
    "tests/unit/test_header.py": "Strict header rules, all 65,536 BF16/F16 bit patterns",
    "tests/unit/test_selection.py": "View/ROI validation, read planner, budgets",
    "tests/property/test_roi_property.py": "Property: random N-D ROIs equal an independent NumPy slice",
    "tests/property/test_header_fuzz.py": "Property: corrupted files fail closed, never more permissive than upstream",
    "tests/security/test_paths.py": "Shard path rules, symlink and root containment",
    "tests/security/test_mutation.py": "Mutation/replacement during reads and jobs (AC-014, B11)",
    "tests/perf/test_large_sparse.py": "Oversized sparse file RSS (AC-002)",
}


def os_tag() -> str:
    return {"darwin": "macos", "win32": "windows"}.get(sys.platform, sys.platform)


def mib(n: int | None) -> str:
    return "n/a" if n is None else f"{n / 2**20:.1f} MiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "evidence" / "g0")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    junit = out / "junit.xml"
    env = dict(os.environ, WEIGHTLAB_EVIDENCE_DIR=str(out))
    started = dt.datetime.now(dt.UTC)
    cmd = [sys.executable, "-m", "pytest", "-q", "-rs", f"--junitxml={junit}", "-o", "junit_family=xunit1"]
    test = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    selfcheck = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_demo_fixtures.py"), "--root", str(ROOT / "tests/fixtures/generated")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip

    cases = ET.parse(junit).getroot().iter("testcase")  # noqa: S314 - our own pytest output
    per_file: dict[str, Counter[str]] = {}
    mutation_rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    for c in cases:
        f = c.get("file") or c.get("classname", "").replace(".", "/") + ".py"
        status = "failed" if c.find("failure") is not None or c.find("error") is not None else (
            "skipped" if c.find("skipped") is not None else "passed")  # fmt: skip
        per_file.setdefault(f, Counter())[status] += 1
        if status == "skipped":
            msg = c.find("skipped")
            skipped.append(f"{c.get('name')}: {msg.get('message', '') if msg is not None else ''}")
        props = {p.get("name"): p.get("value") for p in c.iter("property")}
        if "mutation" in f and props:
            mutation_rows.append((c.get("name", ""), ", ".join(f"{k}={v}" for k, v in props.items())))

    bench_path = out / "benchmark.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else None
    totals = sum(per_file.values(), Counter())
    verdict = "PASS" if test.returncode == 0 and totals["failed"] == 0 else "FAIL"

    lines = [
        f"# G0 evidence: {os_tag()} ({verdict})",
        "",
        f"- Date (UTC): {started:%Y-%m-%d %H:%M}",
        f"- Platform: {platform.platform()} / {platform.machine()} / Python {platform.python_version()}",
        f"- Command: `uv run python scripts/g0_evidence.py` -> `python -m pytest -q -rs --junitxml=junit.xml` (exit {test.returncode})",
        f"- Totals: {totals['passed']} passed, {totals['failed']} failed, {totals['skipped']} skipped",
        "",
        "## WeightLab test results",
        "",
        "| Area | Tests | Passed | Failed | Skipped |",
        "|---|---|---|---|---|",
    ]
    for f, cnt in sorted(per_file.items()):
        lines.append(
            f"| {GROUPS.get(f, f)} (`{f}`) | {sum(cnt.values())} | {cnt['passed']} | {cnt['failed']} | {cnt['skipped']} |"
        )
    lines += ["", "## Mutation and replacement outcomes (AC-014, B11)", "", "| Case | Outcome |", "|---|---|"]
    lines += [f"| `{n}` | {o} |" for n, o in mutation_rows] or ["| (none recorded) | |"]
    lines += ["", "## Memory: oversized sparse file (AC-002)", ""]
    if bench:
        lines += [
            f"- File: {bench['file_bytes']:,} bytes on disk (allocated {bench['file_allocated_bytes']}), one BF16 tensor {bench['shape']}",
            f"- Peak RSS after import: {mib(bench['rss_after_import'])}; after catalog: {mib(bench['rss_after_catalog'])}; "
            f"after exact ROIs: {mib(bench['rss_after_roi'])}; after streaming {bench['scan_bytes'] / 2**30:.1f} GiB: "
            f"{mib(bench['rss_after_scan'])}",
            f"- Catalog {bench['catalog_seconds'] * 1000:.2f} ms; ROIs {bench['roi_seconds'] * 1000:.2f} ms; scan {bench['scan_seconds']:.2f} s",
            f"- ROI values checked against written bit patterns: {bench['roi_correct']}; strided column ROI read {bench['column_stripe_bytes']} bytes",
        ]  # fmt: skip
    else:
        lines.append("- Benchmark did not run (see skipped tests).")
    lines += [
        "",
        "## Supported dtypes and format errors",
        "",
        "- Lossless exact ROI: F64, F32, F16, BF16 (WeightLab decoder), I64, I32, I16, I8, U64, U32, U16, U8, BOOL.",
        "- Rejected with `UNSUPPORTED_DTYPE`: F8_E4M3, F8_E5M2, F8_E8M0, F6_*, F4, C64, C128 and unknown dtype strings.",
        "- Format errors raised by the reader: `INVALID_FORMAT`, `UNSUPPORTED_DTYPE`, `MISSING_SHARD`, `PATH_OUTSIDE_ROOT`, "
        "`FILE_CHANGED`, `INVALID_ROI`, `RESOURCE_LIMIT`, `UNSUPPORTED_PRECISION`, `NOT_FOUND`.",
        "",
        "## Skipped on this OS",
        "",
        *([f"- {s}" for s in skipped] or ["- none"]),
        "",
        "## Fixture self-check (NOT G0 evidence)",
        "",
        f"- `scripts/check_demo_fixtures.py`: {(selfcheck.stdout.strip().splitlines() or ['no output'])[-1]} (exit {selfcheck.returncode})",
        "",
    ]  # fmt: skip
    report = out / f"G0_{os_tag()}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(test.stdout[-2000:])
    print(f"wrote {report.relative_to(ROOT) if report.is_relative_to(ROOT) else report}: {verdict}")
    return test.returncode


if __name__ == "__main__":
    sys.exit(main())
