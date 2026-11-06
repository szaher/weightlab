#!/usr/bin/env python3
"""Run WeightLab's test suite and the diff benchmark, and write the G0.5 evidence (WL-024).

    uv run python scripts/g05_evidence.py --out evidence/g0.5 [--sizes-gib 0.25,1,4,12]
        [--pair BASE DERIVED] [--adapter BASE ADAPTER_DIR]

Writes <out>/G0.5_<os>.md, <out>/junit.xml and <out>/benchmark.json. Exit status is non-zero if
any test fails or a synthetic benchmark metric is wrong. The report states per-OS results only;
it never marks a gate for another OS as passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASE05 = {
    "tests/unit/test_accumulate.py": "Streaming metrics vs exact Fraction reference: any chunking, extreme magnitudes",
    "tests/integration/test_diff_golden.py": "Fixture reports vs generator values, compatibility classes, golden snapshots",
    "tests/integration/test_lineage_and_lora.py": "Lineage evidence paths, LoRA scale/layout, fail-closed adapters",
    "tests/integration/test_cli.py": "CLI exports, exit codes, output safety, escaping, privacy, cancellation",
    "tests/integration/test_review_regressions.py": "PR #2 review regressions: epsilon rollups, limits, I64/U64, LoRA overflow/Gram/memory, export set",
    "tests/perf/test_diff_bounded.py": "Diff peak RSS independent of pair size",
}


def os_tag() -> str:
    return {"darwin": "macos", "win32": "windows"}.get(sys.platform, sys.platform)


def mib(n: float) -> str:
    return f"{n / 2**20:.1f} MiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "evidence" / "g0.5")
    ap.add_argument("--sizes-gib", default="0.25,1,4")
    ap.add_argument("--work", type=Path, help="scratch directory for sparse benchmark files (default: <out>/.work)")
    ap.add_argument("--pair", nargs=2, metavar=("BASE", "DERIVED"))
    ap.add_argument("--adapter", nargs=2, metavar=("BASE", "ADAPTER_DIR"))
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    junit = out / "junit.xml"
    started = dt.datetime.now(dt.UTC)
    cmd = [sys.executable, "-m", "pytest", "-q", "-rs", f"--junitxml={junit}", "-o", "junit_family=xunit1"]
    test = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)

    bench_path = out / "benchmark.json"
    work = args.work or out / ".work"
    bcmd = [sys.executable, str(ROOT / "scripts/run_diff_benchmark.py"), "--dir", str(work),
            "--sizes-gib", args.sizes_gib, "--json", str(bench_path)]  # fmt: skip
    if args.pair:
        bcmd += ["--pair", *args.pair]
    if args.adapter:
        bcmd += ["--adapter", *args.adapter]
    bench_proc = subprocess.run(bcmd, cwd=ROOT, capture_output=True, text=True, check=False)
    for f in (work / "reports").glob("*.json"):
        f.unlink()
    for d in (work / "reports", work):
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    bench = json.loads(bench_path.read_text()) if bench_proc.returncode == 0 else None

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
    ok = test.returncode == 0 and totals["failed"] == 0 and bench is not None
    verdict = "PASS" if ok else "FAIL"

    lines = [
        f"# G0.5 evidence: {os_tag()} ({verdict} on this OS only)",
        "",
        f"- Date (UTC): {started:%Y-%m-%d %H:%M}",
        f"- Platform: {platform.platform()} / {platform.machine()} / Python {platform.python_version()}",
        f"- Command: `uv run python scripts/g05_evidence.py` -> full `pytest` suite (exit {test.returncode}) and "
        f"`scripts/run_diff_benchmark.py` (exit {bench_proc.returncode})",
        f"- Totals (whole suite, Phase 0 + 0.5): {totals['passed']} passed, {totals['failed']} failed, "
        f"{totals['skipped']} skipped",
        "",
        "## Phase 0.5 test results",
        "",
        "| Area | Tests | Passed | Failed | Skipped |",
        "|---|---|---|---|---|",
    ]  # fmt: skip
    for f, label in PHASE05.items():
        cnt = per_file.get(f, Counter())
        lines.append(
            f"| {label} (`{f}`) | {sum(cnt.values())} | {cnt['passed']} | {cnt['failed']} | {cnt['skipped']} |"
        )
    rest = sum((c for f, c in per_file.items() if f not in PHASE05), Counter())
    lines.append(f"| Phase 0 reader suite (regression) | {sum(rest.values())} | {rest['passed']} | {rest['failed']} "
                 f"| {rest['skipped']} |")  # fmt: skip
    lines += ["", "## Benchmark (`weightlab diff`, fresh process per case)", ""]
    if bench:
        lines += [
            f"- Host: {bench['platform']}, {bench['cpu_count']} CPUs",
            "",
            "| Case | Pair bytes read | Diff time | Throughput / side | Peak RSS | RSS growth after import | Class | Complete | Metrics exact |",
            "|---|---|---|---|---|---|---|---|---|",
        ]  # fmt: skip
        for c in bench["cases"]:
            tp = c.get("throughput_mib_s_per_side")
            lines.append(
                f"| {c['case']} | {c['bytes_read'] / 2**30:.2f} GiB | {c['diff_seconds']:.2f} s | "
                f"{f'{tp:.0f} MiB/s' if tp else 'n/a'} | {mib(c['rss_peak'])} | {mib(c['rss_growth_bytes'])} | "
                f"{c['compatibility_class']} | {c['report_complete']} | {c.get('metrics_correct', 'n/a')} |"
            )
        s = bench["summary"]
        if s["synthetic_rss_growth_range_bytes"]:
            lo, hi = s["synthetic_rss_growth_range_bytes"]
            lines += ["", f"- Synthetic pairs span {s['synthetic_size_ratio']:.0f}x in size; RSS growth stays within "
                          f"{mib(lo)}-{mib(hi)} (bounded by chunk buffers, not tensor size)."]  # fmt: skip
        lines.append("- Sparse synthetic files read zero pages from the page cache: throughput measures compute, "
                     "not disk.")  # fmt: skip
        lines.append("- `b05_*` (doc 11 B05): dense 1 GiB BF16 pair of seeded random values. `b06_*` (B06): 8192x8192 "
                     "BF16 base with rank-8/32 LoRA. Expected delta_l2 for both is recomputed with plain NumPy "
                     "(row blocks, float64) by the benchmark harness.")  # fmt: skip
    else:
        lines += ["- Benchmark failed:", "", "```", bench_proc.stderr[-3000:], "```"]
    lines += ["", "## Skipped on this OS", "", *([f"- {s}" for s in skipped] or ["- none"]), ""]
    report = out / f"G0.5_{os_tag()}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(test.stdout[-2000:])
    print(f"wrote {report}: {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
