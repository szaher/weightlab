#!/usr/bin/env python3
"""Run the three evaluation examples and check each report against values the fixture generator wrote.

    python run_examples.py                      # inside the unpacked kit (weightlab on PATH)
    uv run python evaluation-kit/run_examples.py   # inside a clone of the repository

Standard library only. Writes one report directory per example under --out (a new, timestamped directory by
default) and prints PASS/FAIL per check. The expected values come from fixtures/expected_results.json, which
scripts/generate_demo_fixtures.py writes independently of WeightLab. Exit status: 0 if every check passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# Fixed report timestamp so every participant's report.json is comparable byte for byte (up to float rounding).
TIMESTAMP = "2026-11-06T00:00:00Z"
FIXTURE_FILES = [
    "tiny_base.safetensors",
    "tiny_ft.safetensors",
    "tiny_lora/adapter_model.safetensors",
    "tiny_lora_dora_unsupported/adapter_model.safetensors",
]


def default_fixtures() -> Path:
    for cand in (HERE / "fixtures", HERE.parent / "examples" / "fixtures"):  # unpacked kit, then repository
        if (cand / "expected_results.json").is_file():
            return cand
    return HERE / "fixtures"


def frobenius(rows: list[list[float]]) -> float:
    return math.sqrt(math.fsum(v * v for row in rows for v in row))


def close(got: Any, want: float, rel: float = 1e-9) -> bool:
    return isinstance(got, int | float) and math.isclose(got, want, rel_tol=rel, abs_tol=1e-15)


def metric(row: dict[str, Any], name: str) -> Any:
    m = row.get("metrics", {}).get(name)
    return m.get("value") if isinstance(m, dict) else None


class Example:
    def __init__(self, slug: str, title: str, args: list[str], exit_code: int) -> None:
        self.slug, self.title, self.args, self.exit_code = slug, title, args, exit_code


def check_pair(report: dict[str, Any], exp: dict[str, Any]) -> list[tuple[str, bool, str]]:
    lineage = report["compatibility"]["lineage"]
    rows = report["tensors"]
    top = rows[0] if rows else {}
    want_l2 = frobenius(exp["delta_full"])
    others = [r for r in rows[1:] if r.get("status") == "matched"]
    return [
        ("report is complete", report.get("complete") is True, f"complete={report.get('complete')}"),
        ("compatibility class V (ancestry known by construction)", lineage.get("compatibility_class") == "V",
         f"class={lineage.get('compatibility_class')}"),
        (f"{exp['tensor_key']} ranked first", top.get("key") == exp["tensor_key"], f"rank 1 = {top.get('key')}"),
        (f"exactly {len(exp['changed_positions'])} elements changed",
         top.get("counts", {}).get("changed") == len(exp["changed_positions"]),
         f"changed={top.get('counts', {}).get('changed')}"),
        ("delta L2 equals the generator's ||delta_full||", close(metric(top, "delta_l2"), want_l2),
         f"report {metric(top, 'delta_l2')} vs generator {want_l2}"),
        ("delta RMS equals the generator's rms_of_delta",
         close(metric(top, "delta_rms"), exp["rms_delta_vs_difference_of_rms"]["rms_of_delta"]),
         f"report {metric(top, 'delta_rms')}"),
        ("every other tensor is unchanged", bool(others) and all(r["counts"]["changed"] == 0 for r in others),
         f"{len(others)} other matched tensors"),
    ]  # fmt: skip


def check_lora(report: dict[str, Any], exp: dict[str, Any]) -> list[tuple[str, bool, str]]:
    lineage = report["compatibility"]["lineage"]
    rows = report["tensors"]
    out = [
        ("report is complete", report.get("complete") is True, f"complete={report.get('complete')}"),
        ("compatibility class V", lineage.get("compatibility_class") == "V",
         f"class={lineage.get('compatibility_class')}"),
        ("both target modules analyzed", report["completeness"].get("modules_analyzed") == 2,
         f"modules_analyzed={report['completeness'].get('modules_analyzed')}"),
    ]  # fmt: skip
    for module, spec in exp["lora"]["tiny_lora"].items():
        row: dict[str, Any] = next((r for r in rows if str(r.get("key", "")).endswith(f".{module}.weight")), {})
        want = frobenius(spec["delta_full"])
        out.append(
            (
                f"{module}: ||s*B@A|| equals the generator's (rank {spec['rank']}, scale {spec['scale']})",
                close(metric(row, "delta_l2"), want, rel=1e-6),
                f"report {metric(row, 'delta_l2')} vs generator {want}",
            )
        )
    return out


def check_incomplete(report: dict[str, Any], exp: dict[str, Any]) -> list[tuple[str, bool, str]]:
    c = report["completeness"]
    not_compared = report["compatibility"].get("not_compared", [])
    dora = [e for e in not_compared if str(e.get("raw_key", "")).endswith("lora_magnitude_vector")]
    return [
        ("report says it is incomplete", report.get("complete") is False, f"complete={report.get('complete')}"),
        ("the DoRA module is unsupported, not silently dropped", c.get("modules_unsupported") == 1,
         f"modules_unsupported={c.get('modules_unsupported')}"),
        ("nothing was analyzed, so nothing is ranked as if it were", c.get("modules_analyzed") == 0,
         f"modules_analyzed={c.get('modules_analyzed')}"),
        ("the DoRA magnitude key is listed as not compared",
         bool(dora) and dora[0].get("status") == "unsupported_key", json.dumps(dora[:1])),
    ]  # fmt: skip


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", type=Path, default=default_fixtures(), help="directory with expected_results.json")
    ap.add_argument("--out", type=Path, help="new directory for the reports (default: weightlab-eval-<time>)")
    ap.add_argument("--weightlab", default=shutil.which("weightlab") or "", help="weightlab executable")
    args = ap.parse_args()

    fx: Path = args.fixtures.resolve()
    if not (fx / "expected_results.json").is_file():
        print(f"no expected_results.json in {fx}; pass --fixtures", file=sys.stderr)
        return 2
    exe = [args.weightlab] if args.weightlab else [sys.executable, "-m", "weightlab.cli"]
    out: Path = args.out or Path.cwd() / f"weightlab-eval-{dt.datetime.now():%Y%m%d-%H%M%S}"
    if out.exists():
        print(f"{out} already exists; pass a new --out", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    exp = json.loads((fx / "expected_results.json").read_text(encoding="utf-8"))

    version = subprocess.run([*exe, "--version"], capture_output=True, text=True, check=False)
    print(f"weightlab: {(version.stdout or version.stderr).strip()}")
    print(f"platform: {platform.platform()} / {platform.machine()} / Python {platform.python_version()}")
    print(f"fixtures: {fx}\nreports:  {out}\n")

    failures = 0
    for rel in FIXTURE_FILES:  # the files are the ones the expected values describe
        ok = sha256(fx / rel) == exp["file_sha256"][rel]
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] fixture {rel} has the generator's SHA-256")

    base = str(fx / "tiny_base.safetensors")
    examples: list[tuple[Example, Callable[[dict[str, Any], dict[str, Any]], list[tuple[str, bool, str]]]]] = [
        (Example("01_checkpoint_pair", "Checkpoint pair: tiny_base vs tiny_ft",
                 [base, str(fx / "tiny_ft.safetensors")], 0), check_pair),
        (Example("02_lora_adapter", "LoRA adapter: tiny_base + tiny_lora",
                 [base, "--adapter", str(fx / "tiny_lora")], 0), check_lora),
        (Example("03_incomplete_report", "Incomplete report: tiny_base + DoRA adapter (unsupported)",
                 [base, "--adapter", str(fx / "tiny_lora_dora_unsupported")], 3), check_incomplete),
    ]  # fmt: skip
    env = dict(os.environ, WEIGHTLAB_REPORT_TIMESTAMP=TIMESTAMP)
    for ex, check in examples:
        target = out / ex.slug
        cmd = [*exe, "diff", *ex.args, "--out", str(target), "--quiet"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        shown = " ".join(a.replace(str(fx), "fixtures") for a in ex.args)
        print(f"\n{ex.title}\n  $ weightlab diff {shown} --out {target.name}")
        ok = r.returncode == ex.exit_code
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] exit status {r.returncode} (expected {ex.exit_code})")
        if not (target / "report.json").is_file():
            failures += 1
            print(f"  [FAIL] no report written\n{r.stderr.strip()}")
            continue
        report = json.loads((target / "report.json").read_text(encoding="utf-8"))
        for label, passed, detail in check(report, exp):
            failures += not passed
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + ("" if passed else f": {detail}"))
        print(f"  read: {target / 'report.md'}")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    if failures:
        print("Please paste this whole output into question Q14 of the questionnaire.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
