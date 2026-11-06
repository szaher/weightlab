#!/usr/bin/env python3
"""Cross-file consistency sweep for the specification (stdlib only).

1. Every referenced requirement/test/decision ID (FR, AC, SEC, B, ADR, WL, G, T) is defined.
2. Every repository path mentioned in backticks under scripts/, contracts/, docs/, examples/,
   agent/ or .github/ exists.
3. Retired terms from earlier revisions do not reappear outside the changelog/delivery notes.
4. The combined blueprint volume is reproducible from docs/ (scripts/build_documentation.py --check).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY = {"CHANGELOG.md", "DELIVERY_NOTES.md"}

DEFINITIONS = {
    "FR": ("docs/02_FUNCTIONAL_REQUIREMENTS.md", r"\*\*(FR-\d{3}) \["),
    "AC": ("docs/02_FUNCTIONAL_REQUIREMENTS.md", r"\*\*(AC-\d{3}):\*\*"),
    "SEC": ("docs/12_SECURITY_AND_PRIVACY.md", r"^\| (SEC-\d{2}) \|"),
    "B": ("docs/11_PERFORMANCE_BUDGET.md", r"\*\*(B\d{2}) "),
    "ADR": ("docs/16_DECISIONS_RISKS_AND_OPEN_QUESTIONS.md", r"^### (ADR-\d{3}):"),
    "WL": ("docs/15_BACKLOG_AND_TRACEABILITY.md", r"^\| (WL-\d{3}) \|"),
    "G": ("docs/13_TEST_STRATEGY.md", r"\*\*(G\d) "),
    "T": ("docs/06_TRANSFER_RESEARCH_PROTOCOL.md", r"\*\*(T\d):\*\*"),
}
REFERENCE = re.compile(r"(?<![\w-])(FR-\d{3}|AC-\d{3}|SEC-\d{2}|B\d{2}|ADR-\d{3}|WL-\d{3}|G\d|T\d)(?![\w-]|\.\d)")
PATH = re.compile(r"`((?:scripts|contracts|docs|examples|agent|\.github)/[A-Za-z0-9_./-]+)`")
RETIRED = {
    "tiny_math": "renamed to tiny_ft in 1.1",
    "samples_per_block": "per-cell sampling removed in 1.1",
    "vnd.weightlab.tile.v1": "tile format is v2 (status bitfield) in 1.2",
    "user_attested)": "attestation is never class V (1.1)",
    "Citation not verified": "reference 12 verified in 1.2",
    "make_demo_fixtures": "script is generate_demo_fixtures.py",
    "output_dtype": "ROI uses value_encoding (1.2)",
}


def sources() -> list[Path]:
    files = [*ROOT.glob("docs/*.md"), *ROOT.glob("agent/*.md"), *ROOT.glob("examples/*.md"),
             ROOT / "README.md", ROOT / "DELIVERY_NOTES.md", ROOT / "CHANGELOG.md", *ROOT.glob("contracts/*.yaml")]  # fmt: skip
    return [f for f in files if f.exists()]


def main() -> int:
    problems: list[str] = []
    defined: set[str] = set()
    for kind, (rel, pattern) in DEFINITIONS.items():
        found = re.findall(pattern, (ROOT / rel).read_text(encoding="utf-8"), flags=re.M)
        if not found:
            problems.append(f"no {kind} definitions found in {rel}")
        defined.update(found)

    refs = 0
    for f in sources():
        rel = f.relative_to(ROOT).as_posix()
        text = f.read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), 1):
            for ref in REFERENCE.findall(line):
                refs += 1
                if ref not in defined:
                    problems.append(f"{rel}:{n}: undefined ID {ref}")
            for path in PATH.findall(line):
                clean = path.rstrip(".,;:)")
                if "*" in clean or "<" in clean:
                    continue
                if not (ROOT / clean).exists():
                    problems.append(f"{rel}:{n}: missing path {clean}")
            if rel not in HISTORY:
                for term, why in RETIRED.items():
                    if term in line:
                        problems.append(f"{rel}:{n}: retired term {term!r} ({why})")

    build = subprocess.run([sys.executable, str(ROOT / "scripts/build_documentation.py"), "--check"],
                           capture_output=True, text=True)  # fmt: skip
    if build.returncode:
        problems.append(build.stdout.strip())

    for p in problems:
        print("FAIL", p)
    print(f"{'FAIL' if problems else 'PASS'}: {len(defined)} IDs defined, {refs} references checked, "
          f"{len(sources())} files scanned")  # fmt: skip
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
