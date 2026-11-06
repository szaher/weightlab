#!/usr/bin/env python3
"""Build the combined Markdown volume from docs/*.md (the only source of truth).

    python scripts/build_documentation.py          # writes WeightLab_Complete_Blueprint.md
    python scripts/build_documentation.py --check  # exit 1 if the committed volume is not reproducible
    python scripts/build_documentation.py --pdf    # also renders the PDF (needs pandoc + weasyprint)

Never edit WeightLab_Complete_Blueprint.md or the PDF by hand; they are generated.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "WeightLab_Complete_Blueprint.md"
FRONT = """---
title: "WeightLab"
subtitle: "Complete Product & Engineering Blueprint"
author: "Product specification and technical design"
date: "2026-11-05"
---

# WeightLab: Complete Product & Engineering Blueprint

**Design baseline:** 5 November 2026, frozen as 1.2-final (ADR-019)

**Delivery state:** Product documentation and synthetic proof fixtures; **the WeightLab application has not been implemented**.

This volume is generated from `docs/*.md` by `scripts/build_documentation.py`; edit the source documents, not this file. \
It includes the product requirements, functional requirements, architecture, format reader design, mathematical semantics, \
scientific experiment plan, UI specification, API and persistence model, performance constraints, threat model, QA strategy, \
delivery plan, research references and packaging instructions. Machine-readable contracts, the fixture generator/checker and \
agent execution instructions live alongside it in the repository.

**Key constraint:** Default weight inspection runs locally on a normal laptop, without GPU or whole-model RAM loading. \
Running an LLM for evaluations/activation investigations is **optional**, separate and may require more RAM.

---

"""


def render() -> str:
    files = sorted((ROOT / "docs").glob("*.md"))
    return "".join([FRONT] + ["\n\n\\newpage\n\n" + p.read_text(encoding="utf-8") + "\n" for p in files])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify the committed volume equals a fresh build")
    parser.add_argument("--pdf", action="store_true", help="also render WeightLab_Complete_Blueprint.pdf")
    args = parser.parse_args()

    text = render()
    if args.check:
        same = OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") == text
        print("combined volume reproducible" if same else "combined volume is STALE: run scripts/build_documentation.py")
        raise SystemExit(0 if same else 1)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print("Combined docs ->", OUTPUT.relative_to(ROOT), OUTPUT.stat().st_size, "bytes")

    if args.pdf:
        if not (shutil.which("pandoc") and shutil.which("weasyprint")):
            raise SystemExit("--pdf needs pandoc and weasyprint on PATH")
        pdf = ROOT / "WeightLab_Complete_Blueprint.pdf"
        cmd = ["pandoc", str(OUTPUT), "--from", "markdown", "--pdf-engine=weasyprint", "--toc", "-o", str(pdf)]
        subprocess.run(cmd, check=True)
        print("Rendered", pdf.relative_to(ROOT))


if __name__ == "__main__":
    main()
