#!/usr/bin/env python3
"""Source manifest over git-tracked (or added, non-ignored) files.

    python scripts/manifest.py --write   # regenerate MANIFEST.sha256
    python scripts/manifest.py --check   # exit 1 on any missing, extra or changed file

Uses `git ls-files --cached --others --exclude-standard`, so ignored caches and
build products (see .gitignore) are never listed. MANIFEST.sha256 itself is excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.sha256"


def source_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout.decode("utf-8")  # fmt: skip
    files = {f for f in out.split("\0") if f and f != "MANIFEST.sha256" and (ROOT / f).is_file()}
    return sorted(files)


def render() -> str:
    lines = [f"{hashlib.sha256((ROOT / f).read_bytes()).hexdigest()}  {f}" for f in source_files()]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.write:
        MANIFEST.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {MANIFEST.name}: {text.count(chr(10))} files")
        return 0
    current = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
    if current == text:
        print(f"manifest OK: {text.count(chr(10))} files")
        return 0
    old = dict(line.split("  ", 1)[::-1] for line in current.splitlines() if line)
    new = dict(line.split("  ", 1)[::-1] for line in text.splitlines() if line)
    for f in sorted(set(old) | set(new)):
        if old.get(f) != new.get(f):
            print("MISSING" if f not in new else "EXTRA" if f not in old else "CHANGED", f)
    return 1


if __name__ == "__main__":
    sys.exit(main())
