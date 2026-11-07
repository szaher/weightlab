#!/usr/bin/env python3
"""Build the self-contained practitioner evaluation kit (Phase 0.5 task 6) as one zip archive.

    uv run python scripts/build_evaluation_kit.py [--out dist]

Contents: evaluation-kit/ (docs, questionnaire, templates, run_examples.py), the weightlab wheel, the fixtures
the examples use (copied unmodified from examples/fixtures), docs/cli/weightlab-diff.md and SHA256SUMS. The
archive is deterministic (sorted entries, fixed timestamps). After building, the archive is unpacked into a
temporary directory and run_examples.py is run there against the packed fixtures; the build fails if any check
fails. Exit status: 0 on success.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "evaluation-kit"
FIXTURES = ROOT / "examples" / "fixtures"
# Fixtures copied into the archive: the three examples, plus the optional cases named in examples/03.
FIXTURE_ITEMS = [
    "expected_results.json",
    "tiny_base.safetensors",
    "tiny_ft.safetensors",
    "tiny_lora",
    "tiny_lora_dora_unsupported",
    "tiny_precision_a.safetensors",
    "tiny_precision_b.safetensors",
    "compat",
    "malformed",
]
EPOCH = (1980, 1, 1, 0, 0, 0)


def stage(dest: Path, wheel_dir: Path) -> None:
    for src in sorted(KIT.rglob("*")):
        if src.is_file() and "__pycache__" not in src.parts and not src.name.startswith("weightlab-eval-"):
            target = dest / src.relative_to(KIT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    shutil.copy2(ROOT / "docs" / "cli" / "weightlab-diff.md", dest / "weightlab-diff.md")
    for item in FIXTURE_ITEMS:
        src = FIXTURES / item
        if src.is_dir():
            shutil.copytree(src, dest / "fixtures" / item)
        else:
            (dest / "fixtures").mkdir(exist_ok=True)
            shutil.copy2(src, dest / "fixtures" / item)
    (dest / "wheel").mkdir()
    for whl in wheel_dir.glob("*.whl"):
        shutil.copy2(whl, dest / "wheel" / whl.name)
    sums = []
    for f in sorted(p for p in dest.rglob("*") if p.is_file()):
        sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(dest).as_posix()}")
    (dest / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


def write_zip(src: Path, archive: Path, top: str) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(f"{top}/{f.relative_to(src).as_posix()}", date_time=EPOCH)
            info.external_attr = (0o755 if f.suffix == ".py" else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, f.read_bytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    args = ap.parse_args()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    top = f"weightlab-evaluation-kit-{version}"
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"{top}.zip"
    uv = shutil.which("uv")
    if uv is None:
        print("uv is required to build the wheel", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        subprocess.run([uv, "build", "--wheel", "--out-dir", str(t / "wheel")], cwd=ROOT, check=True)
        stage(t / top, t / "wheel")
        write_zip(t / top, archive, top)
        check_dir = t / "check"
        with zipfile.ZipFile(archive) as z:
            z.extractall(check_dir)
        kit = check_dir / top
        # The packed fixtures and runner are checked with the weightlab of this environment.
        r = subprocess.run(
            [sys.executable, str(kit / "run_examples.py"), "--fixtures", str(kit / "fixtures"),
             "--out", str(t / "reports"), "--weightlab", ""],
            cwd=t, capture_output=True, text=True, check=False,
        )  # fmt: skip
        print(r.stdout[-600:])
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            archive.unlink()
            return 1
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f"wrote {archive} ({archive.stat().st_size} bytes, sha256 {digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
