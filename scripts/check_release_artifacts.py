#!/usr/bin/env python3
"""Check PyPI release artifacts without importing the package or downloading dependencies.

Usage:
    python scripts/check_release_artifacts.py --dist dist [--tag v0.1.0a1]
"""

from __future__ import annotations

import argparse
import email
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check(dist: Path, tag: str | None = None) -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        project = tomllib.load(f)["project"]
    version = project["version"]
    expected_tag = f"v{version}"
    if tag is not None and tag != expected_tag:
        raise ValueError(f"tag {tag!r} does not match expected {expected_tag!r}")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:a|b|rc)[0-9]+", version):
        raise ValueError(f"this workflow releases prereleases only, got {version!r}")

    wheels = sorted(dist.glob("weightlab-*.whl"))
    sdists = sorted(dist.glob("weightlab-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(f"expected exactly one wheel and one sdist: {wheels!r}, {sdists!r}")
    wheel, sdist = wheels[0], sdists[0]
    if wheel.name != f"weightlab-{version}-py3-none-any.whl":
        raise ValueError(f"unexpected wheel name: {wheel.name}")
    if sdist.name != f"weightlab-{version}.tar.gz":
        raise ValueError(f"unexpected sdist name: {sdist.name}")

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError("wheel must contain one dist-info METADATA")
        metadata = email.message_from_bytes(archive.read(metadata_files[0]))
        if metadata.get("Name", "").lower().replace("_", "-") != "weightlab":
            raise ValueError("wheel package name is not weightlab")
        if metadata.get("Version") != version:
            raise ValueError("wheel METADATA version differs from pyproject")
        required = {
            "weightlab/__init__.py",
            "weightlab/cli.py",
            "weightlab/ui/static/index.html",
            "weightlab/db/migrations/0001_initial.sql",
            "weightlab/db/migrations/0002_model_files_ctime.sql",
            "weightlab/db/migrations/0003_comparison_reports.sql",
        }
        missing = required - names
        if missing:
            raise ValueError(f"missing wheel package files: {sorted(missing)}")
        if not any(re.fullmatch(r"weightlab/ui/static/assets/.+\.js", p) for p in names):
            raise ValueError("wheel is missing built UI JavaScript")
        if not any(re.fullmatch(r"weightlab/ui/static/assets/.+\.css", p) for p in names):
            raise ValueError("wheel is missing built UI stylesheet")

    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
        prefix = f"weightlab-{version}/"
        if not {prefix + "pyproject.toml", prefix + "README.md", prefix + "weightlab/__init__.py"} <= names:
            raise ValueError("source distribution is missing metadata or package source")

    print(f"PASS: {version} wheel and sdist checked; built UI and database migrations are packaged")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--tag")
    args = parser.parse_args()
    try:
        check(args.dist, args.tag)
    except (ValueError, OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
