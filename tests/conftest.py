"""Session fixtures: regenerate the provided synthetic fixtures into tests/fixtures/generated
(never hand-edited) and confirm they are byte-identical to the committed reference set."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import ROOT

GENERATED = ROOT / "tests" / "fixtures" / "generated"
COMMITTED = ROOT / "examples" / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    if GENERATED.exists():
        shutil.rmtree(GENERATED)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_demo_fixtures.py"),
            "--output",
            str(GENERATED),
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
    )
    expected = json.loads((COMMITTED / "expected_results.json").read_text(encoding="utf-8"))
    differs = [
        rel
        for rel, digest in expected["file_sha256"].items()
        if hashlib.sha256((GENERATED / rel).read_bytes()).hexdigest() != digest
    ]
    assert not differs, f"regenerated fixtures differ from examples/fixtures: {differs}"
    return GENERATED


@pytest.fixture(scope="session")
def expected(fixtures_dir: Path) -> dict[str, Any]:
    return json.loads((fixtures_dir / "expected_results.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]
