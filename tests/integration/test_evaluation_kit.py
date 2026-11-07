"""Practitioner evaluation kit (Phase 0.5 task 6): the examples reproduce, the checks can fail, the documents agree.

The kit's checks compare reports with values from examples/fixtures/expected_results.json (the fixture
generator), so a passing run here is the same evidence a participant sees.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.helpers import ROOT

KIT = ROOT / "evaluation-kit"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load(KIT / "run_examples.py", "wl_eval_run_examples")
builder = _load(ROOT / "scripts" / "build_evaluation_kit.py", "wl_build_evaluation_kit")


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    out = tmp_path_factory.mktemp("kit") / "reports"
    r = subprocess.run(
        [sys.executable, str(KIT / "run_examples.py"), "--out", str(out), "--weightlab", ""],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL CHECKS PASSED" in r.stdout
    assert "[FAIL]" not in r.stdout
    return {d.name: json.loads((d / "report.json").read_text(encoding="utf-8")) for d in out.iterdir()}


def _expected() -> dict[str, Any]:
    exp: dict[str, Any] = json.loads((ROOT / "examples/fixtures/expected_results.json").read_text(encoding="utf-8"))
    return exp


def test_three_examples_reproduce(reports: dict[str, dict[str, Any]]) -> None:
    assert set(reports) == {"01_checkpoint_pair", "02_lora_adapter", "03_incomplete_report"}
    assert reports["03_incomplete_report"]["complete"] is False


def _failed(checks: list[tuple[str, bool, str]]) -> list[str]:
    return [label for label, ok, _ in checks if not ok]


def test_checks_detect_wrong_reports(reports: dict[str, dict[str, Any]]) -> None:
    exp = _expected()
    pair = reports["01_checkpoint_pair"]
    assert _failed(runner.check_pair(pair, exp)) == []
    bad = copy.deepcopy(pair)
    bad["tensors"][0]["counts"]["changed"] = 9
    bad["tensors"][0]["metrics"]["delta_l2"]["value"] *= 1.001
    bad["tensors"][1]["counts"]["changed"] = 1
    bad["compatibility"]["lineage"]["compatibility_class"] = "S"
    assert len(_failed(runner.check_pair(bad, exp))) == 4

    lora = reports["02_lora_adapter"]
    assert _failed(runner.check_lora(lora, exp)) == []
    bad = copy.deepcopy(lora)
    for row in bad["tensors"]:  # e.g. scale alpha/r ignored: every effective delta halves
        row["metrics"]["delta_l2"]["value"] /= 2
    assert len(_failed(runner.check_lora(bad, exp))) == 2

    inc = reports["03_incomplete_report"]
    assert _failed(runner.check_incomplete(inc, exp)) == []
    bad = copy.deepcopy(inc)
    bad["complete"] = True
    bad["compatibility"]["not_compared"] = []
    assert len(_failed(runner.check_incomplete(bad, exp))) == 2


def test_runner_refuses_an_existing_output_directory(tmp_path: Path) -> None:
    r = subprocess.run([sys.executable, str(KIT / "run_examples.py"), "--out", str(tmp_path)],
                       capture_output=True, text=True, check=False)  # fmt: skip
    assert r.returncode == 2
    assert "already exists" in r.stderr


def test_documents_reference_existing_questions_files_and_fixtures() -> None:
    questionnaire = (KIT / "QUESTIONNAIRE.md").read_text(encoding="utf-8")
    defined = {int(n) for n in re.findall(r"^\*\*Q(\d+)\.", questionnaire, flags=re.M)}
    assert defined == set(range(1, 16))
    for doc in sorted(KIT.rglob("*.md")):
        text = doc.read_text(encoding="utf-8")
        cited = {int(n) for n in re.findall(r"\bQ(\d+)\b", text)}
        assert cited <= defined, (doc.name, cited - defined)
        for link in re.findall(r"\]\(([^)#:]+)\)", text):
            assert (doc.parent / link).exists(), (doc.name, link)
    for item in builder.FIXTURE_ITEMS:
        assert (ROOT / "examples" / "fixtures" / item).exists(), item
    for rel in runner.FIXTURE_FILES:
        assert rel in _expected()["file_sha256"], rel
