"""Application conformance on the provided fixtures (G0): WeightLab's own reader, not the self-check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.helpers import ROOT, upstream_accepts
from weightlab import WeightLabError, open_checkpoint

COMMITTED = ROOT / "examples" / "fixtures"
MALFORMED = sorted(__import__("json").loads((COMMITTED / "expected_results.json").read_text())["malformed_expected"])


def accepts(path: Path) -> bool:
    try:
        open_checkpoint(path).close()
        return True
    except WeightLabError:
        return False


@pytest.mark.parametrize("name", MALFORMED)
def test_malformed_fixture_rejected_with_expected_code(fixtures_dir: Path, expected: dict[str, Any], name: str) -> None:
    code = expected["malformed_expected"][name].split(" ")[0]
    with pytest.raises(WeightLabError) as info:
        open_checkpoint(fixtures_dir / "malformed" / name.rstrip("/")).close()
    assert info.value.code == code, info.value.detail


def _single_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.safetensors") if not (p.parent / "model.safetensors.index.json").exists())


def test_every_valid_fixture_opens(fixtures_dir: Path) -> None:
    valid = [p for p in _single_files(fixtures_dir) if "malformed" not in p.parts]
    assert len(valid) >= 12
    for p in valid:
        with open_checkpoint(p) as ck:
            assert ck.metadata(), p
    with open_checkpoint(fixtures_dir / "tiny_sharded") as ck:
        assert {m.key for m in ck.metadata()} == {"demo.layers.0.mlp.up_proj.weight", "demo.vector.weight"}
        assert ck.warnings == []


def test_accept_reject_agrees_with_upstream(fixtures_dir: Path, expected: dict[str, Any]) -> None:
    permitted = expected["upstream_permitted_differences"]
    files = _single_files(fixtures_dir)
    disagreements = []
    for p in files:
        rel = p.relative_to(fixtures_dir / "malformed").as_posix() if "malformed" in p.parts else None
        ours, theirs = accepts(p), upstream_accepts(p)
        if ours != theirs and rel not in permitted:
            disagreements.append((p.relative_to(fixtures_dir).as_posix(), ours, theirs))
    assert not disagreements
    # The recorded differences are real and point the documented way (WeightLab stricter).
    for rel in ("rank_too_high.safetensors", "unknown_entry_field.safetensors"):
        assert not accepts(fixtures_dir / "malformed" / rel)
        assert upstream_accepts(fixtures_dir / "malformed" / rel)
