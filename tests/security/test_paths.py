"""Shard path rules and root containment (doc 04 section 3, SEC-02). Symlink cases are generated per
run because symlinks are not portable in git checkouts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.helpers import write_safetensors
from weightlab import WeightLabError, open_checkpoint
from weightlab.formats.shards import check_relative


@pytest.mark.parametrize(
    "rel",
    ["../x.safetensors", "a/../../x", "/etc/passwd", "C:/x.safetensors", "c:x", "\\\\server\\share\\x",
     "//server/share/x", "sub\\x.safetensors", "", "a\x00b", 5, None, ["x"], "file.safetensors:stream"],
)  # fmt: skip
def test_rejected_relative_paths(rel: object) -> None:
    with pytest.raises(WeightLabError) as info:
        check_relative(rel)
    assert info.value.code == "INVALID_FORMAT"


@pytest.mark.parametrize(
    ("rel", "norm"), [("a.safetensors", "a.safetensors"), ("./sub/a", "sub/a"), ("sub//a", "sub/a")]
)
def test_accepted_relative_paths(rel: str, norm: str) -> None:
    assert check_relative(rel) == norm


def _model(dir_: Path, shard_rel: str = "s.safetensors") -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": shard_rel}}))
    return dir_


def _shard(path: Path) -> None:
    write_safetensors(path, {"w": ("F32", (2,), np.ones(2, "<f4").tobytes())})


def _symlink(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symlinks unavailable here: {e}")


def test_symlink_escape_is_path_outside_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.safetensors"
    _shard(outside)
    model = _model(tmp_path / "model")
    _symlink(outside, model / "s.safetensors")
    with pytest.raises(WeightLabError) as info:
        open_checkpoint(model)
    assert info.value.code == "PATH_OUTSIDE_ROOT"


def test_symlinked_directory_escape(tmp_path: Path) -> None:
    (tmp_path / "elsewhere").mkdir()
    _shard(tmp_path / "elsewhere" / "s.safetensors")
    model = _model(tmp_path / "model", "sub/s.safetensors")
    _symlink(tmp_path / "elsewhere", model / "sub")
    with pytest.raises(WeightLabError) as info:
        open_checkpoint(model)
    assert info.value.code == "PATH_OUTSIDE_ROOT"


def test_symlink_inside_root_is_allowed(tmp_path: Path) -> None:
    model = _model(tmp_path / "model")
    (model / "blobs").mkdir()
    _shard(model / "blobs" / "abc")
    _symlink(model / "blobs" / "abc", model / "s.safetensors")  # Hugging Face cache layout
    with open_checkpoint(model) as ck:
        assert ck.read_roi_2d("w", 0, 1, 0, 2).tolist() == [[1.0, 1.0]]


def test_wider_selected_root_permits_sibling_blobs(tmp_path: Path) -> None:
    _shard(tmp_path / "blobs" / "abc")
    model = _model(tmp_path / "snap")
    _symlink(tmp_path / "blobs" / "abc", model / "s.safetensors")
    with pytest.raises(WeightLabError):
        open_checkpoint(model)
    with open_checkpoint(model, selected_root=tmp_path) as ck:
        assert ck.tensor("w").shape == (2,)


def test_entry_outside_selected_root(tmp_path: Path) -> None:
    _shard(tmp_path / "a" / "model.safetensors")
    (tmp_path / "b").mkdir()
    with pytest.raises(WeightLabError) as info:
        open_checkpoint(tmp_path / "a" / "model.safetensors", selected_root=tmp_path / "b")
    assert info.value.code == "PATH_OUTSIDE_ROOT"


def test_index_rules(tmp_path: Path) -> None:
    model = tmp_path / "m"
    model.mkdir()
    _shard(model / "s.safetensors")
    idx = model / "model.safetensors.index.json"

    def code(doc: object) -> str | None:
        idx.write_text(doc if isinstance(doc, str) else json.dumps(doc))
        try:
            open_checkpoint(model).close()
            return None
        except WeightLabError as e:
            return e.code.value

    assert code({"weight_map": {"w": "s.safetensors"}}) is None
    assert code({"weight_map": {"w": "s.safetensors"}, "metadata": {"total_size": 8}}) is None
    assert code({"weight_map": {"w": "s.safetensors"}, "metadata": {"total_size": "8"}}) == "INVALID_FORMAT"
    assert code({"weight_map": {"w": "s.safetensors"}, "metadata": {"total_size": True}}) == "INVALID_FORMAT"
    assert code({"weight_map": []}) == "INVALID_FORMAT"
    assert code({}) == "INVALID_FORMAT"
    assert code('{"weight_map": {"w": "s.safetensors", "w": "s.safetensors"}}') == "INVALID_FORMAT"
    idx.write_text(json.dumps({"weight_map": {"w": "s.safetensors"}, "metadata": {"total_size": 99}}))
    with open_checkpoint(model) as ck:
        assert any("total_size" in w for w in ck.warnings)  # mismatch is a warning, not an error
