"""`weightlab diff` CLI: exports, exit codes, output safety, untrusted-key escaping, privacy,
read-only inputs and cancellation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tests.helpers import ROOT, write_safetensors
from weightlab.analysis.change_report import run
from weightlab.analysis.diff import Cancelled, RunControl
from weightlab.cli import main


def snapshot(root: Path) -> dict[str, tuple[str, int, int]]:
    return {
        str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_out_dir_writes_all_formats(fixtures_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = snapshot(fixtures_dir)
    out = tmp_path / "rep"
    code = main(
        [
            "diff",
            str(fixtures_dir / "tiny_base.safetensors"),
            str(fixtures_dir / "tiny_ft.safetensors"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "layers.csv",
        "report.json",
        "report.md",
        "roles.csv",
        "tensors.csv",
    ]
    report = json.loads((out / "report.json").read_text())
    table = list(csv.DictReader(io.StringIO((out / "tensors.csv").read_text())))
    assert len(table) == len(report["tensors"]) == 9
    assert table[0]["key"] == "demo.layers.0.mlp.up_proj.weight"
    assert float(table[0]["delta_l2"]) == report["tensors"][0]["metrics"]["delta_l2"]["value"]  # repr round trip
    assert "# WeightLab change report" in capsys.readouterr().out
    assert snapshot(fixtures_dir) == before  # inputs untouched (content, mtime, size)


def test_exit_codes(fixtures_dir: Path, tmp_path: Path) -> None:
    f = fixtures_dir
    assert (
        main(["diff", str(f / "tiny_base.safetensors"), str(f / "compat/shape_mismatch.safetensors"), "--quiet"]) == 3
    )
    assert (
        main(["diff", str(f / "tiny_base.safetensors"), "--adapter", str(f / "tiny_lora_dora_unsupported"), "--quiet"])
        == 3
    )
    assert main(["diff", str(f / "tiny_base.safetensors"), "--quiet"]) == 2  # neither DERIVED nor --adapter
    assert main(["diff", str(f / "malformed/corrupt_json.safetensors"), str(f / "tiny_ft.safetensors"), "--quiet"]) == 2
    assert main(["diff", str(tmp_path / "nope"), str(f / "tiny_ft.safetensors"), "--quiet"]) == 2


def test_refuses_to_write_into_input_directories(
    fixtures_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes((fixtures_dir / "tiny_base.safetensors").read_bytes())
    before = snapshot(model)
    for args in (
        ["--out", str(model)],
        ["--json", str(model / "model.safetensors")],
        ["--md", str(model / "sub" / "r.md")],
    ):
        assert (
            main(
                ["diff", str(model / "model.safetensors"), str(fixtures_dir / "tiny_ft.safetensors"), "--quiet", *args]
            )
            == 2
        )
    assert "refusing to write" in capsys.readouterr().err
    assert snapshot(model) == before


def test_untrusted_keys_are_escaped(tmp_path: Path) -> None:
    keys = ['=HYPERLINK("http://x")', "a|b`c\n<script>", "+1", "@sum"]
    tensors = {k: ("F32", (2,), np.array([1, 2], "<f4").tobytes()) for k in keys}
    write_safetensors(tmp_path / "a" / "model.safetensors", tensors)
    write_safetensors(
        tmp_path / "b" / "model.safetensors", {k: ("F32", (2,), np.array([1, 3], "<f4").tobytes()) for k in keys}
    )
    out = tmp_path / "out"
    assert main(["diff", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(out), "--quiet"]) == 0
    for row in csv.reader(io.StringIO((out / "tensors.csv").read_text())):
        assert row[1][:1] not in ("=", "+", "@", "-"), row[1]
    md = (out / "report.md").read_text()
    assert "<script>" not in md
    assert "a\\|b'c &lt;script&gt;" in md
    assert json.loads((out / "report.json").read_text())["tensors"]  # JSON keeps raw keys losslessly


def test_paths_hidden_unless_requested(fixtures_dir: Path, tmp_path: Path) -> None:
    args = ["diff", str(fixtures_dir / "tiny_base.safetensors"), str(fixtures_dir / "tiny_ft.safetensors"), "--quiet"]
    main([*args, "--out", str(tmp_path / "a")])
    main([*args, "--out", str(tmp_path / "b"), "--include-paths"])
    secret = str(fixtures_dir)
    escaped = json.dumps(secret)[1:-1]  # as it appears inside JSON (Windows backslashes are escaped)
    assert all(secret not in p.read_text() and escaped not in p.read_text() for p in (tmp_path / "a").iterdir())
    assert escaped in (tmp_path / "b" / "report.json").read_text()


def test_verified_consistency_hashes_inputs(fixtures_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    assert (
        main(
            [
                "diff",
                str(fixtures_dir / "tiny_base.safetensors"),
                str(fixtures_dir / "tiny_ft.safetensors"),
                "--consistency",
                "verified",
                "--json",
                str(out),
                "--quiet",
            ]
        )
        == 0
    )
    r = json.loads(out.read_text())
    assert r["settings"]["consistency"] == "verified"
    assert all(f["sha256"] for side in r["provenance"]["inputs"].values() for f in side["files"])


def test_library_cancel_raises_and_reports_nothing(fixtures_dir: Path) -> None:
    ctl = RunControl()
    ctl.progress = lambda done, total, key: ctl.cancel.set()
    with pytest.raises(Cancelled):
        run(fixtures_dir / "tiny_base.safetensors", fixtures_dir / "tiny_ft.safetensors", control=ctl)


def _sparse_pair(tmp: Path, gib: float) -> tuple[Path, Path]:
    cols = 8192
    rows = int(gib * 2**30) // (cols * 2)
    paths = []
    for name in ("a", "b"):
        h = json.dumps({"w": {"dtype": "BF16", "shape": [rows, cols], "data_offsets": [0, rows * cols * 2]}}).encode()
        h += b" " * (-len(h) % 8)
        p = tmp / name / "model.safetensors"
        p.parent.mkdir(parents=True)
        with p.open("wb") as f:
            f.write(len(h).to_bytes(8, "little") + h)
            f.truncate(8 + len(h) + rows * cols * 2)
        paths.append(p)
    return paths[0], paths[1]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT delivery to a child differs on Windows (deferred)")
def test_ctrl_c_writes_nothing_and_exits_130(tmp_path: Path) -> None:
    a, b = _sparse_pair(tmp_path, 2.0)
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "weightlab.cli",
            "diff",
            str(a),
            str(b),
            "--out",
            str(out),
            "--quiet",
            "--chunk-elements",
            "65536",
        ],
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(1.0)
    proc.send_signal(signal.SIGINT)
    _, err = proc.communicate(timeout=120)
    assert proc.returncode == 130, err
    assert "cancelled: no report written" in err
    assert not out.exists()
