"""Reader hardening (doc 12 section 3): only regular files are read; FIFOs, devices and directories are
refused without blocking (open uses O_NONBLOCK, then the handle must be S_ISREG)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import ROOT

PROBE = """
import sys
from pathlib import Path
from weightlab.core.errors import WeightLabError
from weightlab.formats.fileio import PositionedFile
try:
    PositionedFile(Path(sys.argv[1]))
    print("OPENED")
except WeightLabError as e:
    print(e.code.value)
"""


def _probe(path: Path) -> str:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-c", PROBE, str(path)], capture_output=True, text=True, timeout=20, env=env)
    return r.stdout.strip()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX FIFOs and device nodes")
def test_fifo_refused_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "model.safetensors"
    os.mkfifo(fifo)
    assert _probe(fifo) == "INVALID_FORMAT"  # a blocking open would hit the 20 s timeout instead


@pytest.mark.skipif(not Path("/dev/zero").exists(), reason="no /dev/zero")
def test_character_device_refused(tmp_path: Path) -> None:
    assert _probe(Path("/dev/zero")) == "INVALID_FORMAT"


def test_directory_refused(tmp_path: Path) -> None:
    assert _probe(tmp_path) == "INVALID_FORMAT"
