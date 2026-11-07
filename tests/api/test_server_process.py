"""`weightlab start` as a real process on a real socket (doc 12 section 8: token, host, logs, loopback)."""

from __future__ import annotations

import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import ROOT

# Windows: CTRL_C_EVENT reaches every process on the console, pytest and the evidence scripts included. The server
# gets its own process group instead, and CTRL_BREAK_EVENT (SIGBREAK, which uvicorn handles) is sent to that group.
NEW_GROUP: dict[str, Any]
if sys.platform == "win32":
    NEW_GROUP = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    STOP = signal.CTRL_BREAK_EVENT
else:
    NEW_GROUP = {}
    STOP = signal.SIGINT
URL = re.compile(r"http://127\.0\.0\.1:(\d+)/#token=([A-Za-z0-9_-]{43,})")


def _get(port: int, path: str, headers: dict[str, str]) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path, headers=headers)
        r = conn.getresponse()
        return r.status, r.read()
    finally:
        conn.close()


def test_start_serves_loopback_with_a_fresh_token(tmp_path: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    tokens = []
    for _ in range(2):  # restart rotates the token
        proc = subprocess.Popen([sys.executable, "-m", "weightlab.cli", "start", "--port", "0", "--data-dir",
                                 str(tmp_path / "data")], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=env, **NEW_GROUP)  # fmt: skip
        try:
            line, deadline = "", time.monotonic() + 30
            assert proc.stdout is not None
            while not URL.search(line) and time.monotonic() < deadline:
                line += proc.stdout.readline()
            m = URL.search(line)
            assert m, line
            port, token = int(m.group(1)), m.group(2)
            tokens.append(token)
            for _ in range(50):  # wait until uvicorn accepts
                try:
                    status, body = _get(port, "/api/v1/health", {"Host": f"127.0.0.1:{port}"})
                    break
                except ConnectionRefusedError:
                    time.sleep(0.1)
            assert status == 200
            assert json.loads(body)["offline"] is True
            host = {"Host": f"127.0.0.1:{port}"}
            assert _get(port, "/api/v1/projects", host)[0] == 401
            assert _get(port, "/api/v1/projects", {**host, "Authorization": f"Bearer {token}"})[0] == 200
            assert (
                _get(port, "/api/v1/projects", {"Host": "evil.example", "Authorization": f"Bearer {token}"})[0] == 403
            )
            assert _get(port, f"/api/v1/projects?token={token}", host)[0] == 401  # query-string tokens are not a thing
        finally:
            proc.send_signal(STOP)
            try:
                out, err = proc.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
        assert token not in err  # logs (stderr) never carry the token
        assert (line + out).count(token) == 1  # printed once, in the terminal URL
    assert tokens[0] != tokens[1]


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "localhost"])
def test_start_refuses_non_loopback_hosts(host: str, tmp_path: Path) -> None:
    from weightlab.cli import main

    assert main(["start", "--host", host, "--data-dir", str(tmp_path)]) == 2
