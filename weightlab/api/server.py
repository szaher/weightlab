"""`weightlab start`: serve the local API on loopback (doc 18 section 3, doc 12 section 4).

The socket is bound before the app is built, so `--port 0` gets a free port and the Host allowlist
names the real one. The URL with the `#token=` fragment goes to stdout (the terminal) only; uvicorn's
access log is off and its own logging carries no headers, so the token never reaches a log.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import TextIO

from weightlab.api.config import ServerConfig, check_host, default_data_dir


def bind(host: str, port: int) -> socket.socket:
    check_host(host)
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    if sys.platform != "win32":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(64)
    sock.set_inheritable(True)
    return sock


def serve(host: str = "127.0.0.1", port: int = 0, data_dir: Path | None = None, out: TextIO = sys.stdout) -> None:
    import uvicorn

    from weightlab.api.app import create_app

    sock = bind(host, port)
    config = ServerConfig(port=sock.getsockname()[1], data_dir=data_dir or default_data_dir(), host=host)
    app = create_app(config)
    out.write(
        f"WeightLab is running locally (offline, no telemetry). Open:\n\n  {config.url()}\n\n"
        "The #token fragment is a per-launch secret: it never leaves this machine and is not logged.\n"
        f"Data directory: {config.data_dir}\nPress Ctrl+C to stop.\n"
    )
    out.flush()
    uv = uvicorn.Config(app, log_level="warning", access_log=False, proxy_headers=False, server_header=False,
                        date_header=False, lifespan="on", ws="none")  # fmt: skip
    uvicorn.Server(uv).run(sockets=[sock])
