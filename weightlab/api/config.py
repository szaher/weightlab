"""Server configuration: loopback-only binding, per-launch token, data directory, default settings."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOOPBACK_HOSTS = ("127.0.0.1", "::1")  # doc 12 section 4: nothing else, never 0.0.0.0 or ::
API_PREFIX = "/api/v1"
MAX_BODY_BYTES = 1024 * 1024  # doc 12 section 4: bounded JSON bodies, default 1 MiB
MAX_IN_FLIGHT = 16  # concurrent requests before 429 RATE_LIMITED

# doc 18 section 5, restricted to the fields of the frozen Settings schema.
DEFAULT_SETTINGS: dict[str, Any] = {
    "offline": True,
    "telemetry_enabled": False,
    "memory_budget_mib": 1024,
    "max_threads": 2,
    "max_active_tiles": 2,
    "max_heavy_jobs": 1,
    "roi_max_cells": 4096,
    "max_tile_cells": 65536,
    "direct_tile_max_mib": 64,
    "pyramid_exact_max_mib": 1024,
    "tile_cache_memory_mib": 128,
    "tile_cache_disk_mib": 1024,
    "max_scan_read_mib_without_confirmation": 512,
    "max_job_seconds": 600,
    "max_lora_scan_gmac": 50,
    "export_absolute_paths": False,
}
READ_ONLY_SETTINGS = ("offline", "telemetry_enabled", "max_tile_cells")


def check_host(host: str) -> str:
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"--host must be one of {', '.join(LOOPBACK_HOSTS)} (loopback only); got {host!r}")
    return host


def default_data_dir() -> Path:
    env = os.environ.get("WEIGHTLAB_DATA_DIR")
    if env:
        return Path(env)
    from platformdirs import user_data_dir

    return Path(user_data_dir("WeightLab", appauthor=False))


def new_token() -> str:
    return secrets.token_urlsafe(32)  # doc 09: random per launch, never stored or exported


@dataclass(frozen=True)
class ServerConfig:
    port: int
    data_dir: Path
    token: str = field(default_factory=new_token, repr=False)
    host: str = "127.0.0.1"
    max_body_bytes: int = MAX_BODY_BYTES
    max_in_flight: int = MAX_IN_FLIGHT

    def __post_init__(self) -> None:
        check_host(self.host)
        if not 0 < self.port < 65536:
            raise ValueError("port must be a bound TCP port (1-65535)")
        if len(self.token) < 32:
            raise ValueError("token too short")

    @property
    def allowed_hosts(self) -> frozenset[str]:
        """Exact Host header values accepted (DNS-rebinding defence, doc 12 section 4)."""
        return frozenset({f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}"})

    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/#token={self.token}"
