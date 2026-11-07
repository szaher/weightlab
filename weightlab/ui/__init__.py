"""The browser UI's production build (frontend/, built by Vite into weightlab/ui/static and shipped in the wheel).

`load_assets` reads the build once, at server start, into an in-memory map from exact URL path to bytes.
Requests are answered by dictionary lookup only: no request path is ever joined to a filesystem path, so
traversal (`..`, encoded or not), directory listing and symlink tricks have nothing to act on, and any path
not in the map fails closed. The shell is public by design: it holds no secret (doc 09: the token arrives
in the URL fragment and stays in the browser).
"""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"
INDEX = "index.html"
MAX_TOTAL_BYTES = 16 * 1024 * 1024
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}
SUBDIRS = ("assets",)  # the only directory below the build root that is served


@dataclass(frozen=True)
class Asset:
    body: bytes
    content_type: str


def _read(path: Path) -> Asset | None:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):  # no symlinks, devices or directories
        return None
    ctype = CONTENT_TYPES.get(path.suffix.lower())
    if ctype is None or not NAME.match(path.name):
        return None
    return Asset(path.read_bytes(), ctype)


def load_assets(root: Path = STATIC_DIR) -> dict[str, Asset]:
    """URL path -> asset. "/" is index.html; other top-level files and files in assets/ keep their names.
    Returns {} when the UI was not built (the API still works)."""
    out: dict[str, Asset] = {}
    if not root.is_dir() or root.is_symlink():
        return out
    total = 0
    for directory, prefix in [(root, "/"), *((root / d, f"/{d}/") for d in SUBDIRS)]:
        if not directory.is_dir() or directory.is_symlink():
            continue
        for entry in sorted(directory.iterdir()):
            asset = _read(entry)
            if asset is None:
                continue
            total += len(asset.body)
            if total > MAX_TOTAL_BYTES:
                raise ValueError(f"UI build larger than {MAX_TOTAL_BYTES} bytes")
            out[prefix + entry.name] = asset
    index = out.pop("/" + INDEX, None)
    if index is None:
        return {}
    out["/"] = index
    return out
