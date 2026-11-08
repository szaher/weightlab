"""Download the pinned real checkpoints of models.json into a local directory OUTSIDE the repository.

    uv run python evidence/phase1-real-models/fetch_models.py --dest ~/weightlab-models [--tier initial|all]

Each file comes from https://huggingface.co/<repo>/resolve/<revision>/<file> (an immutable commit, not a branch),
is written to <dest>/<name>/<file> via a temporary name, and its size and SHA-256 are checked against models.json
before it is kept. Existing files with the right hash are not downloaded again. The record of every file (URL,
size, SHA-256) goes to <dest>/fetched.json and is printed. Nothing is uploaded; weights are never committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHUNK = 8 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "weightlab-validation"})
    with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
        while b := r.read(CHUNK):
            f.write(b)
    os.replace(tmp, dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--tier", choices=["initial", "all"], default="initial")
    args = ap.parse_args()
    dest = args.dest.expanduser().resolve()
    if dest == REPO or REPO in dest.parents:
        sys.exit("refusing to download model weights inside the repository")
    spec = json.loads((HERE / "models.json").read_text())
    record = []
    for m in spec["models"]:
        if args.tier == "initial" and m["tier"] != "initial":
            continue
        d = dest / m["name"]
        d.mkdir(parents=True, exist_ok=True)
        for name, want in m["files"].items():
            url = f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{name}"
            p = d / name
            if not (p.exists() and want.get("sha256") and sha256(p) == want["sha256"]):
                print(f"downloading {url}", flush=True)
                fetch(url, p)
            got = sha256(p)
            size = p.stat().st_size
            if want.get("sha256") and (got != want["sha256"] or size != want["size"]):
                p.unlink()
                sys.exit(f"{p}: size/sha256 mismatch ({size}, {got})")
            record.append({"model": m["name"], "repo": m["repo"], "revision": m["revision"], "license": m["license"],
                           "file": name, "url": url, "size": size, "sha256": got})  # fmt: skip
    (dest / "fetched.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
