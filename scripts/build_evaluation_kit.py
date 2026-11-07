#!/usr/bin/env python3
"""Build the self-contained practitioner evaluation kit (Phase 0.5 task 6) as one zip archive.

    uv run python scripts/build_evaluation_kit.py [--out dist]

Contents: evaluation-kit/ (docs, questionnaire, templates, run_examples.py), the weightlab wheel, the fixtures
the examples use (copied unmodified from examples/fixtures), docs/cli/weightlab-diff.md and SHA256SUMS. The
archive is deterministic (sorted entries, fixed timestamps).

Self-test, as a participant would run it: the archive is unpacked into a temporary directory, the wheel *from the
archive* is installed into a new virtual environment (`uv venv` + `uv pip install`; dependencies come from PyPI or
the uv cache), and that environment's python runs the packed run_examples.py with that environment's `weightlab`
executable, from outside the repository and with PYTHONPATH/VIRTUAL_ENV cleared. The build checks that
`weightlab` imports from the new environment, then runs `weightlab start` from it (the packaged migrations create
a new database; /health and an authenticated route must answer), and fails, removing the archive, if anything
does not pass.
Exit status: 0 on success.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "evaluation-kit"
FIXTURES = ROOT / "examples" / "fixtures"
# Fixtures copied into the archive: the three examples, plus the optional cases named in examples/03.
FIXTURE_ITEMS = [
    "expected_results.json",
    "tiny_base.safetensors",
    "tiny_ft.safetensors",
    "tiny_lora",
    "tiny_lora_dora_unsupported",
    "tiny_precision_a.safetensors",
    "tiny_precision_b.safetensors",
    "compat",
    "malformed",
]
EPOCH = (1980, 1, 1, 0, 0, 0)
URL = re.compile(r"http://127\.0\.0\.1:(\d+)/#token=([A-Za-z0-9_-]{43,})")


def stage(dest: Path, wheel_dir: Path) -> None:
    for src in sorted(KIT.rglob("*")):
        if src.is_file() and "__pycache__" not in src.parts and not src.name.startswith("weightlab-eval-"):
            target = dest / src.relative_to(KIT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    shutil.copy2(ROOT / "docs" / "cli" / "weightlab-diff.md", dest / "weightlab-diff.md")
    for item in FIXTURE_ITEMS:
        src = FIXTURES / item
        if src.is_dir():
            shutil.copytree(src, dest / "fixtures" / item)
        else:
            (dest / "fixtures").mkdir(exist_ok=True)
            shutil.copy2(src, dest / "fixtures" / item)
    (dest / "wheel").mkdir()
    for whl in wheel_dir.glob("*.whl"):
        shutil.copy2(whl, dest / "wheel" / whl.name)
    sums = []
    for f in sorted(p for p in dest.rglob("*") if p.is_file()):
        sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(dest).as_posix()}")
    (dest / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


def write_zip(src: Path, archive: Path, top: str) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(f"{top}/{f.relative_to(src).as_posix()}", date_time=EPOCH)
            info.external_attr = (0o755 if f.suffix == ".py" else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, f.read_bytes())


def self_test(uv: str, kit: Path, work: Path) -> bool:
    """Install the archive's wheel into a fresh venv and run the packed examples with that venv's CLI."""
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"}}
    venv = work / "venv"
    bindir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    python = bindir / ("python.exe" if sys.platform == "win32" else "python")
    cli = bindir / ("weightlab.exe" if sys.platform == "win32" else "weightlab")
    wheels = sorted((kit / "wheel").glob("*.whl"))
    if len(wheels) != 1:
        print(f"expected one wheel in the archive, found {len(wheels)}", file=sys.stderr)
        return False
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    subprocess.run([uv, "venv", "--quiet", "--python", py, str(venv)], cwd=work, env=env, check=True)
    subprocess.run([uv, "pip", "install", "--quiet", "--python", str(python), str(wheels[0])], cwd=work, env=env,
                   check=True)  # fmt: skip
    origin = subprocess.run([str(python), "-c", "import weightlab; print(weightlab.__file__)"], cwd=work, env=env,
                            capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    if not Path(origin).resolve().is_relative_to(venv.resolve()):
        print(f"weightlab imported from {origin}, not from the test environment", file=sys.stderr)
        return False
    print(f"self-test: {wheels[0].name} installed into a fresh venv; weightlab from {Path(origin).parent}")
    r = subprocess.run(
        [str(python), str(kit / "run_examples.py"), "--fixtures", str(kit / "fixtures"),
         "--out", str(work / "reports"), "--weightlab", str(cli)],
        cwd=work, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    print(r.stdout[-600:])
    if r.returncode != 0:
        print(r.stdout + r.stderr, file=sys.stderr)
        return False
    return api_smoke_test(cli, work, env)


def _get(port: int, path: str, token: str | None = None) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        headers = {"Host": f"127.0.0.1:{port}"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def api_smoke_test(cli: Path, work: Path, env: dict[str, str]) -> bool:
    """`weightlab start` from the installed wheel: the packaged migrations create and migrate a new database,
    and the API answers /health and an authenticated route. Checks the wheel's package data, not the repo's."""
    data = work / "api-data"
    # Windows: own process group, stopped with CTRL_BREAK_EVENT (CTRL_C_EVENT would reach this script too).
    stop: int
    if sys.platform == "win32":
        flags, stop = subprocess.CREATE_NEW_PROCESS_GROUP, signal.CTRL_BREAK_EVENT
    else:
        flags, stop = 0, signal.SIGINT
    proc = subprocess.Popen(
        [str(cli), "start", "--port", "0", "--data-dir", str(data)],
        cwd=work, env={**env, "PYTHONUNBUFFERED": "1"}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=flags,
    )  # fmt: skip
    ok = False
    try:
        assert proc.stdout is not None
        line, deadline = "", time.monotonic() + 60
        while not (m := URL.search(line)) and time.monotonic() < deadline and proc.poll() is None:
            line += proc.stdout.readline()
        if not m:
            print(f"weightlab start printed no URL: {line[-300:]!r}", file=sys.stderr)
            return False
        port, token = int(m.group(1)), m.group(2)
        for _ in range(100):
            try:
                status, body = _get(port, "/api/v1/health")
                break
            except ConnectionRefusedError:
                time.sleep(0.1)
        else:
            print("API never accepted a connection", file=sys.stderr)
            return False
        health = json.loads(body)
        projects = _get(port, "/api/v1/projects", token)
        db = data / "weightlab.sqlite3"
        ok = status == 200 and health.get("status") == "ok" and projects == (200, b"[]") and db.is_file()
        print(f"self-test: weightlab start -> health {status} {health.get('status')}, projects {projects[0]}, "
              f"database created: {db.is_file()}")  # fmt: skip
        if not ok:
            print(f"API smoke test failed: health={body!r} projects={projects!r}", file=sys.stderr)
        return ok
    finally:
        proc.send_signal(stop)
        try:
            proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    args = ap.parse_args()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    top = f"weightlab-evaluation-kit-{version}"
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"{top}.zip"
    uv = shutil.which("uv")
    if uv is None:
        print("uv is required to build the wheel", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        subprocess.run([uv, "build", "--wheel", "--out-dir", str(t / "wheel")], cwd=ROOT, check=True)
        stage(t / top, t / "wheel")
        write_zip(t / top, archive, top)
        check_dir = t / "check"
        with zipfile.ZipFile(archive) as z:
            z.extractall(check_dir)
        kit = check_dir / top
        if not self_test(uv, kit, t):
            archive.unlink()
            return 1
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f"wrote {archive} ({archive.stat().st_size} bytes, sha256 {digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
