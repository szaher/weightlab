#!/usr/bin/env python3
"""Measure Compare report jobs on the pinned real checkpoints and check them against `weightlab diff`.

    uv run --with psutil python evidence/alpha2-compare/measure_real_compare.py \
        --models ~/weightlab-models --out evidence/alpha2-compare/results

Uses a real `weightlab start` process (fresh data directory, loopback, bearer token), the checkpoints fetched by
evidence/phase1-real-models/fetch_models.py (checked against their pinned SHA-256 first), and for each pair:
create a comparison, run a `comparison_report` job with consistency best_effort and then verified, record wall
time, bytes read and the server's peak RSS while the job runs, then compare the stored report with the
`report.json` that `weightlab diff` writes for the same files (tensor order and every value, rollups, totals,
compatibility, completeness). Exits non-zero on any mismatch. RSS is sampled every 50 ms with psutil; it is an
observation on this machine, not a memory guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import psutil

REPO = Path(__file__).resolve().parents[2]
MIB = 1024 * 1024
URL_RE = re.compile(r"http://127\.0\.0\.1:(\d+)/#token=([A-Za-z0-9_-]{43,})")
PAIRS = [("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"), ("Qwen2.5-Math-1.5B", "Qwen2.5-Math-1.5B-Instruct")]
BUDGET = {"max_working_set_mib": 1024, "max_threads": 1, "max_wall_seconds": 600}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


class Server:
    def __init__(self, data_dir: Path) -> None:
        cmd = [sys.executable, "-c", "from weightlab.cli import main; raise SystemExit(main())", "start"]
        cmd += ["--port", "0", "--data-dir", str(data_dir)]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO)
        out, deadline = "", time.monotonic() + 90
        assert self.proc.stdout is not None
        while not (m := URL_RE.search(out)):
            line = self.proc.stdout.readline()
            if (not line and self.proc.poll() is not None) or time.monotonic() > deadline:
                raise RuntimeError(f"weightlab start failed:\n{out}")
            out += line
        self.port, self.token = int(m.group(1)), m.group(2)
        threading.Thread(target=self._drain, daemon=True).start()
        self.ps = psutil.Process(self.proc.pid)
        self.peak = 0
        self._stop = threading.Event()
        threading.Thread(target=self._sample, daemon=True).start()
        for _ in range(200):
            try:
                if self.call("GET", "/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.1)

    def _drain(self) -> None:
        assert self.proc.stdout is not None
        for _ in self.proc.stdout:
            pass

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self.ps.memory_info().rss)
            except psutil.Error:
                return
            time.sleep(0.05)

    def rss(self) -> int:
        return int(self.ps.memory_info().rss)

    def call(self, method: str, path: str, body: Any = None) -> tuple[int, bytes]:
        data = None if body is None else json.dumps(body).encode()
        h = {"Host": f"127.0.0.1:{self.port}", "Accept": "application/json", "Authorization": f"Bearer {self.token}"}
        if data is not None:
            h["Content-Type"] = "application/json"
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/v1{path}", data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def json(self, method: str, path: str, body: Any = None, expect: int = 200) -> Any:
        st, raw = self.call(method, path, body)
        if st != expect:
            raise AssertionError(f"{method} {path} -> {st}: {raw[:500]!r}")
        return json.loads(raw) if raw else None

    def stop(self) -> None:
        self._stop.set()
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_job(s: Server, cid: str, consistency: str) -> tuple[dict[str, Any], dict[str, Any]]:
    s.peak = s.rss()
    rss_before = s.rss()
    t0 = time.monotonic()
    job = s.json("POST", "/jobs", {"kind": "comparison_report", "comparison_id": cid, "consistency": consistency,
                                   "budget": BUDGET}, expect=202)  # fmt: skip
    while job["state"] in ("created", "queued", "running"):
        time.sleep(0.1)
        job = s.json("GET", f"/jobs/{job['job_id']}")
    wall = time.monotonic() - t0
    if job["state"] != "completed":
        raise AssertionError(f"{consistency} job ended {job['state']}: {job.get('error')}")
    rep = s.json("GET", f"/comparisons/{cid}/report")
    m = {
        "consistency": consistency,
        "state": job["state"],
        "published": job["result"]["published"],
        "complete": rep["complete"],
        "wall_seconds": round(wall, 2),
        "bytes_scanned": job.get("bytes_scanned"),
        "estimated_bytes": job.get("estimated_bytes"),
        "rss_before_mib": round(rss_before / MIB, 1),
        "peak_rss_mib": round(s.peak / MIB, 1),
        "report_json_bytes": len(json.dumps(rep["report"])),
    }
    return m, rep


def parity(api_report: dict[str, Any], cli_report: dict[str, Any]) -> list[str]:
    """Every field the Compare screen shows, compared exactly (no tolerance)."""
    bad = []
    if [t["key"] for t in api_report["tensors"]] != [t["key"] for t in cli_report["tensors"]]:
        bad.append("tensor ranking order differs")
    for name in ("tensors", "layers", "roles", "totals", "completeness", "caveats", "warnings"):
        if api_report[name] != cli_report[name]:
            bad.append(f"{name} differ")
    for name in ("lineage", "counts"):
        if api_report["compatibility"][name] != cli_report["compatibility"][name]:
            bad.append(f"compatibility.{name} differ")
    if api_report["complete"] != cli_report["complete"]:
        bad.append("complete differs")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    models: Path = args.models.expanduser().resolve()
    fetched = {(e["model"], e["file"]): e for e in json.loads((models / "fetched.json").read_text())}
    names = sorted({n for p in PAIRS for n in p})
    for n in names:
        f = models / n / "model.safetensors"
        log(f"verifying {n} SHA-256")
        if sha256(f) != fetched[(n, "model.safetensors")]["sha256"]:
            raise SystemExit(f"{f}: SHA-256 differs from the pinned manifest")

    failures: list[str] = []
    results: dict[str, Any] = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
            "python": platform.python_version(),
        },
        "budget": BUDGET,
        "pairs": [],
    }
    with tempfile.TemporaryDirectory(prefix="wl-compare-") as tmp:
        s = Server(Path(tmp) / "data")
        try:
            results["idle_rss_mib"] = round(s.rss() / MIB, 1)
            version = s.json("GET", "/health").get("version")
            results["weightlab_version"] = version
            project = s.json("POST", "/projects", {"name": "real-compare"}, expect=201)["id"]
            ids = {}
            for n in names:
                f = models / n / "model.safetensors"
                body = {"entry_path": str(f), "selected_root": str(f.parent), "display_name": n}
                ids[n] = s.json("POST", f"/projects/{project}/models", body, expect=201)["id"]
            for a, b in PAIRS:
                log(f"pair {a} -> {b}")
                base = {"project_id": project, "model_a_id": ids[a], "model_b_id": ids[b], "user_attestation": False}
                st, raw = s.call("POST", "/comparisons", {**base, "policy": "strict"})
                strict_status, strict_job = st, None
                if st == 201:
                    cmp_ = json.loads(raw)
                    st2, raw2 = s.call(
                        "POST", "/jobs", {"kind": "comparison_report", "comparison_id": cmp_["id"], "budget": BUDGET}
                    )
                    strict_job = {"status": st2, "code": json.loads(raw2)["error"]["code"] if st2 != 202 else None}
                    if st2 == 202:  # Class V/L: strict is allowed; let it finish before the measured runs
                        job = json.loads(raw2)
                        while job["state"] in ("created", "queued", "running"):
                            time.sleep(0.1)
                            job = s.json("GET", f"/jobs/{job['job_id']}")
                cmp_ = s.json("POST", "/comparisons", {**base, "policy": "descriptive_only"}, expect=201)
                entry: dict[str, Any] = {
                    "a": a,
                    "b": b,
                    "class": cmp_["lineage"]["compatibility_class"],
                    "counts": cmp_["counts"],
                    "strict_policy": {"create_status": strict_status, "report_job": strict_job},
                    "runs": [],
                }
                rep = None
                for consistency in ("best_effort", "verified"):
                    m, rep = run_job(s, cmp_["id"], consistency)
                    log(f"  {consistency}: {m['wall_seconds']} s, peak RSS {m['peak_rss_mib']} MiB")
                    entry["runs"].append(m)
                assert rep is not None
                with tempfile.TemporaryDirectory(prefix="wl-diff-") as d:
                    t0 = time.monotonic()
                    cmd = [sys.executable, "-c", "from weightlab.cli import main; raise SystemExit(main())", "diff",
                           str(models / a / "model.safetensors"), str(models / b / "model.safetensors"), "--out", d]  # fmt: skip
                    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
                    entry["cli_diff"] = {"exit": proc.returncode, "wall_seconds": round(time.monotonic() - t0, 2)}
                    cli = json.loads((Path(d) / "report.json").read_text())
                bad = parity(rep["report"], cli)
                entry["parity_with_weightlab_diff"] = bad or "identical"
                entry["tensors_compared"] = len(rep["report"]["tensors"])
                entry["top5_relative_l2"] = [
                    {"key": t["key"], "relative_l2": t["metrics"]["relative_l2"]["value"]}
                    for t in rep["report"]["tensors"][:5]
                ]
                failures += [f"{a} vs {b}: {x}" for x in bad]
                if proc.returncode != 0:
                    failures.append(f"{a} vs {b}: weightlab diff exited {proc.returncode}")
                results["pairs"].append(entry)
            results["peak_rss_mib_overall"] = max(r["peak_rss_mib"] for p in results["pairs"] for r in p["runs"])
        finally:
            s.stop()
    results["failures"] = failures
    results["verdict"] = "PASS" if not failures else "FAIL"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "real_compare.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log(f"{results['verdict']}: wrote {args.out / 'real_compare.json'}")
    for f in failures:
        log(f"  FAIL {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
