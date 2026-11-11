"""comparison_report jobs and GET /comparisons/{id}/report (ADR-020, contract 0.4.0).

Acceptance (ROADMAP P0.1): the API's report equals `weightlab diff`'s on the deterministic fixtures and an incomplete
report is never shown as complete. Plus the ADR-020 invariants: atomic publication with the job's terminal state,
submission-ordered replacement, cancellable hashing, drift and legacy records failing closed, the strict policy for
unverified classes, snapshot freshness and path privacy. Every response is validated against the contract by `Api`;
report bodies additionally against weightlab/schemas/change_report.schema.json.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest
from fastapi.testclient import TestClient
from safetensors.numpy import save_file

from tests.api.conftest import PORT, TOKEN, import_model
from tests.api.contract import Api
from tests.helpers import ROOT
from weightlab.analysis import change_report
from weightlab.analysis.change_report import Options, run
from weightlab.analysis.compat import compare_structure, structure_fingerprint
from weightlab.api import reports
from weightlab.api.app import create_app
from weightlab.api.config import DEFAULT_SETTINGS, ServerConfig
from weightlab.api.reports import ComparisonReport
from weightlab.core.errors import ErrorCode, WeightLabError
from weightlab.db import Database, store
from weightlab.formats.checkpoint import open_checkpoint
from weightlab.jobs import Canceled, JobContext

BUDGET = {"max_working_set_mib": 512, "max_threads": 1, "max_wall_seconds": 120}
SCHEMA = json.loads((ROOT / "weightlab/schemas/change_report.schema.json").read_text(encoding="utf-8"))
BASE, FT = "tiny_base.safetensors", "tiny_ft.safetensors"
PA, PB = "tiny_precision_a.safetensors", "tiny_precision_b.safetensors"


# -- helpers -----------------------------------------------------------------------------------
def compare(api: Api, project: str, root: Path, a: str, b: str, expect: int = 201, **extra: Any) -> Any:
    ma = import_model(api, project, root, a)
    mb = import_model(api, project, root, b)
    body = {"project_id": project, "model_a_id": ma["id"], "model_b_id": mb["id"], **extra}
    return api.post("/comparisons", json_body=body, expect=expect)


def submit(api: Api, comparison_id: str, expect: int = 202, **extra: Any) -> Any:
    body = {"kind": "comparison_report", "comparison_id": comparison_id, "budget": BUDGET, **extra}
    return api.post("/jobs", json_body=body, expect=expect)


def finish(api: Api, job_id: str, t: float = 120) -> dict[str, Any]:
    end = time.monotonic() + t
    while time.monotonic() < end:
        j = api.get("/jobs/{job_id}", job_id=job_id, expect=200)
        if j["state"] not in ("created", "queued", "running"):
            return j  # type: ignore[no-any-return]
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def report(api: Api, comparison_id: str, expect: int = 200) -> Any:
    return api.get("/comparisons/{comparison_id}/report", comparison_id=comparison_id, expect=expect)


def without_time(r: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(r))
    out["provenance"]["created_at"] = "T"
    return out


def edit_compat(config: ServerConfig, cid: str, fn: Callable[[dict[str, Any]], None]) -> None:
    with sqlite3.connect(config.data_dir / "weightlab.sqlite3") as conn:
        compat = json.loads(
            conn.execute("SELECT compatibility_json FROM comparisons WHERE id = ?", (cid,)).fetchone()[0]
        )
        fn(compat)
        conn.execute("UPDATE comparisons SET compatibility_json = ? WHERE id = ?", (json.dumps(compat), cid))


class _Runner:
    def _progress(self, *_: Any) -> None:
        pass


def context(on_tick: Callable[[JobContext, int], None] | None = None) -> JobContext:
    ctx = JobContext(_Runner(), "job_direct", 1, time.monotonic() + 60)  # type: ignore[arg-type]
    if on_tick is not None:
        orig = ctx.tick

        def tick(n: int) -> None:
            on_tick(ctx, n)
            orig(n)

        ctx.tick = tick  # type: ignore[method-assign]
    return ctx


def job_spec(cid: str, project: str, consistency: str = "best_effort", seq: int = 10**9) -> dict[str, Any]:
    return {"comparison_id": cid, "project_id": project, "consistency": consistency, "job_seq": seq}


# -- numerical equivalence and completeness ------------------------------------------------------
@pytest.mark.parametrize(
    ("a", "b", "policy", "complete", "cls"),
    [(BASE, FT, "strict", True, "V"), (PA, PB, "descriptive_only", False, "S")],
)
def test_report_equals_the_cli_report(
    api: Api, project: str, models_dir: Path, a: str, b: str, policy: str, complete: bool, cls: str
) -> None:
    cmp = compare(api, project, models_dir, a, b, policy=policy)
    assert cmp["policy"] == policy
    report(api, cmp["id"], expect=404)  # none yet
    job = submit(api, cmp["id"])
    assert job["kind"] == "comparison_report"
    assert job["estimated_bytes"] > 0
    done = finish(api, job["job_id"])
    assert done["state"] == "completed", done
    assert done["result"] == {"kind": "comparison_report", "comparison_id": cmp["id"], "complete": complete,
                              "published": True}  # fmt: skip
    assert done["progress"] == 1.0

    got = report(api, cmp["id"])
    assert got["job_id"] == done["job_id"]
    assert got["complete"] is complete
    assert got["consistency"] == "best_effort"
    assert got["current_model_status"] == {"model_a": "ready", "model_b": "ready"}
    body = got["report"]
    jsonschema.validate(body, SCHEMA)
    assert body["complete"] is complete  # an incomplete report is labeled incomplete, never complete
    assert body["compatibility"]["lineage"]["compatibility_class"] == cls
    assert body["compatibility"]["lineage"] == cmp["lineage"]
    assert body["compatibility"]["counts"] == cmp["counts"]
    cli = run(models_dir / a, models_dir / b, selected_root=models_dir, options=Options())
    assert without_time(body) == without_time(cli)  # every rank, metric, rollup and caveat


def test_strict_policy_refuses_unverified_classes(api: Api, project: str, models_dir: Path) -> None:
    cmp = compare(api, project, models_dir, PA, PB)  # default policy strict, Class S
    assert cmp["policy"] == "strict"
    e = submit(api, cmp["id"], expect=409)
    assert e["error"]["code"] == "LINEAGE_UNVERIFIED"
    assert "descriptive_only" in e["error"]["message"]


def test_attested_comparison_reports_with_the_same_lineage(api: Api, project: str, models_dir: Path) -> None:
    cmp = compare(api, project, models_dir, PA, PB, user_attestation=True, policy="descriptive_only")
    assert cmp["lineage"]["lineage_evidence"] == "user_attested"
    done = finish(api, submit(api, cmp["id"])["job_id"])
    assert done["state"] == "completed", done
    assert report(api, cmp["id"])["report"]["compatibility"]["lineage"] == cmp["lineage"]


def test_verified_consistency_hashes_inputs(api: Api, project: str, models_dir: Path) -> None:
    cmp = compare(api, project, models_dir, BASE, FT)
    plain = finish(api, submit(api, cmp["id"])["job_id"])
    verified = finish(api, submit(api, cmp["id"], consistency="verified")["job_id"])
    assert verified["state"] == "completed"
    files = sum((models_dir / n).stat().st_size for n in (BASE, FT))
    assert verified["estimated_bytes"] == plain["estimated_bytes"] + files
    assert verified["bytes_scanned"] == verified["estimated_bytes"]  # every hashed chunk ticks the job (ADR-020)
    assert plain["bytes_scanned"] <= plain["estimated_bytes"]  # the estimate is an upper bound
    got = report(api, cmp["id"])
    assert got["job_id"] == verified["job_id"]  # the later-submitted completed job replaced the report
    assert got["consistency"] == "verified"
    assert got["report"]["settings"]["consistency"] == "verified"
    assert all(
        f["sha256"] for side in ("base", "derived") for f in got["report"]["provenance"]["inputs"][side]["files"]
    )


# -- request validation and limits ---------------------------------------------------------------
def test_request_validation_and_limits(
    api: Api, project: str, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmp = compare(api, project, models_dir, BASE, FT)
    src = {"kind": "tensor", "model_id": cmp["model_a_id"], "tensor_id": "tns_x"}
    api.post("/jobs", json_body={"kind": "comparison_report", "budget": BUDGET}, expect=422)
    submit(api, cmp["id"], expect=422, source=src)
    submit(api, cmp["id"], expect=422, model_id=cmp["model_a_id"])
    e = submit(api, cmp["id"], expect=422, budget={**BUDGET, "max_working_set_mib": 64})
    assert e["error"]["code"] == "RESOURCE_LIMIT"
    assert e["error"]["details"]["required_mib"] == 128 + -(-cmp_entries(cmp) * reports.PER_ENTRY_BYTES // 2**20)
    monkeypatch.setattr(reports, "MAX_REPORT_ENTRIES", cmp_entries(cmp) - 1)
    e = submit(api, cmp["id"], expect=422)
    assert e["error"]["code"] == "RESOURCE_LIMIT"
    assert e["error"]["details"]["entries"] == cmp_entries(cmp)
    monkeypatch.undo()
    monkeypatch.setattr(reports, "MAX_REPORT_JSON_BYTES", 100)  # checked after the scan, before publication
    done = finish(api, submit(api, cmp["id"])["job_id"])
    assert done["state"] == "failed"
    assert done["error"]["code"] == "RESOURCE_LIMIT"
    report(api, cmp["id"], expect=404)
    assert submit(api, "cmp_doesnotexist", expect=404)["error"]["code"] == "NOT_FOUND"
    report(api, "cmp_doesnotexist", expect=404)


def cmp_entries(cmp: dict[str, Any]) -> int:
    return sum(cmp["counts"].values())  # type: ignore[no-any-return]


# -- drift, legacy records -----------------------------------------------------------------------
def test_changed_model_is_refused(api: Api, project: str, models_dir: Path) -> None:
    cmp = compare(api, project, models_dir, BASE, FT)
    ft = models_dir / FT
    data = bytearray(ft.read_bytes())
    data[-1] ^= 0xFF
    ft.write_bytes(bytes(data))
    assert submit(api, cmp["id"], expect=409)["error"]["code"] == "FILE_CHANGED"
    report(api, cmp["id"], expect=404)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda c: c.update(structure_sha256="0" * 64),  # same counts, different structure
        lambda c: c["lineage"].update(reasons=["a different assessment"]),
        lambda c: c["counts"].update(matched=c["counts"]["matched"] + 1),
    ],
    ids=["structure", "lineage", "counts"],
)
def test_drift_from_the_recorded_comparison_fails_closed(
    api: Api, project: str, models_dir: Path, config: ServerConfig, tamper: Callable[[dict[str, Any]], None]
) -> None:
    cmp = compare(api, project, models_dir, BASE, FT)
    first = finish(api, submit(api, cmp["id"])["job_id"])
    edit_compat(config, cmp["id"], tamper)
    done = finish(api, submit(api, cmp["id"], consistency="verified")["job_id"])
    assert done["state"] == "failed"
    assert done["error"]["code"] == "CONFLICT"
    assert done["result"] is None
    assert report(api, cmp["id"])["job_id"] == first["job_id"]  # the last valid report is kept


def test_structure_fingerprint_sees_changes_that_keep_counts(models_dir: Path) -> None:
    with open_checkpoint(models_dir / BASE) as a, open_checkpoint(models_dir / FT) as b:
        s = compare_structure(a, b)
        ident = {"model_a_id": "mod_a", "model_b_id": "mod_b"}
        base = structure_fingerprint(s, ident)
        counts = s.counts()
        assert structure_fingerprint(compare_structure(a, b), ident) == base  # deterministic
        assert structure_fingerprint(s, {**ident, "model_b_id": "mod_c"}) != base  # comparison identity
        e = s.entries[0]
        assert e.a is not None
        orig_shape, orig_dtype = e.a.shape, e.a.dtype
        object.__setattr__(e.a, "shape", (*orig_shape, 1))
        assert structure_fingerprint(s, ident) != base
        assert s.counts() == counts  # equal aggregate counts, different structure
        object.__setattr__(e.a, "shape", orig_shape)
        object.__setattr__(e.a, "dtype", "F64" if orig_dtype != "F64" else "F32")
        assert structure_fingerprint(s, ident) != base
        object.__setattr__(e.a, "dtype", orig_dtype)
        e.tied_to = "some.other.key"
        assert structure_fingerprint(s, ident) != base


@pytest.mark.parametrize("variant", ["no_record_version", "no_attestation"])
def test_comparisons_from_before_adr_020_must_be_recreated(
    api: Api, project: str, models_dir: Path, config: ServerConfig, variant: str
) -> None:
    cmp = compare(api, project, models_dir, PA, PB, user_attestation=True, policy="descriptive_only")

    def legacy(c: dict[str, Any]) -> None:  # what POST /comparisons stored before ADR-020
        c.pop("record_version")
        c.pop("structure_sha256")
        if variant == "no_attestation":
            c.pop("user_attestation")

    edit_compat(config, cmp["id"], legacy)
    e = submit(api, cmp["id"], expect=409)
    assert e["error"]["code"] == "CONFLICT"
    assert "create a new comparison" in e["error"]["message"]
    db = Database(config.data_dir / "weightlab.sqlite3")  # a job queued before the upgrade fails the same way
    with pytest.raises(WeightLabError) as ei:
        ComparisonReport(db).run(context(), job_spec(cmp["id"], project))
    assert ei.value.code is ErrorCode.CONFLICT
    report(api, cmp["id"], expect=404)


# -- snapshot freshness and privacy ------------------------------------------------------------------
def test_stored_report_is_a_snapshot_with_current_status(api: Api, project: str, models_dir: Path) -> None:
    cmp = compare(api, project, models_dir, BASE, FT)
    done = finish(api, submit(api, cmp["id"])["job_id"])
    ft = models_dir / FT
    data = bytearray(ft.read_bytes())
    data[-1] ^= 0xFF
    ft.write_bytes(bytes(data))
    got = report(api, cmp["id"])
    assert got["job_id"] == done["job_id"]  # still served, as the historical snapshot it is
    assert got["current_model_status"] == {"model_a": "ready", "model_b": "changed"}


UNUSUAL = "Ünïcode dir with spaces #%&'"


def test_no_absolute_path_reaches_reports_or_errors(
    api: Api, project: str, models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / UNUSUAL / "nested ✓"
    root.mkdir(parents=True)
    names = ["base model (copy) ✓.safetensors", "ft model; rm -rf.safetensors"]
    shutil.copy(models_dir / BASE, root / names[0])
    shutil.copy(models_dir / FT, root / names[1])
    cmp = compare(api, project, root, names[0], names[1], policy="descriptive_only")
    done = finish(api, submit(api, cmp["id"], consistency="verified")["job_id"])
    assert done["state"] == "completed", done
    raw = api.client.get(f"/api/v1/comparisons/{cmp['id']}/report", headers=api.headers).text
    for secret in (str(root), str(tmp_path), UNUSUAL):
        assert secret not in raw
        assert json.dumps(secret)[1:-1] not in raw
    assert names[0] in raw  # file names (not paths) are provenance, as in the CLI

    def leak(*_: Any, **__: Any) -> Any:
        raise WeightLabError(ErrorCode.FILE_CHANGED, f"{root / names[1]} changed under {Path.home()}")

    monkeypatch.setattr(reports, "pair_report", leak)
    failed = finish(api, submit(api, cmp["id"])["job_id"])
    assert failed["state"] == "failed"
    msg = failed["error"]["message"]
    assert str(root) not in msg
    assert str(Path.home()) not in msg
    assert reports.REDACTED in msg


# -- atomic publication, cancellation, deletion, crash -------------------------------------------
def test_deleting_the_project_deletes_its_reports(
    api: Api, project: str, models_dir: Path, config: ServerConfig
) -> None:
    cmp = compare(api, project, models_dir, BASE, FT)
    assert finish(api, submit(api, cmp["id"])["job_id"])["state"] == "completed"
    api.delete("/projects/{project_id}", project_id=project, expect=204)
    with sqlite3.connect(config.data_dir / "weightlab.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM comparison_reports").fetchone()[0] == 0


def test_comparison_deleted_at_the_commit_boundary_publishes_nothing(
    api: Api, project: str, models_dir: Path, config: ServerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmp = compare(api, project, models_dir, BASE, FT)
    orig = ComparisonReport.run

    def run_then_delete(self: ComparisonReport, ctx: JobContext, spec: dict[str, Any]) -> dict[str, Any]:
        out = orig(self, ctx, spec)
        with sqlite3.connect(config.data_dir / "weightlab.sqlite3") as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM comparisons WHERE id = ?", (cmp["id"],))
        return out

    monkeypatch.setattr(ComparisonReport, "run", run_then_delete)
    done = finish(api, submit(api, cmp["id"])["job_id"])
    assert done["state"] == "failed"  # never a false successful job
    assert done["error"]["code"] == "NOT_FOUND"
    with sqlite3.connect(config.data_dir / "weightlab.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM comparison_reports").fetchone()[0] == 0


def test_crash_inside_the_commit_transaction_rolls_back_both(
    api: Api, project: str, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmp = compare(api, project, models_dir, BASE, FT)
    first = finish(api, submit(api, cmp["id"])["job_id"])
    orig = store.publish_comparison_report

    def write_then_crash(*a: Any, **k: Any) -> bool:
        orig(*a, **k)  # the report row is written inside the transaction ...
        raise sqlite3.OperationalError("disk I/O error")  # ... and the transaction dies before COMMIT

    monkeypatch.setattr(store, "publish_comparison_report", write_then_crash)
    done = finish(api, submit(api, cmp["id"], consistency="verified")["job_id"])
    assert done["state"] == "failed"
    assert done["result"] is None
    got = report(api, cmp["id"])
    assert got["job_id"] == first["job_id"]  # the candidate report was rolled back with the completed state
    assert got["consistency"] == "best_effort"


def test_restart_after_a_crash_exposes_no_interrupted_report(
    api: Api, project: str, models_dir: Path, config: ServerConfig
) -> None:
    cmp = compare(api, project, models_dir, BASE, FT)
    db = Database(config.data_dir / "weightlab.sqlite3")
    with db.session() as conn, store.transaction(conn):  # the state a crash mid-job leaves: a running row only
        conn.execute("INSERT INTO jobs (id, project_id, kind, spec_json, state, progress, created_at, started_at) "
                     "VALUES ('job_crashed', ?, 'comparison_report', ?, 'running', 0.5, '2026-01-01T00:00:00Z', "
                     "'2026-01-01T00:00:00Z')",
                     (project, json.dumps({"dedup_key": "x", "memory_bytes": 1, "estimated_bytes": 1,
                                           "wall_seconds": 1, **job_spec(cmp["id"], project)})))  # fmt: skip
    with TestClient(create_app(config), base_url=f"http://127.0.0.1:{PORT}") as c2:
        api2 = Api(c2, TOKEN)
        assert api2.get("/jobs/{job_id}", job_id="job_crashed", expect=200)["state"] == "interrupted"
        report(api2, cmp["id"], expect=404)


def test_cancel_is_linearized_with_completion(api: Api, project: str, models_dir: Path, config: ServerConfig) -> None:
    cmp = compare(api, project, models_dir, BASE, FT)
    first = finish(api, submit(api, cmp["id"])["job_id"])
    db = Database(config.data_dir / "weightlab.sqlite3")
    ctx = context()
    ctx.cancel.set()
    with pytest.raises(Canceled):  # canceled before work: nothing computed, no commit hook
        ComparisonReport(db).run(ctx, job_spec(cmp["id"], project))
    assert ctx.commit_hook is None
    after = api.post("/jobs/{job_id}/cancel", job_id=first["job_id"], expect=200)
    assert after["state"] == "completed"  # completion took the lock first: a later cancel changes nothing
    assert after["result"] == first["result"]
    assert report(api, cmp["id"])["job_id"] == first["job_id"]


# -- cancellable, deadline-bounded hashing -------------------------------------------------------------
@pytest.fixture
def big_pair(tmp_path: Path) -> Path:
    """Two 12 MiB single-tensor files: three 4 MiB hash chunks each."""
    root = tmp_path / "big"
    root.mkdir()
    x = np.arange(3 * 1024 * 1024, dtype=np.float32)
    save_file({"model.layers.0.mlp.up_proj.weight": x}, str(root / "a.safetensors"))
    save_file({"model.layers.0.mlp.up_proj.weight": x + np.float32(1)}, str(root / "b.safetensors"))
    return root


def _calls_from(fn_name: str) -> bool:
    return any(f.function == fn_name for f in inspect.stack())


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
@pytest.mark.parametrize("mode", ["verified", "attested"])
def test_hashing_is_interrupted_between_chunks(
    api: Api, project: str, big_pair: Path, config: ServerConfig, monkeypatch: pytest.MonkeyPatch, stop: str,
    mode: str,
) -> None:  # fmt: skip
    cmp = compare(api, project, big_pair, "a.safetensors", "b.safetensors", policy="descriptive_only",
                  user_attestation=mode == "attested")  # fmt: skip
    scans: list[int] = []
    orig_stats = change_report.pair_stats
    monkeypatch.setattr(change_report, "pair_stats", lambda *a, **k: scans.append(1) or orig_stats(*a, **k))
    ticks: list[bool] = []

    def on_tick(ctx: JobContext, _n: int) -> None:
        ticks.append(_calls_from("sha256"))
        if len(ticks) == 2:  # inside hashing, after the second of three chunks of the first file
            if stop == "cancel":
                ctx.cancel.set()
            else:
                ctx.deadline = 0.0

    ctx = context(on_tick)
    consistency = "verified" if mode == "verified" else "best_effort"
    db = Database(config.data_dir / "weightlab.sqlite3")
    if stop == "cancel":
        with pytest.raises(Canceled):
            ComparisonReport(db).run(ctx, job_spec(cmp["id"], project, consistency))
    else:
        with pytest.raises(WeightLabError) as ei:
            ComparisonReport(db).run(ctx, job_spec(cmp["id"], project, consistency))
        assert ei.value.code is ErrorCode.RESOURCE_LIMIT
    assert ticks == [True, True]  # stopped at the chunk boundary, inside sha256, before any third read
    assert scans == []  # no tensor was analyzed
    assert ctx.commit_hook is None  # nothing can be published
    report(api, cmp["id"], expect=404)


def test_sharded_checkpoints_hash_every_shard_with_ticks(
    api: Api, project: str, models_dir: Path, config: ServerConfig
) -> None:
    shutil.copytree(models_dir / "tiny_sharded", models_dir / "tiny_sharded_b")
    cmp = compare(api, project, models_dir, "tiny_sharded", "tiny_sharded_b", policy="descriptive_only")
    hashed: list[int] = []
    ctx = context(lambda _c, n: hashed.append(n) if _calls_from("sha256") else None)
    db = Database(config.data_dir / "weightlab.sqlite3")
    ComparisonReport(db).run(ctx, job_spec(cmp["id"], project, "verified"))
    shards = [p.stat().st_size for d in ("tiny_sharded", "tiny_sharded_b")
              for p in sorted((models_dir / d).glob("*.safetensors"))]  # fmt: skip
    assert sum(hashed) == 2 * sum(shards)  # before and after, every shard of both sides
    assert ctx.commit_hook is not None


# -- submission ordering under concurrency -----------------------------------------------------------
@pytest.fixture
def api2(config: ServerConfig) -> Iterator[Api]:
    """An app with two heavy-job workers, so two report jobs of one comparison run concurrently."""
    db = Database(config.data_dir / "weightlab.sqlite3")
    with db.session() as conn:
        store.put_settings(conn, {**DEFAULT_SETTINGS, "max_heavy_jobs": 2})
    with TestClient(create_app(config, db), base_url=f"http://127.0.0.1:{PORT}") as c:
        yield Api(c, TOKEN)


@pytest.fixture
def gates(monkeypatch: pytest.MonkeyPatch) -> dict[str, threading.Event]:
    """Hold each job (by consistency) after its scan until its gate opens."""
    g = {"best_effort": threading.Event(), "verified": threading.Event()}
    started = {k: threading.Event() for k in g}
    orig = reports.pair_report

    def gated(a: Any, b: Any, lin: Any, opts: Options, control: Any) -> Any:
        out = orig(a, b, lin, opts, control)
        started[opts.consistency.value].set()
        if not g[opts.consistency.value].wait(30):
            raise AssertionError("gate never opened")
        if getattr(g[opts.consistency.value], "fail", False):
            raise WeightLabError(ErrorCode.FILE_CHANGED, "injected failure")
        return out

    monkeypatch.setattr(reports, "pair_report", gated)
    g["_started"] = started  # type: ignore[assignment]
    return g


def _two_jobs(api: Api, project: str, models_dir: Path, gates: dict[str, Any]) -> tuple[str, str, str]:
    cmp = compare(api, project, models_dir, BASE, FT)
    older = submit(api, cmp["id"])["job_id"]  # best_effort, submitted first
    newer = submit(api, cmp["id"], consistency="verified")["job_id"]
    for k in ("best_effort", "verified"):
        assert gates["_started"][k].wait(30)  # both are running concurrently
    return cmp["id"], older, newer


def test_older_job_finishing_last_does_not_overwrite(api2: Api, gates: dict[str, Any], models_dir: Path) -> None:
    project = api2.post("/projects", json_body={"name": "p"}, expect=201)["id"]
    cid, older, newer = _two_jobs(api2, project, models_dir, gates)
    gates["verified"].set()
    assert finish(api2, newer)["result"]["published"] is True
    gates["best_effort"].set()
    late = finish(api2, older)
    assert late["state"] == "completed"
    assert late["result"]["published"] is False  # computed, but superseded by a later submission
    assert report(api2, cid)["job_id"] == newer


def test_newer_job_finishing_last_replaces(api2: Api, gates: dict[str, Any], models_dir: Path) -> None:
    project = api2.post("/projects", json_body={"name": "p"}, expect=201)["id"]
    cid, older, newer = _two_jobs(api2, project, models_dir, gates)
    gates["best_effort"].set()
    assert finish(api2, older)["result"]["published"] is True
    assert report(api2, cid)["job_id"] == older
    gates["verified"].set()
    assert finish(api2, newer)["result"]["published"] is True
    assert report(api2, cid)["job_id"] == newer


@pytest.mark.parametrize("how", ["fail", "cancel"])
def test_failed_or_canceled_newer_job_keeps_the_older_report(
    api2: Api, gates: dict[str, Any], models_dir: Path, how: str
) -> None:
    project = api2.post("/projects", json_body={"name": "p"}, expect=201)["id"]
    cid, older, newer = _two_jobs(api2, project, models_dir, gates)
    gates["best_effort"].set()
    assert finish(api2, older)["result"]["published"] is True
    if how == "fail":
        gates["verified"].fail = True  # type: ignore[attr-defined]
        gates["verified"].set()
    else:
        threading.Timer(0.3, gates["verified"].set).start()
        api2.post("/jobs/{job_id}/cancel", job_id=newer, expect=200)
    end = finish(api2, newer)
    assert end["state"] == ("failed" if how == "fail" else "canceled")
    assert end["result"] is None
    assert report(api2, cid)["job_id"] == older
