"""The bounded job runner (doc 03 section 7, doc 09 section 5) with a controllable job kind: state machine,
deduplication, cancellation before and during work (with measured latency), restart, timeouts, memory admission,
queue capacity, failures, progress, and shutdown without stray threads."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.api.contract import component
from weightlab.api.config import DEFAULT_SETTINGS
from weightlab.core.errors import ErrorCode, WeightLabError, fail
from weightlab.db import Database, store
from weightlab.jobs import MAX_PENDING, JobContext, JobRunner

MIB = 1024 * 1024


class Scan:
    """A fake scan of `chunks` chunks; `gate` holds it before the first chunk until set."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.gate.set()
        self.started: list[str] = []
        self.chunk_s = 0.005

    def run(self, ctx: JobContext, spec: dict[str, Any]) -> dict[str, Any]:
        self.started.append(ctx.job_id)
        while not self.gate.wait(0.01):
            ctx.check()
        for _ in range(int(spec["chunks"])):
            time.sleep(self.chunk_s)
            ctx.tick(1)
        if spec.get("fail"):
            raise fail(ErrorCode.FILE_CHANGED, "the source changed")
        if spec.get("crash"):
            raise RuntimeError("secret internal detail")
        return {"kind": "pyramid_build", "pyramid_id": "f" * 64}


def spec(key: str, chunks: int = 5, mem_mib: int = 64, wall: float = 60, **extra: Any) -> dict[str, Any]:
    return {"dedup_key": key, "memory_bytes": mem_mib * MIB, "estimated_bytes": chunks, "wall_seconds": wall,
            "chunks": chunks, **extra}  # fmt: skip


@pytest.fixture
def env(tmp_path: Path) -> Iterator[tuple[JobRunner, Scan, Database, str]]:
    db = Database(tmp_path / "db.sqlite3")
    with db.session() as conn:
        pid = store.create_project(conn, "p")["id"]
    settings = dict(DEFAULT_SETTINGS)
    scan = Scan()
    runner = JobRunner(db, lambda: settings, {"pyramid_build": scan})
    runner.start()
    yield runner, scan, db, pid
    scan.gate.set()
    runner.shutdown()
    assert runner.alive_workers == 0


def wait(
    runner: JobRunner, job_id: str, states: tuple[str, ...] = ("completed", "failed", "canceled"), t: float = 10
) -> dict[str, Any]:
    end = time.monotonic() + t
    while time.monotonic() < end:
        j = runner.get(job_id)
        assert j is not None
        if j["state"] in states:
            return j
        time.sleep(0.005)
    raise AssertionError(f"job {job_id} stayed {runner.get(job_id)}")


def valid(job: dict[str, Any]) -> dict[str, Any]:
    component("Job").validate(job)
    return job


def test_lifecycle_to_completed_with_typed_result_and_monotonic_progress(env: Any) -> None:
    runner, scan, _, pid = env
    scan.chunk_s = 0.05
    job, dedup = runner.submit(pid, "pyramid_build", spec("k1", chunks=10))
    assert not dedup
    assert valid(job)["state"] in ("queued", "running")
    seen = []
    while (j := runner.get(job["job_id"]))["state"] != "completed":
        seen.append(j["progress"])
        valid(j)
        time.sleep(0.01)
    assert seen == sorted(seen)
    assert all(p < 1 for p in seen)
    done = valid(j)
    assert done["progress"] == 1.0
    assert done["result"] == {"kind": "pyramid_build", "pyramid_id": "f" * 64}
    assert done["error"] is None
    assert done["bytes_scanned"] == 10
    assert done["started_at"]
    assert done["finished_at"]
    assert "poll_after_ms" not in done


def test_concurrent_identical_requests_share_one_job(env: Any) -> None:
    runner, scan, _, pid = env
    scan.gate.clear()
    out: list[tuple[dict[str, Any], bool]] = []
    threads = [threading.Thread(target=lambda: out.append(runner.submit(pid, "pyramid_build", spec("same"))))
               for _ in range(12)]  # fmt: skip
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ids = {j["job_id"] for j, _ in out}
    assert len(ids) == 1
    assert sorted(d for _, d in out) == [False] + [True] * 11
    assert all(valid(j)["deduplicated"] == d for j, d in out)
    other, d = runner.submit(pid, "pyramid_build", spec("different"))
    assert not d
    assert other["job_id"] not in ids
    scan.gate.set()
    wait(runner, ids.pop())
    _again, d = runner.submit(pid, "pyramid_build", spec("same"))  # a finished job is not joined
    assert not d


def test_cancel_before_execution_never_runs_the_job(env: Any) -> None:
    runner, scan, _, pid = env
    scan.gate.clear()
    first, _ = runner.submit(pid, "pyramid_build", spec("a"))
    wait(runner, first["job_id"], ("running",))
    queued, _ = runner.submit(pid, "pyramid_build", spec("b"))
    assert runner.get(queued["job_id"])["state"] == "queued"  # type: ignore[index]
    c = valid(runner.cancel(queued["job_id"]))  # type: ignore[arg-type]
    assert c["state"] == "canceled"
    assert c["result"] is None
    assert runner.cancel(queued["job_id"])["state"] == "canceled"  # type: ignore[index]  # idempotent
    scan.gate.set()
    wait(runner, first["job_id"])
    time.sleep(0.1)
    assert queued["job_id"] not in scan.started


def test_cancel_during_an_active_scan_is_prompt_and_publishes_nothing(env: Any) -> None:
    runner, scan, _, pid = env
    scan.chunk_s = 0.02
    job, _ = runner.submit(pid, "pyramid_build", spec("long", chunks=10_000))
    wait(runner, job["job_id"], ("running",))
    time.sleep(0.1)
    t0 = time.monotonic()
    c = runner.cancel(job["job_id"])
    latency = time.monotonic() - t0
    assert c is not None
    assert c["state"] == "canceled"
    assert latency < 0.5  # one chunk (20 ms) plus bookkeeping
    assert c["result"] is None
    assert c["error"] is None
    assert 0 < c["progress"] < 1
    assert runner.cancel(job["job_id"])["state"] == "canceled"  # type: ignore[index]
    with pytest.raises(AssertionError):
        wait(runner, job["job_id"], ("completed",), t=0.3)


def test_timeout_fails_with_resource_limit(env: Any) -> None:
    runner, scan, _, pid = env
    scan.chunk_s = 0.02
    job, _ = runner.submit(pid, "pyramid_build", spec("slow", chunks=1000, wall=0.2))
    j = valid(wait(runner, job["job_id"]))
    assert j["state"] == "failed"
    assert j["error"]["code"] == "RESOURCE_LIMIT"
    assert j["result"] is None


def test_failures_carry_the_error_and_never_a_traceback(env: Any) -> None:
    runner, _, _, pid = env
    a, _ = runner.submit(pid, "pyramid_build", spec("f1", fail=True))
    b, _ = runner.submit(pid, "pyramid_build", spec("f2", crash=True))
    ja, jb = valid(wait(runner, a["job_id"])), valid(wait(runner, b["job_id"]))
    assert (ja["state"], ja["error"]["code"]) == ("failed", "FILE_CHANGED")
    assert jb["state"] == "failed"
    assert "secret" not in str(jb)
    assert ja["result"] is None and jb["result"] is None  # noqa: PT018


def test_memory_reservations_admit_jobs_one_at_a_time(env: Any) -> None:
    runner, _scan, _, pid = env
    with pytest.raises(WeightLabError) as e:
        runner.submit(pid, "pyramid_build", spec("huge", mem_mib=2048))  # above memory_budget_mib (1024)
    assert e.value.code is ErrorCode.RESOURCE_LIMIT


def test_queue_capacity_is_bounded(env: Any) -> None:
    runner, scan, _, pid = env
    scan.gate.clear()
    first, _ = runner.submit(pid, "pyramid_build", spec("q0"))
    wait(runner, first["job_id"], ("running",))
    for i in range(MAX_PENDING):
        runner.submit(pid, "pyramid_build", spec(f"q{i + 1}"))
    with pytest.raises(WeightLabError) as e:
        runner.submit(pid, "pyramid_build", spec("one too many"))
    assert e.value.code is ErrorCode.RATE_LIMITED


def test_restart_marks_running_jobs_interrupted_and_requeues_queued_ones(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    with db.session() as conn:
        pid = store.create_project(conn, "p")["id"]
    settings = dict(DEFAULT_SETTINGS)
    scan = Scan()
    scan.gate.clear()
    r1 = JobRunner(db, lambda: settings, {"pyramid_build": scan})
    r1.start()
    running, _ = r1.submit(pid, "pyramid_build", spec("r"))
    wait(r1, running["job_id"], ("running",))
    queued, _ = r1.submit(pid, "pyramid_build", spec("q"))
    # simulate a crash: the threads stop without the runner recording anything
    r1._stopping = True
    with r1._cv:
        r1._cv.notify_all()
    for p in r1._running.values():
        p.ctx.cancel.set()  # type: ignore[union-attr]
    for t in r1._threads:
        t.join(5)
    with db.session() as conn:  # whatever the old process managed to write, the row says running
        conn.execute("UPDATE jobs SET state = 'running', finished_at = NULL WHERE id = ?", (running["job_id"],))
    scan2 = Scan()
    r2 = JobRunner(db, lambda: settings, {"pyramid_build": scan2})
    r2.start()
    try:
        j = valid(r2.get(running["job_id"]))  # type: ignore[arg-type]
        assert j["state"] == "interrupted"
        assert j["result"] is None
        assert valid(wait(r2, queued["job_id"]))["state"] == "completed"  # queued work survives a restart
        again, d = r2.submit(pid, "pyramid_build", spec("r"))  # the interrupted job is not joined: a new one
        assert not d
        assert again["job_id"] != running["job_id"]
    finally:
        r2.shutdown()
    assert r2.alive_workers == 0


def test_shutdown_interrupts_running_jobs_and_joins_every_worker(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    with db.session() as conn:
        pid = store.create_project(conn, "p")["id"]
    scan = Scan()
    scan.chunk_s = 0.01
    r = JobRunner(db, lambda: dict(DEFAULT_SETTINGS), {"pyramid_build": scan})
    r.start()
    job, _ = r.submit(pid, "pyramid_build", spec("s", chunks=100_000))
    wait(r, job["job_id"], ("running",))
    before = threading.active_count()
    r.shutdown()
    assert r.alive_workers == 0
    assert threading.active_count() < before
    assert r.get(job["job_id"])["state"] == "interrupted"  # type: ignore[index]
    with pytest.raises(WeightLabError):
        r.submit(pid, "pyramid_build", spec("after"))


def test_project_cancellation_stops_its_jobs(env: Any) -> None:
    runner, scan, db, pid = env
    with db.session() as conn:
        other = store.create_project(conn, "q")["id"]
    scan.chunk_s = 0.01
    mine, _ = runner.submit(pid, "pyramid_build", spec("m", chunks=100_000))
    theirs, _ = runner.submit(other, "pyramid_build", spec("t", chunks=3))
    wait(runner, mine["job_id"], ("running",))
    runner.cancel_project(pid)
    assert runner.get(mine["job_id"])["state"] == "canceled"  # type: ignore[index]
    assert wait(runner, theirs["job_id"])["state"] == "completed"


def test_a_running_job_can_grow_its_reservation_only_within_both_budgets(env: Any) -> None:
    runner, _scan, _, pid = env
    seen: list[tuple[bool, bool, bool, int]] = []

    class Grow:
        def run(self, ctx: JobContext, spec: dict[str, Any]) -> dict[str, Any]:
            base = ctx.memory_bytes
            over_job = ctx.grow_memory(65 * MIB, 128 * MIB)  # 64 + 65 > the job's own 128 MiB
            over_all = ctx.grow_memory(961 * MIB, 2048 * MIB)  # 64 + 961 > memory_budget_mib (1024)
            ok = ctx.grow_memory(64 * MIB, 128 * MIB)
            seen.append((over_job, over_all, ok, ctx.memory_bytes - base))
            return {"kind": "pyramid_build", "pyramid_id": "f" * 64}

    runner.kinds["grow"] = Grow()
    job, _ = runner.submit(pid, "grow", spec("g"))
    assert wait(runner, job["job_id"])["state"] == "completed"
    assert seen == [(False, False, True, 64 * MIB)]
    assert not runner._running  # the grown reservation ended with the job
