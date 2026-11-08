"""Bounded background jobs (doc 03 section 7, doc 09 section 5, doc 11 section 6) on the existing `jobs` table.

States: created -> queued -> running -> completed | failed | canceled; queued -> canceled; running -> interrupted
when the app stops or restarts. `result` is written only for completed jobs; failed jobs carry the doc 09 error
object; canceled and interrupted jobs carry neither a result nor a partial one.

* Workers: `max_heavy_jobs` threads (default 1), started with the app and joined when it stops; nothing outlives
  the runner. A queued job starts only when its memory reservation fits: the sum of running reservations stays
  within `memory_budget_mib`.
* Queue: at most `MAX_PENDING` created/queued jobs; more is 429 RATE_LIMITED.
* Deduplication: identical pending jobs (same project, kind and canonical identity) share one job; the second and
  later submitters get it with `deduplicated: true`.
* Cancellation is cooperative between bounded chunks (a few MiB of reads), so its latency is one chunk; canceling a
  queued job is immediate; cancel is idempotent.
* Budgets: the job's wall time is min(budget.max_wall_seconds, max_job_seconds); exceeding it fails the job with
  RESOURCE_LIMIT at the next chunk. Progress is completed work / estimated work, monotonic, and 1.0 only when
  completed.
* Restart: rows left `running` by a previous process are marked `interrupted`; `created`/`queued` rows are queued
  again (their spec is self-contained).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from weightlab.core.errors import ErrorCode, WeightLabError, fail
from weightlab.db import Database
from weightlab.db.schema import now
from weightlab.db.store import dumps, new_id, transaction

MIB = 1024 * 1024
MAX_PENDING = 16
PROGRESS_INTERVAL_S = 0.2
POLL_AFTER_MS = 500
PENDING = ("created", "queued", "running")
TERMINAL = ("completed", "failed", "canceled", "interrupted")


class Canceled(Exception):
    """The job was canceled (or the runner is stopping); nothing is published."""

    def __init__(self, interrupted: bool = False) -> None:
        super().__init__("canceled")
        self.interrupted = interrupted


class JobContext:
    """Handed to a job's work function: progress, cancellation and the wall-time deadline, checked per chunk."""

    def __init__(self, runner: JobRunner, job_id: str, total_bytes: int, deadline: float) -> None:
        self._runner = runner
        self.job_id = job_id
        self.total = max(1, total_bytes)
        self.done = 0
        self.deadline = deadline
        self.cancel = threading.Event()
        self.interrupt = False
        self._last = 0.0

    def tick(self, nbytes: int) -> None:
        self.done += nbytes
        self.check()
        t = time.monotonic()
        if t - self._last >= PROGRESS_INTERVAL_S:
            self._last = t
            self._runner._progress(self.job_id, min(self.done / self.total, 0.999), self.done)

    def check(self) -> None:
        if self.cancel.is_set():
            raise Canceled(self.interrupt)
        if time.monotonic() > self.deadline:
            raise fail(ErrorCode.RESOURCE_LIMIT, "the job exceeded its wall-time budget (max_wall_seconds)")


class JobKind(Protocol):
    def run(self, ctx: JobContext, spec: dict[str, Any]) -> dict[str, Any]:
        """Do the work and publish the result atomically; return the typed JobResult. Raise to fail."""
        ...


@dataclass
class _Pending:
    job_id: str
    project_id: str
    kind: str
    dedup_key: str
    memory_bytes: int
    total_bytes: int
    wall_seconds: float
    spec: dict[str, Any]
    ctx: JobContext | None = None
    done: threading.Event = field(default_factory=threading.Event)


class JobRunner:
    def __init__(
        self,
        db: Database,
        settings: Callable[[], dict[str, Any]],
        kinds: dict[str, JobKind],
    ) -> None:
        self.db = db
        self.settings = settings
        self.kinds = kinds
        self._cv = threading.Condition()
        self._queue: list[_Pending] = []
        self._running: dict[str, _Pending] = {}
        self._by_key: dict[str, _Pending] = {}
        self._threads: list[threading.Thread] = []
        self._stopping = False
        self._started = False

    # -- lifecycle ------------------------------------------------------------------------------------
    def start(self) -> None:
        with self.db.session() as conn, transaction(conn):
            conn.execute("UPDATE jobs SET state = 'interrupted', finished_at = ? WHERE state = 'running'", (now(),))
            rows = conn.execute(
                "SELECT id, project_id, kind, spec_json FROM jobs WHERE state IN ('created', 'queued') "
                "ORDER BY created_at, id"
            ).fetchall()
            for r in rows:
                spec = json.loads(r["spec_json"])
                p = self._pending_from(r["id"], r["project_id"], r["kind"], spec)
                self._queue.append(p)
                self._by_key[p.dedup_key] = p
                conn.execute("UPDATE jobs SET state = 'queued' WHERE id = ?", (r["id"],))
        n = max(1, int(self.settings()["max_heavy_jobs"]))
        self._started = True
        for i in range(n):
            t = threading.Thread(target=self._worker, name=f"weightlab-job-{i}", daemon=False)
            t.start()
            self._threads.append(t)

    def shutdown(self, timeout: float = 30.0) -> None:
        """Stop workers: running jobs are interrupted at their next chunk; queued ones stay queued in the database
        and are queued again on the next start. Returns after every worker thread has exited."""
        with self._cv:
            self._stopping = True
            for p in self._running.values():
                if p.ctx is not None:
                    p.ctx.interrupt = True
                    p.ctx.cancel.set()
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout)
        self._threads = [t for t in self._threads if t.is_alive()]

    @property
    def alive_workers(self) -> int:
        return sum(t.is_alive() for t in self._threads)

    # -- submission -----------------------------------------------------------------------------------
    @staticmethod
    def _pending_from(job_id: str, project_id: str, kind: str, spec: dict[str, Any]) -> _Pending:
        return _Pending(
            job_id, project_id, kind, spec["dedup_key"], int(spec["memory_bytes"]), int(spec["estimated_bytes"]),
            float(spec["wall_seconds"]), spec,
        )  # fmt: skip

    def submit(self, project_id: str, kind: str, spec: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Queue a job, or join the identical pending one. `spec` must hold dedup_key, memory_bytes,
        estimated_bytes and wall_seconds besides what the kind needs. Returns (job view, deduplicated)."""
        if kind not in self.kinds:
            raise fail(ErrorCode.VALIDATION_ERROR, f"job kind {kind} is not available")
        budget = int(self.settings()["memory_budget_mib"]) * MIB
        if int(spec["memory_bytes"]) > budget:
            raise fail(
                ErrorCode.RESOURCE_LIMIT,
                f"the job needs ~{int(spec['memory_bytes']) // MIB} MiB of working memory, above "
                f"memory_budget_mib ({budget // MIB} MiB)",
            )
        with self._cv:
            if self._stopping:
                raise fail(ErrorCode.RATE_LIMITED, "the server is stopping")
            existing = self._by_key.get(spec["dedup_key"])
            if existing is not None:
                view = self.get(existing.job_id)
                if view is not None and view["state"] in PENDING:
                    return {**view, "deduplicated": True}, True
            if len(self._queue) >= MAX_PENDING:
                raise fail(ErrorCode.RATE_LIMITED, f"the job queue is full ({MAX_PENDING} pending jobs)")
            job_id = new_id("job")
            with self.db.session() as conn:
                with transaction(conn):
                    conn.execute(
                        "INSERT INTO jobs (id, project_id, kind, spec_json, state, progress, created_at) "
                        "VALUES (?, ?, ?, ?, 'created', 0, ?)",
                        (job_id, project_id, kind, dumps(spec), now()),
                    )
                with transaction(conn):
                    conn.execute("UPDATE jobs SET state = 'queued' WHERE id = ?", (job_id,))
            p = self._pending_from(job_id, project_id, kind, spec)
            self._queue.append(p)
            self._by_key[p.dedup_key] = p
            self._cv.notify_all()
        view = self.get(job_id)
        assert view is not None
        return {**view, "deduplicated": False}, False

    def find_pending(self, dedup_key: str) -> dict[str, Any] | None:
        with self._cv:
            p = self._by_key.get(dedup_key)
        if p is None:
            return None
        view = self.get(p.job_id)
        return view if view is not None and view["state"] in PENDING else None

    # -- inspection and cancellation ---------------------------------------------------------------
    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.db.session() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return None if row is None else job_view(row)

    def cancel(self, job_id: str, wait_s: float = 2.0) -> dict[str, Any] | None:
        """Idempotent. A queued job is canceled at once; a running job at its next chunk (waits up to `wait_s`)."""
        target: _Pending | None = None
        with self._cv:
            for p in self._queue:
                if p.job_id == job_id:
                    self._queue.remove(p)
                    self._forget(p)
                    self._finish(p, "canceled")
                    break
            else:
                target = self._running.get(job_id)
                if target is not None and target.ctx is not None:
                    target.ctx.cancel.set()
        if target is not None:
            target.done.wait(wait_s)
        return self.get(job_id)

    def cancel_project(self, project_id: str, wait_s: float = 10.0) -> None:
        """Before a project is deleted: no job of it may keep running or start."""
        waiting: list[_Pending] = []
        with self._cv:
            for p in [p for p in self._queue if p.project_id == project_id]:
                self._queue.remove(p)
                self._forget(p)
                self._finish(p, "canceled")
            for p in self._running.values():
                if p.project_id == project_id and p.ctx is not None:
                    p.ctx.cancel.set()
                    waiting.append(p)
        for p in waiting:
            p.done.wait(wait_s)

    # -- workers --------------------------------------------------------------------------------------
    def _admissible(self) -> _Pending | None:
        budget = int(self.settings()["memory_budget_mib"]) * MIB
        used = sum(p.memory_bytes for p in self._running.values())
        for p in self._queue:
            if used + p.memory_bytes <= budget:
                return p
        return None

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._stopping and (p := self._admissible()) is None:
                    self._cv.wait(0.5)
                if self._stopping:
                    return
                assert p is not None
                self._queue.remove(p)
                limit = min(p.wall_seconds, float(self.settings()["max_job_seconds"]))
                p.ctx = JobContext(self, p.job_id, p.total_bytes, time.monotonic() + limit)
                self._running[p.job_id] = p
            self._run(p)

    def _run(self, p: _Pending) -> None:
        assert p.ctx is not None
        with self.db.session() as conn, transaction(conn):
            conn.execute("UPDATE jobs SET state = 'running', started_at = ? WHERE id = ?", (now(), p.job_id))
        state, result, error = "failed", None, None
        try:
            result = self.kinds[p.kind].run(p.ctx, p.spec)
            state = "completed"
        except Canceled as c:
            state = "interrupted" if c.interrupted else "canceled"
        except WeightLabError as e:
            error = {"code": e.code.value, "message": e.detail[:2000], "retryable": e.code is ErrorCode.RATE_LIMITED}
        except Exception:  # never a traceback to the client; the job fails closed
            error = {"code": ErrorCode.CONFLICT.value, "message": "internal error; the job did not complete",
                     "retryable": False}  # fmt: skip
        with self._cv:
            self._running.pop(p.job_id, None)
            self._forget(p)
            self._finish(p, state, result=result, error=error, done=p.ctx.done)
            self._cv.notify_all()
        p.done.set()

    def _forget(self, p: _Pending) -> None:
        if self._by_key.get(p.dedup_key) is p:
            del self._by_key[p.dedup_key]

    def _finish(
        self,
        p: _Pending,
        state: str,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        done: int | None = None,
    ) -> None:
        with self.db.session() as conn, transaction(conn):
            row = conn.execute("SELECT spec_json FROM jobs WHERE id = ?", (p.job_id,)).fetchone()
            if row is None:  # its project was deleted meanwhile
                return
            spec = json.loads(row["spec_json"])
            if done is not None:
                spec["bytes_scanned"] = done
            conn.execute(
                "UPDATE jobs SET state = ?, progress = ?, result_json = ?, error_json = ?, finished_at = ?, "
                "spec_json = ? WHERE id = ?",
                (state, 1.0 if state == "completed" else _progress_of(conn, p.job_id),
                 dumps(result) if state == "completed" and result is not None else None,
                 dumps(error) if state == "failed" and error is not None else None,
                 now(), dumps(spec), p.job_id),
            )  # fmt: skip

    def _progress(self, job_id: str, progress: float, done: int) -> None:
        with contextlib.suppress(sqlite3.Error), self.db.session() as conn, transaction(conn):
            row = conn.execute("SELECT spec_json, progress FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return
            spec = json.loads(row["spec_json"])
            spec["bytes_scanned"] = done
            conn.execute(
                "UPDATE jobs SET progress = ?, spec_json = ? WHERE id = ? AND state = 'running'",
                (max(float(row["progress"]), progress), dumps(spec), job_id),
            )


def _progress_of(conn: sqlite3.Connection, job_id: str) -> float:
    row = conn.execute("SELECT progress FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return float(row["progress"]) if row else 0.0


def job_view(row: sqlite3.Row) -> dict[str, Any]:
    """The contract `Job` object of one row."""
    spec = json.loads(row["spec_json"])
    state = row["state"]
    out: dict[str, Any] = {
        "job_id": row["id"],
        "kind": row["kind"],
        "state": state,
        "progress": float(row["progress"]),
        "estimated_bytes": int(spec.get("estimated_bytes", 0)),
        "bytes_scanned": int(spec.get("bytes_scanned", 0)),
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "result": json.loads(row["result_json"]) if state == "completed" and row["result_json"] else None,
        "error": json.loads(row["error_json"]) if state == "failed" and row["error_json"] else None,
    }
    if state in PENDING:
        out["poll_after_ms"] = POLL_AFTER_MS
    return out
