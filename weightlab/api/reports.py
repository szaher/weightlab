"""The comparison_report job and GET /comparisons/{id}/report (ADR-020, contract 0.4.0).

A comparison (POST /comparisons) records structure and lineage only. This job computes its change report by
calling `change_report.pair_report`, the function behind `weightlab diff`, on the two imported models, so every
number, rollup, caveat and completeness flag is the CLI's by construction; nothing here computes a metric.

* Fail closed on drift: both models are reopened with their import fingerprints (FILE_CHANGED otherwise), and the
  report's structural counts and lineage must equal what the comparison recorded; a mismatch is CONFLICT and the
  user creates a new comparison. A report never silently describes different bytes than its comparison.
* Bounded: the diff streams one chunk per side (G0.5 bound: under REPORT_WORKING_SET of growth); progress and
  cancellation go through the job context at every chunk; the wall-time budget is enforced there too.
* Atomic: the report is published in one transaction only after the job's last check, replacing any earlier
  report of the comparison; canceled, failed and interrupted jobs leave the earlier report (if any) untouched.
* Private: paths are redacted exactly as the CLI does by default (`include_paths=False`).
"""

from __future__ import annotations

from typing import Any

from weightlab.analysis.change_report import Options, pair_report
from weightlab.analysis.diff import Cancelled, RunControl
from weightlab.analysis.lineage import LineageInputs
from weightlab.api.catalog import open_model
from weightlab.api.errors import ApiError
from weightlab.api.schemas import JobRequest
from weightlab.core.errors import ErrorCode, WeightLabError, fail
from weightlab.db import Database, store
from weightlab.formats.base import Consistency
from weightlab.formats.checkpoint import Checkpoint
from weightlab.jobs import MIB, Canceled, JobContext, JobRunner

KIND = "comparison_report"
REPORT_WORKING_SET = 128 * MIB  # tests/perf/test_diff_bounded.py: diff memory growth stays below this


def _attested(compat: dict[str, Any]) -> bool:
    """The attestation the comparison was created with. Rows written before ADR-020 did not store it; their
    lineage evidence says whether it was given (a verified pair that was also attested then reassesses
    differently and fails the drift check, so the user recreates the comparison: fail closed)."""
    if "user_attestation" in compat:
        return bool(compat["user_attestation"])
    return bool(compat["lineage"].get("lineage_evidence") == "user_attested")


def _capped_lineage(lineage: dict[str, Any]) -> dict[str, Any]:
    """The lineage as POST /comparisons stores it (relations.lineage_view caps the reasons)."""
    return {**lineage, "reasons": [r[:500] for r in list(lineage["reasons"])[:100]]}


def _open_pair(db: Database, comparison_id: str, project_id: str | None = None) -> tuple[dict[str, Any], Checkpoint,
                                                                                         Checkpoint]:  # fmt: skip
    with db.session() as conn:
        row = store.get_comparison(conn, comparison_id, project_id)
        if row is None:
            raise ApiError(ErrorCode.NOT_FOUND, "comparison not found")
        a = open_model(conn, row["model_a_id"], row["project_id"])
        try:
            b = open_model(conn, row["model_b_id"], row["project_id"])
        except BaseException:
            a.close()
            raise
    return row, a, b


def _analyzable_bytes(db: Database, comparison_id: str, a: Checkpoint, b: Checkpoint) -> int:
    with db.session() as conn:
        _, entries = store.list_entries(conn, comparison_id, (), limit=2**62, offset=0)
    ma = {m.key: m.byte_length for m in a.metadata()}
    mb = {m.key: m.byte_length for m in b.metadata()}
    keys = [e["raw_key"] for e in entries if e["detail_json"].get("analyzable")]
    return sum(ma.get(k, 0) + mb.get(k, 0) for k in keys)


def _file_bytes(ck: Checkpoint) -> int:
    return sum(f.size for f in ck.file_info())


def submit(db: Database, runner: JobRunner, req: JobRequest) -> tuple[dict[str, Any], bool]:
    """Validate and queue (or join) a comparison_report job; caps on `req.budget` are checked by the caller."""
    assert req.comparison_id is not None
    if req.budget.max_working_set_mib * MIB < REPORT_WORKING_SET:
        raise ApiError(
            ErrorCode.RESOURCE_LIMIT,
            f"a comparison report needs {REPORT_WORKING_SET // MIB} MiB of working memory, above the job budget "
            f"max_working_set_mib ({req.budget.max_working_set_mib} MiB)",
            status=422,
            details={"required_mib": REPORT_WORKING_SET // MIB},
        )
    row, a, b = _open_pair(db, req.comparison_id)  # FILE_CHANGED now rather than after queueing
    try:
        estimated = _analyzable_bytes(db, req.comparison_id, a, b)
        files = _file_bytes(a) + _file_bytes(b)
    finally:
        b.close()
        a.close()
    if req.consistency == "verified":
        estimated += 2 * files  # one hash of every input file before and one after the scan
    elif _attested(row["compatibility_json"]):
        estimated += files  # attestation names exact bytes: both sides are hashed once, as the CLI does
    from weightlab.api.tiles import dedup_key  # local: tiles imports this module

    spec = {
        "dedup_key": dedup_key(row["project_id"], KIND, {"comparison_id": row["id"], "consistency": req.consistency}),
        "memory_bytes": REPORT_WORKING_SET,
        "estimated_bytes": estimated,
        "wall_seconds": req.budget.max_wall_seconds,
        "budget": req.budget.model_dump(),
        "project_id": row["project_id"],
        "comparison_id": row["id"],
        "consistency": req.consistency,
    }
    return runner.submit(row["project_id"], KIND, spec)


class ComparisonReport:
    """The comparison_report job kind."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def run(self, ctx: JobContext, spec: dict[str, Any]) -> dict[str, Any]:
        cid = spec["comparison_id"]
        try:
            row, a, b = _open_pair(self.db, cid, spec["project_id"])
        except ApiError as e:
            raise WeightLabError(e.code, e.message) from None
        compat = row["compatibility_json"]
        last = 0

        def progress(done: int, _total: int, _key: str) -> None:
            nonlocal last
            ctx.tick(done - last)  # raises Canceled / RESOURCE_LIMIT between chunks
            last = done

        opts = Options(consistency=Consistency(spec["consistency"]))
        try:
            with a, b:
                ctx.check()
                report = pair_report(a, b, LineageInputs(user_attestation=_attested(compat)), opts,
                                     RunControl(progress=progress))  # fmt: skip
        except Cancelled:  # not raised with this control (its event is never set); fail closed if it ever is
            raise Canceled() from None
        if report["compatibility"]["counts"] != compat["counts"] or _capped_lineage(
            report["compatibility"]["lineage"]
        ) != _capped_lineage(compat["lineage"]):
            raise fail(
                ErrorCode.CONFLICT,
                "the models no longer match this comparison's recorded structure or lineage; create a new comparison",
            )
        ctx.check()  # a cancel that arrived during the last chunk still wins
        with self.db.session() as conn:
            store.put_comparison_report(conn, cid, ctx.job_id, report)
        return {"kind": KIND, "comparison_id": cid, "complete": bool(report["complete"])}


def report_view(row: dict[str, Any]) -> dict[str, Any]:
    """The contract `ComparisonReport` object of one stored report."""
    return {
        "comparison_id": row["comparison_id"],
        "job_id": row["job_id"],
        "complete": bool(row["complete"]),
        "created_at": row["created_at"],
        "report": row["report_json"],
    }
