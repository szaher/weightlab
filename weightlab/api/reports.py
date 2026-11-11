"""The comparison_report job and GET /comparisons/{id}/report (ADR-020, contract 0.4.0).

A comparison (POST /comparisons) records structure and lineage only. This job computes its change report by
calling `change_report.pair_report`, the function behind `weightlab diff`, on the two imported models, so every
number, rollup, caveat and completeness flag is the CLI's by construction; nothing here computes a metric.

* Records: only comparisons created under ADR-020 (record_version 2: `user_attestation` and `structure_sha256`
  recorded) can be reported. Older records are refused with CONFLICT ("recreate the comparison"): their attestation
  input is unknown and lineage trust is never reconstructed by guessing.
* Policy (FR-035): `strict` comparisons report Class V and L only; Class S and X need `descriptive_only`, the
  user's acknowledgement that the result is descriptive inspection of unverified coordinates (LINEAGE_UNVERIFIED).
* Fail closed on drift: both models are reopened with their import fingerprints (FILE_CHANGED); the canonical
  structure fingerprint (every key, status, shape, dtype and tie, plus the model ids) and the lineage must equal the
  comparison's record, else CONFLICT.
* Bounded: REPORT_WORKING_SET for the streaming diff plus PER_ENTRY_BYTES per comparison entry (measured: ~9 KiB
  peak per entry for report objects and JSON, ADR-020), at most MAX_REPORT_ENTRIES entries and MAX_REPORT_JSON_BYTES
  of stored JSON; larger work is refused before it is queued. Hashing and diff reads tick the job context per chunk
  (cancellation, wall time).
* Atomic and ordered: the report is written by the job's commit hook, inside the transaction that marks the job
  completed (weightlab/jobs.py). It replaces the stored report only if the job was submitted later (job_seq), so an
  earlier job finishing later never overwrites a newer report; its result then says `published: false`.
* Private: provenance carries file names only (as the CLI by default); every string of the report and every error
  message has the models' absolute roots and entry paths and the user's home directory replaced.
* Snapshot: a stored report describes the bytes read by its job. GET reads no tensor bytes; it reports each model's
  current quick-fingerprint status (stat only) so a client can show that files changed since the report.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi.responses import Response

from weightlab.analysis.change_report import Options, pair_report
from weightlab.analysis.compat import compare_structure, structure_fingerprint
from weightlab.analysis.diff import Cancelled, RunControl
from weightlab.analysis.lineage import LineageInputs
from weightlab.api.catalog import live_status, open_model
from weightlab.api.errors import ApiError
from weightlab.api.relations import COMPARISON_RECORD_VERSION, structure_identity
from weightlab.api.schemas import JobRequest
from weightlab.core.errors import ErrorCode, WeightLabError, fail
from weightlab.db import Database, store
from weightlab.db.store import dumps
from weightlab.formats.base import Consistency
from weightlab.formats.checkpoint import Checkpoint
from weightlab.jobs import MIB, Canceled, JobContext, JobRunner

KIND = "comparison_report"
REPORT_WORKING_SET = 128 * MIB  # tests/perf/test_diff_bounded.py: diff memory growth stays below this
PER_ENTRY_BYTES = 16 * 1024  # measured ~9 KiB peak per entry (report objects + JSON); 1.8x margin for the copies
MAX_REPORT_ENTRIES = 32_768  # ~640 MiB reservation, ~90 MiB JSON
MAX_REPORT_JSON_BYTES = 128 * MIB
UNVERIFIED_CLASSES = ("S", "X")
REDACTED = "<redacted path>"


# -- helpers -----------------------------------------------------------------------------------
def _sensitive(models: list[dict[str, Any]]) -> list[str]:
    """Absolute path strings that must not appear in a report or an error, longest first."""
    out: set[str] = set()
    for m in models:
        for p in (m["selected_root"], m["entry_path"]):
            out |= {str(p), os.path.realpath(p), str(Path(p).resolve())}
    home = os.path.expanduser("~")
    if home not in ("~", "/", ""):
        out |= {home, os.path.realpath(home)}
    return sorted((s for s in out if len(s) > 1), key=len, reverse=True)


def scrub(text: str, sensitive: list[str]) -> str:
    for s in sensitive:
        if s in text:
            text = text.replace(s, REDACTED)
    return text


def _scrub_obj(obj: Any, sensitive: list[str]) -> Any:
    if isinstance(obj, str):
        return scrub(obj, sensitive)
    if isinstance(obj, list):
        return [_scrub_obj(x, sensitive) for x in obj]
    if isinstance(obj, dict):
        return {scrub(k, sensitive): _scrub_obj(v, sensitive) for k, v in obj.items()}
    return obj


def _check_record(compat: dict[str, Any], policy_error: bool = True) -> None:
    if compat.get("record_version", 1) < COMPARISON_RECORD_VERSION:
        raise fail(
            ErrorCode.CONFLICT,
            "this comparison was created before comparison reports (ADR-020) and does not record its attestation and "
            "structure; create a new comparison of the same models to report it",
        )
    cls = compat["lineage"]["compatibility_class"]
    if policy_error and compat.get("policy", "strict") == "strict" and cls in UNVERIFIED_CLASSES:
        raise fail(
            ErrorCode.LINEAGE_UNVERIFIED,
            f"Class {cls} comparison with policy strict: a change report of unverified coordinates is descriptive "
            "inspection only (FR-035); create the comparison with policy descriptive_only to acknowledge that",
        )


def _comparison(conn: sqlite3.Connection, comparison_id: str, project_id: str | None) -> dict[str, Any]:
    row = store.get_comparison(conn, comparison_id, project_id)
    if row is None:
        raise ApiError(ErrorCode.NOT_FOUND, "comparison not found")
    return row


def _open_pair(db: Database, row: dict[str, Any]) -> tuple[Checkpoint, Checkpoint, list[str]]:
    with db.session() as conn:
        models = [store.get_model(conn, row[k], row["project_id"]) for k in ("model_a_id", "model_b_id")]
        if any(m is None for m in models):
            raise ApiError(ErrorCode.NOT_FOUND, "model not found")
        sensitive = _sensitive([m for m in models if m is not None])
        try:
            a = open_model(conn, row["model_a_id"], row["project_id"])
            try:
                b = open_model(conn, row["model_b_id"], row["project_id"])
            except BaseException:
                a.close()
                raise
        except ApiError as e:
            raise ApiError(e.code, scrub(e.message, sensitive), status=e.status) from None
        except WeightLabError as e:
            raise WeightLabError(e.code, scrub(e.detail, sensitive)) from None
    return a, b, sensitive


# -- submission --------------------------------------------------------------------------------
def submit(db: Database, runner: JobRunner, req: JobRequest) -> tuple[dict[str, Any], bool]:
    """Validate and queue (or join) a comparison_report job; caps on `req.budget` are checked by the caller."""
    assert req.comparison_id is not None
    with db.session() as conn:
        row = _comparison(conn, req.comparison_id, None)
        entries, analyzable = store.comparison_entry_stats(conn, row["id"], row["model_a_id"], row["model_b_id"])
        files = sum(int(f["size_bytes"]) for k in ("model_a_id", "model_b_id") for f in store.model_files(conn, row[k]))
    try:
        _check_record(row["compatibility_json"])
    except WeightLabError as e:
        raise ApiError(e.code, e.detail) from None
    if entries > MAX_REPORT_ENTRIES:
        raise ApiError(ErrorCode.RESOURCE_LIMIT, f"the comparison has {entries} entries; a report covers at most "
                       f"{MAX_REPORT_ENTRIES}", status=422, details={"entries": entries,
                                                                     "max_entries": MAX_REPORT_ENTRIES})  # fmt: skip
    memory = REPORT_WORKING_SET + entries * PER_ENTRY_BYTES
    if memory > req.budget.max_working_set_mib * MIB:
        need = -(-memory // MIB)
        raise ApiError(
            ErrorCode.RESOURCE_LIMIT,
            f"a report of {entries} entries needs ~{need} MiB of working memory, above the job budget "
            f"max_working_set_mib ({req.budget.max_working_set_mib} MiB)",
            status=422,
            details={"required_mib": need},
        )
    a, b, _ = _open_pair(db, row)  # FILE_CHANGED now rather than after queueing
    a.close()
    b.close()
    # An upper bound: verified hashes every input file before and after the scan; otherwise lineage assessment
    # hashes each file at most once (attestation, generator fixtures, hub-cache blobs; digests are cached).
    estimated = analyzable + files * (2 if req.consistency == "verified" else 1)
    from weightlab.api.tiles import dedup_key  # local: tiles imports this module

    with db.session() as conn:
        seq = store.next_report_seq(conn)
    spec = {
        "dedup_key": dedup_key(row["project_id"], KIND, {"comparison_id": row["id"], "consistency": req.consistency}),
        "memory_bytes": memory,
        "estimated_bytes": estimated,
        "wall_seconds": req.budget.max_wall_seconds,
        "budget": req.budget.model_dump(),
        "project_id": row["project_id"],
        "comparison_id": row["id"],
        "consistency": req.consistency,
        "job_seq": seq,  # a deduplicated submission joins the earlier job and keeps that job's (earlier) number
    }
    return runner.submit(row["project_id"], KIND, spec)


# -- the job -----------------------------------------------------------------------------------
class ComparisonReport:
    """The comparison_report job kind."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def run(self, ctx: JobContext, spec: dict[str, Any]) -> dict[str, Any]:
        cid = spec["comparison_id"]
        try:
            with self.db.session() as conn:
                row = _comparison(conn, cid, spec["project_id"])
            compat = row["compatibility_json"]
            _check_record(compat)
            a, b, sensitive = _open_pair(self.db, row)
        except ApiError as e:
            raise WeightLabError(e.code, e.message) from None
        last = 0

        def progress(done: int, _total: int, _key: str) -> None:
            nonlocal last
            ctx.tick(done - last)  # raises Canceled / RESOURCE_LIMIT between chunks
            last = done

        consistency = spec["consistency"]
        try:
            with a, b:
                a.hash_tick = b.hash_tick = ctx.tick  # every hash chunk (lineage, attestation, verified) ticks too
                ctx.check()
                identity = structure_identity(row["model_a_id"], row["model_b_id"])
                if structure_fingerprint(compare_structure(a, b), identity) != compat["structure_sha256"]:
                    raise fail(ErrorCode.CONFLICT, "the models' tensor structure no longer matches this comparison's "
                                                   "record; create a new comparison")  # fmt: skip
                lin = LineageInputs(user_attestation=bool(compat["user_attestation"]))
                opts = Options(consistency=Consistency(consistency))
                report = pair_report(a, b, lin, opts, RunControl(progress=progress))
        except Cancelled:  # not raised with this control (its event is never set); fail closed if it ever is
            raise Canceled() from None
        except WeightLabError as e:
            raise WeightLabError(e.code, scrub(e.detail, sensitive)) from None
        if report["compatibility"]["counts"] != compat["counts"] or _capped_lineage(
            report["compatibility"]["lineage"]
        ) != _capped_lineage(compat["lineage"]):
            raise fail(ErrorCode.CONFLICT, "the models' lineage assessment no longer matches this comparison's "
                                           "record; create a new comparison")  # fmt: skip
        text = dumps(_scrub_obj(report, sensitive))
        if len(text.encode("utf-8")) > MAX_REPORT_JSON_BYTES:
            raise fail(ErrorCode.RESOURCE_LIMIT, f"the report exceeds {MAX_REPORT_JSON_BYTES // MIB} MiB of JSON")
        complete = bool(report["complete"])
        del report

        def publish(conn: sqlite3.Connection) -> dict[str, Any]:
            if conn.execute("SELECT 1 FROM comparisons WHERE id = ?", (cid,)).fetchone() is None:
                raise fail(ErrorCode.NOT_FOUND, "the comparison was deleted while its report was computed")
            published = store.publish_comparison_report(conn, cid, ctx.job_id, int(spec["job_seq"]), consistency,
                                                        complete, text)  # fmt: skip
            return {"kind": KIND, "comparison_id": cid, "complete": complete, "published": published}

        ctx.on_commit(publish)
        return {"kind": KIND, "comparison_id": cid, "complete": complete, "published": False}  # replaced by publish


def _capped_lineage(lineage: dict[str, Any]) -> dict[str, Any]:
    """The lineage as POST /comparisons stores it (relations.lineage_view caps the reasons)."""
    return {**lineage, "reasons": [r[:500] for r in list(lineage["reasons"])[:100]]}


# -- GET /comparisons/{id}/report --------------------------------------------------------------
def report_response(db: Database, comparison_id: str) -> Response:
    """The contract `ComparisonReport`, assembled around the stored JSON text without re-parsing it."""
    with db.session() as conn:
        row = _comparison(conn, comparison_id, None)
        rep = store.get_comparison_report(conn, comparison_id)
        if rep is None:
            raise ApiError(ErrorCode.NOT_FOUND, "no completed comparison_report job for this comparison")
        status = {}
        for side, key in (("model_a", "model_a_id"), ("model_b", "model_b_id")):
            m = store.get_model(conn, row[key], row["project_id"])
            status[side] = "disconnected" if m is None else live_status(conn, m)
    head = {
        "comparison_id": comparison_id,
        "job_id": rep["job_id"],
        "complete": bool(rep["complete"]),
        "consistency": rep["consistency"],
        "created_at": rep["created_at"],
        "current_model_status": status,
    }
    body = dumps(head)[:-1] + ',"report":' + rep["report_json"] + "}"
    return Response(body.encode("utf-8"), media_type="application/json")
