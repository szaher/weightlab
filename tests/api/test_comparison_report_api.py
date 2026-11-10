"""comparison_report jobs and GET /comparisons/{id}/report (ADR-020, contract 0.4.0).

The acceptance criterion of ROADMAP P0.1: the API's report equals `weightlab diff`'s (same rankings, values,
compatibility, completeness) on the deterministic fixtures, and an incomplete report is never shown as complete.
Every response is validated against the contract by the `Api` client; the report body additionally against
weightlab/schemas/change_report.schema.json.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from tests.api.conftest import import_model
from tests.api.contract import Api
from tests.helpers import ROOT
from weightlab.analysis.change_report import Options, run
from weightlab.api.reports import ComparisonReport
from weightlab.db import Database, store
from weightlab.jobs import Canceled, JobContext

BUDGET = {"max_working_set_mib": 512, "max_threads": 1, "max_wall_seconds": 120}
SCHEMA = json.loads((ROOT / "weightlab/schemas/change_report.schema.json").read_text(encoding="utf-8"))


def _compare(api: Api, project: str, root: Path, a: str, b: str, **extra: Any) -> dict[str, Any]:
    ma = import_model(api, project, root, a)
    mb = import_model(api, project, root, b)
    body = {"project_id": project, "model_a_id": ma["id"], "model_b_id": mb["id"], **extra}
    return api.post("/comparisons", json_body=body, expect=201)  # type: ignore[no-any-return]


def _submit(api: Api, comparison_id: str, expect: int = 202, **extra: Any) -> Any:
    body = {"kind": "comparison_report", "comparison_id": comparison_id, "budget": BUDGET, **extra}
    return api.post("/jobs", json_body=body, expect=expect)


def _finish(api: Api, job_id: str, t: float = 60) -> dict[str, Any]:
    end = time.monotonic() + t
    while time.monotonic() < end:
        j = api.get("/jobs/{job_id}", job_id=job_id, expect=200)
        if j["state"] not in ("created", "queued", "running"):
            return j  # type: ignore[no-any-return]
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def _without_time(report: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(report))
    out["provenance"]["created_at"] = "T"
    return out  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("a", "b", "complete", "cls"),
    [
        ("tiny_base.safetensors", "tiny_ft.safetensors", True, "V"),
        ("tiny_precision_a.safetensors", "tiny_precision_b.safetensors", False, "S"),
    ],
)
def test_report_equals_the_cli_report(
    api: Api, project: str, models_dir: Path, a: str, b: str, complete: bool, cls: str
) -> None:
    cmp = _compare(api, project, models_dir, a, b)
    api.get("/comparisons/{comparison_id}/report", comparison_id=cmp["id"], expect=404)  # none yet
    job = _submit(api, cmp["id"])
    assert job["kind"] == "comparison_report"
    assert job["estimated_bytes"] > 0
    done = _finish(api, job["job_id"])
    assert done["state"] == "completed", done
    assert done["result"] == {"kind": "comparison_report", "comparison_id": cmp["id"], "complete": complete}
    assert done["progress"] == 1.0

    got = api.get("/comparisons/{comparison_id}/report", comparison_id=cmp["id"], expect=200)
    assert got["job_id"] == done["job_id"]
    assert got["complete"] is complete
    report = got["report"]
    jsonschema.validate(report, SCHEMA)
    assert report["complete"] is complete  # an incomplete report is labeled incomplete, never complete
    assert report["compatibility"]["lineage"]["compatibility_class"] == cls
    assert report["compatibility"]["lineage"] == cmp["lineage"]
    assert report["compatibility"]["counts"] == cmp["counts"]
    assert report["provenance"]["privacy"]["paths_included"] is False
    assert str(models_dir) not in json.dumps(report)  # no local path leaks into the stored report

    cli = run(models_dir / a, models_dir / b, selected_root=models_dir, options=Options())
    assert _without_time(report) == _without_time(cli)  # every rank, metric, rollup and caveat


def test_attested_comparison_reports_with_the_same_lineage(api: Api, project: str, models_dir: Path) -> None:
    cmp = _compare(
        api, project, models_dir, "tiny_precision_a.safetensors", "tiny_precision_b.safetensors", user_attestation=True
    )
    assert cmp["lineage"]["lineage_evidence"] == "user_attested"
    done = _finish(api, _submit(api, cmp["id"])["job_id"])
    assert done["state"] == "completed", done
    got = api.get("/comparisons/{comparison_id}/report", comparison_id=cmp["id"], expect=200)
    assert got["report"]["compatibility"]["lineage"] == cmp["lineage"]


def test_verified_consistency_hashes_inputs(api: Api, project: str, models_dir: Path) -> None:
    cmp = _compare(api, project, models_dir, "tiny_base.safetensors", "tiny_ft.safetensors")
    plain = _finish(api, _submit(api, cmp["id"])["job_id"])
    verified = _finish(api, _submit(api, cmp["id"], consistency="verified")["job_id"])
    assert verified["state"] == "completed"
    files = sum(p.stat().st_size for p in (models_dir / "tiny_base.safetensors", models_dir / "tiny_ft.safetensors"))
    assert verified["estimated_bytes"] == plain["estimated_bytes"] + 2 * files
    got = api.get("/comparisons/{comparison_id}/report", comparison_id=cmp["id"], expect=200)
    assert got["job_id"] == verified["job_id"]  # the newer completed report replaced the earlier one
    assert got["report"]["settings"]["consistency"] == "verified"


def test_request_validation(api: Api, project: str, models_dir: Path) -> None:
    cmp = _compare(api, project, models_dir, "tiny_base.safetensors", "tiny_ft.safetensors")
    src = {"kind": "tensor", "model_id": cmp["model_a_id"], "tensor_id": "tns_x"}
    api.post("/jobs", json_body={"kind": "comparison_report", "budget": BUDGET}, expect=422)
    _submit(api, cmp["id"], expect=422, source=src)
    _submit(api, cmp["id"], expect=422, model_id=cmp["model_a_id"])
    e = _submit(api, cmp["id"], expect=422, budget={**BUDGET, "max_working_set_mib": 64})
    assert e["error"]["code"] == "RESOURCE_LIMIT"
    assert e["error"]["details"]["required_mib"] == 128
    e = _submit(api, "cmp_doesnotexist", expect=404)
    assert e["error"]["code"] == "NOT_FOUND"
    api.get("/comparisons/{comparison_id}/report", comparison_id="cmp_doesnotexist", expect=404)


def test_changed_model_is_refused(api: Api, project: str, models_dir: Path) -> None:
    cmp = _compare(api, project, models_dir, "tiny_base.safetensors", "tiny_ft.safetensors")
    ft = models_dir / "tiny_ft.safetensors"
    data = bytearray(ft.read_bytes())
    data[-1] ^= 0xFF
    ft.write_bytes(bytes(data))
    e = _submit(api, cmp["id"], expect=409)
    assert e["error"]["code"] == "FILE_CHANGED"
    api.get("/comparisons/{comparison_id}/report", comparison_id=cmp["id"], expect=404)


def test_drift_from_the_recorded_comparison_fails_closed(api: Api, project: str, models_dir: Path, config: Any) -> None:
    cmp = _compare(api, project, models_dir, "tiny_base.safetensors", "tiny_ft.safetensors")
    with sqlite3.connect(config.data_dir / "weightlab.sqlite3") as conn:  # simulate a stale record
        compat = json.loads(conn.execute("SELECT compatibility_json FROM comparisons WHERE id = ?",
                                         (cmp["id"],)).fetchone()[0])  # fmt: skip
        compat["counts"]["matched"] += 1
        conn.execute("UPDATE comparisons SET compatibility_json = ? WHERE id = ?", (json.dumps(compat), cmp["id"]))
    done = _finish(api, _submit(api, cmp["id"])["job_id"])
    assert done["state"] == "failed"
    assert done["error"]["code"] == "CONFLICT"
    assert done["result"] is None
    api.get("/comparisons/{comparison_id}/report", comparison_id=cmp["id"], expect=404)


def test_deleting_the_project_deletes_its_reports(api: Api, project: str, models_dir: Path, config: Any) -> None:
    cmp = _compare(api, project, models_dir, "tiny_base.safetensors", "tiny_ft.safetensors")
    assert _finish(api, _submit(api, cmp["id"])["job_id"])["state"] == "completed"
    api.delete("/projects/{project_id}", project_id=project, expect=204)
    with sqlite3.connect(config.data_dir / "weightlab.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM comparison_reports").fetchone()[0] == 0


class _Runner:
    def _progress(self, *_: Any) -> None:
        pass


def test_cancel_publishes_nothing_and_keeps_the_earlier_report(
    api: Api, project: str, models_dir: Path, config: Any
) -> None:
    cmp = _compare(api, project, models_dir, "tiny_base.safetensors", "tiny_ft.safetensors")
    first = _finish(api, _submit(api, cmp["id"])["job_id"])
    db = Database(config.data_dir / "weightlab.sqlite3")
    spec = {"comparison_id": cmp["id"], "project_id": project, "consistency": "best_effort"}
    ctx = JobContext(_Runner(), "job_canceled", 1, time.monotonic() + 60)  # type: ignore[arg-type]
    ctx.cancel.set()
    with pytest.raises(Canceled):
        ComparisonReport(db).run(ctx, spec)
    late = threading.Event()  # a cancel that lands after the last chunk still wins over publishing
    ctx = JobContext(_Runner(), "job_late", 1, time.monotonic() + 60)  # type: ignore[arg-type]
    orig = ctx.tick

    def tick(n: int) -> None:
        orig(n)
        late.set()
        ctx.cancel.set()

    ctx.tick = tick  # type: ignore[method-assign]
    with pytest.raises(Canceled):
        ComparisonReport(db).run(ctx, spec)
    assert late.is_set()
    with db.session() as conn:
        row = store.get_comparison_report(conn, cmp["id"])
    assert row is not None
    assert row["job_id"] == first["job_id"]
