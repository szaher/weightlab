"""ADR-020 resource limits for large comparison reports: a 20,000-entry pair runs through the API within its declared
reservation, the stored JSON and the HTTP body stay within MAX_REPORT_JSON_BYTES, and a comparison above
MAX_REPORT_ENTRIES is refused before any job is queued (no unbounded allocation)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from tests.api.conftest import import_model
from tests.api.contract import Api
from weightlab.api import reports

N = 20_000
BUDGET = {"max_working_set_mib": 1024, "max_threads": 1, "max_wall_seconds": 600}


def _pair(root: Path, n: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    a = {f"model.layers.{i // 7}.mlp.t{i % 7}.weight": np.full(4, i % 13, dtype=np.float32) for i in range(n)}
    save_file(a, str(root / "a.safetensors"))
    save_file({k: v + np.float32(0.5) for k, v in a.items()}, str(root / "b.safetensors"))


def _comparison(api: Api, project: str, root: Path) -> dict[str, Any]:
    ma = import_model(api, project, root, "a.safetensors")
    mb = import_model(api, project, root, "b.safetensors")
    body = {"project_id": project, "model_a_id": ma["id"], "model_b_id": mb["id"], "policy": "descriptive_only"}
    return api.post("/comparisons", json_body=body, expect=201)  # type: ignore[no-any-return]


def test_twenty_thousand_entry_report_within_limits(api: Api, project: str, tmp_path: Path) -> None:
    root = tmp_path / "wide"
    _pair(root, N)
    cmp = _comparison(api, project, root)
    assert sum(cmp["counts"].values()) == N
    job = api.post("/jobs", json_body={"kind": "comparison_report", "comparison_id": cmp["id"], "budget": BUDGET},
                   expect=202)  # fmt: skip
    end = time.monotonic() + 600
    while (j := api.get("/jobs/{job_id}", job_id=job["job_id"], expect=200))["state"] in ("queued", "running"):
        assert time.monotonic() < end
        time.sleep(0.2)
    assert j["state"] == "completed", j
    r = api.client.get(f"/api/v1/comparisons/{cmp['id']}/report", headers=api.headers)
    assert r.status_code == 200
    assert len(r.content) <= reports.MAX_REPORT_JSON_BYTES + 4096
    body = json.loads(r.content)
    assert len(body["report"]["tensors"]) == N
    assert body["complete"] is True
    assert BUDGET["max_working_set_mib"] * 2**20 >= reports.REPORT_WORKING_SET + N * reports.PER_ENTRY_BYTES


def test_comparison_above_the_entry_cap_is_refused_before_queueing(api: Api, project: str, tmp_path: Path) -> None:
    root = tmp_path / "too_wide"
    _pair(root, reports.MAX_REPORT_ENTRIES + 1)
    cmp = _comparison(api, project, root)
    e = api.post("/jobs", json_body={"kind": "comparison_report", "comparison_id": cmp["id"], "budget": BUDGET},
                 expect=422)  # fmt: skip
    assert e["error"]["code"] == "RESOURCE_LIMIT"
    assert e["error"]["details"]["max_entries"] == reports.MAX_REPORT_ENTRIES
