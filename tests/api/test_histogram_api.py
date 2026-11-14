"""tensor_histogram jobs over HTTP (ADR-021, contract 0.5.0).

Every response is validated against the contract by the `Api` client. Results are compared with the analysis
function called directly on the same file and with an independent numpy count against the published edges; jobs
are checked for exact byte accounting (one or two passes, plus verified hashing), refusal of invalid binning before
queueing, file changes, cancellation and wall time (no partial result), and deduplication.
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import import_model, tensors_by_key
from tests.api.contract import Api, component
from tests.helpers import reference_array, write_safetensors
from weightlab import open_checkpoint
from weightlab.analysis.histogram import Binning, histogram
from weightlab.api import histograms

BUDGET = {"max_working_set_mib": 512, "max_threads": 1, "max_wall_seconds": 120}
N = 300_000


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("histmodels")
    rng = np.random.default_rng(5)
    w = rng.standard_t(3, N).astype("<f4")
    w[:6] = [0.0, -0.0, np.nan, np.inf, -np.inf, 1e-40]  # zero, negative zero, non-finite, F32 subnormal
    ints = np.array([-(2**63), 2**63 - 1, 0, 0, 7, -7, 2**53 + 1], dtype="<i8")
    write_safetensors(root / "m" / "model.safetensors", {
        "w": ("F32", (600, N // 600), w.tobytes()),
        "i": ("I64", (ints.size,), ints.tobytes()),
        "c": ("F32", (4,), np.full(4, 0.5, "<f4").tobytes()),
    })  # fmt: skip
    return root


class Env:
    def __init__(self, client: TestClient, api: Api, root: Path) -> None:
        self.client, self.api, self.root = client, api, root
        self.project = api.post("/projects", json_body={"name": "hist"}, expect=201)["id"]
        self.model = import_model(api, self.project, root / "m", "model.safetensors")["id"]
        self.t = {k: v["id"] for k, v in tensors_by_key(api, self.model).items()}

    def body(self, key: str, **extra: Any) -> dict[str, Any]:
        src = {"kind": "tensor", "model_id": self.model, "tensor_id": self.t[key]}
        return {"kind": "tensor_histogram", "source": src, "budget": BUDGET, **extra}

    def submit(self, key: str, expect: int = 202, **extra: Any) -> dict[str, Any]:
        return self.api.post("/jobs", json_body=self.body(key, **extra), expect=expect)  # type: ignore[no-any-return]

    def finish(self, job_id: str, t: float = 60) -> dict[str, Any]:
        end = time.monotonic() + t
        while time.monotonic() < end:
            j = self.api.get("/jobs/{job_id}", job_id=job_id, expect=200)
            if j["state"] not in ("created", "queued", "running"):
                return j  # type: ignore[no-any-return]
            time.sleep(0.02)
        raise AssertionError("job did not finish")


@pytest.fixture
def env(client: TestClient, api: Api, models: Path) -> Env:
    return Env(client, api, models)


def _direct(root: Path, key: str, b: Binning) -> dict[str, Any]:
    with open_checkpoint(root / "m" / "model.safetensors") as ck:
        return histogram(ck, key, b)


def test_default_histogram_equals_the_analysis_and_an_independent_count(env: Env) -> None:
    j = env.submit("w")
    assert j["kind"] == "tensor_histogram"
    assert j["estimated_bytes"] == 2 * N * 4
    done = env.finish(j["job_id"])
    assert done["state"] == "completed", done.get("error")
    assert done["bytes_scanned"] == 2 * N * 4
    assert done["progress"] == 1
    r = done["result"]
    assert (r["model_id"], r["tensor_id"], r["consistency"]) == (env.model, env.t["w"], "best_effort")
    h = r["histogram"]
    assert h == _direct(env.root, "w", Binning())
    vals = reference_array(env.root / "m" / "model.safetensors", "w").reshape(-1)
    fin = vals[np.isfinite(vals)].astype(np.float64)
    edges = np.array(h["bins"]["edges"])
    ref = [int(np.count_nonzero((fin >= a) & ((fin < b) if i < len(edges) - 2 else (fin <= b))))
           for i, (a, b) in enumerate(itertools.pairwise(edges))]  # fmt: skip
    assert h["bins"]["counts"] == ref
    c = h["counts"]
    assert (c["nan"], c["pos_inf"], c["neg_inf"], c["negative_zeros"]) == (1, 1, 1, 1)
    assert c["zeros"] == int(np.count_nonzero(fin == 0))
    assert c["finite"] == fin.size
    assert h["min"] == float(fin.min())
    assert h["max"] == float(fin.max())
    assert any(b["exponent"] == math.frexp(1e-40)[1] - 1 for b in h["magnitude"]["buckets"])


def test_symmetric_explicit_and_integer_histograms(env: Env) -> None:
    sym = env.finish(env.submit("w", histogram={"bins": 33, "range": "symmetric"})["job_id"])["result"]["histogram"]
    assert sym == _direct(env.root, "w", Binning(33, "symmetric"))
    assert sym["bins"]["lo"] == -sym["bins"]["hi"]
    j = env.submit("w", histogram={"bins": 8, "range": "explicit", "lo": -1, "hi": 1})
    assert j["estimated_bytes"] == N * 4  # one pass
    ex = env.finish(j["job_id"])["result"]["histogram"]
    assert ex == _direct(env.root, "w", Binning(8, "explicit", -1, 1))
    assert ex["bins"]["below"] > 0
    assert ex["bins"]["above"] > 0
    assert ex["bins"]["below"] + sum(ex["bins"]["counts"]) + ex["bins"]["above"] == ex["counts"]["finite"]
    ih = env.finish(env.submit("i", histogram={"bins": 4})["job_id"])["result"]["histogram"]
    assert ih["value_domain"] == "integer"
    assert ih["magnitude"] is None
    assert ih["min"] == str(-(2**63))
    assert ih["max"] == str(2**63 - 1)
    assert ih["bins"]["edges"][-1] == str(2**63)
    assert sum(ih["bins"]["counts"]) == 7
    const = env.finish(env.submit("c")["job_id"])["result"]["histogram"]
    assert const["bins"]["edges"] == [0.5, 0.5]
    assert const["bins"]["counts"] == [4]


def test_verified_hashes_before_and_after(env: Env) -> None:
    size = (env.root / "m" / "model.safetensors").stat().st_size
    j = env.submit("c", consistency="verified")
    assert j["estimated_bytes"] == 2 * 4 * 4 + 2 * size
    done = env.finish(j["job_id"])
    assert done["state"] == "completed"
    assert done["result"]["consistency"] == "verified"
    assert done["bytes_scanned"] == 2 * 4 * 4 + 2 * size


@pytest.mark.parametrize(
    ("key", "options", "match"),
    [
        ("w", {"range": "explicit", "lo": 1, "hi": -1}, "lo <= hi"),
        ("i", {"range": "explicit", "lo": 0.5, "hi": 3}, "integer lo and hi"),
        ("i", {"range": "explicit", "lo": 0, "hi": 2**60}, "integer lo and hi"),
    ],
)
def test_invalid_binning_is_refused_before_queueing(env: Env, key: str, options: dict[str, Any], match: str) -> None:
    e = env.submit(key, expect=422, histogram=options)
    assert e["error"]["code"] == "VALIDATION_ERROR"
    assert match in e["error"]["message"]


def test_request_shape_errors(env: Env) -> None:
    src = env.body("w")["source"]
    for body in (
        {**env.body("w"), "view": {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}},
        {**env.body("w"), "model_id": env.model},
        {**env.body("w"), "histogram": {"bins": 0}},
        {"kind": "pyramid_build", "source": src, "budget": BUDGET, "histogram": {}},
        {**env.body("w"), "source": {"kind": "delta", "comparison_id": "cmp_x", "entry_id": "ent_x",
                                     "transform": "b_minus_a"}},
    ):  # fmt: skip
        e = env.api.post("/jobs", json_body=body, expect=422)
        assert e["error"]["code"] == "VALIDATION_ERROR"
    env.api.post("/jobs", json_body={**env.body("w"), "source": {**src, "tensor_id": "tns_doesnotexist"}},
                 expect=404)  # fmt: skip


def test_a_file_changed_after_import_is_refused(env: Env, models: Path, tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(models / "m", root)
    mid = import_model(env.api, env.project, root, "model.safetensors")["id"]
    tid = tensors_by_key(env.api, mid)["c"]["id"]
    p = root / "model.safetensors"
    data = bytearray(p.read_bytes())
    data[-1] ^= 0xFF
    p.write_bytes(bytes(data))
    body = {"kind": "tensor_histogram", "source": {"kind": "tensor", "model_id": mid, "tensor_id": tid},
            "budget": BUDGET}  # fmt: skip
    e = env.api.post("/jobs", json_body=body, expect=409)
    assert e["error"]["code"] == "FILE_CHANGED"


def test_a_job_whose_source_changed_while_queued_fails_closed(env: Env) -> None:
    """The run re-checks the fingerprints recorded at submission (here: a spec recorded for other file bytes)."""
    from weightlab.core.errors import WeightLabError

    svc = env.client.app.app.state.tiles  # type: ignore[attr-defined]  # inside the security Guard
    spec = {"model_id": env.model, "tensor_id": env.t["c"], "project_id": env.project, "consistency": "best_effort",
            "files": [{"relpath": "model.safetensors", "size": 1, "mtime_ns": 0, "ctime_ns": 0, "file_id": "0:0"}],
            "binning": Binning().identity()}  # fmt: skip
    with pytest.raises(WeightLabError, match="changed after the job was queued"):
        histograms.TensorHistogram(svc).run(_Ctx(), spec)  # type: ignore[arg-type]


class _Ctx:
    job_id = "job_test"

    def tick(self, _: int) -> None:
        pass

    def check(self) -> None:
        pass


@pytest.fixture
def slow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every chunk of a histogram scan take 20 ms, so cancellation and wall time land mid-scan."""
    real = histograms.histogram

    def slowed(ck: Any, key: str, b: Binning, tick: Any = None) -> dict[str, Any]:
        def t(n: int) -> None:
            time.sleep(0.02)
            if tick is not None:
                tick(n)

        return real(ck, key, b, tick=t)

    monkeypatch.setattr(histograms, "histogram", slowed)
    monkeypatch.setattr("weightlab.analysis.histogram.CHUNK_ELEMENTS", 4096)


def test_cancel_mid_scan_leaves_no_result(env: Env, slow: None) -> None:
    j = env.submit("w")
    end = time.monotonic() + 30
    while env.api.get("/jobs/{job_id}", job_id=j["job_id"], expect=200)["state"] != "running":
        assert time.monotonic() < end
        time.sleep(0.01)
    c = env.api.post("/jobs/{job_id}/cancel", job_id=j["job_id"], expect=200)
    done = env.finish(c["job_id"])
    assert done["state"] == "canceled"
    assert done["result"] is None
    component("Job").validate(done)


def test_wall_time_fails_with_resource_limit_and_no_result(env: Env, slow: None) -> None:
    j = env.api.post("/jobs", json_body={**env.body("w"), "budget": {**BUDGET, "max_wall_seconds": 1}}, expect=202)
    done = env.finish(j["job_id"])
    assert done["state"] == "failed"
    assert done["result"] is None
    assert done["error"]["code"] == "RESOURCE_LIMIT"


def test_identical_pending_submissions_share_one_job(env: Env, slow: None) -> None:
    a = env.submit("w", histogram={"bins": 10})
    b = env.submit("w", histogram={"bins": 10})
    c = env.submit("w", histogram={"bins": 11})
    assert b["job_id"] == a["job_id"]
    assert b["deduplicated"] is True
    assert c["job_id"] != a["job_id"]
    for x in (a, c):
        env.api.post("/jobs/{job_id}/cancel", job_id=x["job_id"], expect=200)
