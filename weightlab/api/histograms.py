"""The tensor_histogram job (ADR-021, contract 0.5.0): the exact value histogram of one tensor.

The numbers come from `analysis.histogram.histogram`; this module only validates, bounds and queues the scan.

* Exact only: every element is counted; there is no sampled mode. `min_max` and `symmetric` read the tensor twice
  (value range, then bins), `explicit` once. `estimated_bytes` is that scan plus, when `verified`, two full hashes
  of the files holding the tensor.
* Bounded: the reservation is `histogram.WORKING_SET` (chunked decode and temporaries, independent of tensor size);
  the scan ticks the job context after every chunk, so cancellation and wall time stop it within one chunk.
* Fail closed: the tensor's file fingerprints are recorded at submission; a job whose source changed before it ran
  fails with FILE_CHANGED, and every chunk read is checked (best_effort). `verified` also hashes the inputs before
  and after the scan and fails with FILE_CHANGED on any difference.
* The histogram is the job's result (`Job.result`), written in the transaction that marks the job completed;
  canceled, failed and interrupted jobs carry no histogram, partial or otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from weightlab.analysis.histogram import WORKING_SET, Binning, histogram, scan_bytes, validate_for
from weightlab.api.errors import ApiError
from weightlab.api.schemas import JobRequest, TensorSource
from weightlab.core.errors import ErrorCode, WeightLabError, fail
from weightlab.jobs import MIB, JobContext

if TYPE_CHECKING:
    from weightlab.api.tiles import TileService

KIND = "tensor_histogram"


def binning_of(req: JobRequest) -> Binning:
    o = req.histogram
    try:
        return Binning() if o is None else Binning(o.bins, o.range, o.lo, o.hi)
    except WeightLabError as e:
        raise ApiError(e.code, e.detail, details={"field": "histogram"}) from None


def submit(svc: TileService, req: JobRequest) -> tuple[dict[str, Any], bool]:
    """Validate and queue (or join) a tensor_histogram job; caps on `req.budget` are checked by the caller."""
    src = req.source
    if not isinstance(src, TensorSource):
        raise ApiError(ErrorCode.VALIDATION_ERROR, f"{KIND} is available for kind=tensor sources only",
                       details={"field": "source.kind"})  # fmt: skip
    binning = binning_of(req)
    if req.budget.max_working_set_mib * MIB < WORKING_SET:
        raise ApiError(ErrorCode.RESOURCE_LIMIT, f"a histogram needs {WORKING_SET // MIB} MiB of working memory, above "
                       f"the job budget max_working_set_mib ({req.budget.max_working_set_mib} MiB)", status=422,
                       details={"required_mib": WORKING_SET // MIB})  # fmt: skip
    t, ck = svc._open(src.model_id, src.tensor_id)
    with ck:
        key = t["raw_key"]
        try:
            validate_for(binning, ck.tensor(key).dtype)  # integer tensors need integer bounds: refuse now
        except WeightLabError as e:
            raise ApiError(e.code, e.detail, details={"field": "histogram"}) from None
        files = ck.fingerprints([key])
        estimated = scan_bytes(ck, key, binning)
        if req.consistency == "verified":
            estimated += 2 * ck.input_bytes([key])
    from weightlab.api.tiles import dedup_key  # local: tiles imports this module

    identity = {"model_id": src.model_id, "tensor_id": src.tensor_id, "files": files,
                "binning": binning.identity(), "consistency": req.consistency}  # fmt: skip
    spec = {
        "dedup_key": dedup_key(t["project_id"], KIND, identity),
        "memory_bytes": WORKING_SET,
        "estimated_bytes": estimated,
        "wall_seconds": req.budget.max_wall_seconds,
        "budget": req.budget.model_dump(),
        "project_id": t["project_id"],
        "model_id": src.model_id,
        "tensor_id": src.tensor_id,
        "files": files,
        "binning": binning.identity(),
        "consistency": req.consistency,
    }
    return svc.runner.submit(t["project_id"], KIND, spec)


class TensorHistogram:
    """The tensor_histogram job kind."""

    def __init__(self, svc: TileService) -> None:
        self.svc = svc

    def run(self, ctx: JobContext, spec: dict[str, Any]) -> dict[str, Any]:
        try:
            t, ck = self.svc._open(spec["model_id"], spec["tensor_id"], spec["project_id"])
        except ApiError as e:
            raise WeightLabError(e.code, e.message) from None
        b = spec["binning"]
        binning = Binning(b["bins"], b["range"], b["lo"], b["hi"])
        with ck:
            key = t["raw_key"]
            if ck.fingerprints([key]) != spec["files"]:
                raise fail(ErrorCode.FILE_CHANGED, "the tensor's file changed after the job was queued")
            ck.hash_tick = ctx.tick
            verified = spec["consistency"] == "verified"
            ctx.check()
            before = ck.input_digests([key], tick=ctx.tick) if verified else None
            hist = histogram(ck, key, binning, tick=ctx.tick)
            if verified and ck.input_digests([key], tick=ctx.tick) != before:
                raise fail(ErrorCode.FILE_CHANGED, "the tensor's file content changed during the job (verified)")
            ck.check()
            ctx.check()  # a cancel that arrived during the last chunk still wins
        return {"kind": KIND, "model_id": spec["model_id"], "tensor_id": spec["tensor_id"],
                "consistency": spec["consistency"], "histogram": hist}  # fmt: skip
