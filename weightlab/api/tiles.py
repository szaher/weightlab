"""POST /tiles, GET /tiles/{cache_key}/metadata, the pyramid_build job and job submission (comparison_report jobs
are implemented in api/reports.py) (doc 09 sections 3 and 5, doc 03 section 5).

Quality decision table (doc 09 section 3), applied in this order; `covered` is the source bytes under the tile,
`limit` = pyramid_exact_max_mib applied to the tensor's on-disk size:

| condition                                   | auto                   | exact       | row_sampled         |
|---------------------------------------------|------------------------|-------------|---------------------|
| covered <= direct_tile_max_mib (all L0-L3)  | 200 exact, direct read | same        | 422 (not needed)    |
| pyramid ready                               | 200 exact from pyramid | same        | 422                 |
| pyramid not ready, source <= limit          | 202 + pyramid_build    | 202 + job   | 422                 |
| pyramid not ready, source > limit           | 200 row_sampled        | 202 + job   | 200 row_sampled     |

Levels 0-3 are always direct reads (doc 03: "Levels 0-3 always qualify for every supported dtype"; the pyramid
starts at level 4), even if direct_tile_max_mib is set below the 32 MiB an F64 L3 tile can cover.

Only `kind = tensor` sources are served in this release; delta / LoRA / virtual-effective tiles need their own
paired passes (a later milestone) and are answered with 422 VALIDATION_ERROR, never with a substitute.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any

from fastapi.responses import JSONResponse, Response

from weightlab.analysis import pyramid as pyr
from weightlab.analysis import sampling
from weightlab.analysis import tile_codec as tc
from weightlab.analysis.tile import MIB, TILE_FORMAT, Tile, TileBudget, compute_tile, tile_geometry
from weightlab.api import reports
from weightlab.api.catalog import open_model
from weightlab.api.config import DEFAULT_SETTINGS
from weightlab.api.errors import ApiError
from weightlab.api.schemas import Budget, JobRequest, TensorSource, TileRequest
from weightlab.api.schemas import View as ViewModel
from weightlab.cache import (
    BuildInProgress,
    CachedTile,
    DiskUnavailable,
    MemoryTileCache,
    PyramidCache,
    Reservation,
    TileMetadataStore,
)
from weightlab.core.errors import ErrorCode, WeightLabError, fail
from weightlab.db import Database, store
from weightlab.formats.checkpoint import Checkpoint
from weightlab.formats.dtypes import lookup
from weightlab.formats.selection import View, resolve_view
from weightlab.jobs import JobContext, JobRunner

TILE_SLOT_WAIT_S = 10.0


def to_view(v: ViewModel) -> View:
    return View(v.row_axis, v.col_axis, {int(k): i for k, i in v.fixed_indices.items()}, v.transpose)


def view_from_json(v: dict[str, Any]) -> View:
    return View(v["row_axis"], v["col_axis"], {int(k): int(i) for k, i in v["fixed_indices"].items()}, v["transpose"])


MEDIA = {TILE_FORMAT: "binary", "application/*": "binary", "*/*": "binary", "application/json": "json"}


def negotiate(accept: str | None) -> str:
    """'binary' (the default) or 'json' from an Accept header, by q-value; on a tie the binary default wins.
    406 NOT_ACCEPTABLE when neither is acceptable."""
    if not accept or not accept.strip():
        return "binary"
    best: dict[str, float] = {}
    for part in accept.split(","):
        fields = [f.strip() for f in part.split(";")]
        kind = MEDIA.get(fields[0].lower())
        q = 1.0
        for f in fields[1:]:
            if f.lower().startswith("q="):
                try:
                    q = float(f[2:])
                except ValueError:
                    q = 0.0
        if kind is not None and q > 0:
            best[kind] = max(best.get(kind, 0.0), q)
    if not best:
        raise ApiError(ErrorCode.NOT_ACCEPTABLE, f"tiles are served as {TILE_FORMAT} or application/json")
    return "json" if best.get("json", 0.0) > best.get("binary", 0.0) else "binary"


@dataclass
class Resolved:
    """A tensor tile request resolved against the catalog and the open checkpoint."""

    project_id: str
    model_id: str
    tensor_id: str
    key: str
    view: View
    rows: int
    cols: int
    esize: int
    source_bytes: int
    files: list[dict[str, object]]
    pyramid_identity: dict[str, Any]
    pyramid_id: str


def dedup_key(project_id: str, kind: str, identity: Any) -> str:
    raw = json.dumps(
        {"project_id": project_id, "kind": kind, "identity": identity}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class TileService:
    def __init__(self, db: Database, data_dir: Any) -> None:
        self.db = db
        s = self.settings()
        self.memory = MemoryTileCache(int(s["tile_cache_memory_mib"]) * MIB)
        self.metadata = TileMetadataStore()
        self.pyramids = PyramidCache(
            data_dir / "cache" / "pyramids",
            disk_capacity=lambda: int(self.settings()["tile_cache_disk_mib"]) * MIB,
            session_capacity=lambda: int(self.settings()["memory_budget_mib"]) * MIB // 4,
        )
        self.runner = JobRunner(
            db, self.settings, {"pyramid_build": PyramidBuild(self), reports.KIND: reports.ComparisonReport(db)}
        )
        self._slots_lock = threading.Lock()
        self._active = 0
        self._slot_cv = threading.Condition(self._slots_lock)

    def settings(self) -> dict[str, Any]:
        with self.db.session() as conn:
            return store.get_settings(conn, DEFAULT_SETTINGS)

    def settings_changed(self) -> None:
        """Caps apply at once: a lowered tile_cache_memory_mib / tile_cache_disk_mib / memory budget evicts now."""
        self.memory.set_capacity(int(self.settings()["tile_cache_memory_mib"]) * MIB)
        self.pyramids.enforce_caps()

    # -- concurrency -----------------------------------------------------------------------------------
    def _acquire(self) -> None:
        limit = int(self.settings()["max_active_tiles"])
        with self._slot_cv:
            if not self._slot_cv.wait_for(lambda: self._active < limit, TILE_SLOT_WAIT_S):
                raise ApiError(ErrorCode.RATE_LIMITED, f"{limit} tile computations are already running")
            self._active += 1

    def _release(self) -> None:
        with self._slot_cv:
            self._active -= 1
            self._slot_cv.notify()

    # -- resolution ------------------------------------------------------------------------------------
    def _open(self, model_id: str, tensor_id: str, project_id: str | None = None) -> tuple[dict[str, Any], Checkpoint]:
        with self.db.session() as conn:
            model = store.get_model(conn, model_id, project_id)
            if model is None:
                raise ApiError(ErrorCode.NOT_FOUND, "model not found")
            t = store.get_tensor(conn, tensor_id, model_id)
            if t is None:
                raise ApiError(ErrorCode.NOT_FOUND, "tensor not found in this model")
            ck = open_model(conn, model_id)
        return {**t, "project_id": model["project_id"]}, ck

    @staticmethod
    def resolve(t: dict[str, Any], ck: Checkpoint, view: View) -> Resolved:
        key = t["raw_key"]
        meta = ck.tensor(key)
        rv = resolve_view(meta.shape, view)
        files = ck.fingerprints([key])
        ident = tc.pyramid_identity(source_kind="tensor", files=files, tensor_keys=[key], view=view)
        return Resolved(
            t["project_id"], t["model_id"], t["id"], key, view, rv.rows, rv.cols, lookup(meta.dtype).size,
            int(t["byte_length"]), files, ident, tc.cache_key(ident),
        )  # fmt: skip

    # -- POST /tiles -------------------------------------------------------------------------------------
    def tile(self, req: TileRequest, accept: str | None) -> Response:
        fmt = negotiate(accept)
        src = req.source
        if not isinstance(src, TensorSource):
            raise ApiError(
                ErrorCode.VALIDATION_ERROR,
                f"heatmap tiles of kind={src.kind} sources are not available in this release (they need a paired "
                "pass, a later milestone); exact values of these sources come from /tensors/roi",
                details={"field": "source.kind"},
            )
        settings = self.settings()
        t, ck = self._open(src.model_id, src.tensor_id)
        with ck:
            r = self.resolve(t, ck, to_view(req.view))
            spec, mode = req.tile, req.quality.mode
            g = tile_geometry(r.rows, r.cols, spec.level, spec.x, spec.y, spec.size)
            covered = g.covered_cells * r.esize
            direct_limit = int(settings["direct_tile_max_mib"]) * MIB
            source_dict = src.model_dump()

            def exact_key() -> str:
                return tc.cache_key(tc.tile_identity(
                    source_kind="tensor", files=r.files, tensor_keys=[r.key], view=r.view, level=spec.level,
                    x=spec.x, y=spec.y, size=spec.size, aggregation=req.aggregation, quality_mode="exact"))  # fmt: skip

            def not_needed(why: str) -> ApiError:
                return ApiError(ErrorCode.VALIDATION_ERROR, f"row_sampled is not available here: {why}",
                                details={"field": "quality.mode"})  # fmt: skip

            # 1. direct read
            if covered <= direct_limit or spec.level < pyr.BASE_LEVEL:
                if mode == "row_sampled":
                    raise not_needed("this tile is exact by a direct read")
                key = exact_key()
                hit = self.memory.get(key)
                if hit is not None:
                    return self._respond(hit.meta, hit.body, source_dict, r, fmt, key)
                budget = TileBudget.from_settings(settings)
                if spec.level < pyr.BASE_LEVEL:  # L0-L3 always direct (at most 32 MiB covered)
                    budget = TileBudget(budget.max_tile_cells, max(budget.direct_tile_max_bytes, 32 * MIB),
                                        max(budget.max_read_bytes, 128 * MIB), budget.band_cells,
                                        budget.memory_bytes)  # fmt: skip
                self._acquire()
                try:
                    tile = compute_tile(ck, r.key, r.view, level=spec.level, x=spec.x, y=spec.y, size=spec.size,
                                        aggregation=req.aggregation, budget=budget)  # fmt: skip
                finally:
                    self._release()
                return self._serve(tile, key, source_dict, r, fmt)

            # 2. pyramid ready
            with self.pyramids.open(r.pyramid_id, r.pyramid_identity) as h:
                if h is not None:
                    if mode == "row_sampled":
                        raise not_needed("the exact pyramid is ready")
                    key = exact_key()
                    hit = self.memory.get(key)
                    if hit is not None:
                        self.pyramids.add_owner(r.pyramid_id, r.project_id)
                        return self._respond(hit.meta, hit.body, source_dict, r, fmt, key)
                    try:
                        tile = pyr.pyramid_tile(h.blob, h.pyramid, level=spec.level, x=spec.x, y=spec.y,
                                                size=spec.size, aggregation=req.aggregation, view_rows=r.rows,
                                                view_cols=r.cols, esize=r.esize)  # fmt: skip
                    except pyr.PyramidCorrupt:
                        self.pyramids.report_corrupt(r.pyramid_id)  # rebuilt below, never served
                    else:
                        self.pyramids.add_owner(r.pyramid_id, r.project_id)
                        return self._serve(tile, key, source_dict, r, fmt)

            # 3./4. pyramid not ready
            limit = int(settings["pyramid_exact_max_mib"]) * MIB
            if r.source_bytes <= limit:
                if mode == "row_sampled":
                    raise not_needed(
                        f"sampling is only for sources above pyramid_exact_max_mib ({limit // MIB} MiB); "
                        "request auto or exact for the exact pyramid"
                    )
                return self._job_response(r, ck, settings)
            if mode == "exact":
                return self._job_response(r, ck, settings)
            return self._sampled(req, r, ck, settings, source_dict, fmt)

    def _sampled(
        self, req: TileRequest, r: Resolved, ck: Checkpoint, settings: dict[str, Any], source: dict[str, Any], fmt: str
    ) -> Response:
        spec, q = req.tile, req.quality
        budget = TileBudget.from_settings(settings)
        plan = sampling.plan_sample(
            ck, r.key, r.view, level=spec.level, x=spec.x, y=spec.y, size=spec.size,
            rows_per_block=q.sampled_rows_per_block, seed=q.seed, max_read_bytes=budget.direct_tile_max_bytes,
            max_cells=budget.direct_tile_max_bytes,
        )  # fmt: skip
        if plan.complete:  # every covered row selected: that is an exact scan within the sampling budget
            key = tc.cache_key(tc.tile_identity(
                source_kind="tensor", files=r.files, tensor_keys=[r.key], view=r.view, level=spec.level, x=spec.x,
                y=spec.y, size=spec.size, aggregation=req.aggregation, quality_mode="exact"))  # fmt: skip
            exact_budget = TileBudget(budget.max_tile_cells, max(budget.direct_tile_max_bytes, plan.read_bytes),
                                      budget.max_read_bytes, budget.band_cells, budget.memory_bytes)  # fmt: skip
            self._acquire()
            try:
                tile = compute_tile(ck, r.key, r.view, level=spec.level, x=spec.x, y=spec.y, size=spec.size,
                                    aggregation=req.aggregation, budget=exact_budget)  # fmt: skip
            finally:
                self._release()
            return self._serve(tile, key, source, r, fmt)
        op = {"sampled_rows_per_block": plan.rows_per_block, "sampling_policy": plan.policy}
        key = tc.cache_key(tc.tile_identity(
            source_kind="tensor", files=r.files, tensor_keys=[r.key], view=r.view, level=spec.level, x=spec.x,
            y=spec.y, size=spec.size, aggregation=req.aggregation, quality_mode="row_sampled", seed=plan.seed,
            operation=op))  # fmt: skip
        hit = self.memory.get(key)
        if hit is not None:
            return self._respond(hit.meta, hit.body, source, r, fmt, key)
        self._acquire()
        try:
            tile = sampling.sampled_tile(
                ck, r.key, r.view, level=spec.level, x=spec.x, y=spec.y, size=spec.size,
                aggregation=req.aggregation, plan=plan, pyramid_id=r.pyramid_id,
            )  # fmt: skip
        finally:
            self._release()
        return self._serve(tile, key, source, r, fmt)

    # -- responses ---------------------------------------------------------------------------------------
    def _serve(self, tile: Tile, key: str, source: dict[str, Any], r: Resolved, fmt: str) -> Response:
        meta = tc.metadata(tile, cache_key=key, source=source, view=r.view)
        tc.validate_metadata(meta, tile.geometry.rows, tile.geometry.cols)
        body = tc.encode_binary(tile.values, tile.status, tile.geometry.level)
        cached = CachedTile({k: v for k, v in meta.items() if k != "source"}, body)
        cached.meta["_project_ids"] = [r.project_id]
        self.memory.put(key, cached)
        return self._respond(cached.meta, body, source, r, fmt, key)

    def _respond(
        self, meta_wo_source: dict[str, Any], body: bytes, source: dict[str, Any], r: Resolved, fmt: str, key: str
    ) -> Response:
        meta = {k: v for k, v in meta_wo_source.items() if not k.startswith("_")}
        meta["source"] = source
        owners = meta_wo_source.setdefault("_project_ids", [])
        if r.project_id not in owners:
            owners.append(r.project_id)
        self.metadata.put(key, meta, r.project_id)
        rows, cols, level = meta["rows"], meta["cols"], meta["level"]
        if fmt == "json":
            values, status = tc.decode_binary(body, rows, cols, level)
            return JSONResponse(tc.to_json(meta, values, status))
        return Response(content=body, media_type=TILE_FORMAT, headers=tc.headers(meta))

    def tile_metadata(self, cache_key: str) -> dict[str, Any]:
        meta = self.metadata.get(cache_key)
        if meta is None:
            raise ApiError(ErrorCode.NOT_FOUND, "no tile with this cache key was served in this session")
        return meta

    # -- jobs ------------------------------------------------------------------------------------------
    def _job_response(self, r: Resolved, ck: Checkpoint, settings: dict[str, Any]) -> JSONResponse:
        budget = Budget(
            max_working_set_mib=int(settings["memory_budget_mib"]),
            max_threads=int(settings["max_threads"]),
            max_wall_seconds=int(settings["max_job_seconds"]),
        )
        view, _ = self.submit_pyramid(r, ck, "best_effort", budget)
        return JSONResponse(view, status_code=202)

    def submit_pyramid(
        self, r: Resolved, ck: Checkpoint, consistency: str, budget: Budget
    ) -> tuple[dict[str, Any], bool]:
        plan = pyr.stream_plan(ck, r.key, r.view)
        nbytes = pyr.estimate_bytes(r.rows, r.cols)
        where = self.pyramids.placement(nbytes)
        if where is None:
            raise fail(
                ErrorCode.RESOURCE_LIMIT,
                f"the exact pyramid would need {nbytes} bytes, more than tile_cache_disk_mib and the session "
                "memory store allow",
            )
        mem = pyr.build_working_set(r.rows, r.cols, plan.orientation, ck.limits.read_chunk_bytes)
        if where == "memory":
            mem += nbytes
        if mem > budget.max_working_set_mib * MIB:
            raise fail(
                ErrorCode.RESOURCE_LIMIT,
                f"the pyramid build needs ~{-(-mem // MIB)} MiB of working memory, above the job budget "
                f"max_working_set_mib ({budget.max_working_set_mib} MiB)",
            )
        read_bytes = r.rows * r.cols * r.esize
        if consistency == "verified":
            read_bytes += 2 * ck.input_bytes([r.key])
        spec = {
            "dedup_key": dedup_key(r.project_id, "pyramid_build", {"pyramid_id": r.pyramid_id,
                                                                  "consistency": consistency}),
            "memory_bytes": mem,
            "estimated_bytes": read_bytes,
            "wall_seconds": budget.max_wall_seconds,
            "budget": budget.model_dump(),
            "project_id": r.project_id,
            "model_id": r.model_id,
            "tensor_id": r.tensor_id,
            "raw_key_sha256": hashlib.sha256(r.key.encode()).hexdigest(),
            "view": tc.view_json(r.view),
            "consistency": consistency,
            "pyramid_id": r.pyramid_id,
            "pyramid_bytes": nbytes,
            "placement": where,  # what memory_bytes assumed; the build never silently changes it (see run)
        }  # fmt: skip
        return self.runner.submit(r.project_id, "pyramid_build", spec)

    def submit_job(self, req: JobRequest) -> tuple[dict[str, Any], bool]:
        settings = self.settings()
        caps: dict[str, int] = {
            "memory_budget_mib": int(settings["memory_budget_mib"]),
            "max_threads": int(settings["max_threads"]),
            "max_job_seconds": int(settings["max_job_seconds"]),
        }
        b = req.budget
        if (
            b.max_working_set_mib > caps["memory_budget_mib"]
            or b.max_threads > caps["max_threads"]
            or (b.max_wall_seconds > caps["max_job_seconds"])
        ):
            raise ApiError(ErrorCode.RESOURCE_LIMIT, "the job budget exceeds the server's caps", status=422,
                           details=dict(caps))  # fmt: skip
        if req.kind == reports.KIND:
            return reports.submit(self.db, self.runner, req)
        if req.kind != "pyramid_build":
            raise ApiError(
                ErrorCode.VALIDATION_ERROR,
                f"job kind {req.kind} is not available in this release; only pyramid_build and "
                f"{reports.KIND} are implemented",
                details={"field": "kind"},
            )
        src = req.source
        if not isinstance(src, TensorSource):
            raise ApiError(ErrorCode.VALIDATION_ERROR, "pyramid_build is available for kind=tensor sources only",
                           details={"field": "source.kind"})  # fmt: skip
        if req.view is None:
            raise ApiError(ErrorCode.VALIDATION_ERROR, "pyramid_build needs a view", details={"field": "view"})
        if req.model_id is not None:
            raise ApiError(ErrorCode.VALIDATION_ERROR, "pyramid_build takes its model from source, not model_id",
                           details={"field": "model_id"})  # fmt: skip
        t, ck = self._open(src.model_id, src.tensor_id)
        with ck:
            r = self.resolve(t, ck, to_view(req.view))
            try:
                return self.submit_pyramid(r, ck, req.consistency, b)
            except WeightLabError as e:
                if e.code is ErrorCode.RESOURCE_LIMIT:  # /jobs declares 422, not 413
                    raise ApiError(e.code, e.detail, status=422) from None
                raise

    def delete_project(self, project_id: str, purge_cache: bool) -> None:
        self.runner.cancel_project(project_id)
        self.metadata.remove_project(project_id)
        if purge_cache:
            self.pyramids.purge_owner(project_id)
            self.memory.remove_where(lambda t: project_id in t.meta.get("_project_ids", []))


class PyramidBuild:
    """The pyramid_build job kind: one streaming build, published atomically only on success."""

    WAIT_SLICE_S = 0.05

    def __init__(self, svc: TileService) -> None:
        self.svc = svc

    def _reserve(self, ctx: JobContext, spec: dict[str, Any], r: Resolved) -> Reservation | None:
        """Reserve space for the build, or None when the pyramid became available while waiting.

        * Builds are coordinated by pyramid id, across projects: when another job holds the reservation this job
          waits (cancellable, within its wall time) and then uses the published pyramid. If that build was
          canceled or failed instead (e.g. its project was deleted), this job builds it itself, so deleting one
          project never takes away a build another project needs.
        * Memory: a job admitted for disk placement may fall back to the session store only after re-admission
          for the pyramid's size, within both its own budget and memory_budget_mib; otherwise it fails with
          RESOURCE_LIMIT and allocates nothing."""
        nbytes = int(spec["pyramid_bytes"])
        cache = self.svc.pyramids
        while True:
            ctx.check()
            with cache.open(r.pyramid_id, r.pyramid_identity) as h:
                if h is not None:
                    return None
            try:
                try:
                    return cache.reserve(r.pyramid_id, nbytes, allow_memory=spec.get("placement") == "memory")
                except DiskUnavailable:
                    limit = int(spec["budget"]["max_working_set_mib"]) * MIB if "budget" in spec else 0
                    if not ctx.grow_memory(nbytes, limit):
                        raise fail(
                            ErrorCode.RESOURCE_LIMIT,
                            "the disk cache cannot take the pyramid now, and holding it in memory "
                            f"({-(-nbytes // MIB)} MiB) would exceed the job's memory budget "
                            "(max_working_set_mib) or memory_budget_mib",
                        ) from None
                    return cache.reserve(r.pyramid_id, nbytes, allow_memory=True)
            except BuildInProgress:
                cache.wait_for_build(r.pyramid_id, self.WAIT_SLICE_S)

    def run(self, ctx: JobContext, spec: dict[str, Any]) -> dict[str, Any]:
        try:
            t, ck = self.svc._open(spec["model_id"], spec["tensor_id"], spec["project_id"])
        except ApiError as e:
            raise WeightLabError(e.code, e.message) from None
        with ck:
            r = self.svc.resolve(t, ck, view_from_json(spec["view"]))
            if r.pyramid_id != spec["pyramid_id"]:
                raise fail(ErrorCode.FILE_CHANGED, "the source changed after the job was queued")
            result = {"kind": "pyramid_build", "pyramid_id": r.pyramid_id}
            with self.svc.pyramids.open(r.pyramid_id, r.pyramid_identity) as h:
                if h is not None:
                    self.svc.pyramids.add_owner(r.pyramid_id, r.project_id)
                    return result
            res = self._reserve(ctx, spec, r)
            if res is None:  # another job (possibly another project's) built it meanwhile: share its result
                self.svc.pyramids.add_owner(r.pyramid_id, r.project_id)
                return result
            sink: pyr.FileSink | pyr.MemorySink | None = None
            try:
                sink = pyr.FileSink(res.tmp_path) if res.tmp_path is not None else pyr.MemorySink(res.nbytes)
                verified = spec["consistency"] == "verified"
                before = ck.input_digests([r.key], tick=ctx.tick) if verified else None
                built = pyr.build_pyramid(ck, r.key, r.view, r.pyramid_identity, r.pyramid_id, sink, ctx)
                if verified and ck.input_digests([r.key], tick=ctx.tick) != before:
                    raise fail(ErrorCode.FILE_CHANGED, "the source's content changed during the job (verified)")
                ck.check()
                ctx.check()  # a cancel that arrived during the last chunk still wins
                self.svc.pyramids.publish(res, sink, built.total_bytes, owner=r.project_id)
            except BaseException:
                self.svc.pyramids.abort(res, sink)
                raise
        return result
