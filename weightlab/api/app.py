"""FastAPI application: routes of contracts/openapi.yaml 0.4.0 (Phase 1 Task 2; comparison reports: ADR-020).

Implemented: /health, /projects (+ /{id}), /projects/{id}/models, /models/{id}/relink, /models/{id}/tensors
(+ /{tensor_id}), /tensors/roi (all four source kinds), /tiles (+ /{cache_key}/metadata; the complete quality
decision table for kind=tensor sources, see api/tiles.py), /jobs (+ /{id}, /{id}/cancel; kinds pyramid_build and
comparison_report), /comparisons (+ /{id}, /{id}/entries, /{id}/report), /projects/{id}/adapters,
/projects/{id}/bookmarks (+ /{bookmark_id}), /settings.
Deferred (not registered, so they answer 404): /exports* (export jobs), /dialogs/select-directory (optional
capability, not advertised). `IMPLEMENTED` and `DEFERRED` list them; tests check both against
the contract.

Bodies are parsed by the strict JSON reader (no NaN/Infinity, no duplicate keys) and validated by the
closed Pydantic models in `schemas`; any error becomes the doc 09 section 7 envelope.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any

from fastapi import FastAPI, Path, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from weightlab import __version__
from weightlab.api import catalog, relations, reports, roi
from weightlab.api.config import API_PREFIX, DEFAULT_SETTINGS, ServerConfig
from weightlab.api.errors import ApiError, envelope, from_weightlab, request_id
from weightlab.api.schemas import (
    AdapterImport,
    BookmarkCreate,
    ComparisonRequest,
    JobRequest,
    ModelImport,
    ModelRelink,
    ProjectCreate,
    RoiRequest,
    SavedView,
    SettingsPatch,
    TileRequest,
)
from weightlab.api.security import Guard
from weightlab.api.tiles import TileService
from weightlab.core.errors import ErrorCode, WeightLabError
from weightlab.db import Database, quick_health, store
from weightlab.formats.base import strict_json
from weightlab.ui import Asset, load_assets

ID = r"^[a-z]{3}_[A-Za-z0-9]{1,64}$"
IdPath = Annotated[str, Path(pattern=ID, max_length=68)]

IMPLEMENTED: frozenset[tuple[str, str]] = frozenset(
    {
        ("get", "/health"),
        ("get", "/projects"),
        ("post", "/projects"),
        ("get", "/projects/{project_id}"),
        ("delete", "/projects/{project_id}"),
        ("get", "/projects/{project_id}/models"),
        ("post", "/projects/{project_id}/models"),
        ("post", "/models/{model_id}/relink"),
        ("get", "/models/{model_id}/tensors"),
        ("get", "/models/{model_id}/tensors/{tensor_id}"),
        ("post", "/tensors/roi"),
        ("post", "/comparisons"),
        ("get", "/comparisons/{comparison_id}"),
        ("get", "/comparisons/{comparison_id}/entries"),
        ("get", "/comparisons/{comparison_id}/report"),
        ("get", "/projects/{project_id}/comparisons"),
        ("get", "/projects/{project_id}/adapters"),
        ("post", "/projects/{project_id}/adapters"),
        ("get", "/projects/{project_id}/bookmarks"),
        ("post", "/projects/{project_id}/bookmarks"),
        ("delete", "/projects/{project_id}/bookmarks/{bookmark_id}"),
        ("get", "/settings"),
        ("patch", "/settings"),
        ("post", "/tiles"),
        ("get", "/tiles/{cache_key}/metadata"),
        ("post", "/jobs"),
        ("get", "/jobs/{job_id}"),
        ("post", "/jobs/{job_id}/cancel"),
    }
)
DEFERRED: dict[tuple[str, str], str] = {
    ("post", "/exports"): "export jobs (a later milestone)",
    ("get", "/exports/{export_id}"): "export jobs (a later milestone)",
    ("get", "/exports/{export_id}/download"): "export jobs (a later milestone)",
    ("post", "/dialogs/select-directory"): "optional capability native_directory_dialog, not advertised",
}


def _rid(request: Request) -> str:
    state = request.scope.get("state") or {}
    return str(state.get("request_id") or request_id())


def _json(content: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(content, status_code=status)


async def parse[M: BaseModel](request: Request, model: type[M]) -> M:
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype != "application/json":
        raise ApiError(ErrorCode.VALIDATION_ERROR, "Content-Type must be application/json")
    raw = await request.body()
    try:
        data = strict_json(raw, "request body")
    except WeightLabError as e:
        raise ApiError(ErrorCode.VALIDATION_ERROR, e.detail) from None
    try:
        return model.model_validate(data)
    except ValidationError as e:
        first = e.errors(include_input=False, include_url=False)[0]
        loc = ".".join(str(x) for x in first["loc"]) or "body"
        raise ApiError(
            ErrorCode.VALIDATION_ERROR,
            f"{loc}: {first['msg']}"[:500],
            details={"field": loc[:200], "error": first["type"], "errors": e.error_count()},
        ) from None


def create_app(config: ServerConfig, database: Database | None = None, assets: dict[str, Asset] | None = None) -> Guard:
    """The API behind the Guard, which also serves the built UI shell (`assets`, default: the packaged build)."""
    db = database or Database(config.data_dir / "weightlab.sqlite3")

    def run(fn: Callable[..., Any], *args: Any) -> Any:
        """One database session per unit of work, on the calling (worker) thread."""
        with db.session() as conn:
            return fn(conn, *args)

    async def work(fn: Callable[..., Any], *args: Any) -> Any:
        return await run_in_threadpool(run, fn, *args)

    tiles = TileService(db, config.data_dir)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        tiles.runner.start()
        try:
            yield
        finally:
            await run_in_threadpool(tiles.runner.shutdown)

    app = FastAPI(
        title="WeightLab Local API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    app.state.tiles = tiles  # caches and the job runner, for diagnostics and the benchmark

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _json(envelope(exc, _rid(request)), exc.status)

    @app.exception_handler(WeightLabError)
    async def _wl_error(request: Request, exc: WeightLabError) -> JSONResponse:
        err = from_weightlab(exc)
        return _json(envelope(err, _rid(request)), err.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {"loc": (), "msg": "invalid request", "type": "invalid"}
        loc = ".".join(str(x) for x in first.get("loc", ())) or "request"
        err = ApiError(
            ErrorCode.VALIDATION_ERROR,
            f"{loc}: {first.get('msg', 'invalid')}"[:500],
            details={"field": loc[:200], "error": str(first.get("type", "invalid"))},
        )
        return _json(envelope(err, _rid(request)), 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: ErrorCode.NOT_FOUND, 405: ErrorCode.NOT_FOUND, 406: ErrorCode.NOT_ACCEPTABLE}.get(
            exc.status_code, ErrorCode.VALIDATION_ERROR
        )
        status = 404 if exc.status_code == 405 else exc.status_code  # no method oracle for deferred routes
        return _json(
            envelope(
                ApiError(code, "not found" if status == 404 else "request rejected", status=status), _rid(request)
            ),
            status,
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        # The contract's error enum has no INTERNAL code (spec gap, recorded); CONFLICT is the closest
        # schema-valid choice. No exception text or trace reaches the client.
        err = ApiError(ErrorCode.CONFLICT, f"internal error; request {_rid(request)} was not completed", status=500)
        return _json(envelope(err, _rid(request)), 500)

    p = API_PREFIX

    # -- health / settings ---------------------------------------------------------------------
    @app.get(f"{p}/health")
    def health() -> JSONResponse:
        with db.session() as conn:
            ok = quick_health(conn)  # constant cost; full integrity checks belong to `weightlab doctor`
        return _json(
            {"status": "ok" if ok else "degraded", "version": __version__[:64], "offline": True, "capabilities": []}
        )

    @app.get(f"{p}/settings")
    def get_settings() -> JSONResponse:
        with db.session() as conn:
            return _json(store.get_settings(conn, DEFAULT_SETTINGS))

    @app.patch(f"{p}/settings")
    async def patch_settings(request: Request) -> JSONResponse:
        patch = await parse(request, SettingsPatch)

        def apply(conn: Any) -> dict[str, Any]:
            current = store.get_settings(conn, DEFAULT_SETTINGS)
            current.update(patch.changes())
            store.put_settings(
                conn, {k: v for k, v in current.items() if k not in ("offline", "telemetry_enabled", "max_tile_cells")}
            )
            return store.get_settings(conn, DEFAULT_SETTINGS)

        out = await work(apply)
        tiles.settings_changed()
        return _json(out)

    # -- projects ------------------------------------------------------------------------------
    @app.get(f"{p}/projects")
    def list_projects() -> JSONResponse:
        with db.session() as conn:
            return _json(store.list_projects(conn))

    @app.post(f"{p}/projects")
    async def create_project(request: Request) -> JSONResponse:
        body = await parse(request, ProjectCreate)
        return _json(await work(store.create_project, body.name), 201)

    def _project(conn: Any, project_id: str) -> dict[str, Any]:
        proj = store.get_project(conn, project_id)
        if proj is None:
            raise ApiError(ErrorCode.NOT_FOUND, "project not found")
        return proj

    @app.get(f"{p}/projects/{{project_id}}")
    def get_project(project_id: IdPath) -> JSONResponse:
        with db.session() as conn:
            return _json(_project(conn, project_id))

    @app.delete(f"{p}/projects/{{project_id}}")
    def delete_project(project_id: IdPath, purge_cache: Annotated[bool, Query()] = True) -> Response:
        # Jobs of the project are canceled first; purge_cache removes its pyramids and cached tiles. Nothing here
        # opens, moves or deletes a checkpoint file.
        with db.session() as conn:
            if store.get_project(conn, project_id) is None:
                raise ApiError(ErrorCode.NOT_FOUND, "project not found")
        tiles.delete_project(project_id, purge_cache)
        with db.session() as conn:
            if not store.delete_project(conn, project_id):
                raise ApiError(ErrorCode.NOT_FOUND, "project not found")
        return Response(status_code=204)

    # -- models and catalog --------------------------------------------------------------------
    @app.get(f"{p}/projects/{{project_id}}/models")
    def list_models(project_id: IdPath) -> JSONResponse:
        with db.session() as conn:
            _project(conn, project_id)
            return _json([catalog.model_view(conn, m["id"]) for m in store.list_models(conn, project_id)])

    @app.post(f"{p}/projects/{{project_id}}/models")
    async def import_model(project_id: IdPath, request: Request) -> JSONResponse:
        body = await parse(request, ModelImport)

        def do(conn: Any) -> dict[str, Any]:
            _project(conn, project_id)
            return catalog.import_model(conn, project_id, body)

        return _json(await work(do), 201)

    @app.post(f"{p}/models/{{model_id}}/relink")
    async def relink(model_id: IdPath, request: Request) -> JSONResponse:
        body = await parse(request, ModelRelink)
        return _json(await work(catalog.relink, model_id, body))

    @app.get(f"{p}/models/{{model_id}}/tensors")
    def list_tensors(
        model_id: IdPath,
        q: Annotated[str | None, Query(max_length=256)] = None,
        dtype: Annotated[list[str] | None, Query(max_length=13)] = None,
        role: Annotated[list[Annotated[str, Query(max_length=16)]] | None, Query(max_length=12)] = None,
        layer: Annotated[int | None, Query(ge=0, le=100000)] = None,
        rank: Annotated[int | None, Query(ge=0, le=32)] = None,
        min_numel: Annotated[int | None, Query(ge=0, le=2**53 - 1)] = None,
        sort: Annotated[str, Query(pattern="^(raw_key|numel|byte_length|layer_index)$")] = "raw_key",
        order: Annotated[str, Query(pattern="^(asc|desc)$")] = "asc",
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        offset: Annotated[int, Query(ge=0, le=2**53 - 1)] = 0,
    ) -> JSONResponse:
        dtypes = dtype or []
        allowed = {"F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U64", "U32", "U16", "U8", "BOOL"}
        if any(d not in allowed for d in dtypes):
            raise ApiError(ErrorCode.VALIDATION_ERROR, "dtype: unknown dtype filter", details={"field": "dtype"})
        with db.session() as conn:
            if store.get_model(conn, model_id) is None:
                raise ApiError(ErrorCode.NOT_FOUND, "model not found")
            total, rows = store.query_tensors(
                conn,
                model_id,
                q=q,
                dtypes=dtypes,
                roles=role or [],
                layer=layer,
                rank=rank,
                min_numel=min_numel,
                sort=sort,
                order=order,
                limit=limit,
                offset=offset,
            )
            return _json({"total": total, "items": [catalog.tensor_view(t) for t in rows]})

    @app.get(f"{p}/models/{{model_id}}/tensors/{{tensor_id}}")
    def get_tensor(model_id: IdPath, tensor_id: IdPath) -> JSONResponse:
        with db.session() as conn:
            if store.get_model(conn, model_id) is None:
                raise ApiError(ErrorCode.NOT_FOUND, "model not found")
            t = store.get_tensor(conn, tensor_id, model_id)
            if t is None:
                raise ApiError(ErrorCode.NOT_FOUND, "tensor not found")
            return _json(catalog.tensor_view(t))

    @app.post(f"{p}/tensors/roi")
    async def read_roi(request: Request) -> JSONResponse:
        body = await parse(request, RoiRequest)
        return _json(await work(roi.compute, body))

    # -- tiles and jobs --------------------------------------------------------------------------
    @app.post(f"{p}/tiles")
    async def post_tile(request: Request) -> Response:
        body = await parse(request, TileRequest)
        accept = request.headers.get("accept")
        return await run_in_threadpool(tiles.tile, body, accept)

    @app.get(f"{p}/tiles/{{cache_key}}/metadata")
    def tile_metadata(cache_key: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")]) -> JSONResponse:
        return _json(tiles.tile_metadata(cache_key))

    @app.post(f"{p}/jobs")
    async def post_job(request: Request) -> JSONResponse:
        body = await parse(request, JobRequest)
        view, _ = await run_in_threadpool(tiles.submit_job, body)
        return _json(view, 202)

    @app.get(f"{p}/jobs/{{job_id}}")
    def get_job(job_id: IdPath) -> JSONResponse:
        view = tiles.runner.get(job_id)
        if view is None:
            raise ApiError(ErrorCode.NOT_FOUND, "job not found")
        return _json(view)

    @app.post(f"{p}/jobs/{{job_id}}/cancel")
    async def cancel_job(job_id: IdPath) -> JSONResponse:
        view = await run_in_threadpool(tiles.runner.cancel, job_id)
        if view is None:
            raise ApiError(ErrorCode.NOT_FOUND, "job not found")
        return _json(view)

    # -- comparisons ---------------------------------------------------------------------------
    @app.post(f"{p}/comparisons")
    async def create_comparison(request: Request) -> JSONResponse:
        body = await parse(request, ComparisonRequest)
        return _json(await work(relations.create_comparison, body), 201)

    def _comparison(conn: Any, comparison_id: str) -> dict[str, Any]:
        row = store.get_comparison(conn, comparison_id)
        if row is None:
            raise ApiError(ErrorCode.NOT_FOUND, "comparison not found")
        return row

    @app.get(f"{p}/comparisons/{{comparison_id}}")
    def get_comparison(comparison_id: IdPath) -> JSONResponse:
        with db.session() as conn:
            return _json(relations.comparison_view(_comparison(conn, comparison_id), conn))

    @app.get(f"{p}/comparisons/{{comparison_id}}/entries")
    def list_entries(
        comparison_id: IdPath,
        status: Annotated[list[str] | None, Query(max_length=6)] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        offset: Annotated[int, Query(ge=0, le=2**53 - 1)] = 0,
    ) -> JSONResponse:
        allowed = {"matched", "missing_in_a", "missing_in_b", "tied", "shape_mismatch", "dtype_mismatch"}
        if any(s not in allowed for s in status or []):
            raise ApiError(ErrorCode.VALIDATION_ERROR, "status: unknown entry status", details={"field": "status"})
        with db.session() as conn:
            _comparison(conn, comparison_id)
            total, rows = store.list_entries(conn, comparison_id, status or [], limit, offset)
            return _json({"total": total, "items": [relations.entry_view(r) for r in rows]})

    @app.get(f"{p}/projects/{{project_id}}/comparisons")
    def list_comparisons(project_id: IdPath) -> JSONResponse:
        with db.session() as conn:
            _project(conn, project_id)
            return _json([relations.comparison_view(r, conn) for r in store.list_comparisons(conn, project_id)])

    @app.get(f"{p}/comparisons/{{comparison_id}}/report")
    def get_report(comparison_id: IdPath) -> Response:
        return reports.report_response(db, comparison_id)

    # -- adapters ------------------------------------------------------------------------------
    @app.get(f"{p}/projects/{{project_id}}/adapters")
    def list_adapters(project_id: IdPath) -> JSONResponse:
        with db.session() as conn:
            _project(conn, project_id)
            return _json([relations.adapter_view(r) for r in store.list_adapters(conn, project_id)])

    @app.post(f"{p}/projects/{{project_id}}/adapters")
    async def import_adapter(project_id: IdPath, request: Request) -> JSONResponse:
        body = await parse(request, AdapterImport)

        def do(conn: Any) -> dict[str, Any]:
            _project(conn, project_id)
            return relations.import_adapter(conn, project_id, body)

        return _json(await work(do), 201)

    # -- bookmarks -----------------------------------------------------------------------------
    @app.get(f"{p}/projects/{{project_id}}/bookmarks")
    def list_bookmarks(project_id: IdPath) -> JSONResponse:
        with db.session() as conn:
            _project(conn, project_id)
            return _json(store.list_bookmarks(conn, project_id))

    @app.post(f"{p}/projects/{{project_id}}/bookmarks")
    async def create_bookmark(project_id: IdPath, request: Request) -> JSONResponse:
        body = await parse(request, BookmarkCreate)

        def do(conn: Any) -> dict[str, Any]:
            _project(conn, project_id)
            check_view_ownership(conn, project_id, body.view)
            return store.insert_bookmark(conn, project_id, body.view.model_dump(exclude_unset=True), body.note)

        return _json(await work(do), 201)

    @app.delete(f"{p}/projects/{{project_id}}/bookmarks/{{bookmark_id}}")
    def delete_bookmark(project_id: IdPath, bookmark_id: IdPath) -> Response:
        with db.session() as conn:
            _project(conn, project_id)
            if not store.delete_bookmark(conn, project_id, bookmark_id):
                raise ApiError(ErrorCode.NOT_FOUND, "bookmark not found")
        return Response(status_code=204)

    return Guard(app, config, load_assets() if assets is None else assets)


def check_view_ownership(conn: Any, project_id: str, view: SavedView) -> None:
    """Every ID in a saved view must belong to the project; otherwise 404 (no existence oracle)."""
    src = view.source
    nf = ApiError(ErrorCode.NOT_FOUND, "referenced object not found in this project")
    if src.kind == "tensor":
        if (
            store.get_model(conn, src.model_id, project_id) is None
            or store.get_tensor(conn, src.tensor_id, src.model_id) is None
        ):
            raise nf
    elif src.kind == "delta":
        if (
            store.get_comparison(conn, src.comparison_id, project_id) is None
            or store.get_entry(conn, src.comparison_id, src.entry_id) is None
        ):
            raise nf
    else:
        row = store.get_adapter(conn, src.adapter_id, project_id)
        if row is None or relations.adapter_module(row, src.module_id) is None:
            raise nf
