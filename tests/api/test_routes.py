"""Route table vs the frozen contract: implemented + deferred == contract, exactly; nothing extra is served."""

from __future__ import annotations

from typing import Any

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tests.api.contract import Api, component, contract_routes
from weightlab.api.app import DEFERRED, IMPLEMENTED
from weightlab.api.config import API_PREFIX


def served_routes(client: TestClient) -> set[tuple[str, str]]:
    app: Any = client.app.app  # type: ignore[attr-defined]  # Guard wraps the FastAPI app
    out = set()
    for r in app.routes:
        if isinstance(r, APIRoute):
            assert r.path.startswith(API_PREFIX), r.path
            out |= {(m.lower(), r.path[len(API_PREFIX) :]) for m in r.methods}
    return out


def test_implemented_plus_deferred_is_exactly_the_contract(client: TestClient) -> None:
    contract = contract_routes()
    assert IMPLEMENTED.isdisjoint(DEFERRED)
    assert IMPLEMENTED | set(DEFERRED) == contract
    assert served_routes(client) == IMPLEMENTED  # no undocumented route, no deferred route half-served


def test_deferred_routes_answer_404_envelope(api: Api, client: TestClient) -> None:
    for (method, path), why in DEFERRED.items():
        url = API_PREFIX + path.format(cache_key="0" * 64, job_id="job_1", export_id="exp_1")
        r = client.request(method, url, headers=api.headers, json={} if method == "post" else None)
        assert r.status_code == 404, (method, path, why)
        component("Error").validate(r.json())
        assert r.json()["error"]["code"] == "NOT_FOUND"


def test_docs_and_schema_routes_are_disabled(api: Api, client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json", f"{API_PREFIX}/docs", f"{API_PREFIX}/openapi.json"):
        r = client.get(path, headers=api.headers)
        assert r.status_code == 404, path


def test_method_not_allowed_is_not_an_oracle(api: Api, client: TestClient) -> None:
    for method in ("put", "patch", "delete"):
        r = client.request(method, f"{API_PREFIX}/projects", headers=api.headers, json={})
        assert r.status_code == 404, method
        component("Error").validate(r.json())
