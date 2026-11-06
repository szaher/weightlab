"""Frozen-contract helpers for API tests: JSON Schema validators built from contracts/openapi.yaml itself
(never from the implementation), and a client wrapper that checks every response against the schema
the contract declares for its path, method and status."""

from __future__ import annotations

import functools
import json
from typing import Any

import jsonschema
import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from tests.helpers import ROOT

BASE = "urn:weightlab:openapi"
PREFIX = "/api/v1"


@functools.cache
def spec() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@functools.cache
def _registry() -> Registry:  # type: ignore[type-arg]
    return Registry().with_resource(BASE, Resource.from_contents(spec(), default_specification=DRAFT202012))


def _pointer(*parts: str) -> str:
    return "#/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in parts)


@functools.cache
def validator_for(pointer: str) -> Draft202012Validator:
    return Draft202012Validator({"$ref": BASE + pointer}, registry=_registry())


def component(name: str) -> Draft202012Validator:
    return validator_for(_pointer("components", "schemas", name))


def contract_routes() -> set[tuple[str, str]]:
    return {(m, path) for path, ops in spec()["paths"].items() for m in ops if m in ("get", "post", "patch", "delete")}


# Statuses that doc 09's behavioural rules require but the frozen contract does not declare for the route
# (ownership 404 on list/create routes, 422 for an invalid query filter). Bodies are still validated
# against `Error`. Listed in evidence/phase1/ as contract gaps for the next contract revision.
CONTRACT_GAPS: frozenset[tuple[str, str, int]] = frozenset({
    ("get", "/projects/{project_id}/models", 404),
    ("get", "/projects/{project_id}/adapters", 404),
    ("get", "/projects/{project_id}/bookmarks", 404),
    ("post", "/projects/{project_id}/bookmarks", 404),
    ("delete", "/projects/{project_id}/bookmarks/{bookmark_id}", 404),
    ("get", "/comparisons/{comparison_id}/entries", 422),
    ("post", "/models/{model_id}/relink", 422),
    ("post", "/models/{model_id}/relink", 403),
    ("post", "/projects/{project_id}/adapters", 403),
    ("post", "/projects/{project_id}/adapters", 413),
})  # fmt: skip


def response_validator(method: str, path: str, status: int) -> Draft202012Validator | None:
    op = spec()["paths"][path][method]
    resp = op["responses"].get(str(status))
    if resp is None and (method, path, status) in CONTRACT_GAPS and status >= 400:
        return component("Error")
    assert resp is not None, f"contract declares no {status} response for {method.upper()} {path}"
    if "$ref" in resp:
        name = resp["$ref"].rsplit("/", 1)[-1]
        resp = spec()["components"]["responses"][name]
        content = resp.get("content", {})
        if "application/json" not in content:
            return None
        return validator_for(_pointer("components", "responses", name, "content", "application/json", "schema"))
    if "application/json" not in resp.get("content", {}):
        return None
    return validator_for(_pointer("paths", path, method, "responses", str(status), "content", "application/json",
                                  "schema"))  # fmt: skip


class Api:
    """Wraps a TestClient: `api.post("/projects", json=..., expect=201)` asserts the status, validates the
    body against the contract for that route and status, and returns the parsed JSON."""

    def __init__(self, client: Any, token: str) -> None:
        self.client = client
        self.headers = {"Authorization": f"Bearer {token}"}
        self.seen_error_bodies: list[str] = []

    def call(self, method: str, template: str, *, expect: int, params: Any = None, json_body: Any = None,
             raw: bytes | None = None, headers: dict[str, str] | None = None, **path: str) -> Any:  # fmt: skip
        url = PREFIX + template.format(**path)
        hdrs = {**self.headers, **(headers or {})}
        if raw is not None:
            r = self.client.request(method, url, content=raw, headers={"Content-Type": "application/json", **hdrs})
        else:
            r = self.client.request(method, url, params=params, json=json_body, headers=hdrs)
        assert r.status_code == expect, (method, url, r.status_code, r.text[:800])
        if r.status_code == 204:
            assert r.content == b""
            return None
        body = r.json()
        if r.status_code >= 400:
            self.seen_error_bodies.append(r.text)
        v = response_validator(method, template, r.status_code)
        if v is not None:
            v.validate(body)
        else:
            component("Error").validate(body) if r.status_code >= 400 else None
        json.dumps(body, allow_nan=False)  # never NaN/Infinity on the wire
        return body

    def get(self, template: str, **kw: Any) -> Any:
        return self.call("get", template, **kw)

    def post(self, template: str, **kw: Any) -> Any:
        return self.call("post", template, **kw)

    def patch(self, template: str, **kw: Any) -> Any:
        return self.call("patch", template, **kw)

    def delete(self, template: str, **kw: Any) -> Any:
        return self.call("delete", template, **kw)


def is_valid(validator: Draft202012Validator, instance: Any) -> bool:
    try:
        validator.validate(instance)
    except jsonschema.ValidationError:
        return False
    return True
