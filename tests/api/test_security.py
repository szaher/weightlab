"""Local API hardening (doc 12 sections 4 and 8, SEC-01/05/11): Host allowlist, Origin, token, body and
concurrency limits, headers, strict bodies, loopback-only binding."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import PORT, TOKEN
from tests.api.contract import Api, component
from weightlab.api.app import IMPLEMENTED
from weightlab.api.config import API_PREFIX, ServerConfig, check_host
from weightlab.api.security import Guard

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _err(r: Any, status: int, code: str) -> None:
    assert r.status_code == status, (r.status_code, r.text)
    component("Error").validate(r.json())
    assert r.json()["error"]["code"] == code


@pytest.mark.parametrize(("method", "path"), sorted(IMPLEMENTED - {("get", "/health")}))
def test_token_required_on_every_route_but_health(client: TestClient, method: str, path: str) -> None:
    url = API_PREFIX + path.format(
        project_id="prj_x", model_id="mod_x", tensor_id="tns_x", comparison_id="cmp_x", bookmark_id="bmk_x"
    )
    for headers in (
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": TOKEN},
        {"Authorization": f"bearer {TOKEN}"},
    ):
        r = client.request(method, url, headers=headers, json={})
        _err(r, 401, "UNAUTHORIZED")
        assert r.headers["www-authenticate"] == "Bearer"


def test_health_needs_no_token_and_reveals_nothing(client: TestClient, tmp_path: Path) -> None:
    r = client.get(f"{API_PREFIX}/health")
    assert r.status_code == 200
    assert set(r.json()) == {"status", "version", "offline", "capabilities"}
    assert TOKEN not in r.text
    assert str(tmp_path) not in r.text


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        f"evil.example:{PORT}",
        f"127.0.0.1:{PORT + 1}",
        "127.0.0.1",
        f"localhost.evil.example:{PORT}",
        f"0.0.0.0:{PORT}",
        f"[::]:{PORT}",
        "",
    ],
)
def test_foreign_host_rejected_before_routing(client: TestClient, host: str) -> None:
    for path in (f"{API_PREFIX}/health", f"{API_PREFIX}/projects", "/anything"):
        r = client.get(path, headers={**AUTH, "Host": host})
        _err(r, 403, "FORBIDDEN_ORIGIN")


@pytest.mark.parametrize("host", [f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"[::1]:{PORT}"])
def test_loopback_hosts_behave_identically(client: TestClient, host: str) -> None:
    r = client.get(f"{API_PREFIX}/projects", headers={**AUTH, "Host": host})
    assert r.status_code == 200
    r = client.get(f"{API_PREFIX}/projects", headers={**AUTH, "Host": host, "Origin": f"http://{host}"})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        "null",
        f"http://127.0.0.1:{PORT + 1}",
        f"https://127.0.0.1:{PORT}",
        f"http://localhost:{PORT}",
    ],
)
def test_cross_origin_rejected_even_with_valid_token(client: TestClient, origin: str) -> None:
    r = client.get(f"{API_PREFIX}/projects", headers={**AUTH, "Origin": origin})  # Host is 127.0.0.1:PORT
    _err(r, 403, "FORBIDDEN_ORIGIN")


def test_no_cors_headers_ever(client: TestClient) -> None:
    probes = [
        client.options(
            f"{API_PREFIX}/projects", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"}
        ),
        client.get(f"{API_PREFIX}/projects", headers=AUTH),
        client.get(f"{API_PREFIX}/health", headers={"Origin": f"http://127.0.0.1:{PORT}"}),
        client.post(
            f"{API_PREFIX}/projects", headers={**AUTH, "Origin": f"http://127.0.0.1:{PORT}"}, json={"name": "a"}
        ),
    ]
    for r in probes:
        assert not [h for h in r.headers if h.lower().startswith("access-control-")], dict(r.headers)


def test_security_headers_on_every_response(client: TestClient) -> None:
    for r in (
        client.get(f"{API_PREFIX}/health"),
        client.get(f"{API_PREFIX}/projects"),
        client.get(f"{API_PREFIX}/projects", headers=AUTH),
        client.get("/", headers=AUTH),
    ):
        h = r.headers
        assert h["cache-control"] == "no-store"
        assert h["referrer-policy"] == "no-referrer"
        assert h["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in h["content-security-policy"]
        assert h["x-request-id"].startswith("req_")
        assert "server" not in h
        assert TOKEN not in str(dict(h))


def test_oversized_bodies_rejected(client: TestClient) -> None:
    big = b'{"name": "' + b"a" * (1024 * 1024) + b'"}'
    r = client.post(f"{API_PREFIX}/projects", headers={**AUTH, "Content-Type": "application/json"}, content=big)
    _err(r, 413, "RESOURCE_LIMIT")

    def chunks() -> Any:  # no Content-Length: counted while streaming
        for _ in range(20):
            yield b"x" * 65536

    r = client.post(f"{API_PREFIX}/projects", headers={**AUTH, "Content-Type": "application/json"}, content=chunks())
    _err(r, 413, "RESOURCE_LIMIT")


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        (b'{"name": "a", "name": "b"}', "duplicate key"),
        (b'{"name": NaN}', "NaN token"),
        (b"[1, 2]", "not an object"),
        (b'{"name": "a"', "truncated"),
        (b"\xff\xfe", "not UTF-8"),
        (b'{"name": "\\ud800"}', "lone surrogate"),
        (b'{"name": "a", "extra": 1}', "unknown field"),
        (b"{}", "missing field"),
        (b'{"name": 5}', "wrong type"),
    ],
)
def test_strict_bodies(api: Api, raw: bytes, why: str) -> None:
    body = api.post("/projects", raw=raw, expect=422)
    assert body["error"]["code"] == "VALIDATION_ERROR", why


def test_json_content_type_required(client: TestClient) -> None:
    for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
        r = client.post(f"{API_PREFIX}/projects", headers={**AUTH, "Content-Type": ctype}, content=b'{"name":"a"}')
        _err(r, 422, "VALIDATION_ERROR")


def test_in_flight_limit_answers_429(tmp_path: Path) -> None:
    release = asyncio.Event()

    async def slow_app(scope: Any, receive: Any, send: Any) -> None:
        await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    guard = Guard(slow_app, ServerConfig(port=PORT, data_dir=tmp_path, token=TOKEN, max_in_flight=2))
    scope = {
        "type": "http",
        "method": "GET",
        "path": f"{API_PREFIX}/projects",
        "headers": [(b"host", f"127.0.0.1:{PORT}".encode()), (b"authorization", f"Bearer {TOKEN}".encode())],
    }

    async def one() -> int:
        sent: list[Any] = []

        async def receive() -> Any:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(m: Any) -> None:
            sent.append(m)

        await guard(dict(scope), receive, send)
        return int(sent[0]["status"])

    async def scenario() -> list[int]:
        tasks = [asyncio.create_task(one()) for _ in range(2)]
        await asyncio.sleep(0.05)
        third = await one()  # two are in flight
        release.set()
        return [third, *(await asyncio.gather(*tasks))]

    third, *done = asyncio.run(scenario())
    assert third == 429
    assert done == [200, 200]


def test_non_loopback_bind_refused(tmp_path: Path) -> None:
    for host in ("0.0.0.0", "::", "192.168.1.5", "localhost", "example.com"):
        with pytest.raises(ValueError, match="loopback"):
            check_host(host)
        with pytest.raises(ValueError, match="loopback"):
            ServerConfig(port=PORT, data_dir=tmp_path, host=host)
    assert check_host("127.0.0.1") == "127.0.0.1"
    assert check_host("::1") == "::1"


def test_token_is_fresh_per_config(tmp_path: Path) -> None:
    a, b = ServerConfig(port=PORT, data_dir=tmp_path), ServerConfig(port=PORT, data_dir=tmp_path)
    assert a.token != b.token
    assert len(a.token) >= 43  # token_urlsafe(32)
    assert a.token not in repr(a)
