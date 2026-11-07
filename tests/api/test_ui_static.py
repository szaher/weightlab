"""The UI shell served by the Guard (Phase 1 Task 3; doc 09 header, doc 12 sections 4 and 8, SEC-01/09).

The shell is public and token-free, answered from an in-memory map of the build; every API route keeps its
token, Host and Origin checks; anything that is not an exact shell path fails closed with 404."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import PORT, TOKEN
from weightlab.api.app import create_app
from weightlab.api.config import API_PREFIX, ServerConfig
from weightlab.api.security import CSP
from weightlab.ui import STATIC_DIR, Asset, load_assets

INDEX = b'<!doctype html><html><head><script type="module" src="/assets/app-1.js"></script></head><body></body></html>'


@pytest.fixture
def build(tmp_path: Path) -> Path:
    """A synthetic UI build, independent of whether the real frontend was built."""
    root = tmp_path / "static"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_bytes(INDEX)
    (root / "assets" / "app-1.js").write_bytes(b"console.log('weightlab')")
    (root / "assets" / "app-1.css").write_bytes(b"body{}")
    (root / "favicon.svg").write_bytes(b"<svg/>")
    (root / "assets" / "notes.unknownext").write_bytes(b"x")  # unknown type: never served
    (root / "private").mkdir()
    (root / "private" / "secret.txt").write_bytes(b"secret")  # not a served directory
    return root


@pytest.fixture
def ui(config: ServerConfig, build: Path) -> Iterator[TestClient]:
    with TestClient(create_app(config, assets=load_assets(build)), base_url=f"http://127.0.0.1:{PORT}") as c:
        yield c


def test_shell_and_assets_are_served_without_a_token(ui: TestClient) -> None:
    r = ui.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/html; charset=utf-8"
    assert r.content == INDEX
    js = ui.get("/assets/app-1.js")
    assert (js.status_code, js.headers["content-type"]) == (200, "text/javascript; charset=utf-8")
    assert ui.get("/assets/app-1.css").headers["content-type"] == "text/css; charset=utf-8"
    assert ui.get("/favicon.svg").headers["content-type"] == "image/svg+xml"
    head = ui.head("/")
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == str(len(INDEX))


def test_shell_responses_carry_the_strict_headers(ui: TestClient) -> None:
    for path in ("/", "/assets/app-1.js"):
        h = ui.get(path).headers
        assert h["content-security-policy"] == CSP
        assert "unsafe-inline" not in h["content-security-policy"]
        assert h["cache-control"] == "no-store"
        assert h["referrer-policy"] == "no-referrer"
        assert h["x-content-type-options"] == "nosniff"
        assert h["x-frame-options"] == "DENY"
        assert not [k for k in h if k.lower().startswith("access-control-")]
        assert "set-cookie" not in h


@pytest.mark.parametrize(
    "path",
    [
        "/index.html",
        "/assets",
        "/assets/",
        "/assets/../index.html",
        "/assets/%2e%2e/index.html",
        "/assets/..%2findex.html",
        "/assets/notes.unknownext",
        "/private/secret.txt",
        "/private/",
        "/static/index.html",
        "//assets/app-1.js",
        "/assets/app-1.js/",
        "/ASSETS/app-1.js",
        "/weightlab.sqlite3",
        "/data/weightlab.sqlite3",
        "/explore",
        "/docs",
        "/openapi.json",
        "/api",
        "/api/v2/projects",
    ],
)
def test_unknown_paths_fail_closed(ui: TestClient, path: str) -> None:
    r = ui.get(path)
    assert r.status_code == 404, (path, r.status_code)
    assert r.json()["error"]["code"] == "NOT_FOUND"
    assert b"secret" not in r.content


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def test_shell_paths_accept_only_get_and_head(ui: TestClient, method: str) -> None:
    for path in ("/", "/assets/app-1.js"):
        r = ui.request(method, path, headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 404, (method, path)


def test_shell_keeps_host_and_origin_checks(ui: TestClient) -> None:
    for path in ("/", "/assets/app-1.js"):
        assert ui.get(path, headers={"Host": "evil.example"}).status_code == 403
        assert ui.get(path, headers={"Host": f"rebind.example:{PORT}"}).status_code == 403
        assert ui.get(path, headers={"Origin": "http://evil.example"}).status_code == 403
        assert ui.get(path, headers={"Origin": f"http://127.0.0.1:{PORT}"}).status_code == 200


def test_static_paths_do_not_bypass_api_authentication(ui: TestClient) -> None:
    for path in (
        f"{API_PREFIX}/projects",
        f"{API_PREFIX}/projects/",
        f"{API_PREFIX}/projects?x=/assets/app-1.js",
        f"{API_PREFIX}/../assets/app-1.js/../../api/v1/projects",
        f"{API_PREFIX}/settings",
    ):
        r = ui.get(path)
        assert r.status_code in (401, 404), (path, r.status_code)
    assert ui.get(f"{API_PREFIX}/projects").status_code == 401
    assert ui.get(f"{API_PREFIX}/projects", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_loader_skips_symlinks_and_unknown_files(build: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.js"
    outside.write_bytes(b"leak")
    (build / "assets" / "link.js").symlink_to(outside)
    assets = load_assets(build)
    assert set(assets) == {"/", "/assets/app-1.js", "/assets/app-1.css", "/favicon.svg"}
    assert all(isinstance(a, Asset) for a in assets.values())


def test_no_build_means_no_shell_but_a_working_api(config: ServerConfig, tmp_path: Path) -> None:
    assert load_assets(tmp_path / "missing") == {}
    with TestClient(create_app(config, assets={}), base_url=f"http://127.0.0.1:{PORT}") as c:
        r = c.get("/")
        assert r.status_code == 404
        assert "not part of this build" in r.json()["error"]["message"]
        assert c.get(f"{API_PREFIX}/health").status_code == 200


def test_packaged_build_is_present_and_self_contained() -> None:
    """The committed production build: an index with external module scripts only (the CSP has no
    'unsafe-inline'), every referenced asset present, no external URL, and no token placeholder."""
    assets = load_assets(STATIC_DIR)
    assert "/" in assets, "frontend build missing: run `npm ci && npm run build` in frontend/"
    html = assets["/"].body.decode()
    assert "<script>" not in html
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "inline script would violate the CSP"
    assert "<style" not in html
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert refs
    for ref in refs:
        assert ref.startswith("/"), ref
        assert not ref.startswith("//"), ref
        assert ref in assets, ref
    for path, a in assets.items():  # stylesheets load nothing external (fonts, images, @import)
        if a.content_type.startswith("text/css"):
            css = a.body.decode()
            assert "@import" not in css, path
            assert not re.search(r"url\(\s*['\"]?(?:https?:)?//", css), path
    # JS may contain URLs as plain strings (React error links); the browser E2E proves no request leaves loopback.
