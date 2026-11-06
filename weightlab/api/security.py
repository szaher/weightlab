"""Request guard: pure ASGI middleware applied before routing (doc 12 sections 4 and 8, SEC-01/05/11).

Order of checks, each failing closed with the error envelope:
1. Host is exactly 127.0.0.1:PORT, localhost:PORT or [::1]:PORT (DNS rebinding) -> 403 FORBIDDEN_ORIGIN.
2. Origin, when present (including "null"), equals http://<Host> -> otherwise 403 FORBIDDEN_ORIGIN.
3. Only /api/v1/* is served (no UI in Task 2) -> otherwise 404.
4. Bearer token on every route except GET /api/v1/health, compared in constant time -> 401 UNAUTHORIZED.
5. Body at most max_body_bytes (Content-Length and actual bytes) -> 413 RESOURCE_LIMIT.
6. At most max_in_flight concurrent requests -> 429 RATE_LIMITED.
Every response gets no-store caching, no-referrer, nosniff, frame denial and the doc 12 CSP; no CORS
header is ever sent (any that a handler might set is stripped).
"""

from __future__ import annotations

import hmac
import threading
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from weightlab.api.config import API_PREFIX, ServerConfig
from weightlab.api.errors import ApiError, envelope_bytes, request_id
from weightlab.core.errors import ErrorCode

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)
SECURITY_HEADERS = [
    (b"cache-control", b"no-store"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"content-security-policy", CSP.encode()),
]
_STRIP = {
    b"server",
    b"cache-control",
    b"referrer-policy",
    b"x-content-type-options",
    b"x-frame-options",
    b"content-security-policy",
    b"cross-origin-resource-policy",
}
HEALTH = f"{API_PREFIX}/health"


class Guard:
    def __init__(self, app: ASGIApp, config: ServerConfig) -> None:
        self.app = app
        self.config = config
        self._token = f"Bearer {config.token}".encode()
        self._lock = threading.Lock()
        self._in_flight = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        rid = request_id()
        if scope["type"] != "http":  # no WebSocket in R1 (doc 09 section 8)
            await send({"type": "websocket.close", "code": 1008})
            return
        headers = _headers(scope)
        try:
            self._check(scope, headers)
            body = await self._read_body(receive, headers)
        except ApiError as err:
            await _reply(send, err, rid)
            return
        with self._lock:
            if self._in_flight >= self.config.max_in_flight:
                busy = True
            else:
                busy = False
                self._in_flight += 1
        if busy:
            await _reply(send, ApiError(ErrorCode.RATE_LIMITED, "too many requests in flight"), rid)
            return
        scope.setdefault("state", {})["request_id"] = rid
        try:
            await self.app(scope, _replay(body), _secured(send, rid))
        finally:
            with self._lock:
                self._in_flight -= 1

    def _check(self, scope: Scope, headers: dict[bytes, list[bytes]]) -> None:
        hosts = headers.get(b"host", [])
        host = hosts[0].decode("latin-1") if len(hosts) == 1 else ""
        if host not in self.config.allowed_hosts:
            raise ApiError(ErrorCode.FORBIDDEN_ORIGIN, "request host is not this local WeightLab server")
        origins = headers.get(b"origin")
        if origins is not None and (len(origins) != 1 or origins[0].decode("latin-1") != f"http://{host}"):
            raise ApiError(ErrorCode.FORBIDDEN_ORIGIN, "cross-origin requests are not accepted")
        path = scope.get("path", "")
        if path != API_PREFIX and not path.startswith(API_PREFIX + "/"):
            raise ApiError(ErrorCode.NOT_FOUND, "not found")
        if path == HEALTH and scope.get("method") == "GET":
            return
        auth = headers.get(b"authorization", [])
        if len(auth) != 1 or not hmac.compare_digest(auth[0], self._token):
            raise ApiError(ErrorCode.UNAUTHORIZED, "missing or invalid bearer token")

    async def _read_body(self, receive: Receive, headers: dict[bytes, list[bytes]]) -> bytes:
        limit = self.config.max_body_bytes
        lengths = headers.get(b"content-length", [])
        if len(lengths) > 1:
            raise ApiError(ErrorCode.VALIDATION_ERROR, "conflicting Content-Length headers")
        if lengths:
            try:
                declared = int(lengths[0])
            except ValueError:
                raise ApiError(ErrorCode.VALIDATION_ERROR, "invalid Content-Length") from None
            if declared < 0:
                raise ApiError(ErrorCode.VALIDATION_ERROR, "invalid Content-Length")
            if declared > limit:
                raise ApiError(ErrorCode.RESOURCE_LIMIT, f"request body above {limit} bytes", details={"limit": limit})
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > limit:
                raise ApiError(ErrorCode.RESOURCE_LIMIT, f"request body above {limit} bytes", details={"limit": limit})
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        return b"".join(chunks)


def _headers(scope: Scope) -> dict[bytes, list[bytes]]:
    out: dict[bytes, list[bytes]] = {}
    for k, v in scope.get("headers", []):
        out.setdefault(k.lower(), []).append(v)
    return out


def _replay(body: bytes) -> Receive:
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def _secured(send: Send, rid: str) -> Send:
    async def wrapped(message: Message) -> None:
        if message["type"] == "http.response.start":
            kept = [
                (k, v)
                for k, v in message.get("headers", [])
                if k.lower() not in _STRIP and not k.lower().startswith(b"access-control-")
            ]
            message["headers"] = [*kept, *SECURITY_HEADERS, (b"x-request-id", rid.encode())]
        await send(message)

    return wrapped


async def _reply(send: Send, err: ApiError, rid: str) -> None:
    body = envelope_bytes(err, rid)
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    if err.code is ErrorCode.UNAUTHORIZED:
        headers.append((b"www-authenticate", b"Bearer"))
    await _secured(send, rid)({"type": "http.response.start", "status": err.status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
