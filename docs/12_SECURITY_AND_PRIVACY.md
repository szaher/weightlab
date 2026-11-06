# 12 — Security, Privacy and Threat Model

## 1. Assets and trust boundaries

**Assets:** Proprietary/local model weights, names/paths revealing user info, project metadata, experiment prompts/answers, computed parameter extracts, local storage and machine availability.  
**Boundaries:** Browser UI -> localhost API -> filesystem reader; separately optional local evaluation subprocess; user-selected files untrusted; any future network connector explicitly untrusted and disabled by default.

## 2. Threats and mitigations

| ID | Threat | Control | Test |
|---|---|---|---|
| SEC-01 | Malicious website probes localhost service | Random startup token delivered via URL fragment (never in HTML or logs), Host allowlist (DNS rebinding), strict same-origin and CORS, CSRF/Origin validation, bound loopback only | Cross-origin and rebinding requests denied; token absent from served HTML and logs |
| SEC-02 | Path traversal from index/model path | Canonical root, relative-only shard names, symlink/reparse-point containment | `../secret` + Windows path tests |
| SEC-03 | Arbitrary deserialization code | SafeTensors only; no `torch.load`/pickle/custom model repo code | Upload .pt rejected |
| SEC-04 | Malformed header causes OOM/overflow | Official validator, upper limits, checked arithmetic, independent file size ranges | Fuzz and property tests |
| SEC-05 | DoS through huge ROI/tile/job | Request budgets, concurrency caps, timeouts, cancellation | 413/422/429 + low RSS |
| SEC-06 | Checkpoint changes, is truncated or replaced after import or during a read | Positioned reads (no mmap in API process, so truncation is a short read not SIGBUS); handle and path identity compared per batch (detects truncate, append, size/mtime change, replace/rename, symlink swap); `consistency=verified` hashes before/after for snapshot guarantees; fail closed with `FILE_CHANGED`; stale cache invalidation | B11 cases on each OS; backend survives; limitation for same-size, same-mtime rewrites documented |
| SEC-07 | Sensitive paths or weights leaked in report | Redacted path default, explicit raw tensor export confirmation, no telemetry | Export privacy tests |
| SEC-08 | Arbitrary process/code launched by inference plugin | Explicit install/enable and declared executable, no shell string interpolation, process isolation | hostile plugin config denied |
| SEC-09 | Browser malicious script injection via model metadata | Render escaped text, CSP, no `dangerouslySetInnerHTML` from filenames | XSS fixtures |
| SEC-10 | Runaway disk cache | Hard disk cap, eviction, temp cleanup, free-space checks | cache saturation test |
| SEC-11 | Binary response confusion / huge payload | Validated payload shapes and `Content-Length`, response limits | malformed tile server tests |
| SEC-12 | Software supply chain | Locked dependencies, SBOM, scanning, signed release planned | CI security gate |

## 3. Filesystem access

User types or pastes the model path (the optional native dialog of doc 09 section 1.1 only fills the same field). Server resolves canonical path, enforces selected root for shard links, only reads regular files, denies device files/pipes/sockets, applies file size/metadata limits. When OS permissions prevent read, return explicit permission error; never ask for administrator privileges by default.

## 4. Localhost API hardening

Bind to `127.0.0.1` by default; `--host ::1` is the only other accepted value (IPv6 loopback), never `0.0.0.0` or `::`. Token: `secrets.token_urlsafe(32)`, new on every launch (restart rotates it), handed to the browser only through the URL fragment printed in the terminal (doc 09). The UI reads it, stores it in `sessionStorage`, then calls `history.replaceState` so it is removed from the address bar and history; any "copy link" action in the UI produces URLs without the fragment. Responses carry `Referrer-Policy: no-referrer`.

- **Host allowlist (DNS rebinding):** requests are accepted only when `Host` is exactly `127.0.0.1:PORT`, `localhost:PORT` or `[::1]:PORT`; anything else gets 403 `FORBIDDEN_ORIGIN` before routing.
- **Origin:** checked on every request that carries it, in addition to the bearer token: it must equal the served origin; otherwise 403. No CORS headers are sent at all (no wildcard, no reflection).
- **CSP for the static UI:** `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'`. No external CDN or fonts.
- Disable the FastAPI Swagger/OpenAPI routes in release builds unless authenticated. Bounded JSON bodies (default 1 MiB) and Pydantic field limits mirroring `contracts/openapi.yaml`. `Cache-Control: no-store` for API responses and downloads. If the user's browser is shared, note other local apps may still have local OS access.
- The offline demo makes no network request; installation itself may need the network (or a local wheelhouse), which is outside the runtime offline guarantee.

## 5. Privacy policy

- R1 works without accounts, remote resources or tracking.
- No silent network requests even for version checks.
- No automatic download of models, dataset or fonts.
- Local file paths are not transmitted to external services.
- Export includes paths only if user explicitly selects option; UI warns if raw weight values are included.
- Optional usage collection requires explicit opt-in, independent of core functionality.
- Project deletion never alters source model files.

## 6. Optional evaluation risks

Inference can execute third-party native libraries; pinned binaries, checksum verification, subprocess isolation, no arbitrary local script plugins by default. Model/tokenizer code from remote repos must never execute silently (`trust_remote_code=False` where applicable). If a plugin cannot enforce the viewer's resource caps, mark it unsupported rather than giving an untrue safety guarantee.

## 7. Release security gate

All high-severity threat cases have automated regression tests; dependencies scanned; no open network listeners; no cookies or tokens exported; sensitive error messages sanitized; CI includes path traversal on Windows and POSIX; known security exceptions documented with severity, owner and remediation before beta release.

## 8. End-to-end security tests (R1)

- Token: generated per launch with sufficient entropy, different after restart, required on every route except `/health`, absent from served HTML, server logs, `Referer` and browser history after load; copied links contain no fragment.
- Host: requests with `Host: evil.example`, a rebinding hostname or a wrong port are rejected before routing; IPv4 and IPv6 loopback behave identically; the server refuses non-loopback `--host` values.
- Origin: a cross-origin request with a valid token is still rejected; no CORS headers are ever present.
- Files: reads outside the selected root (`../`, absolute, symlink and Windows reparse-point escapes) fail with `PATH_OUTSIDE_ROOT`; cache and export downloads require the token and are confined to app directories.
- Limits: oversized bodies 413, too many in-flight requests 429, unknown fields 422.
- Offline: the Playwright run blocks all non-loopback network access and the demo completes.
