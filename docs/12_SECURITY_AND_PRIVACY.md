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
| SEC-06 | Checkpoint changes or is truncated after import or during a read | Fingerprints/stats, positioned reads (no mmap in API process, so truncation is a short read not SIGBUS), re-`fstat` per batch, fail closed with `FILE_CHANGED`, stale cache invalidation | File truncation and replacement during job; backend process survives |
| SEC-07 | Sensitive paths or weights leaked in report | Redacted path default, explicit raw tensor export confirmation, no telemetry | Export privacy tests |
| SEC-08 | Arbitrary process/code launched by inference plugin | Explicit install/enable and declared executable, no shell string interpolation, process isolation | hostile plugin config denied |
| SEC-09 | Browser malicious script injection via model metadata | Render escaped text, CSP, no `dangerouslySetInnerHTML` from filenames | XSS fixtures |
| SEC-10 | Runaway disk cache | Hard disk cap, eviction, temp cleanup, free-space checks | cache saturation test |
| SEC-11 | Binary response confusion / huge payload | Validated payload shapes and `Content-Length`, response limits | malformed tile server tests |
| SEC-12 | Software supply chain | Locked dependencies, SBOM, scanning, signed release planned | CI security gate |

## 3. Filesystem access

User chooses model root via native path picker or an explicit path. Server resolves canonical path, enforces selected root for shard links, only reads regular files, denies device files/pipes/sockets, applies file size/metadata limits. When OS permissions prevent read, return explicit permission error; never ask for administrator privileges by default.

## 4. Localhost API hardening

Bind to `127.0.0.1` and optionally `::1` only, never `0.0.0.0`. Random opaque token generated on launch and handed to the browser only through the URL fragment printed in the terminal (doc 09). No CORS wildcard; verify Origin for browser mutating requests. Use restrictive CSP (`default-src 'self'`, no remote scripts), no external CDN, disable FastAPI Swagger/OpenAPI route in public release unless authenticated. Defend against DNS rebinding using Host allowlist for localhost and loopback addresses. Normalize/limit JSON body sizes and Pydantic fields. Set no-store for sensitive API responses. If user's browser is shared, note other local apps may still have local OS access.

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
