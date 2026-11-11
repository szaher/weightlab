# 09 — Local API Contract

**Base URL:** `http://127.0.0.1:{port}/api/v1`  
**Protocol:** JSON metadata + compact binary payloads for heatmap tiles, no external API dependencies.  
**Security:** Random local bearer token (`secrets.token_urlsafe(32)`) generated at each launch and delivered in the URL fragment printed by `weightlab start` (`http://127.0.0.1:PORT/#token=...`, the pattern Jupyter uses). Browsers never send the fragment in HTTP requests, so it does not reach server logs; the UI moves it to `sessionStorage`, removes it from the address bar and history with `history.replaceState`, and sends `Authorization: Bearer` on every API call. The static UI shell contains no secret. Host allowlist, Origin checks and CSP are specified in doc 12 section 4. Token is not stored in project exports (ADR-016).  
**Versioning:** `/api/v1` with `schema_version` on every durable record. `contracts/openapi.yaml` (0.3.0, frozen with blueprint 1.2-final; 0.4.0 adds comparison reports, ADR-020) is binding: every request body is a closed schema (`additionalProperties: false`) with size limits; unknown fields, unknown source kinds and contradictory fields are rejected with 422. Clients tolerate unknown optional *response* fields.  
**Ownership:** IDs are opaque server-issued strings matching `^[a-z]{3}_[A-Za-z0-9]{1,64}$`. Any ID referenced in a request must belong to the request's project; otherwise the response is 404, never 403, so the API is not an existence oracle across projects.

This document adds behavioral rules the schema cannot express. `contracts/examples/api_examples.json` holds positive and negative payloads that `scripts/check_contracts.py` validates against the schema.

## 1. Endpoints (R1)

| Method | Path | Purpose | Notes |
|---|---|---|---|
| GET | `/health` | version, status, offline, capabilities | unauthenticated; no paths or project data |
| GET/POST | `/projects` | list / create | |
| GET/DELETE | `/projects/{id}` | get / delete | delete removes app metadata and (default) cache, never checkpoint files |
| GET/POST | `/projects/{id}/models` | list / import in place | typed absolute path; no upload (section 1.1) |
| POST | `/models/{model_id}/relink` | reconnect a moved model | all per-file quick fingerprints must match, else 409 `FILE_CHANGED` |
| GET | `/models/{model_id}/tensors` | catalog page | filters `q`, `dtype[]`, `role[]`, `layer`, `rank`, `min_numel`; `sort`, `order`, `limit`, `offset` |
| GET | `/models/{model_id}/tensors/{tensor_id}` | tensor metadata | shape, dtype, numel, shard relpath, file byte extent, heuristic role/layer |
| POST | `/tensors/roi` | exact ROI, lossless typed values | section 2 |
| POST | `/tiles` | heatmap tile | section 3 |
| GET | `/tiles/{cache_key}/metadata` | full tile metadata | |
| POST | `/comparisons` | compatibility + lineage assessment | optional `user_attestation` |
| GET | `/comparisons/{id}` | summary with counts and `LineageAssessment` | |
| GET | `/comparisons/{id}/entries` | matched / missing / tied / mismatched rows | paged, filter by status |
| GET | `/projects/{id}/comparisons` | project's comparisons, newest first, with `report` summary and `latest_report_job_id` | ADR-020 (UI discovery after a reload) |
| GET | `/comparisons/{id}/report` | change report of the latest completed `comparison_report` job | ADR-020; 404 until one completed |
| GET/POST | `/projects/{id}/adapters` | list / import PEFT adapter | per-module mapping, native scale, support status |
| GET/POST | `/projects/{id}/bookmarks` | saved views | typed `SavedView` |
| DELETE | `/projects/{id}/bookmarks/{bookmark_id}` | remove bookmark | |
| POST | `/jobs` | queue bounded job | deduplicated, section 5 |
| GET | `/jobs/{id}` | state, progress, typed result | |
| POST | `/jobs/{id}/cancel` | idempotent cancel | |
| POST | `/exports` | export views (JSON/PNG) | job; result carries `export_id` |
| GET | `/exports/{id}` | export record | |
| GET | `/exports/{id}/download` | export bytes | section 6 |
| GET/PATCH | `/settings` | resource and privacy settings | PATCH validates hard bounds; `offline`/`telemetry_enabled` read-only |
| POST | `/dialogs/select-directory` | optional native folder picker | only with capability `native_directory_dialog` |

Tensor IDs are deterministic API-safe opaque IDs mapping to exact raw keys. Never use unescaped arbitrary tensor names as URL path segments.

### 1.1 Importing a model: typed path, no upload

Browsers do not reveal local absolute paths to a web page, and uploading multi-GB checkpoints through HTTP would contradict the read-in-place design. R1 therefore requires **typed or pasted absolute paths** (`entry_path` and `selected_root`) as the mandatory import method. The server resolves and validates them (doc 12 section 3). An optional server-side native folder dialog (`POST /dialogs/select-directory`) runs a platform helper on the host (`osascript` on macOS, PowerShell on Windows, `zenity`/`kdialog` on Linux) when present and returns the chosen path; the first-run workflow never depends on it. Drag-and-drop shows an explanation instead of importing, because dropped files expose no path. Windows paths are accepted as `C:\...` and long-path `\\?\C:\...`; UNC paths (`\\server\share\...`) are accepted with a "network location, reads may be slow" warning. Permission errors return 403 with the OS reason and never suggest elevation.

## 2. ROI request: exact and lossless

`POST /tensors/roi`:

```json
{
  "source": {"kind": "tensor", "model_id": "mod_a", "tensor_id": "tns_0009"},
  "view": {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": false},
  "roi": {"r0": 10, "r1": 14, "c0": 20, "c1": 24},
  "value_encoding": "auto"
}
```

Response (abridged):

```json
{
  "schema_version": 2,
  "source": {"kind": "tensor", "model_id": "mod_a", "tensor_id": "tns_0009"},
  "roi": {"r0": 10, "r1": 14, "c0": 20, "c1": 24},
  "shape": [4, 4], "source_dtype": "BF16",
  "value_encoding": "float_json", "coverage": "complete", "numeric_precision": "lossless",
  "values": [[0.1015625, -0.0, 0.25, 0.5]],
  "nonfinite_mask": null, "warnings": []
}
```

**Source, compute and display dtypes.** The source dtype is what the file stores. For raw tensor sources the ROI is **lossless**:

| Source dtype | `value_encoding` (auto) | Exactness |
|---|---|---|
| F16, BF16, F32, F64 | `float_json`: JSON number that parses to exactly the source value as IEEE float64; the serializer emits the shortest round-trip representation and `-0.0` for negative zero | lossless (F16/BF16/F32 embed exactly in float64) |
| I8, I16, I32, U8, U16, U32 | `int_json` | lossless (within ±(2^53−1)) |
| I64, U64 | `int_decimal_string`: decimal strings, always, regardless of magnitude | lossless |
| BOOL | `bool_json` | lossless |

`value_encoding: "hex_bits"` returns every element as its raw source bit pattern (lowercase hex at source width, e.g. BF16 `"0x7fc1"`), which also preserves NaN payloads; it is available for `kind=tensor` only. In `auto` mode non-finite floats are `null` with `nonfinite_mask` (`nan`, `pos_inf`, `neg_inf`); JSON `NaN`/`Infinity` literals are forbidden. Display rounding (for example 6 significant digits in a table cell) is a client presentation choice; the tooltip and copy action show the full value.

Derived sources (`delta`, `lora_delta`, `virtual_effective`) are computed in float64 and reported with `numeric_precision: "rounded"`; a value whose computation overflows although its inputs are finite is `null` with mask `overflow` and `numeric_precision: "overflowed"`, never presented as a source Inf or as an exact finite value. A request whose result cannot be represented under these rules returns 422 `UNSUPPORTED_PRECISION`.

ROI max cells (`roi_max_cells`) is 4096 by default, configurable within a hard upper bound; this is the numeric-table budget, separate from the heatmap tile budget `max_tile_cells` = 65,536 (doc 03 section 5). Coordinates are half-open, at most 2^53−1 (JavaScript safe integers), and validated before any file read. Every axis that is neither `row_axis` nor `col_axis` must appear exactly once in `fixed_indices`; `row_axis != col_axis`.

## 3. Heatmap tile request

`POST /tiles` (all fields typed in `TileRequest`):

```json
{
  "source": {"kind": "tensor", "model_id": "mod_a", "tensor_id": "tns_0009"},
  "view": {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": false},
  "tile": {"level": 4, "x": 0, "y": 1, "size": 256},
  "aggregation": "mean_abs",
  "quality": {"mode": "auto"}
}
```

Tiles are **lossy visualizations**: display values are float32 (`value_precision: float32_visualization`) and exact numbers always come from section 2. Coverage, however, is never misrepresented: every tile is either `exact` (all covered elements aggregated) or `row_sampled` (labeled).

**Quality decision table.** Let *covered bytes* be the source bytes under the tile at this level, *limit* = `pyramid_exact_max_mib` (default 1024 MiB) applied to the source tensor's on-disk size (for paired sources, the larger input), and *pyramid* the pyramid with this tile's identity (doc 03 section 5).

| Condition | `auto` | `exact` | `row_sampled` |
|---|---|---|---|
| covered bytes <= `direct_tile_max_mib` (default 64 MiB; always true for L0–L3) | 200 exact by direct read | same | 422 (not needed) |
| pyramid ready | 200 exact from pyramid | same | 422 |
| pyramid not ready, source <= limit | 202 + `pyramid_build` job | 202 + job | 422 |
| pyramid not ready, source > limit | 200 `row_sampled`, labeled | 202 + job | 200 `row_sampled` |

Jobs are **deduplicated** by pyramid identity: concurrent requests for the same pyramid receive the same `job_id` (`deduplicated: true`). While a job is pending, the UI shows a "building exact overview, x%" placeholder with the I/O estimate, never a sampled tile in its place, and may show already-available lower levels. After `completed` the client repeats the request and gets 200 exact. Canceling the job returns all waiters to 202-on-request; nothing marks a tile exact unless the pyramid completed. A pyramid enters the disk cache only if it fits `tile_cache_disk_mib` after LRU eviction; otherwise it is kept in memory for the session or the job is rejected with `RESOURCE_LIMIT`.

**Response formats**, chosen by `Accept`:

- `application/vnd.weightlab.tile.v2` (default, binary): `rows*cols` float32 little-endian display values in row-major order, immediately followed by `rows*cols` uint8 **status bitfields**. Body length is exactly `rows*cols*5` bytes.
- `application/json`: only for tiles with <= 16,384 cells (128x128), for tests and debugging (`TileJson`). It carries the identical information: `values[i][j]` is the float32 display value when the cell is drawable and `null` otherwise; `status` is the same bitfield.

**Status bitfield** (one byte per cell; `TileStatus`):

| Bit | Name | Meaning |
|---|---|---|
| 1 | `NO_FINITE` | no finite element in the cell's block; value is a placeholder and is never drawn |
| 2 | `HAS_NAN` | block contains at least one NaN |
| 4 | `HAS_POS_INF` | block contains at least one +Inf |
| 8 | `HAS_NEG_INF` | block contains at least one −Inf |
| 16 | `NOT_REPRESENTABLE` | a finite aggregate exists but cannot be shown as float32 (e.g. F64 1e300, or a delta of 6.8e38) or its computation overflowed; value placeholder, exact value via ROI or stats |
| 32, 64, 128 | reserved | must be 0; clients reject tiles that set them |

A cell is drawable iff `(status & 17) == 0`. The kind of non-finite content is never lost: an L0 NaN cell is `3` (`NO_FINITE|HAS_NAN`), +Inf `5`, −Inf `9`; a finite block containing a NaN and a +Inf is `6` with its finite aggregate as value; an all-non-finite block containing all three kinds is `15`. Legend and hover derive their non-finite labels from these bits. Golden cases: `edge.nonfinite_blocks.weight` in `examples/fixtures/expected_results.json`.

Binary metadata headers: `X-WeightLab-Rows`, `X-WeightLab-Cols`, `X-WeightLab-Row-Start`, `X-WeightLab-Row-End`, `X-WeightLab-Col-Start`, `X-WeightLab-Col-End`, `X-WeightLab-Level`, `X-WeightLab-Quality` (`exact` or `row_sampled`), `X-WeightLab-Value-Precision` (`float32_visualization`), `X-WeightLab-Sampled-Rows`, `X-WeightLab-Aggregation`, `X-WeightLab-Cache-Key`. Full metadata (source identity, view, true block coverage, finite/non-finite counts, invalid pair count, sampling policy) is at `GET /tiles/{cache_key}/metadata`. The cache key is the SHA-256 of the canonical tile identity (doc 03 section 5 and doc 10 section 6). Color is client-side so colors can change without disk reread.

## 4. Derived sources

`Source` is a discriminated union on `kind` with exactly these variants:

| `kind` | Required fields | Meaning |
|---|---|---|
| `tensor` | `model_id`, `tensor_id` | raw tensor |
| `delta` | `comparison_id`, `entry_id`, `transform` (`b_minus_a`, `abs_b_minus_a`, `relative_abs`), optional `epsilon` | elementwise B − A of one matched comparison entry, computed in a paired pass |
| `lora_delta` | `adapter_id`, `module_id`, `experiment_multiplier` | `multiplier * native_scale * B @ A` |
| `virtual_effective` | `adapter_id`, `module_id`, `experiment_multiplier`, `research_mode` | base W + the LoRA delta, visualization only |

```json
{"kind": "delta", "comparison_id": "cmp_001", "entry_id": "ent_0042", "transform": "b_minus_a"}
```

The adapter's experiment multiplier is separate from its native PEFT scale and both are stored. `virtual_effective` requires the adapter's base lineage to be class V, or `research_mode: true`, which is recorded in provenance; otherwise 409 `LINEAGE_UNVERIFIED`. LoRA sources honor the module's `transpose` flag (supported `fan_in_fan_out` only).

## 5. Jobs

`POST /jobs`:

```json
{
  "kind": "tensor_stats",
  "source": {"kind": "tensor", "model_id": "mod_a", "tensor_id": "tns_0009"},
  "consistency": "best_effort",
  "budget": {"max_working_set_mib": 768, "max_threads": 2, "max_wall_seconds": 600}
}
```

Kinds: `tensor_stats`, `pyramid_build` (with `source` and `view`), `compare_stats`, `file_hash` (with `model_id`, no `source`), `comparison_report` (with `comparison_id`, no `source`, `view` or `model_id`; ADR-020); exports create jobs of kind `export`.

`comparison_report` (ADR-020) computes the comparison's change report with the same code as `weightlab diff` (per-tensor metrics, layer and role rollups, compatibility, caveats, `complete`); its `JobResult` is `{kind, comparison_id, complete, published}`. Submission errors: 422 `RESOURCE_LIMIT` when the comparison has more than 32,768 entries or `max_working_set_mib` is below `128 MiB + 16 KiB x entries`; 409 `LINEAGE_UNVERIFIED` for a Class S/X comparison with policy `strict` (create it with `descriptive_only`); 409 `CONFLICT` for a comparison created before ADR-020 (recreate it); 409 `FILE_CHANGED` when either model changed since import. The job fails with `CONFLICT` when the models' structure fingerprint, counts or lineage no longer match the comparison's record. The report is written in the same transaction that marks the job `completed`; it replaces the stored report only when the job was submitted later than the stored one's job (`published: false` otherwise). Canceled, failed and interrupted jobs never publish. Hashing (lineage, attestation, `verified`) checks cancellation and wall time per 4 MiB chunk. An incomplete report (`complete: false`) is stored and served as incomplete. `GET /comparisons/{id}/report` returns the stored snapshot with `consistency`, `created_at` and `current_model_status` (a stat of each model's files; no tensor bytes are read). Response 202 is a `Job`. States: `created → queued → running → completed | failed | canceled`; `queued → canceled`; `running → interrupted` when the app restarts. Timeouts and budget overruns end as `failed` with `RESOURCE_LIMIT`. `result` is a typed `JobResult` and is non-null only for `completed`; canceled, failed and interrupted jobs never carry a partial result. `POST /jobs/{id}/cancel` is idempotent. Identical pending jobs (same kind and identity) are deduplicated. `budget` values are upper bounds requested by the client: the server rejects with 422 `RESOURCE_LIMIT` (server caps in `details`) any value above its configured caps and never raises its own limits on request. `consistency: "verified"` hashes every input file before and after the job (two full reads) and fails with `FILE_CHANGED` on any difference (doc 03 section 6).

## 6. Export

`POST /exports` takes typed `SavedView`s, `format` `json|png` (Markdown is R2), `strip_absolute_paths=true` by default and `include_raw_values=false` by default (true requires an explicit confirmation in the UI). Model bytes are never included in R1. The file is written only under the configured export directory with a server-generated name (`^[A-Za-z0-9._-]{1,128}$`). The completed job's result carries `export_id`; `GET /exports/{id}` returns the record (state, size, SHA-256) and `GET /exports/{id}/download` returns the bytes with `Content-Disposition: attachment` and `Cache-Control: no-store`. Because the download requires the bearer token like every route, the UI fetches it and saves it through a Blob object URL; plain links and query-string tokens are not used.

## 7. Error envelope

```json
{
  "error": {
    "code": "RESOURCE_LIMIT",
    "message": "Exact tile scan would exceed the configured disk read budget.",
    "details": {"estimated_read_mib": 4096, "allowed_mib": 1024},
    "retryable": true,
    "recommended_action": "Queue the exact overview job or zoom in."
  },
  "request_id": "req_0123"
}
```

`code` is one of the enum in `Error` (OpenAPI); `details` is a flat map of scalars. No stack traces, model bytes, directory listings outside approved root or arbitrary local path disclosures in error payloads.

## 8. WebSocket vs polling

Use ordinary polling in R1 with 500-1000ms adaptive interval for jobs (`poll_after_ms`). WebSocket is unnecessary for first release and complicates security. If future streaming is added, secure with same origin/token and bounded message size.

## 9. Contract test list

- Static: `scripts/check_contracts.py` (OpenAPI valid, all refs resolve, every positive example accepted and every negative example rejected, fixture status values valid).
- Generated client: a client generated from `contracts/openapi.yaml` completes every first-run E2E journey (doc 08 section 10) using only described routes.
- Behavior: catalog filters and pagination correct against the fixture catalog; settings PATCH round-trip and bound rejection; relink after move and refusal after change; project delete leaves checkpoint files byte-identical; export download requires the token and returns the recorded SHA-256; all four source kinds round-trip; `comparison_report` results equal `weightlab diff` on the fixture pairs (complete and incomplete) and drift (structure, lineage, counts), changed files, legacy records, strict policy, cancellation (including during hashing), deadlines, crashes and deletion at the commit boundary, opposite completion orders of concurrent jobs, 20,000-entry reports and path redaction leave no wrong report; unknown/missing/contradictory fields, cross-project IDs and oversized bodies rejected; coordinate overflow beyond 2^53−1 rejected; UTF-8 special tensor keys (`tiny_edge.safetensors`); stale fingerprint; lossless ROI for `tiny_precision_*` (U64 max, I64 min, 2^53±1, F64 extremes, signed zero) and `overflowed` for the overflowing delta; binary tile body length `rows*cols*5`, byte order, status bitfields and JSON parity for the non-finite golden blocks; cold/warm tile requests below and above the pyramid limit, duplicate concurrent requests deduplicated, cancel and restart with no false exact badge; job budget above server caps rejected; explicit 404/409/413/422/429 errors; security cases from doc 12 section 8.
