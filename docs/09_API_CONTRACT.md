# 09 — Local API Contract

**Base URL:** `http://127.0.0.1:{port}/api/v1`  
**Protocol:** JSON metadata + optional compact binary payloads for heatmap tiles, no external API dependencies.  
**Security:** Random local bearer token generated at each launch and delivered in the URL fragment printed by `weightlab start` (`http://127.0.0.1:PORT/#token=...`, the pattern Jupyter uses). Browsers never send the fragment in HTTP requests, so it does not reach server logs; the UI moves it to `sessionStorage`, removes it from the address bar and sends `Authorization: Bearer` on every API call. The static UI shell contains no secret. Bind only loopback; Host allowlist; strict Origin checking, request size limits and CSP. Token is not stored in project exports (ADR-016).  
**Versioning:** `/api/v1` with `schema_version` on every durable record. Unknown optional fields should be tolerated by clients, missing required fields rejected.

The formal YAML (`contracts/openapi.yaml`) is the minimum implementable subset; this document is authoritative for additional behavioral constraints and future endpoints.

## 1. Catalog endpoints

| Method | Path | Body/query | Success | Errors |
|---|---|---|---|---|
| GET | `/health` | none | version, status, offline, capabilities | 500 |
| POST | `/projects` | name | project ID | 400 |
| GET | `/projects` | none | projects | 500 |
| POST | `/projects/{id}/models` | local path and selected root | model metadata | 400/403/404/409/413 |
| GET | `/projects/{id}/models` | none | models | 404 |
| GET | `/models/{model_id}/tensors` | query, role, page, limit | tensor summaries | 404/422 |
| GET | `/models/{model_id}/tensors/{tensor_id}` | none | tensor metadata | 404 |
| POST | `/comparisons` | model_a, model_b, policy | compatibility analysis | 409/422 |
| GET | `/comparisons/{id}` | none | comparison, unmatched pairs | 404 |
| POST/GET | `/projects/{id}/adapters` | base model ID + adapter path | validated adapter / module mapping | 404/409/422 |
| POST/GET | `/projects/{id}/bookmarks` | view configuration and note | saved bookmarks | 404/422 |
| DELETE | `/projects/{id}/bookmarks/{bookmark_id}` | none | no content | 404 |

Tensor IDs are deterministic API-safe opaque IDs mapping to exact raw keys. Never use unescaped arbitrary tensor names as URL path segments.

## 2. ROI request

`POST /tensors/roi` request:

```json
{
  "model_id": "mod_a",
  "tensor_id": "tns_0009",
  "view": {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": false},
  "roi": {"r0": 10, "r1": 14, "c0": 20, "c1": 24},
  "output_dtype": "float32"
}
```

Response:

```json
{
  "schema_version": 1,
  "tensor": {"model_id": "mod_a", "tensor_id": "tns_0009", "raw_key": "model.layers.0.mlp.up_proj.weight"},
  "roi": {"r0": 10, "r1": 14, "c0": 20, "c1": 24},
  "shape": [4, 4], "dtype": "float32",
  "exact": true, "values": [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2],[1.3,1.4,1.5,1.6]],
  "nonfinite_mask": null, "warnings": []
}
```

If any non-finite exists, `values` uses JSON `null` at that position and `nonfinite_mask` provides string enum (`nan`, `pos_inf`, `neg_inf`) for each affected element; JSON `NaN`/`Infinity` is forbidden. ROI max cells (`roi_max_cells`) is 4096 by default, configurable within a hard upper bound; this is the numeric-table budget, separate from the heatmap tile budget `max_tile_cells` = 65,536 (doc 03 section 5). Coordinates are half-open and validated before file read.

## 3. Heatmap tile request

`POST /tiles` with:

```json
{
  "source": {"kind": "tensor", "model_id": "mod_a", "tensor_id": "tns_0009"},
  "view": {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": false},
  "tile": {"level": 2, "x": 0, "y": 1, "size": 256},
  "aggregation": "mean_abs",
  "quality": {"mode": "auto", "seed": 42},
  "comparison_id": null,
  "color": {"mode": "client_side"}
}
```

`quality.mode`: `auto` (default: exact from direct read or pyramid when available; otherwise a row-sampled preview, labeled), `exact` (202 + job if the pyramid is not built yet), `row_sampled` (with optional `sampled_rows_per_block`). Response formats, chosen by `Accept`, both defined in `contracts/openapi.yaml`:

- `application/vnd.weightlab.tile.v1` (default, binary): `rows*cols` float32 little-endian values in row-major order, immediately followed by `rows*cols` uint8 status codes. Body length is exactly `rows*cols*5` bytes. Status: `0` finite, `1` block contains NaN, `2` contains +Inf, `3` contains -Inf, `4` contains several non-finite kinds, `5` no finite data (value is a NaN placeholder and must not be drawn as a number). For aggregate cells the value is the aggregate over finite elements and the status reports non-finite elements in the block.
- `application/json`: only for tiles with <= 16,384 cells (128x128), for tests and debugging; `TileJson` with `values` (`null` where status != 0) and `status`.

Binary metadata headers: `X-WeightLab-Rows`, `X-WeightLab-Cols`, `X-WeightLab-Row-Start`, `X-WeightLab-Row-End`, `X-WeightLab-Col-Start`, `X-WeightLab-Col-End`, `X-WeightLab-Level`, `X-WeightLab-Quality` (`exact` or `row_sampled`), `X-WeightLab-Sampled-Rows`, `X-WeightLab-Aggregation`, `X-WeightLab-Cache-Key`. Full metadata (source identity, true block coverage, finite/non-finite counts, sampling policy) is at `GET /tiles/{cache_key}/metadata`. Cache key must include file fingerprints, coordinates, aggregation, scope, scale/LoRA config, app numerical version and sample seed. Color is client-side so colors can change without disk reread.

## 4. Comparison source

For derived source:

```json
{
  "source": {
    "kind": "delta",
    "comparison_id": "cmp_001",
    "transform": "b_minus_a"
  },
  "tile": {"level": 0, "x": 0, "y": 0, "size": 256},
  "aggregation": "mean",
  "quality": {"mode": "exact"}
}
```

For virtual LoRA delta/effective weight:

```json
{
  "source": {
    "kind": "lora_delta",
    "base_model_id": "mod_a",
    "adapter_id": "adp_001",
    "target_tensor_id": "tns_0009",
    "experiment_multiplier": 1.0
  }
}
```

Adapter experiment scale is separate from native PEFT scale and must be stored. Unverified mapping returns `409 LINEAGE_UNVERIFIED` for virtual merge unless research-mode flag is explicitly recorded.

## 5. Statistics jobs

`POST /jobs`:

```json
{
  "kind": "tensor_stats",
  "source": {"kind": "tensor", "model_id": "mod_a", "tensor_id": "tns_0009"},
  "quality": {"mode": "exact"},
  "budget": {"max_working_set_mib": 768, "max_threads": 2, "max_wall_seconds": 600}
}
```

Response `202`: `{ "job_id": "job_...", "state": "queued", "poll_after_ms": 500 }`. `GET /jobs/{id}` returns state, progress (0..1), estimated bytes, actual bytes scanned, timestamps, error object and result reference. `POST /jobs/{id}/cancel` idempotent; canceled state must not imply a partial exact result. `budget` values are upper bounds requested by the client: the server rejects with 422 `RESOURCE_LIMIT` (server caps in `details`) any value above its configured caps (`memory_budget_mib`, `max_threads`, `max_job_seconds`) and never raises its own limits on request. Jobs that exceed limits during execution return `RESOURCE_LIMIT` with safe suggestions.

## 6. Export

`POST /exports` accepts JSON manifest assembly with model IDs, views and charts, `format` `json|png|markdown` (R1 JSON/PNG, Markdown R2), `strip_absolute_paths=true` default, `include_model_bytes=false` immutable in R1. Return local export job/path under configured export directory, not an arbitrary user-controlled filesystem write.

## 7. Error envelope

```json
{
  "error": {
    "code": "RESOURCE_LIMIT",
    "message": "Exact tile scan would exceed the configured disk read budget.",
    "details": {"estimated_read_mib": 4096, "allowed_mib": 1024},
    "retryable": true,
    "recommended_action": "Use sampled preview or queue a bounded exact job."
  },
  "request_id": "req_0123"
}
```

No stack traces, model bytes, directory listings outside approved root or arbitrary local path disclosures in error payloads.

## 8. WebSocket vs polling

Use ordinary polling in R1 with 500-1000ms adaptive interval for jobs. WebSocket is unnecessary for first release and complicates security. If future streaming is added, secure with same origin/token and bounded message size.

## 9. Contract test list

Validate required/optional fields, coordinate overflow, UTF-8 special tensor keys, concurrent requests, stale fingerprint, nonfinite values, huge shape, 3D slice invalid axes, canceled jobs, binary byte order and body length `rows*cols*5`, status channel codes, job budget above server caps rejected, explicit 413/422/409 errors, no CORS wildcard, auth missing or invalid, token absent from server logs, and closed localhost binding.
