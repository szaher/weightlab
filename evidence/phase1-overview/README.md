# Phase 1 PR #8: whole-tensor heatmap explorer (evidence)

Branch `phase-1-whole-tensor-ui`, based on `main` at 61709f8 (merged PR #7). No backend, contract, fixture, frozen document or spec script changed: the UI uses `POST /tiles`, `GET /tiles/{cache_key}/metadata`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel` and `POST /tensors/roi` as frozen in `contracts/openapi.yaml`.

## What works

Explore → select a 2D tensor (or a 2D view of a higher-rank one). The inspector has two tabs on one shared view (axes, slice, transpose):

- **Overview**: the whole view as a Canvas heatmap built from 256 × 256-cell tiles.
  - Pan by dragging or Shift+arrows; zoom toward the pointer with Ctrl/⌘/Alt + wheel, toward the selection with the buttons or +/−; Fit and Reset. The viewport stays coherent for very asymmetric views (centred on the short axis).
  - Level: Auto (finest level whose cells are ≥ 1 CSS px) or any explicit L0…top level; more than 48 tiles for the screen is refused with an explanation.
  - Aggregation: mean, mean_abs, rms, min, max, each explained ("RMS … is not a standard deviation", "mean_abs … magnitudes, not signed weights").
  - Axes in raw view coordinates; hover and selection show half-open raw bounds, the aggregate (labelled float32 visualization), the cell's status bits in words and its coverage.
  - Disclosures: an **EXACT coverage** / **ROW-SAMPLED preview** / **MIXED** badge (outlined solid vs dashed, plus text), rows per block, sampling policy and seed from the tile metadata, source dtype, level and block size, server warnings, a visible-region scale legend, and counts of NaN/±Inf/non-representable cells with patterns.
  - `202`: a "BUILDING exact overview" badge and a job panel with state, progress bar, bytes read of the estimate and Cancel (one request). Completion re-requests the tiles; canceled, failed (with the error code and a recovery action) and interrupted builds are explained, with "Build again".
  - Above `pyramid_exact_max_mib`: the sampled preview, a seed field, and "Build the exact overview…" which first shows the estimated read (the tensor's size) and only then sends `quality.mode: exact`. A transposed view, which the server cannot row-sample, shows the server's reason with "Build the exact overview instead" and "Show the untransposed view".
- **Exact ROI**: the existing explorer, unchanged in how it reads and shows values. "Inspect exact values" opens it on exactly the selected block or rectangle (same view). Regions above `roi_max_cells` are not truncated: the overview offers "Zoom the overview to this region" or an explicitly labelled smaller top-left window.

Rank 0/1 and empty tensors keep the Exact ROI representation only.

## Tests

| Suite | Result (macOS 26, M3 Pro) |
|---|---|
| `npm run lint`, `npm run typecheck` | clean |
| Vitest (`npm test`) | 190 passed (71 new: binary decoding and malformed responses, geometry incl. 257 × 263 and asymmetric views, LRU/scheduler/stale responses/revalidation/429, labels, painting, 13 component tests) |
| Playwright, checkout (`npx playwright test`) | 14 passed, incl. `overview.spec.ts` |
| Playwright, installed wheel (`WEIGHTLAB_CMD=[wheel venv]/bin/weightlab`) | 14 passed |
| `perf.spec.ts` (`WEIGHTLAB_UI_PERF=1`) | passed, numbers below |

`overview.spec.ts` (real server, offline, non-loopback requests blocked): import a synthetic checkpoint; the overview of a 1024 × 2304 F32 tensor; L4 triggers a real `pyramid_build` job (progress panel, then exact); three L4 mean_abs aggregates equal an independent computation from the test's own generator (relative error < 1e-5, the NaN block keeps its finite aggregate); zoom and pan; a selected block opens Exact ROI on `rows [160, 176) × columns [320, 336)` and three cells equal the file's bytes (independent decoder); a 6,400-element rectangle is refused for exact reads and shrunk explicitly; a slow answer of the previous aggregation arrives after switching tensors and is not drawn; the 512 MiB sparse tensor shows a labelled row-sampled preview, seed 7 switches to seeded SplitMix64; transposed sampling shows the server's limitation; an explicit exact build shows "Estimated read: 512.0 MiB", is canceled with exactly one cancel request and nothing becomes exact; page-measured tile concurrency ≤ 6 (2 = the server's `max_active_tiles`); `FILE_CHANGED` after the file's mtime changes; axe (WCAG 2.2 AA tags) clean.

## Browser performance (`browser_perf_macos.json`)

Synthetic sparse checkpoint: `layers.0.mlp.up_proj.weight` 8192 × 28672 F32 (896 MiB, below the 1024 MiB pyramid limit) and `model.embed_tokens.weight` 32768 × 16384 F32 (2 GiB, above it). Chromium (Playwright 1.65), macOS 26, M3 Pro.

| Measurement | Value |
|---|---|
| Tensor selected → job panel shown | 148 ms |
| Tensor selected → exact overview (incl. the 896 MiB pyramid build) | 2.98 s |
| Zoom ×32 + 40 pans on the exact tensor, all tiles settled | 1.99 s |
| 2 GiB tensor selected → labelled sampled preview | 1.11 s |
| Zoom ×16 + 24 pans on the sampled tensor, settled | 2.39 s |
| Peak tile requests in flight | 2 (bound: min(6, `max_active_tiles` = 2)) |
| Decoded-tile cache at the end | 7.7 MiB, 27 tiles (bound 64 MiB) |
| JS heap used at the end | 8 MiB |

The decoded-tile cache is the browser's (this tab); the server's pyramid cache (disk LRU, `tile_cache_disk_mib`) is separate and covered in `evidence/phase1-pyramids/`.

## Screenshots (`screenshots/`, from `overview.spec.ts` with `WEIGHTLAB_SCREENSHOTS`)

- `overview-01-building.png`: L4 requested, pyramid job queued, nothing shown as exact.
- `overview-02-exact.png`: exact overview with legend, coverage and precision disclosures.
- `overview-03-roi-drilldown.png`: Exact ROI opened from the selected block.
- `overview-04-sampled.png`: row-sampled preview with rows, policy, server warning and the exact-build action.

## Limitations

- Not run on Linux or Windows locally; CI runs the same Vitest and Playwright suites (checkout and installed wheel) on ubuntu, and the Python suite on all three OSes.
- Browser support verified in Chromium only (the Playwright project).
- The scale is the visible region's; there is no whole-tensor statistic (stats jobs are not built).
- Tiles of a namespace already decoded in this tab are reused after one re-validated tile; a file changed *while* the overview is open is detected on the next tile request, not instantly.
- Hover readouts round the float32 visualization to 6 significant digits; exact values are only in Exact ROI.
- Comparison/LoRA tiles, stats, exports and other job kinds remain out of scope.
