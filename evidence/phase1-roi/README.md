# Phase 1 Task 4A: interactive exact ROI explorer (evidence)

Branch `phase-1-exact-roi-explorer`, based on `main` at e67b254 (merged PR #4). Spec 1.2-final is frozen (ADR-019). No file under `docs/*.md`, `contracts/`, `examples/fixtures/` or the spec scripts was edited.

**Scope.** This is the exact-window vertical slice of Phase 1 Task 4. The heatmap always shows one exact, bounded region read with `POST /tensors/roi`. It is **not** a whole-tensor overview. Tiles, pyramids, sampled previews, jobs, stats and exports are Task 4B and later, and the UI never calls their (deferred) routes.

## What works

Select a tensor in Explore. Next to its metadata you get an **Exact values** panel:

- **Region banner.** `Rows [r0, r1) of R · Columns [c0, c1) of C`, plus "Exact region of N of M cells (x%): this is a window, not the whole tensor". The slice is named for rank > 2.
- **Heatmap tab.**
  - Drawing: Canvas 2D, no chart library. Row and column indices are drawn on the axes. Cells are snapped to device pixels, so they stay crisp at any `devicePixelRatio`.
  - Interaction: hover readout; click, or the keyboard (arrows, Shift+arrows by 8, Page Up/Down, Home/End, +/−), to select a cell. The selected cell gets a two-tone outline.
  - Legend: the scale range, what the scale was computed from, and every non-colour cell kind with its count.
- **Values tab.** An accessible table with one row per cell: row, column, exact value, dtype, state and a copy button. It shows 256 cells per page, can be limited to non-finite cells, and follows the selection.
- **Selected cell.** Exact value, dtype, state, view coordinates and the full tensor index, with **Copy exact value** and **Copy coordinates**.
- **Navigation.**
  - Pan buttons move by half a window; arrow keys move the window when the selection reaches an edge.
  - Zoom halves or doubles the window and keeps the selected cell in place.
  - "Go to cell" (with validation), "Reset to origin", and Transpose (rank ≥ 2).
- **Rank > 2.** Row-axis and column-axis selectors; picking the other axis swaps them. A fixed-index field for every other axis. An invalid index sends nothing and explains why.
- **Rank 0** shows the scalar value. **Rank 1** shows one row of indices. A **zero-size** tensor shows an explanation and sends no request.
- **Colour.** Scales:
  - auto: min–max of the finite values in the loaded region only, labelled so;
  - sign-symmetric: ±max |value| of the region;
  - fixed min/max, with clipped cells counted.

  Palettes: sequential viridis, or diverging blue–white–red. No rainbow.

  Special regions: an all-zero or constant region is drawn mid-colour and stated in text; a region with nothing to colour says so.
- **Non-finite cells.** NaN, +Inf, −Inf and overflow each get their own pattern (`//`, `+`, `−`, `×`) and a text state, never a scale colour or a zero.
- **64-bit integers.**
  - I64 and U64 values are kept as the server's decimal strings and are never converted to `Number`.
  - Values beyond ±(2^53 − 1) are drawn as a dot pattern, with "colour withheld" in the legend, the table and the cell panel.
- **States.**
  - Ordinary states: loading, no tensor selected, empty tensor, unsupported view (client-side), and settings unavailable.
  - Server errors: `INVALID_ROI`, `RESOURCE_LIMIT` (the budget), `UNSUPPORTED_PRECISION`, `UNSUPPORTED_DTYPE`, `FILE_CHANGED` (re-import or relink), and `NOT_FOUND`.
  - Connection and response errors: auth expired or no token, backend unavailable, and an unexpected or invalid response.
  - All of these go through the existing `ErrorPanel` and `describeError`, with an ROI context.

## Exactness rules (`frontend/src/features/roi/values.ts`)

A response is drawn only after the whole response validates against the request that is current **now**:

- same tensor;
- same bounds;
- `shape` = bounds;
- `coverage: complete`;
- `source_dtype` = the catalog dtype;
- `value_encoding` = that dtype's lossless auto encoding;
- every row the right length;
- the mask consistent with the null values;
- `overflow` only with `numeric_precision: overflowed`.

Anything else is a `BAD_RESPONSE` error, and none of the response is shown.

| Encoding | Shown and copied as |
|---|---|
| `float_json` (F64/F32/F16/BF16) | the server's float64 value, printed as the shortest round-trip text (`-0.0` keeps its sign, integral values get `.0`). Its `Number(...)` is bit-identical to the source value |
| `int_json` (I8..I32, U8..U32) | checked to be safe integers, printed exactly |
| `int_decimal_string` (I64/U64) | the string itself, range-checked with `BigInt`; no `Number` conversion for display or copy |
| `bool_json` | `true` / `false` |
| non-finite (mask) | `NaN`, `+Infinity`, `−Infinity`, `overflow`. Copy gives `NaN`, `Infinity`, `-Infinity` |

`coverage` and `numeric_precision` are shown as two separate badges. Derived-source precision (`rounded` or `overflowed`) would be shown as given; the UI never relabels it as lossless.

## Resource bounds

- The window is 64 × 64 by default and never exceeds the server's `roi_max_cells` (`GET /settings`, default 4,096). The UI computes no limit of its own. If zooming out would exceed the budget, the button is disabled and a note names the budget.
- At most one ROI request is in flight per explorer. Every pan, zoom, slice, transpose or tensor change **aborts** it (`AbortController` in `useResource`) and sends the next request after 120 ms of quiet, so rapid panning sends one request for where you stop.
- Every read the explorer issues gets a **generation**. A result is shown only for the read that produced it, once that read has settled successfully: never while a read or retry is pending, never next to an error, and never from an earlier read of the same region. Going A → B → A therefore needs a new successful read of A; old values for A are not shown again (review fix, below). An old window's values never appear under new bounds (unit-tested, including a mutation check: removing the guard fails 3 tests).
- There is **no client value cache**. Every window shown is a fresh read, checked by the server against the import fingerprint. A changed checkpoint therefore always appears as `FILE_CHANGED`, never as stale values. The heatmap reuses one canvas, and its backing store is released when it unmounts.
- There is no automatic scanning: the colour scale uses only the loaded region.

## Review fixes (PR #5 review)

| Finding | Fix | Regression tests |
|---|---|---|
| **P1:** subnormal F64 regions produced NaN colour positions. Halving both operands underflowed: `[0, 5e-324]` and `[-5e-324, 5e-324]` gave `rgb(undefined, …)` | `normalize` uses the plain difference whenever the span `hi − lo` is finite (exact near zero, no underflow). Operands are halved only when the span itself overflows; that span is ≥ ~9e307, so halving cannot underflow it. The palette throws on a non-finite position, and `cellFill` draws any uncomputable position as "colour withheld" rather than inventing a colour. | `scale.test.ts`: the reported regions give exactly `[0, 1]`, `[0, 1]` and `[0, 1]` (plus `[0, 0.5, 1]` cases). Every cell of 8 extreme or subnormal regions × 5 scales × 2 palettes is a valid `rgb(r, g, b)`. Both tests fail with the old formula. |
| **P1:** returning to an earlier ROI could briefly redisplay its old values (A → B pending → file changes → back to A) | Each issued read has a generation that is part of the resource key. Only the settled successful answer of the current read is drawn: not while loading, not with an error, and not from an earlier read of the same region. | `RoiExplorer.test.tsx`: the reported sequence shows no heatmap, values table or selected value while A's second read is pending, and after it returns `FILE_CHANGED`. This test failed before the fix. A second test checks that returning to A shows the new read's numbers. |
| **P2:** a raw tensor response claiming `rounded` or `overflowed` precision was accepted | For `source.kind = "tensor"`, `parseRoi` rejects anything but `numeric_precision: "lossless"` (doc 09 §2), including an `overflow` mask. The selected-cell label reads "Value (exact)" only for lossless results. | `values.test.ts`: rounded, overflowed, overflowed with an overflow mask, and unknown precision are each `BAD_RESPONSE`. |

## Backend change (one bug found by the E2E)

If a checkpoint was edited after import so that it no longer parses (here, one byte appended), `POST /tensors/roi` returned `INVALID_FORMAT` ("trailing bytes after the last tensor") instead of `FILE_CHANGED`. `open_model` now maps a parse failure to `FILE_CHANGED` when the files' stat fingerprint differs from the import, the same way it already handled missing files. A regression test in `tests/api/test_roi_explorer.py` fails without the fix.

## Screenshots (`screenshots/`, captured by `roi.spec.ts` from a real server, offline)

| File | Shows |
|---|---|
| `roi-01-heatmap-selected.png` | 257 × 263 F32 tensor: 64 × 64 exact window, bounds banner, axes, legend, selected cell (5, 7) |
| `roi-02-values-table.png` | Values tab after jumping to (256, 262), with exact copy |
| `roi-03-nonfinite.png` | `demo.special_nonfinite.weight`: NaN, +Inf and −Inf patterns, legend counts |
| `roi-04-nonfinite-values.png` | Values tab filtered to non-finite cells |
| `roi-05-slice-controls.png` | Rank-3 `demo.cube.weight`: row/column axis selectors, fixed index, tensor index `[1, 0, 3]` |
| `roi-06-u64-exact.png` | U64 `18446744073709551615` exact, colour withheld |
| `roi-07-file-changed.png` | Checkpoint changed on disk: `FILE_CHANGED` with re-import or relink guidance, no values |

## Commands run (macOS 26, arm64; Node 24.21.0, Python 3.12)

| Command | Result |
|---|---|
| `npm run lint` | 0 problems |
| `npm run typecheck` | 0 errors |
| `npm test` (Vitest) | **119 passed** in 12 files (57 before; 62 new: `values` 26, `view` 8, `scale`/`paint` 14 [includes devicePixelRatio 1, 1.25, 1.5, 2, 3], `RoiExplorer` 14) |
| `npm run build` | `index.html` + 1 JS (304 kB, 94 kB gzip) + 1 CSS; two builds byte-identical |
| `npx playwright test` against `uv run weightlab start` | **13 passed** (a11y 5, journey 1, **roi 1**, security 6) in 42 s |
| `npx playwright test` with `WEIGHTLAB_CMD` = the installed wheel's `weightlab` | **13 passed** in 40 s |
| `uv run ruff format --check . && uv run ruff check .` | clean |
| `uv run mypy --platform {linux,darwin,win32}` | no issues (42 files each) |
| `uv run python scripts/phase1_evidence.py --out evidence/phase1` | **444 passed**, 0 failed, 0 skipped (11 new: `tests/api/test_roi_explorer.py`) |
| `uv run python scripts/build_evaluation_kit.py` | `ALL CHECKS PASSED`; wheel smoke test `UI: shell + 3 assets served` |
| `python scripts/check_spec.py` | `PASS: 154 IDs defined, 281 references checked, 24 files scanned` |
| `python scripts/check_contracts.py` (spec-tool venv) | `PASS: OpenAPI 0.3.0-spec-1.2-final, experiment schema v2, 68 example checks` |
| `uv run python scripts/check_demo_fixtures.py --root examples/fixtures` | `PASS: 170 fixture self-checks` |

**Flakiness observed on this machine.** Two earlier full E2E runs, made while unrelated virtual machines held the load average at 13–15, each had one test time out. One was the ROI spec, the other the security hostile-names test; in both, the page waited on requests that answered very slowly. Each test passed when run alone (15 s and 3 s), the journey and ROI specs passed together (31 s), and the two final full runs above passed. As in PR #4, CI runs the suite on a dedicated Ubuntu runner, now against the checkout and against the installed wheel.

Linux and Windows were not run locally. In CI, the three-OS `test` matrix runs the Python suite (including the new ROI tests) and the installed-wheel package test. The `frontend` job on Ubuntu runs lint, typecheck, unit tests, the rebuild comparison, and the E2E suite twice: checkout, then installed wheel.

## Numerical correctness evidence

Expected values never come from WeightLab's own reader or serializer:

- **E2E (`frontend/tests/e2e/roi.spec.ts`).**
  - `readTensor` decodes each tensor straight from the fixture bytes in Node: the header JSON plus a `DataView` over the payload. F16 is decoded from its bits; BF16 by shifting into a float32; I64/U64 with `getBigInt64`/`getBigUint64`.
  - Displayed and copied text is compared numerically as float64 with `Object.is`, so −0 and NaN count, and as exact strings for 64-bit integers.
  - Covered:
    - `q_proj` 257 × 263 cells (5, 7), (6, 8), (6, 262), (42, 52) and (256, 262), including through transpose;
    - all 8 cells of `special_nonfinite`;
    - the rank-3 cube slices, which also match `expected_results.json` ([[42, 43, 44], [50, 51, 52]]);
    - the rank-1 vector, the scalar 3.5, and all 8 `bf16_specials`;
    - every `p.i64` value (−2^63 … 2^63−1, including 2^53 + 1), U64 2^64−1, and all 8 `p.f64` values including −0.0, 5e-324 and ±1.797e308.
- **Python (`tests/api/test_roi_explorer.py`).**
  - The exact request bodies the UI sends are checked against `np.frombuffer` over the fixture bytes, using bitwise float64 equality:
    - default and far-corner windows of the ragged 257 × 263 tensor, plus orientation;
    - transpose;
    - rank 0, rank 1, and two rank-3 views;
    - non-finite masks;
    - BF16/F16 specials and 4 × 4 tensors;
    - F64, I64, U64, I32 and BOOL from `tiny_precision_a`.
  - Also: the 4,160-cell budget, invalid bounds, zero-size, and file-changed handling.
- **Vitest.** `values.test.ts` parses wire JSON built from the generator's `expected_results.json`: the 8 × 8 ramp in row-major orientation, `f64_a`, `i64`, `u64`, `i32` and the `bf16_specials_bits`. It also covers 16 malformed or mismatched responses that must be rejected. `RoiExplorer.test.tsx` uses an independent oracle (value = C-order flat index / 4 − 1) to check pan, zoom, transpose, fixed index and axis changes.

## Security and privacy

- **PR #4 guarantees kept.** Same-origin requests through the existing `request()` client: Bearer token from `sessionStorage`, `credentials: omit`, `redirect: error`. The CSP and the Host/Origin checks are unchanged, and the security E2E passes. No token appears in URLs or logs (checked by E2E).
- **Network.** The ROI E2E runs with every non-loopback request blocked and records zero blocked requests, zero console errors and zero CSP violations.
- **Bounded, permitted requests.** It asserts that every ROI request had ≤ 4,096 cells and that no `/tiles`, `/jobs`, `/stats` or `/exports` request was made.
- **Text rendering.** Tensor names and values are rendered as React text; there is no `dangerouslySetInnerHTML` (lint rule).
- **Styles.** Canvas sizes are set through CSSOM (`element.style`), which CSP `style-src 'self'` allows; there are no inline style attributes in served HTML.
- **No upload, no writes.** Tensor values never leave the loopback server and page. Nothing is uploaded or written to model files; reads are positioned reads on the server.
- **Copy is explicit.** Copy writes to the clipboard only when the user clicks.

## Accessibility

- **axe-core** (WCAG 2.0/2.1/2.2 A/AA tags) reports no violations on the heatmap, the Values table and the slice controls. The existing light, dark and phone-width checks still pass.
- One axe finding was fixed during this work, in the shared `CopyButton`: the visible "Copied" was missing from the button's name.
- **Keyboard.** The heatmap is a focusable `role="application"` with described keys. Tabs follow the WAI-ARIA pattern (arrow keys, Home/End, roving `tabindex`). Every pan, zoom and copy action is a labelled button. The selected-cell panel is an `aria-live` region.
- **Colour.** Never the only signal: patterns plus text for special cells, and exact numbers in the table.
- No manual screen-reader audit has been done.

## Known limitations

- **One window at a time.** At most `roi_max_cells` cells (≤ 65,536); there is no whole-tensor overview. A 4096 × 14336 tensor is explored 64 × 64 at a time.
- **Settings.** `roi_max_cells` cannot be edited in the UI yet (the Settings screen is read-only). The API's `PATCH /settings` can change it.
- **Colour scale.** Auto covers the visible region only, by design. Robust percentiles and a whole-tensor scale need the stats jobs of Task 4B.
- **Request timing.** No client cache, so returning to a window re-reads it (one local positioned read). The 120 ms quiet period is fixed.
- **Copy formats.** Copy is per cell; there is no region export (exports are out of scope). `hex_bits` (NaN payloads) is supported by the API but not offered in the UI.
- **Not yet supported:** pointer-drag ROI selection and wheel zoom; zoom uses buttons and the keyboard.
- **Not tested:** Firefox and Safari (E2E runs Chromium only), and Linux/Windows locally.

## Next milestone

**Phase 1 Task 4B:** tiled whole-tensor overview, exact pyramid, sampled previews and bounded background job infrastructure.
