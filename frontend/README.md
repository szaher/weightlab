# WeightLab browser UI

This is the React and TypeScript UI for Phase 1 Tasks 3 and 4A (exact ROI explorer). Vite builds it into `../weightlab/ui/static`. The Python package ships that directory, and `weightlab start` serves it on the same loopback origin as the API. Node.js is needed only to build and test the UI. An installed WeightLab never runs Node.

## Commands

```bash
npm ci                              # exact versions from package-lock.json
npm run lint                        # ESLint (typescript-eslint strict; react-hooks; no dangerouslySetInnerHTML)
npm run typecheck                   # tsc -b, strict
npm test                            # Vitest unit and component tests (jsdom)
npm run build                       # typecheck + production build into ../weightlab/ui/static
npx playwright install chromium     # once
npm run e2e                         # Playwright against real `weightlab start` processes
```

The E2E suite starts `uv run --locked weightlab start` from the repository root, each time with a fresh temporary data directory. To use another command, such as an installed wheel, set `WEIGHTLAB_CMD='["/path/to/venv/bin/weightlab"]'`. To save the evidence screenshots, set `WEIGHTLAB_SCREENSHOTS=<dir>`.

The build output is committed. After changing anything under `src/`, run `npm run build` and commit `weightlab/ui/static` along with the change. CI rebuilds the UI and fails if the result differs from the committed build.

## Layout

| Path | Responsibility |
|---|---|
| `src/app/` | Shell: top bar, navigation, auth and reconnect screen, section routing |
| `src/features/workspace/` | First-launch welcome, project workspace |
| `src/features/projects/` | Project list and creation |
| `src/features/models/` | Import form, relink form, model status and warnings |
| `src/features/catalog/` | Explore view, server-side tensor catalog, metadata inspector |
| `src/features/roi/` | Exact ROI explorer: view/window rules (`view.ts`), response validation and exact text (`values.ts`), colour scales (`scale.ts`), Canvas drawing (`paint.ts`), heatmap, values table, legend, selected cell |
| `src/features/settings/` | Theme and read-only display of the backend settings |
| `src/lib/api/` | Same-origin client, error envelope, typed endpoints, catalog query serialization |
| `src/lib/auth/` | Token bootstrap from the URL fragment into `sessionStorage` |
| `src/lib/errors/` | API errors as explanation, code, corrective action and request ID |
| `tests/e2e/` | Journey, security, accessibility and exact-ROI E2E (`roi.spec.ts` decodes the fixtures' bytes itself) |

## Rules this code keeps

- **Access token.** The token comes only from the `#token=` fragment. It is stored in `sessionStorage` and removed from the URL with `history.replaceState`. It is sent only in the `Authorization` header and is never placed in localStorage, cookies, query strings or logs.
- **Network.** All requests are relative `/api/v1/...` paths on the page's own origin: no CORS, no external resources, and no fonts beyond the system fonts. The CSP allows only `'self'`.
- **Untrusted text.** Model and tensor names are rendered as text, and tensor names never appear in URLs (opaque `tensor_id` only).
- **Catalog queries.** Filtering, sorting and paging happen on the server. The browser holds one bounded page (at most 200 rows).
- **Exact values only.** The ROI explorer draws only a response that validates against the request that is current now (same tensor, view and bounds, `coverage: complete`, the dtype's lossless encoding). Float values are the server's float64 numbers, shown with the shortest round-trip text; I64/U64 are never converted to `Number` for display or copy; non-finite cells are patterns plus text, never zeros. Colour is withheld for integers beyond ±(2^53 − 1).
- **Bounded reads.** One ROI request at most in flight; every pan, zoom, slice or tensor change aborts it, and the next is sent after 120 ms of quiet. Windows never exceed the server's `roi_max_cells`. There is no client value cache, so a changed checkpoint surfaces as `FILE_CHANGED` instead of stale values.
- **No invented data.** The UI shows no fake progress, metrics or sample data. Features that are not built yet appear as "planned" and are not clickable.
