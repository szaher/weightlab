# Phase 1 Task 3: UI shell, model import and tensor catalog (evidence)

Branch `phase-1-ui-shell`, based on `main` at 27fe55f (merged PR #3). Spec 1.2-final is frozen (ADR-019). No file under `docs/*.md`, `contracts/`, `examples/fixtures/` or the spec scripts was edited.

## What works

`weightlab start` serves the API and a React UI on one loopback origin. In the browser a user can:

- create, select and delete projects. Deletion is confirmed first, and the dialog says that checkpoint files are not touched;
- import a `.safetensors` file, a sharded folder or a `.safetensors.index.json` by absolute path, with an explicit `selected_root`;
- see each model's status (ready, changed or disconnected) and relink a moved model;
- search, filter (dtype, role, layer, rank, minimum parameters), sort (name, parameters, bytes, layer) and page the tensor catalog. All of this runs on the server, one bounded page at a time;
- inspect one tensor's metadata: the full raw key with a copy button, shape, rank, dtype, elements, bytes, shard-relative path, byte offsets, and the heuristic role and layer;
- close and restart WeightLab without losing anything. All state lives in the SQLite database; the browser keeps only the session token and the theme.

Every screen uses real API data. There is no mock data, no sample project, and no fake CPU, RAM or progress display.

## Screenshots

These were captured by the E2E suite (`WEIGHTLAB_SCREENSHOTS`) from a real server, with all non-loopback requests blocked.

| File | Shows |
|---|---|
| `screenshots/01-welcome.png` | First launch, no projects |
| `screenshots/02-workspace-imported.png` | Project with `tiny_base.safetensors` imported |
| `screenshots/03-explore-inspector.png` | Search `q_proj`, selected tensor in the inspector |
| `screenshots/04-catalog-3k.png` | 3,001-tensor synthetic model on its last page (200 rows per page) |
| `screenshots/05-error-not-found.png` | Nonexistent path: explanation, `NOT_FOUND`, request ID |
| `screenshots/06-changed-disconnected.png` | A changed model and a disconnected model |
| `screenshots/07-delete-confirm.png` | Delete confirmation: "Your checkpoint files are not deleted or modified." |
| `screenshots/08-explore-light.png`, `08-explore-dark.png` | Light and dark themes |
| `screenshots/09-phone-explore.png` | 390 px wide: no horizontal page overflow |

## Commands run (macOS 26, arm64; Node 24.21.0, Python 3.12)

| Command | Result |
|---|---|
| `npm ci` / `npm install` (lockfile v3, exact versions) | installed |
| `npm run lint` | 0 problems |
| `npm run typecheck` (`tsc -b`, strict, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`) | 0 errors |
| `npm test` (Vitest) | **57 passed** in 8 files |
| `npm run build` | `index.html` + 1 JS (267 kB, 83 kB gzip) + 1 CSS; two builds byte-identical |
| `npx playwright test` against `uv run weightlab start` | **12 passed** (journey 1, security 6, accessibility 5) |
| `npx playwright test` with `WEIGHTLAB_CMD` = the installed wheel's `weightlab` | **12 passed** |
| `uv run ruff format --check . && uv run ruff check .` | clean |
| `uv run mypy --platform {linux,darwin,win32}` | no issues (42 files each) |
| `uv run python scripts/phase1_evidence.py --out evidence/phase1` | **433 passed**, 0 failed, 0 skipped (32 of them new: `tests/api/test_ui_static.py`) |
| `uv run python scripts/build_evaluation_kit.py` (wheel into a fresh venv, then `weightlab start`) | `health 200 ok, projects 200, without token 401, database created: True, UI: shell + 3 assets served` |
| `python scripts/check_spec.py` | `PASS: 154 IDs defined, 281 references checked, 24 files scanned` |
| `python scripts/check_contracts.py` (spec-tool venv) | `PASS: OpenAPI 0.3.0-spec-1.2-final, experiment schema v2, 68 example checks` |
| `python scripts/check_demo_fixtures.py --root examples/fixtures` | `PASS: 170 fixture self-checks` |
| `python scripts/manifest.py --check` | `manifest OK: 322 files` |

Linux and Windows have not been run locally. In CI the `test` matrix (all three OSes) runs the Python suite and the installed-wheel package test, which now checks the UI too. The new `frontend` job on Ubuntu runs lint, typecheck, unit tests, a rebuild compared with the committed build, and the Playwright E2E suite.

## Security and offline verification

| Requirement | How it is verified |
|---|---|
| Token read from the fragment, stored in `sessionStorage` only, fragment removed with `replaceState` | Unit tests in `src/lib/auth/token.test.ts` (7). The journey E2E checks after load: the URL has no `token=`, `sessionStorage` holds the token, `localStorage` and cookies do not, and the rendered DOM does not contain it |
| A fresh token after a restart; an old link is refused with a reconnect screen | Journey E2E (restart step) |
| Missing or wrong token: no project data | Security E2E (5 header variants plus a `?token=` query) and the existing `test_security.py` |
| Host and Origin checks still apply to the API and the UI shell | Security E2E (raw HTTP with forged `Host`/`Origin`) and `test_ui_static.py` |
| The token is not in the served HTML or the server logs | Security E2E: shell body; server stderr has no token; stdout carries it only in the single printed launch URL. Also checked in `test_server_process.py` |
| Static routes cannot bypass the API's authentication | Static files are served only by exact match from an in-memory map built at start-up; a request path is never joined to the filesystem. 20 traversal and unknown-path cases plus 5 methods in `test_ui_static.py`, and raw-HTTP cases in the security E2E. No directory listing |
| Strict CSP unchanged: no inline script or style, no external resources | The CSP header is identical to doc 12 §4. A test inspects the committed build for inline scripts, external references and CSS imports. The E2E fails on any console error, so a CSP violation fails it |
| Model and tensor names rendered as text, not HTML | Security E2E imports a checkpoint whose keys contain `<img onerror>`, `<script>`, `javascript:`, a 600-character name, Unicode and URL metacharacters, plus an HTML project name and display name. Nothing executes, no element is injected, every name appears with its exact characters, and the page does not scroll horizontally. Lint rule: no `dangerouslySetInnerHTML` |
| No network access beyond loopback | Every E2E context aborts and records any request not addressed to 127.0.0.1/localhost/[::1]. The full journey completes with zero blocked requests. A deliberate `fetch("https://example.com")` from the page fails |
| Deleting a project leaves source files untouched | Journey E2E: the SHA-256 of both checkpoints is the same before and after deletion |
| Dropped files are never read or uploaded | `App.test.tsx`: the drop event's default action is prevented and an explanation is shown. The UI has no file input |

## Accessibility (target: WCAG 2.2 AA for the implemented flows)

- **Automated checks:** axe-core (`@axe-core/playwright`, rule tags wcag2a/aa, 21a/aa, 22aa) finds no violations on the welcome, workspace (with inline field errors), explore with inspector, settings and delete-dialog screens, in both light and dark themes and at phone width.
- **Keyboard:** the skip link comes first. Searching and selecting a tensor works by keyboard alone, with a visible focus outline (tested). The dialog takes focus on Cancel, closes on Escape and returns focus to the button that opened it (tested).
- **Statuses:** always an icon and a word, never colour alone. Sortable headers carry `aria-sort`. Live regions announce result counts, busy states and errors.
- **Motion:** `prefers-reduced-motion` turns off transitions.

Automated checks do not replace a manual screen-reader audit, which has not been done.

## Decisions and deviations (frozen spec not edited)

1. **The UI shell is public.** The Guard serves GET/HEAD for `/` and the exact build files (`/assets/*`, `/favicon.svg`) without a token, after the Host and Origin checks. This matches doc 09: "The static UI shell contains no secret". Every `/api/v1` route except `/health` still requires the bearer token. Any other path returns 404, including SPA-style deep links: the UI uses in-memory navigation, not URL routes.
2. **The catalog is paged, not virtualized.** Doc 08 §5 asks for a "virtualized" list. The server pages it instead (25/50/100/200 rows), so the browser never holds more than 200 rows, which bounds memory and DOM size as virtualization would. Sorting is always global, on the server.
3. **No RAM/CPU budget or task indicator in the top bar** (doc 08 §1). No API reports these yet, and the task forbids fake metrics. The top bar shows project, model, LOCAL/OFFLINE (from `/health`) and the backend connection.
4. **No "Explore demo" button** (doc 08 §6). There is no demo endpoint, and the wheel ships no fixture. The welcome screen says that a repository clone has `examples/fixtures/tiny_base.safetensors`. The first-launch flow is one screen, not the three-step wizard, because resource settings are not editable yet.
5. **No "recent paths" list** (doc 08 §5). Keeping absolute paths in the browser would need a privacy decision. It is left for a later task.
6. **Settings are read-only in the UI.** The values only govern features that do not exist yet (tiles, jobs, exports). PATCH `/settings` exists and is tested in the backend.
7. **Model status comes from `GET /projects/{id}/models`.** That response is computed live from a stat of each recorded file, so reloading the Workspace shows the current status.
8. **The production build is committed** in `weightlab/ui/static`. A clone runs `uv run weightlab start` with no Node, and the wheel and the evaluation kit include the UI. CI rebuilds it and fails if the committed build is stale; the build is byte-reproducible.

## Known limitations

- **Platforms tested:** macOS 26 arm64 for everything. Chromium only. Firefox and Safari have not been tested.
- **Model status updates on refresh:** it is checked when the model list loads, not pushed. A file that changes while the page is open shows up after the next reload or action.
- **Relinking a changed file is refused with `FILE_CHANGED`,** as designed. The UI then tells the user to import the model again; it never offers a way around the check.
- **Clipboard:** copying the raw key uses the Clipboard API. On failure the key stays selectable (`user-select: all`).
- **Stale detection** uses quick fingerprints. Same-size rewrites that keep the mtime go unnoticed, as in Phase 1 Tasks 1–2.
- **Not built (later milestones):** heatmaps and tiles, ROI viewer, stats jobs, cancellation, exports, Compare, Adapters.
