# Alpha 2 — Compare dashboard evidence

Screenshots captured by the browser E2E suite (`frontend/tests/e2e/compare.spec.ts`) against a real `weightlab start`, with every non-loopback request blocked. They are not mockups.

Reproduce:

```bash
cd frontend
npm run build
WEIGHTLAB_SCREENSHOTS=../evidence/alpha2-compare/screenshots npx playwright test compare
```

| File | What it shows | Inputs |
|---|---|---|
| `compare-01-dashboard.png` | Compatibility header (Class V, B − A, complete, consistency), the layer × role grid and tensor rankings | `tiny_base` vs `tiny_ft` (bundled fixtures) |
| `compare-02-grid-filtered.png` | A grid cell selected by keyboard, filtering the tensor ranking | same |
| `compare-03-role-rankings.png` | Role ranking from the report's own rollups | same |
| `compare-04-tensor-detail.png` | Tensor drill-down: metrics with method and precision; the delta heatmap shown as unavailable | same |
| `compare-05-inspect-in-explore.png` | "Inspect in B" opening the existing single-tensor explorer | same |
| `compare-06-class-s-incomplete.png` | Class S warning, incomplete report, withheld values labelled (never zero) | `tiny_precision_a` vs `tiny_precision_b`, policy descriptive only |
| `compare-07-files-changed.png` | Snapshot freshness: files changed after the report; re-run refused with `FILE_CHANGED` advice; previous report kept | `tiny_ft` modified after the report |
| `compare-08-running.png` | A verified analysis running, with progress and cancel | two 1 GiB sparse F32 checkpoints |
| `compare-09-wide-light.png`, `compare-09-wide-dark.png` | A 3,001-tensor report (500 layers) with a bounded table (100 rows per page) | synthetic zero-valued checkpoint and its copy |

The same test checks that the tensor ranking order and every relative L2 value equal `weightlab diff` on the same files, and runs axe (WCAG 2.x A/AA) on the dashboard in light and dark themes.
