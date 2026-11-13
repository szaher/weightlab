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

## Real checkpoints

[`measure_real_compare.py`](measure_real_compare.py) runs Compare report jobs through a real `weightlab start` process on the pinned Qwen2.5 checkpoints from [the Phase 1 real-model validation](../phase1-real-models/README.md) (each file's SHA-256 is checked against its pin first). It compares every stored report with the `report.json` of `weightlab diff` on the same files: tensor order, every per-tensor value, layer and role rollups, totals, compatibility, completeness, caveats and warnings, compared exactly, without tolerance.

```bash
uv run python evidence/phase1-real-models/fetch_models.py --dest ~/weightlab-models --tier all
uv run --with psutil python evidence/alpha2-compare/measure_real_compare.py \
    --models ~/weightlab-models --out evidence/alpha2-compare/results
```

Recorded run ([`results/real_compare.json`](results/real_compare.json)): `weightlab 0.1.0a2`, Apple M3 Pro (12 cores), 36 GiB RAM, macOS 26.0, Python 3.12.11, default settings, job budget 1024 MiB / 1 thread / 600 s. **Verdict: PASS.**

| Pair (A → B) | Class | Strict policy | Consistency | Wall time | Read | Server peak RSS | Same as `weightlab diff` |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B → Instruct (290 tensors) | S | refused, 409 `LINEAGE_UNVERIFIED` | best effort | 5.3 s | 1.84 GiB | 92.5 MiB | identical |
| | | | verified | 6.2 s | 5.52 GiB | 98.4 MiB | identical |
| Qwen2.5-Math-1.5B → Instruct (338 tensors) | S | refused, 409 `LINEAGE_UNVERIFIED` | best effort | 14.2 s | 5.75 GiB | 98.6 MiB | identical |
| | | | verified | 20.9 s | 17.25 GiB | 98.0 MiB | identical |

Both pairs are Class S (no lineage evidence: Hugging Face checkpoints do not name their base file), so the strict policy refuses them and the measured runs use `descriptive_only`, as the UI asks. Verified runs read each file three times (SHA-256 before, the comparison, SHA-256 after). Idle server RSS was 61.7 MiB; the report JSON is about 0.9 MB and 1.05 MB.

**Caveats.** Timings are with a warm OS page cache (the files had just been hashed), so they understate cold-disk time. RSS is psutil sampling every 50 ms on a 36 GiB machine: an observation, not a memory guarantee, and **not evidence for the 8 GiB laptop target**, which remains unverified. These are API runs; the browser screens were exercised on the bundled fixtures above.
