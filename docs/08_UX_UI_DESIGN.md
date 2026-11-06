# 08 — UX, Wireframes, Interaction and Accessibility

## 1. Information architecture

Top-level navigation:

1. **Workspace:** Import files; models; saved projects; storage status.
2. **Explore:** Model layer tree + tensor inspector + heatmap and numerical sidebar.
3. **Compare:** Model A / B selectors, compatibility and synced heatmap views.
4. **Adapters:** LoRA configuration, modules, native/experimental scale, effective-delta explorer.
5. **Experiments (R3):** Create, run, observe and compare intervention tests.
6. **Settings:** Offline policy, resource limits, caches, display preferences, export policy.

One persistent top bar: project, active models, CPU/RAM budget status and task indicator. Browser title identifies local-only mode.

## 2. Explore view wireframe

```text
┌────────────────────────────────────────────────────────────────────────┐
│ WeightLab   Project: Math Investigation    LOCAL    RAM 412/1024 MiB │
├────────────────┬───────────────────────────────────┬──────────────────┤
│ Model catalog  │ Model: base-qwen ...              │ Tensor details   │
│ Search tensors │ Tensor: layers.8...q_proj.weight  │ 2048 x 2048 F16 │
│ > embeddings   │ [Raw] [Abs] [RMS]    Scale: +/-X  │ Mean, RMS, L2   │
│ > Layer 0      │                                   │ Exact / Sampled │
│ > Layer 1      │   [Heatmap Canvas / 256px Tiles]  │ ROI coordinates │
│ > ...          │                                   │ Finite/NaN      │
│ > Layer 8      │                                   │ [Exact ROI]     │
│   q_proj       │                                   │ [Stats Job]     │
│   k_proj       │ [zoom] [pan] [select ROI] [legend]│ [Export]        │
├────────────────┴───────────────────────────────────┴──────────────────┤
│ Task: Tile scan 38%      row-sampled, 64 rows/blk       [Cancel]     │
└────────────────────────────────────────────────────────────────────────┘
```

## 3. Compare view

Header requires A file and B file plus `compatibility badge` with click-through reasons. Shared tensor selector and four modes: A / B / delta / abs delta. Two-panel synchronized zoom with optional lock, legend that states whether scales shared. Below heatmaps: delta metric strip, per-layer summary list and links to exact ROI. Prominent caution when lineage unverified. Optional "Show alignment details" reveals raw keys/shapes/dtypes/mapping. Never imply "B is better" from delta without an eval result.

## 4. LoRA view

Top: Base checkpoint + Adapter directory -> mapping status. Middle: table (`module`, `base weight`, `rank`, `native scale`, `status`) with filter and unsupported details. Inspector: Raw Base / Adapter A / Adapter B / Effective Delta / Virtual Merged (visualization only). Experimental multiplier slider visually separated and disabled until user chooses to enter experiment mode. All charts disclose `delta=s*B@A` and selected parameter names.

## 5. Interaction specifications

- **Open local model:** a path field (paste or type an absolute path; recent paths offered) is the mandatory method, with an optional "Browse..." button only when the server reports the `native_directory_dialog` capability. Browsers cannot reveal local paths and model bytes are never uploaded, so drag-and-drop shows how to copy a path (Finder Option-Command-C, Explorer Shift+right-click "Copy as path") instead of importing. Windows drive, long-path and UNC paths are accepted (UNC with a slow-network warning); permission errors explain the OS reason without suggesting elevation. Progress during validation; errors actionable.
- **Tensor list:** virtualized for thousands of entries, keyboard searchable, sort shape/size/name; semantic tags heuristic.
- **Pan/zoom:** scroll wheel + modifier with accessible zoom buttons; deterministic coordinate mapping; no approximate hover presented as exact.
- **ROI selection:** pointer drag, display raw bounds while dragging, max preview selection limit, exact-read action for selected region.
- **Tooltip:** tensor key, row/col, displayed exact scalar **or** block range + aggregation + count + sample status; source dtype.
- **Color scale:** automatic robust or explicit min/max, auto sign-symmetric for delta, legend pinned and printed on export.
- **Status:** pages show `ready`, `computing`, `sampled`, `exact`, `stale`, `unsupported`, `canceled`, `failed`.
- **Error:** copyable code, explanation and corrective action. No stack trace in production UI.
- **Cancel:** immediately de-queues or sends cooperative cancellation, returns to usable state and releases memory.

## 6. First-launch wizard (max 3 steps)

1. Welcome: "Analyze LLM weights on this computer. No upload or GPU required." Buttons: **Explore demo**, **Open checkpoint**.
2. Resource mode: Laptop-safe defaults (1 GiB backend working-set budget, <=2 CPU workers) adjustable in Settings.
3. Exploration hint: "This image represents weights, not explanations for model behavior." Single linked 30-second tutorial.

## 7. Empty/error states

- No files: `Start with the included 8x8 fixture`.
- Unsupported format: exact filename and supported import types; `Do not run code from this file`.
- Huge exact scan: `Estimated read: 18.4GB; this could take time` and `Sample preview` / `Queue exact` / `Cancel`.
- Unmatched tensor: side-by-side metadata and candidate mappings, no silent patching.
- Access permission: explain OS path access without suggesting unsafe elevation.
- Network denied: app fully works; no login popup.

## 8. Accessibility

WCAG 2.2 AA design target for core operations. Complete keyboard navigation, visible focus, aria labels for color controls and zoom, never use color as sole signal, text legend table and underlying numeric ROI for a visual heatmap. Respect reduced motion, high contrast, non-colorblind discriminability, scalable fonts and screen reader table alternatives. Pan/zoom action has buttons, not gesture-only access. Dialogs trap focus appropriately and return focus on close.

## 9. Visual language

Dark/light theme; neutral structural chrome and restrained accent colors. Avoid rainbow heatmap because it creates false perceptual boundaries. Signed differences use balanced diverging scale; raw magnitudes sequential scale; pattern and tooltip for non-finite. In model overviews show L2-normalized stats only with clear legends; raw magnitude and normalized metrics must never share an unlabeled chart.

## 10. UX acceptance walkthrough

A new user with no model files starts demo, selects a known tensor, sees 2D heatmap, switches to "Exact ROI", gets correct 4x4 values, opens compare demo, observes known changed elements with zero-centered legend, opens compatibility explanation, saves a JSON/PNG report, closes/reopens project, and can cancel a longer scan. Test must be automated by Playwright with offline network blocked.
