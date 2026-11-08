# Phase 1 practitioner session: real-checkpoint workflow (unassisted)

**Status: not yet run.** No practitioner has attempted this workflow, so nothing here is usability evidence yet. This protocol extends the Phase 0.5 kit in `evaluation-kit/` (same recruitment, ID, consent and handling rules in `evaluation-kit/COORDINATOR.md` sections 2–4). It adds a UI task script for the whole-tensor explorer on real checkpoints. Sessions count towards the 5–10-practitioner gate only when they are run by someone outside the project team.

## Setup (the participant does this, from the instructions alone)

1. Install WeightLab as in `evaluation-kit/README.md` section 1.
2. Download the two initial checkpoints at their pinned revisions, outside any repository:
   `python evidence/phase1-real-models/fetch_models.py --dest ~/weightlab-models --tier initial`
   (about 2 × 943 MiB; each file is checked against the SHA-256 in `models.json`).
3. Start the app with `weightlab start` and open the printed URL.

## Tasks (30 minutes, observed or asynchronous; do not explain the UI)

| # | Task | Success criterion | Record |
|---|---|---|---|
| T1 | Create a project and import `Qwen2.5-0.5B` and `Qwen2.5-0.5B-Instruct` | both show **Ready**, 290 tensors | time; any wrong path or import error |
| T2 | Find every attention query projection | role filter `attention.q`, 48 tensors | did they find the role filter? |
| T3 | Open `model.layers.12.mlp.gate_proj.weight` and switch between all aggregations | five aggregations seen | do they understand mean vs mean\_abs vs rms? |
| T4 | Show this matrix at the finest level | they hit the "L0 would need 76 tiles (limit 48)" message and zoom in instead | is the message understood without help? |
| T5 | Zoom and pan to the block at rows ~300, columns ~500 of `layers.0 q_proj`; select it | the selection reads `rows [300, 304) · columns [500, 504)` at L2 | pan/zoom method used (wheel, drag, keys) |
| T6 | Inspect the selected block's exact values | Exact ROI opens with BF16 source values | do they trust "EXACT" vs "float32 visualization"? |
| T7 | Open `model.embed_tokens.weight` | pyramid job runs (≈1.5 s here), then EXACT coverage | do they notice/understand the job panel? |
| T8 | In a terminal, run `weightlab diff` on the base/Instruct pair | `report.md` produced | do they read class **S** as "shape-compatible only, no causal claim"? |

After the tasks, ask the participant to fill in `evaluation-kit/QUESTIONNAIRE.md` sections C–D, and these three questions:

- Which question about these weights could you not answer with the overview and ROI views?
- Which label, badge or message did you misread first?
- What would you do next with this tool on your own checkpoint?

## Observation sheet (one per participant, stored privately as `P0n-phase1.md`)

- Times for T1–T8, and which tasks they abandoned.
- Every question they asked aloud, word for word.
- Hesitations longer than about 10 s, and where they happened.
- Misreadings of coverage badges (EXACT / ROW-SAMPLED / float32 visualization) or of class S.

Commit only aggregated counts and consented quotes (see `evaluation-kit/COORDINATOR.md` section 4).
