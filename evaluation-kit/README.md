# WeightLab evaluation kit (Phase 0.5 practitioner feedback)

Thank you for trying WeightLab. We are asking 5–10 LLM engineers and researchers to try one command, `weightlab diff`, and tell us which questions it did and did not answer. Your answers help prioritize future graphical checkpoint comparison and analysis workflows.

**Time: about 30 minutes.** Install: 5 min. Three examples: 15 min. Questionnaire: 10 min.

## What WeightLab is, and what this version is not

`weightlab diff` compares two SafeTensors checkpoints, or a base checkpoint and a PEFT LoRA adapter, and writes a change report:

- a compatibility verdict;
- per-tensor, per-layer and per-role change tables (L2, RMS, relative L2, cosine, max |Δ|, non-finite counts);
- JSON, CSV and Markdown files.

It runs on the CPU, reads files only (it never modifies them and never runs a model), needs no network or account, and keeps memory bounded (under 100 MiB on a 0.5B pair).

This kit evaluates the pre-release **command-line comparison tool**, not the separate browser tensor explorer:

- A local browser UI exists for individual checkpoint tensor exploration, heatmaps, and exact ROI inspection. **Graphical side-by-side checkpoint comparison and LoRA overlays are not implemented yet.**
- Only SafeTensors files are read. No `.bin`, GGUF or quantized formats.
- No activations, evaluations or inference.
- Supported LoRA is a documented subset. Anything else is reported as unsupported, never guessed.

Please judge it as a pre-release tool. Do tell us when one of these gaps stops you.

## Privacy

- Nothing is sent anywhere. WeightLab makes no network connections.
- We ask only for your questionnaire answers. You do not need to send model files or reports.
- If you choose to attach a report from your own checkpoint, reports carry no absolute paths by default. They do contain:
  - tensor names, shapes and statistics;
  - file names, sizes and SHA-256 digests.

  Look at a report before you share it.
- Your answers are stored under a participant ID (P01, P02, …), not your name. A quote is used only if you tick the consent box in the questionnaire.

## 1. Install (5 minutes)

You need Python 3.12 or newer on macOS, Linux or Windows. Pick one of the options below.

**A. From the kit archive (recommended).** In the unpacked `weightlab-evaluation-kit-*/` directory:

```bash
# with uv (https://docs.astral.sh/uv/)
uv tool install ./wheel/weightlab-*.whl
# or with a plain virtual environment
python3 -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install ./wheel/weightlab-*.whl
```

The wheel pins its dependencies: numpy, fastapi, pydantic, uvicorn and platformdirs. Installing them needs PyPI access once. After that, everything runs offline.

**B. From a clone of the repository:**

```bash
git clone https://github.com/szaher/weightlab && cd weightlab
uv sync --locked
```

In a clone, prefix the commands below with `uv run`. The examples live in `evaluation-kit/` and the fixtures in `examples/fixtures/`.

Check the install:

```bash
weightlab --version            # weightlab 0.0.1.dev0
weightlab diff --help
```

## 2. Run the three examples (15 minutes)

One command runs all three. It checks each report against values that the fixture generator wrote, independently of WeightLab:

```bash
python run_examples.py                          # kit archive (uses the weightlab on your PATH)
uv run python evaluation-kit/run_examples.py    # repository clone
```

It ends with `ALL CHECKS PASSED`. The reports go into a new directory, `weightlab-eval-<time>/`, one subdirectory per example. If a check fails, paste the whole output into question Q14. A failure is useful feedback, not your mistake.

Then **read each report**. Most questions are about the `report.md` files. Each walkthrough says what the example shows and what to look for:

| # | Example | Walkthrough | Shows |
|---|---|---|---|
| 1 | Checkpoint pair | [`examples/01_checkpoint_pair.md`](examples/01_checkpoint_pair.md) | A fine-tune with 10 known changed values: ranking, metrics, compatibility class |
| 2 | LoRA adapter | [`examples/02_lora_adapter.md`](examples/02_lora_adapter.md) | Effective delta s·(B@A) per module, scale from `lora_alpha`/`r` and `rank_pattern` |
| 3 | Incomplete report | [`examples/03_incomplete_report.md`](examples/03_incomplete_report.md) | An unsupported (DoRA) adapter: what the tool refuses to claim |

The fixtures are tiny synthetic files, 8×8 matrices with known values, so that every number can be checked. They say nothing about a real model's behaviour.

**Optional, and the most useful:** run `weightlab diff` on a pair or adapter you know well, from your own work or public. Each walkthrough ends with the command, and with a pinned public pair to download if you have none.

## 3. Questionnaire (10 minutes)

Open [`QUESTIONNAIRE.md`](QUESTIONNAIRE.md). Copy it, fill in the answers and send it back the way the coordinator asked. Short answers are fine. "I don't know" and "I didn't try" are useful answers too.

## Contents

| File | For |
|---|---|
| `README.md` | Participants: this page |
| `examples/0*.md` | Participants: what each example shows and what to look for |
| `run_examples.py` | Participants: runs and checks the three examples (standard library only) |
| `QUESTIONNAIRE.md` | Participants: the 10-minute questionnaire |
| `COORDINATOR.md` | Coordinator: recruiting, sessions, consent, aggregation |
| `FEEDBACK_SUMMARY_TEMPLATE.md` | Coordinator: the structured summary that the Phase 0.5 gate needs |
| `fixtures/`, `wheel/`, `SHA256SUMS` | Kit archive only (built by `scripts/build_evaluation_kit.py`) |

Reference for every option, metric and exit code: `docs/cli/weightlab-diff.md` (in the archive: `weightlab-diff.md`).
