# 18 — Packaging, Operations, Install and Troubleshooting Specification

## 1. Supported install modes

**Primary:** Python wheel with bundled React static distribution. User installs via `uv tool install weightlab` / `pipx install weightlab` once published, then launches `weightlab start`. For source developer install, `uv sync --locked --extra dev` and `npm ci` per root/frontend lockfiles. Package name and CLI examples are *design targets*, not released package claims. Do not promise a PyPI URL before package release.

**Requirements:** Python 3.11+, user-writable config directory, modern browser, read access to model directory, localhost binding. No root/admin permission. No Docker prerequisite. GPU, CUDA, PyTorch, HF token and internet are not required at runtime for R1. Installation itself may need network access (or a pre-downloaded wheelhouse); the offline guarantee starts once dependencies are installed.

## 2. Development environment

```bash
git clone <your-weightlab-repo-url>
cd weightlab
uv sync --locked --extra dev
npm --prefix frontend ci
npm --prefix frontend run build
uv run weightlab doctor
uv run weightlab demo --output ./demo
uv run weightlab diff ./demo/tiny_base.safetensors ./demo/tiny_ft.safetensors --format md
uv run weightlab start --port 0
```

Above commands are **proposed CLI interface** to implement and test; `<your-weightlab-repo-url>` is intentionally a placeholder. No repository is created by this documentation pack.

## 3. CLI contract

```text
weightlab start [--port 0] [--host 127.0.0.1] [--memory-budget-mib 1024]
weightlab demo [--output PATH] [--seed 7]
weightlab diff BASE [DERIVED | --adapter DIR] [--format json|csv|md] [--output PATH] [--memory-budget-mib 1024]
weightlab doctor [--json]
weightlab benchmark [--profile laptop8g] [--output PATH]
weightlab cache status|clear [--project ID]
weightlab projects list
weightlab version
```

`--host` may accept only loopback addresses in R1. Starting app with an occupied port chooses a safe alternate when `--port 0`. Print the localhost URL including the `#token=` fragment to the terminal only (never to log files), a short privacy assurance and the data directory. Never print bearer token to shared logs by default.

## 4. Runtime directories

Use platform-native application-data dirs (e.g. via platformdirs). Example logical layout:

```text
WeightLab/
  config.json
  weightlab.sqlite3
  logs/
  cache/tiles/
  cache/stats/
  exports/
  experiments/
```

Require stable DB migration/version files and bounded cache deletion. Model weights remain where user selected them. Cross-platform paths normalized using `pathlib` and proper Windows path handling.

## 5. Settings (defaults)

```json
{
  "offline": true,
  "telemetry_enabled": false,
  "memory_budget_mib": 1024,
  "max_threads": 2,
  "max_active_tiles": 2,
  "max_heavy_jobs": 1,
  "roi_max_cells": 4096,
  "max_tile_cells": 65536,
  "direct_tile_max_mib": 64,
  "max_lora_scan_gmac": 50,
  "pyramid_exact_max_mib": 1024,
  "pyramid_base_level": 4,
  "tile_size": 256,
  "tile_cache_memory_mib": 128,
  "tile_cache_disk_mib": 1024,
  "default_tile_quality": "auto",
  "max_scan_read_mib_without_confirmation": 512,
  "max_job_seconds": 600,
  "export_absolute_paths": false
}
```

The settings may be tuned after testing on 8GB laptop, and estimates must be tested rather than assumed.

## 6. Logging and diagnostics

Structured local logs with timestamps, request correlation IDs, error codes, model *opaque IDs* instead of raw paths, memory samples and no raw weights. Default INFO with rotation (e.g., 10MB x 3). Debug logging opt-in. `weightlab doctor` reports version, OS, Python, free RAM and cache, DB integrity, local bind policy, safe reader version, possible BLAS thread count and plugin availability. Diagnostic export defaults to redacting absolute paths and prompts.

## 7. Updates, migrations and failure recovery

Versioned DB schema; transactional migrations with pre-upgrade backup. On incompatible DB version, show readable recovery path without deleting data. Cache can always be purged. File moved -> mark model disconnected, not deleted. System crashes -> reclassify active jobs as `interrupted`, never as success. Undo operations change project configuration only; original model data never modified.

## 8. CI/CD proposal

- PR: lint, formatting, static typing, unit/property/integration/security, frontend component tests, OpenAPI schema validation, and the specification checks of `.github/workflows/spec-checks.yml` (fixture self-check, contract examples, cross-reference check, documentation build reproducibility, manifest check, fixture hashes on Linux/macOS/Windows).
- Nightly: OS/browser matrix, randomized formats, large-sparse and I/O benchmarks.
- Release: reproducible wheel, SBOM, dependency/license report, SHA-256 checksums, changelog, known limitations, tagged version, signed artifact where tooling available.
- R1 first release is beta until one tested clean install each macOS/Windows/Linux and a real 8GB laptop performance report.

## 8a. Source manifest

`MANIFEST.sha256` lists SHA-256 for every file tracked by git (or added and not ignored), excluding the manifest itself. Generate with `python scripts/manifest.py --write`, verify with `python scripts/manifest.py --check`; the spec CI verifies it on a clean clone. Local caches and build products (`.ruff_cache/`, `__pycache__/`, `.venv/`, `node_modules/`, `dist/`, PDFs) are ignored by `.gitignore` and never listed.

## 9. Troubleshooting decisions

| Symptom | Diagnostic | Resolution |
|---|---|---|
| Model opens but no tensors | shard/index mismatch, dtype | show detailed missing key or unsupported dtype |
| Heatmap blank | all-nonfinite, clipping, tile failure | inspect mask/status; no fabricated zeros |
| Memory high | exact scan or browser tile retention | cancel, sampled mode, lower budget, clear cache |
| Delta looks huge | baseline near zero / wrong orientation / incompatible lineage | inspect absolute diff, epsilon and model origin |
| LoRA overlay fails | unsupported adapter type / wrong base / rank mismatch | list exact module errors, no partial merge claim |
| Slow visualization | random I/O, HDD, level too detailed | sampled preview, different zoom, warm cache |
| Browser cannot connect | server stopped, token mismatch, Origin | `weightlab doctor`, new localhost URL |
| Experiment won't run | inference optional plugin or RAM insufficient | keep weight-only app usable; show requirements |

## 10. OSS contributor experience

`CONTRIBUTING.md` must describe fixtures, make target for small check, numerical test conventions, API contract generation, security model, simple demo and issue labels (`bug`, `good first issue`, `numerics`, `reader`, `research`, `performance`, `security`). README states no capability transfer promise and no built-in network requirement.
