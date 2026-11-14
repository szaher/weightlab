# Releasing WeightLab to PyPI

This guide describes the current alpha, **`0.1.0a2` (Alpha 2 — Visual Checkpoint Comparison)**. `0.1.0a1` is already on PyPI and immutable. Publishing is **not** triggered by merging a PR; it requires an explicit GitHub **pre-release**, created by the maintainer. The per-release checklist is in [docs/releases/0.1.0a2.md](docs/releases/0.1.0a2.md#release-checklist).

## 1. Configure PyPI (repository owner, done for 0.1.0a1)

The GitHub workflow uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) with **OIDC**, not an API token.

In PyPI, add a trusted publisher to the existing `weightlab` project. If you do not yet own that PyPI project, set up a [pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) for the name (subject to name availability).

| PyPI field | Value |
| --- | --- |
| PyPI project name | `weightlab` |
| Owner / organization | `szaher` |
| Repository | `weightlab` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

Create a GitHub environment named `pypi` in **Settings → Environments**. Configure required reviewers and restrict its allowed deployment refs to protected release tags if desired. Protect `main` and release tags from unreviewed changes.

The trusted publisher must match **the exact workflow file and environment**. You cannot test OIDC upload to the real PyPI project in a PR.

**Name check:** Confirm on [PyPI](https://pypi.org/project/weightlab/) that the distribution name is available to you or belongs to your account. The CI workflow cannot reserve it.

## 2. Review the release PR and checks

Review version and distribution metadata in `pyproject.toml`, the runtime `weightlab.__version__`, and `uv.lock`. All must agree.

Merge only after both GitHub CI workflows pass. The `package` job checks the built wheel, sdist, compiled browser UI assets, and database migrations. The publisher separately checks package metadata, runs Python tests and refuses a release tag that is not on `main`.

## 3. Create the release deliberately

After this PR is merged into `main` and CI is green:

1. In GitHub → **Releases** → **Draft a new release**, create tag **`v0.1.0a2`**, targeting **`main`**.
2. Mark the GitHub release **Set as a pre-release**. Use [docs/releases/0.1.0a2.md](docs/releases/0.1.0a2.md) as the release notes (it includes the known limitations).
3. Click **Publish release**. This triggers `.github/workflows/publish.yml`.
4. Approve the `pypi` environment deployment if you configured required reviewers.
5. Inspect the GitHub Actions publish logs and the resulting PyPI project page.

Do **not** create a stable `v0.1.0` release via this workflow: its build gate only accepts alpha, beta or RC versions and GitHub pre-releases.

The publish job only downloads the distribution files from the restricted build job and invokes `pypa/gh-action-pypi-publish@release/v1` with `id-token: write`. No long-lived PyPI credentials are stored.

## 4. Install and smoke test

After **successful** publication to PyPI:

```bash
uv tool install "weightlab==0.1.0a2"
weightlab --version
weightlab start
```

Or in a fresh virtual environment:

```bash
python -m pip install "weightlab==0.1.0a2"
weightlab --version
```

Expect `weightlab 0.1.0a2`. Opening `weightlab start` should display the local web UI, with **Compare** enabled in the side navigation.

## Future releases

Use new, never-before-uploaded versions (e.g. `0.1.0a3`, `0.1.0b1`). Update `pyproject.toml`, `weightlab/__init__.py`, `uv.lock`, and user-facing references together. Update `MANIFEST.sha256` with `python scripts/manifest.py --write` and verify `python scripts/manifest.py --check`.

A PyPI release is immutable: you cannot overwrite or republish the same version if the upload goes wrong. Fix the issue and release a new version.

**Licensing:** The repository has no explicit open-source license today. Publishing to PyPI does not grant others a right to redistribute or modify the source. Choose a repository license separately if you want open-source adoption.
