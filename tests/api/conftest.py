"""API test fixtures: an app on a temporary data directory, a contract-checking client, and a project with
the provided fixture checkpoints imported in place."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.api.contract import Api
from weightlab.api.app import create_app
from weightlab.api.config import ServerConfig

PORT = 8765
TOKEN = "test-token-" + "x" * 40


@pytest.fixture
def config(tmp_path: Path) -> ServerConfig:
    return ServerConfig(port=PORT, data_dir=tmp_path / "data", token=TOKEN)


@pytest.fixture
def client(config: ServerConfig) -> Iterator[TestClient]:
    with TestClient(create_app(config), base_url=f"http://127.0.0.1:{PORT}") as c:
        yield c


@pytest.fixture
def api(client: TestClient) -> Api:
    return Api(client, TOKEN)


@pytest.fixture
def project(api: Api) -> str:
    return str(api.post("/projects", json_body={"name": "demo"}, expect=201)["id"])


@pytest.fixture
def models_dir(fixtures_dir: Path, tmp_path: Path) -> Path:
    """A private copy of the fixture tree, so tests may move or modify files."""
    dest = tmp_path / "models"
    shutil.copytree(fixtures_dir, dest)
    return dest


def import_model(api: Api, project: str, root: Path, entry: str, expect: int = 201, **extra: Any) -> Any:
    body = {"entry_path": str(root / entry), "selected_root": str(root), **extra}
    return api.post("/projects/{project_id}/models", project_id=project, json_body=body, expect=expect)


def tensors_by_key(api: Api, model_id: str) -> dict[str, dict[str, Any]]:
    page = api.get("/models/{model_id}/tensors", model_id=model_id, params={"limit": 1000}, expect=200)
    return {t["raw_key"]: t for t in page["items"]}


VIEW2D = {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}
VIEW1D = {"row_axis": None, "col_axis": 0, "fixed_indices": {}, "transpose": False}
