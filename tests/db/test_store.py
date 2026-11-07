"""Phase 1 Task 1: repositories: constraints, cascades, batched catalog inserts, catalog queries."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from weightlab.db import Database, store
from weightlab.db.store import FileRow, TensorRow


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = Database(tmp_path / "w.sqlite3")
    with db.session() as c:
        yield c


def tensors(n: int, tag: str = "") -> Iterator[TensorRow]:
    for i in range(n):
        layer = i % 24 if i % 5 else None
        key = f"model.layers.{layer}.mlp.up_proj.weight.{i}" if layer is not None else f"embed.Ünïcode.{i}"
        yield TensorRow(f"tns_{tag}{i:08x}", key, (i % 7 + 1, 3), "BF16" if i % 2 else "F32", "model.safetensors",
                        (i % 7 + 1) * 3 * 2, (i % 7 + 1) * 3, 1000 + i * 64, 1000 + i * 64 + 42,
                        "mlp.up" if layer is not None else "embedding", layer)  # fmt: skip


def add_model(conn: sqlite3.Connection, project_id: str, n: int = 50) -> str:
    tag = store.new_id("tag")[4:]  # tensor IDs are a global primary key (the API derives them from the model ID)
    return store.insert_model(
        conn, project_id=project_id, display_name="m", selected_root="/r", entry_path="/r/model.safetensors",
        fmt="safetensors", lineage={}, files=[FileRow("model.safetensors", 10, 20, "1:2", None, "size_mtime")],
        tensors=tensors(n, tag),
    )  # fmt: skip


def test_foreign_keys_are_enforced(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        add_model(conn, "prj_does_not_exist")
    assert conn.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 0  # whole insert rolled back


def test_tensor_keys_unique_per_model(conn: sqlite3.Connection) -> None:
    p = store.create_project(conn, "p")["id"]
    mid = add_model(conn, p, 3)
    with pytest.raises(sqlite3.IntegrityError), store.transaction(conn):
        conn.execute(
            "INSERT INTO tensors (id, model_id, raw_key, rank, shape_json, dtype, shard_relpath, byte_length, "
            "numel, file_offset_begin, file_offset_end) VALUES ('tns_dup', ?, ?, 1, '[1]', 'F32', 'x', 4, 1, 0, 4)",
            (mid, "model.layers.1.mlp.up_proj.weight.1"),
        )


def test_check_constraints_reject_unknown_enums(conn: sqlite3.Connection) -> None:
    p = store.create_project(conn, "p")["id"]
    mid = add_model(conn, p, 1)
    with pytest.raises(sqlite3.IntegrityError), store.transaction(conn):
        conn.execute("UPDATE models SET status = 'exploded' WHERE id = ?", (mid,))


def test_project_delete_cascades_to_all_children(conn: sqlite3.Connection) -> None:
    p = store.create_project(conn, "p")["id"]
    other = store.create_project(conn, "other")["id"]
    a, b = add_model(conn, p, 10), add_model(conn, p, 10)
    keep = add_model(conn, other, 5)
    store.insert_comparison(conn, project_id=p, model_a_id=a, model_b_id=b, compatibility={},
                            entries=[{"entry_id": "ent_1", "raw_key": "k", "status": "matched"}])  # fmt: skip
    store.insert_adapter(conn, project_id=p, base_model_id=a, adapter_path="/x", config={}, modules=[],
                         status="ready", lineage={}, files={}, warnings=[])  # fmt: skip
    store.insert_bookmark(conn, p, {"source": {}}, "n")
    assert store.delete_project(conn, p)
    survivors = {
        "models": 1,
        "model_files": 1,
        "tensors": 5,
        "comparisons": 0,
        "comparison_entries": 0,
        "adapters": 0,
        "bookmarks": 0,
    }  # only the other project's model, its file row and its 5 tensors
    for table, want in survivors.items():
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == want, table  # noqa: S608 - constants
    assert conn.execute("SELECT COUNT(*) FROM tensors").fetchone()[0] == 5
    assert store.get_model(conn, keep) is not None
    assert store.delete_project(conn, p) is False


def test_batched_catalog_insert_is_fast(conn: sqlite3.Connection) -> None:
    p = store.create_project(conn, "p")["id"]
    t = time.perf_counter()
    mid = add_model(conn, p, 60_000)  # tens of thousands of keys (doc 10 section 3)
    elapsed = time.perf_counter() - t
    assert conn.execute("SELECT COUNT(*) FROM tensors WHERE model_id = ?", (mid,)).fetchone()[0] == 60_000
    assert elapsed < 10.0, elapsed


def test_catalog_filters_sort_and_paging(conn: sqlite3.Connection) -> None:
    p = store.create_project(conn, "p")["id"]
    mid = add_model(conn, p, 200)
    total, rows = store.query_tensors(conn, mid, limit=1000)
    assert total == 200
    assert [r["raw_key"] for r in rows] == sorted(r["raw_key"] for r in rows)
    total, rows = store.query_tensors(conn, mid, q="üNÏCODE", limit=1000)  # Unicode-aware case folding
    assert total == 40
    assert all("Ünïcode" in r["raw_key"] for r in rows)
    total, _ = store.query_tensors(conn, mid, dtypes=["BF16"], roles=["mlp.up"], layer=3)
    assert total == sum(1 for t in tensors(200) if t.dtype == "BF16" and t.layer_index == 3)
    total, rows = store.query_tensors(conn, mid, min_numel=18, sort="numel", order="desc", limit=5)
    assert total == sum(1 for t in tensors(200) if t.numel >= 18)
    assert [r["numel"] for r in rows] == [21] * 5
    _, rows = store.query_tensors(conn, mid, sort="layer_index", order="desc", limit=1000)
    layers = [r["layer_index"] for r in rows]
    assert layers[: layers.index(None)] == sorted((x for x in layers if x is not None), reverse=True)
    assert set(layers[layers.index(None) :]) == {None}  # nulls last
    pages = [store.query_tensors(conn, mid, limit=37, offset=o)[1] for o in range(0, 200, 37)]
    assert [r["id"] for page in pages for r in page] == [r["id"] for r in store.query_tensors(conn, mid, limit=1000)[1]]


def test_cross_project_lookups_return_none(conn: sqlite3.Connection) -> None:
    p, q = store.create_project(conn, "p")["id"], store.create_project(conn, "q")["id"]
    mid = add_model(conn, p, 2)
    assert store.get_model(conn, mid, project_id=p) is not None
    assert store.get_model(conn, mid, project_id=q) is None
    bm = store.insert_bookmark(conn, p, {}, None)
    assert store.delete_bookmark(conn, q, bm["id"]) is False
    assert store.delete_bookmark(conn, p, bm["id"]) is True


def test_settings_round_trip_ignores_unknown_stored_keys(conn: sqlite3.Connection) -> None:
    defaults = {"a": 1, "b": 2}
    assert store.get_settings(conn, defaults) == defaults
    store.put_settings(conn, {"a": 5, "stale_key": 9})
    assert store.get_settings(conn, defaults) == {"a": 5, "b": 2}


def test_new_ids_match_contract_pattern() -> None:
    import re

    pat = re.compile(r"^[a-z]{3}_[A-Za-z0-9]{1,64}$")
    assert all(pat.match(store.new_id(p)) for p in ("prj", "mod", "cmp", "ent", "adp", "bmk"))
    with pytest.raises(ValueError, match="three lowercase"):
        store.new_id("Bad")
