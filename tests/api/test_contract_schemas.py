"""Generated contract tests: the API's request models agree with contracts/openapi.yaml (frozen, binding).

1. Every contract example (contracts/examples/api_examples.json) for a modelled schema gets its declared
   verdict from the Pydantic model.
2. Differential: from valid seed instances, systematic mutations (drop each required field, add an unknown
   field, wrong type, null, out-of-range numbers, over-long strings, nested recursively) must get the same
   accept/reject verdict from the Pydantic model and from the frozen JSON Schema.
3. Settings: server defaults validate against `Settings`; PATCH bounds equal the contract's bounds.
Known, intended difference: integral floats such as 1.0 where an integer is required. JSON Schema counts
1.0 as an integer; the API (strict) rejects it. Mutations therefore use non-integral floats.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from tests.api.contract import component, is_valid, spec
from tests.helpers import ROOT
from weightlab.api.config import DEFAULT_SETTINGS
from weightlab.api.schemas import REQUEST_MODELS, SETTINGS_BOUNDS, Source

SOURCE = TypeAdapter(Source)
VIEW = {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "transpose": False}
SEEDS: dict[str, list[dict[str, Any]]] = {
    "ProjectCreate": [{"name": "demo"}],
    "ModelImport": [{"entry_path": "/m/model.safetensors", "selected_root": "/m"},
                    {"entry_path": "/m", "selected_root": "/m", "display_name": "base"}],
    "ModelRelink": [{"entry_path": "/n/model.safetensors", "selected_root": "/n"}],
    "Bounds": [{"r0": 0, "r1": 4, "c0": 2, "c1": 9}],
    "View": [VIEW, {"row_axis": 2, "col_axis": 0, "fixed_indices": {"1": 3}, "transpose": True},
             {"row_axis": None, "col_axis": 0, "fixed_indices": {}, "transpose": False}],
    "RoiRequest": [
        {"source": {"kind": "tensor", "model_id": "mod_a", "tensor_id": "tns_1"}, "view": VIEW,
         "roi": {"r0": 0, "r1": 2, "c0": 0, "c1": 2}},
        {"source": {"kind": "delta", "comparison_id": "cmp_1", "entry_id": "ent_2", "transform": "relative_abs",
                    "epsilon": 0.001}, "view": VIEW, "roi": {"r0": 0, "r1": 2, "c0": 0, "c1": 2},
         "value_encoding": "auto"},
        {"source": {"kind": "lora_delta", "adapter_id": "adp_1", "module_id": "amd_1", "experiment_multiplier": 0.5},
         "view": VIEW, "roi": {"r0": 1, "r1": 2, "c0": 0, "c1": 3}},
        {"source": {"kind": "virtual_effective", "adapter_id": "adp_1", "module_id": "amd_1",
                    "experiment_multiplier": -2, "research_mode": True}, "view": VIEW,
         "roi": {"r0": 0, "r1": 1, "c0": 0, "c1": 1}, "value_encoding": "hex_bits"},
    ],
    "ComparisonRequest": [{"project_id": "prj_1", "model_a_id": "mod_a", "model_b_id": "mod_b"},
                          {"project_id": "prj_1", "model_a_id": "mod_a", "model_b_id": "mod_b",
                           "policy": "descriptive_only", "user_attestation": True}],
    "AdapterImport": [{"base_model_id": "mod_a", "adapter_directory": "/adapters/lora"}],
    "SavedView": [{"source": {"kind": "tensor", "model_id": "mod_a", "tensor_id": "tns_1"}, "view": VIEW,
                   "roi": {"r0": 0, "r1": 1, "c0": 0, "c1": 1}, "tile_level": 3, "aggregation": "rms",
                   "color": {"mode": "fixed", "vmin": -1.5, "vmax": 2, "locked_shared_scale": True}}],
    "BookmarkCreate": [{"view": {"source": {"kind": "tensor", "model_id": "mod_a", "tensor_id": "tns_1"},
                                 "view": VIEW}, "note": "look here"}],
    "SettingsPatch": [{"max_threads": 4}, {"memory_budget_mib": 2048, "export_absolute_paths": True,
                                           "tile_cache_disk_mib": 0, "max_lora_scan_gmac": 100000}],
}  # fmt: skip


def pydantic_ok(name: str, instance: Any) -> bool:
    try:
        if name == "Source":
            SOURCE.validate_python(instance)
        else:
            model: type[BaseModel] = REQUEST_MODELS[name]
            model.model_validate(instance)
    except (ValidationError, ValueError):
        return False
    return True


def test_every_request_model_has_seeds_and_a_contract_schema() -> None:
    schemas = spec()["components"]["schemas"]
    assert set(SEEDS) == set(REQUEST_MODELS)
    for name, seeds in SEEDS.items():
        assert name in schemas, name
        for s in seeds:
            assert is_valid(component(name), s), (name, s)
            assert pydantic_ok(name, s), (name, s)


EXAMPLES = json.loads((ROOT / "contracts/examples/api_examples.json").read_text(encoding="utf-8"))["examples"]
MODELLED = [e for e in EXAMPLES if e["schema"] in REQUEST_MODELS or e["schema"] == "Source"]


@pytest.mark.parametrize("example", MODELLED, ids=[f"{e['schema']}:{e['why']}" for e in MODELLED])
def test_contract_examples_get_their_declared_verdict(example: dict[str, Any]) -> None:
    assert pydantic_ok(example["schema"], example["instance"]) is example["valid"], example["why"]


def test_examples_cover_the_modelled_request_schemas() -> None:
    covered = {e["schema"] for e in MODELLED}
    assert {"Source", "RoiRequest", "SettingsPatch", "ModelImport"} <= covered


def _mutations(value: Any, depth: int = 0) -> Iterator[Any]:
    """Each yielded value is `value` with exactly one change, at any depth."""
    if isinstance(value, dict):
        for key in list(value):
            dropped = {k: v for k, v in value.items() if k != key}
            yield dropped
            for bad in (None, "x", -1, 1.5, True, 2**53, "y" * 5000, [], {}):
                yield {**value, key: bad}
            for inner in _mutations(value[key], depth + 1):
                yield {**value, key: inner}
        yield {**value, "unexpected_field": 1}
    elif isinstance(value, list):
        yield [*value, 1]
    elif isinstance(value, bool):
        yield not value
    elif isinstance(value, int):
        yield from (value + 1, value - 1, 0, 2**53 - 1, 2**53, 17, 65537, 100001)
    elif isinstance(value, float):
        yield -value
        yield value * 1000.5


CASES = [
    (name, i, m) for name, seeds in SEEDS.items() for i, s in enumerate(seeds) for m in _mutations(copy.deepcopy(s))
]


def test_differential_pydantic_vs_frozen_json_schema() -> None:
    disagreements = []
    for name, i, instance in CASES:
        want = is_valid(component(name), instance)
        got = pydantic_ok(name, instance)
        if want != got:
            disagreements.append((name, i, json.dumps(instance)[:300], "schema" if want else "pydantic"))
    assert len(CASES) > 1500
    assert not disagreements, f"{len(disagreements)} disagreements, first: {disagreements[:5]}"


def test_default_settings_match_the_settings_schema() -> None:
    component("Settings").validate(DEFAULT_SETTINGS)


def test_settings_patch_bounds_equal_the_contract() -> None:
    props = spec()["components"]["schemas"]["Settings"]["properties"]
    for name, (lo, hi) in SETTINGS_BOUNDS.items():
        assert (props[name]["minimum"], props[name]["maximum"]) == (lo, hi), name
    writable = set(spec()["components"]["schemas"]["SettingsPatch"]["properties"])
    assert writable == set(SETTINGS_BOUNDS) | {"export_absolute_paths"}
