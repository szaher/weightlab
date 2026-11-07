"""Request schemas: closed, strict mirrors of contracts/openapi.yaml 0.3.0 (frozen, binding).

Every model forbids unknown fields and validates strictly (no "1" -> 1 coercion, no bool -> int).
tests/api/test_contract_schemas.py checks these models against the frozen JSON Schema itself: all
contract examples, plus generated mutations of them, must get the same accept/reject verdict.
Only request bodies of implemented routes are modelled here; responses are plain dicts that the
contract tests validate against the frozen response schemas.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

MAX_SAFE_INTEGER = 2**53 - 1

Id = Annotated[str, StringConstraints(pattern=r"^[a-z]{3}_[A-Za-z0-9]{1,64}$", max_length=68)]
SafeInt = Annotated[int, Field(ge=0, le=MAX_SAFE_INTEGER)]
LocalPath = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
Axis = Annotated[int, Field(ge=0, le=31)]
AXIS_KEYS = frozenset(str(i) for i in range(32))


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


# -- projects / models -------------------------------------------------------------------------
class ProjectCreate(Closed):
    name: Annotated[str, StringConstraints(min_length=1, max_length=120)]


class ModelImport(Closed):
    entry_path: LocalPath
    selected_root: LocalPath
    display_name: Annotated[str, StringConstraints(max_length=120)] | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_null_display_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and "display_name" in data and data["display_name"] is None:
            raise ValueError("display_name must be a string when present")
        return data


class ModelRelink(Closed):
    entry_path: LocalPath
    selected_root: LocalPath


# -- views, sources, ROI -----------------------------------------------------------------------
class Bounds(Closed):
    r0: SafeInt
    r1: SafeInt
    c0: SafeInt
    c1: SafeInt


class View(Closed):
    row_axis: Axis | None
    col_axis: Axis | None
    fixed_indices: dict[str, SafeInt] = Field(max_length=30)
    transpose: bool

    @field_validator("fixed_indices")
    @classmethod
    def _axis_keys(cls, v: dict[str, int]) -> dict[str, int]:
        bad = [k for k in v if k not in AXIS_KEYS]
        if bad:
            raise ValueError("fixed_indices keys must be decimal axis numbers 0-31")
        return v


class TensorSource(Closed):
    kind: Literal["tensor"]
    model_id: Id
    tensor_id: Id


class DeltaSource(Closed):
    kind: Literal["delta"]
    comparison_id: Id
    entry_id: Id
    transform: Literal["b_minus_a", "abs_b_minus_a", "relative_abs"]
    epsilon: Annotated[float, Field(gt=0, le=1)] = 1.0e-8


class LoraDeltaSource(Closed):
    kind: Literal["lora_delta"]
    adapter_id: Id
    module_id: Id
    experiment_multiplier: Annotated[float, Field(ge=-16, le=16)]


class VirtualEffectiveSource(Closed):
    kind: Literal["virtual_effective"]
    adapter_id: Id
    module_id: Id
    experiment_multiplier: Annotated[float, Field(ge=-16, le=16)]
    research_mode: bool


Source = Annotated[TensorSource | DeltaSource | LoraDeltaSource | VirtualEffectiveSource, Field(discriminator="kind")]


class RoiRequest(Closed):
    source: Source
    view: View
    roi: Bounds
    value_encoding: Literal["auto", "hex_bits"] = "auto"


# -- comparisons, adapters, bookmarks ----------------------------------------------------------
class ComparisonRequest(Closed):
    project_id: Id
    model_a_id: Id
    model_b_id: Id
    policy: Literal["strict", "descriptive_only"] = "strict"
    user_attestation: bool = False


class AdapterImport(Closed):
    base_model_id: Id
    adapter_directory: LocalPath


class Color(Closed):
    mode: Literal["auto_robust", "fixed", "min_max", "symmetric"]
    vmin: float | None = None
    vmax: float | None = None
    locked_shared_scale: bool = False


Aggregation = Literal["mean", "mean_abs", "rms", "min", "max"]


class SavedView(Closed):
    source: Source
    view: View
    roi: Bounds | None = None
    tile_level: Annotated[int, Field(ge=0, le=52)] | None = None
    aggregation: Aggregation | None = None
    color: Color | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_nulls(cls, data: Any) -> Any:
        if isinstance(data, dict):
            nulls = [k for k in ("roi", "tile_level", "aggregation", "color") if k in data and data[k] is None]
            if nulls:
                raise ValueError(f"{', '.join(nulls)} must not be null")
        return data


class BookmarkCreate(Closed):
    view: SavedView
    note: Annotated[str, StringConstraints(max_length=2000)] | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_null_note(cls, data: Any) -> Any:
        if isinstance(data, dict) and "note" in data and data["note"] is None:
            raise ValueError("note must be a string when present")
        return data


# -- settings ----------------------------------------------------------------------------------
SETTINGS_BOUNDS: dict[str, tuple[int, int]] = {
    "memory_budget_mib": (256, 65536),
    "max_threads": (1, 64),
    "max_active_tiles": (1, 8),
    "max_heavy_jobs": (1, 4),
    "roi_max_cells": (1, 65536),
    "direct_tile_max_mib": (8, 512),
    "pyramid_exact_max_mib": (64, 16384),
    "tile_cache_memory_mib": (16, 4096),
    "tile_cache_disk_mib": (0, 1048576),
    "max_scan_read_mib_without_confirmation": (1, 1048576),
    "max_job_seconds": (1, 86400),
    "max_lora_scan_gmac": (1, 100000),
}


class SettingsPatch(Closed):
    """Bounds equal SETTINGS_BOUNDS (a test keeps the two and the contract in step).

    Any non-empty subset of the writable settings. `offline` and `telemetry_enabled` are read-only
    in R1 and therefore not fields here: sending them is an unknown-field error."""

    memory_budget_mib: Annotated[int, Field(ge=256, le=65536)] | None = None
    max_threads: Annotated[int, Field(ge=1, le=64)] | None = None
    max_active_tiles: Annotated[int, Field(ge=1, le=8)] | None = None
    max_heavy_jobs: Annotated[int, Field(ge=1, le=4)] | None = None
    roi_max_cells: Annotated[int, Field(ge=1, le=65536)] | None = None
    direct_tile_max_mib: Annotated[int, Field(ge=8, le=512)] | None = None
    pyramid_exact_max_mib: Annotated[int, Field(ge=64, le=16384)] | None = None
    tile_cache_memory_mib: Annotated[int, Field(ge=16, le=4096)] | None = None
    tile_cache_disk_mib: Annotated[int, Field(ge=0, le=1048576)] | None = None
    max_scan_read_mib_without_confirmation: Annotated[int, Field(ge=1, le=1048576)] | None = None
    max_job_seconds: Annotated[int, Field(ge=1, le=86400)] | None = None
    max_lora_scan_gmac: Annotated[int, Field(ge=1, le=100000)] | None = None
    export_absolute_paths: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def _non_empty_no_nulls(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data:
                raise ValueError("a settings patch needs at least one field")
            nulls = [k for k, v in data.items() if v is None]
            if nulls:
                raise ValueError(f"{', '.join(sorted(nulls))} must not be null")
        return data

    def changes(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.model_fields_set}


REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "ProjectCreate": ProjectCreate,
    "ModelImport": ModelImport,
    "ModelRelink": ModelRelink,
    "Bounds": Bounds,
    "View": View,
    "RoiRequest": RoiRequest,
    "ComparisonRequest": ComparisonRequest,
    "AdapterImport": AdapterImport,
    "SavedView": SavedView,
    "BookmarkCreate": BookmarkCreate,
    "SettingsPatch": SettingsPatch,
}
