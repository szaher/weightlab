#!/usr/bin/env python3
"""Static contract checks (docs/13 section 6, cross-file sweep item 2).

- contracts/openapi.yaml is valid OpenAPI 3.1 and every $ref resolves.
- contracts/experiment.schema.json is a valid JSON Schema 2020-12.
- contracts/demo_experiment.json validates.
- Every example in contracts/examples/*.json marked valid passes, and every one marked
  invalid is rejected, by its component schema.
- Tile status values recorded in the fixture expectations are valid TileStatus values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import yaml
from openapi_spec_validator import validate as validate_openapi

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    failures: list[str] = []
    spec = yaml.safe_load((ROOT / "contracts/openapi.yaml").read_text(encoding="utf-8"))
    validate_openapi(spec)

    def component(name: str) -> jsonschema.protocols.Validator:
        schema = {"$ref": f"#/components/schemas/{name}", "components": spec["components"]}
        return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())

    experiment_schema = json.loads((ROOT / "contracts/experiment.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(experiment_schema)
    experiment = jsonschema.Draft202012Validator(experiment_schema, format_checker=jsonschema.FormatChecker())
    experiment.validate(json.loads((ROOT / "contracts/demo_experiment.json").read_text(encoding="utf-8")))

    count = 0
    for path, pick in (
        ("contracts/examples/experiment_examples.json", lambda ex: experiment),
        ("contracts/examples/api_examples.json", lambda ex: component(ex["schema"])),
    ):
        for ex in json.loads((ROOT / path).read_text(encoding="utf-8"))["examples"]:
            errors = list(pick(ex).iter_errors(ex["instance"]))
            if ex["valid"] and errors:
                failures.append(f"{path}: expected VALID ({ex['why']}): {errors[0].message[:160]}")
            if not ex["valid"] and not errors:
                failures.append(f"{path}: expected INVALID but accepted ({ex['why']})")
            count += 1

    status = component("TileStatus")
    nb = json.loads((ROOT / "examples/fixtures/expected_results.json").read_text(encoding="utf-8"))["edge"]["nonfinite_blocks"]
    for group in ("level0_cells", "level4_blocks_mean"):
        for cell, v in nb[group].items():
            if list(status.iter_errors(v["status"])):
                failures.append(f"fixture status {group}[{cell}]={v['status']} is not a valid TileStatus")
            drawable = (v["status"] & 17) == 0
            if drawable != (v["value"] is not None):
                failures.append(f"fixture {group}[{cell}]: drawable={drawable} but value={v['value']}")
            count += 1

    for f in failures:
        print("FAIL", f)
    print(f"{'FAIL' if failures else 'PASS'}: OpenAPI {spec['info']['version']}, experiment schema v2, {count} example checks")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
