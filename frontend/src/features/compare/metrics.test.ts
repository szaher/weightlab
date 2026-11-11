// Layer x role cells are aggregated from per-tensor records only. Proof of correctness: aggregating the same
// records over a whole layer (resp. role, resp. everything) must reproduce the report's own layer (role, total)
// rollups, which the backend computes with exact power-of-two scaled sums (weightlab/analysis/accumulate.py).
import { describe, expect, it } from "vitest";
import type { ChangeReport, RollupRow, TensorMetric } from "../../lib/api/compare";
import gridJson from "./fixtures/report_grid.json";
import sJson from "./fixtures/report_s.json";
import vJson from "./fixtures/report_v.json";
import { aggregate, buildGrid, cellKey, domain, formatSci, intensity, metricVal, rollupVal, rootSumSquares, type Val } from "./metrics";

const REPORTS: [string, ChangeReport][] = [
  ["grid (multi-tensor cells)", gridJson as unknown as ChangeReport],
  ["V fixture", vJson as unknown as ChangeReport],
  ["S fixture (withheld)", sJson as unknown as ChangeReport],
];
const ROLLUP: Exclude<TensorMetric, "cosine">[] = ["delta_l2", "delta_rms", "relative_l2", "max_abs_delta"];

function expectSame(got: Val, want: Val) {
  expect(got.kind).toBe(want.kind);
  if (got.kind === "value" && want.kind === "value") {
    if (want.v === 0) expect(got.v).toBe(0);
    else expect(Math.abs(got.v - want.v) / Math.abs(want.v)).toBeLessThan(1e-12);
  }
}

describe.each(REPORTS)("aggregation reproduces the CLI rollups: %s", (_name, r) => {
  const eps = r.settings.epsilon_norm;
  const check = (members: typeof r.tensors, row: RollupRow) => {
    const cell = aggregate(members, eps);
    expect(cell.tensors.length).toBe(row.tensors);
    expect(cell.finitePairs).toBe(Number(row.finite_pairs));
    expect(cell.withheld).toBe(row.withheld_tensors);
    for (const k of ROLLUP) expectSame(cell.metrics[k], rollupVal(row, k));
  };
  it("per layer", () => {
    for (const row of r.layers) check(r.tensors.filter((t) => t.layer === (row.layer ?? null)), row);
  });
  it("per role", () => {
    for (const row of r.roles) check(r.tensors.filter((t) => t.role === row.role), row);
  });
  it("in total", () => check(r.tensors, r.totals));
  it("a one-tensor cell is that tensor's own value, cosine included", () => {
    for (const t of r.tensors) {
      const cell = aggregate([t], eps);
      for (const k of [...ROLLUP, "cosine"] as TensorMetric[]) expectSame(cell.metrics[k], metricVal(t.metrics[k]));
    }
  });
});

describe("the layer x role grid", () => {
  const r = gridJson as unknown as ChangeReport;
  const g = buildGrid(r.tensors, r.settings.epsilon_norm);
  it("has real layer indices, then the unlayered row, and keeps other/unknown roles visible", () => {
    expect(g.layers).toEqual([0, 1, 2, null]);
    expect(g.roles).toEqual(["embedding", "attention.q", "attention.k", "mlp.up", "mlp.down", "norm", "lm_head", "other"]);
  });
  it("cells hold exactly the tensors of their layer and role (weight + bias together)", () => {
    const q1 = g.cells.get(cellKey(1, "attention.q"))!;
    expect(q1.tensors.map((t) => t.key).sort()).toEqual(["model.layers.1.self_attn.q_proj.bias", "model.layers.1.self_attn.q_proj.weight"]);
    const total = [...g.cells.values()].reduce((s, c) => s + c.tensors.length, 0);
    expect(total).toBe(r.tensors.length);
    expect(g.cells.get(cellKey(null, "lm_head"))!.tensors).toHaveLength(1);
  });
  it("cosine of a multi-tensor cell is not aggregated", () => {
    expect(g.cells.get(cellKey(0, "attention.q"))!.metrics.cosine.kind).toBe("not_aggregable");
  });
  it("an unchanged tensor is exactly zero, not missing", () => {
    const up1 = g.cells.get(cellKey(1, "mlp.up"))!;
    expect(up1.metrics.delta_l2).toEqual({ kind: "value", v: 0 });
    expect(up1.metrics.cosine).toEqual({ kind: "value", v: 1 });
  });
  it("a near-zero reference divides by epsilon_norm, as the CLI does", () => {
    const kb = r.tensors.find((t) => t.key === "model.layers.0.self_attn.k_proj.bias")!;
    const cell = aggregate([kb], r.settings.epsilon_norm);
    expectSame(cell.metrics.relative_l2, metricVal(kb.metrics.relative_l2));
  });
});

describe("withheld and undefined values are never numbers", () => {
  const r = sJson as unknown as ChangeReport;
  const f64 = r.tensors.find((t) => t.key === "p.f64")!;
  it("an overflowed tensor metric is withheld", () => {
    expect(metricVal(f64.metrics.delta_l2)).toEqual({ kind: "withheld" });
    expect(intensity(metricVal(f64.metrics.delta_l2), "delta_l2", 0, 1, false)).toBeNull();
  });
  it("a cell containing it is withheld, not partial", () => {
    const cell = aggregate(r.tensors, r.settings.epsilon_norm);
    expect(cell.metrics.delta_l2.kind).toBe("withheld");
    expect(cell.metrics.relative_l2.kind).toBe("withheld");
  });
  it("no finite pairs makes rms / relative / max undefined", () => {
    const t = { ...f64, counts: { ...f64.counts, finite_pairs: 0 }, metrics: { ...f64.metrics, delta_l2: { ...f64.metrics.delta_l2, value: 0, numeric_precision: "rounded" as const } } };
    const cell = aggregate([t], 1e-12);
    expect(cell.metrics.delta_l2).toEqual({ kind: "value", v: 0 });
    expect(cell.metrics.delta_rms.kind).toBe("undefined");
    expect(cell.metrics.relative_l2.kind).toBe("undefined");
  });
});

describe("numerics helpers", () => {
  it("rootSumSquares does not overflow near float64's limit", () => {
    expect(rootSumSquares([1e300, 1e300])).toBeCloseTo(Math.SQRT2 * 1e300, -290);
    expect(rootSumSquares([3, 4])).toBe(5);
    expect(rootSumSquares([])).toBe(0);
  });
  it("formatSci is readable scientific notation", () => {
    expect(formatSci(0)).toBe("0");
    expect(formatSci(0.012345)).toBe("0.01235");
    expect(formatSci(1.5e-7)).toBe("1.5e−7");
    expect(formatSci(123456)).toBe("1.235e5");
  });
  it("colour scale: log domain uses positive values; cosine is coloured by 1 − cosine", () => {
    const vals: Val[] = [{ kind: "value", v: 0 }, { kind: "value", v: 1e-6 }, { kind: "value", v: 1e-2 }, { kind: "withheld" }];
    expect(domain(vals, "relative_l2", true)).toEqual({ lo: 1e-6, hi: 1e-2 });
    expect(intensity({ kind: "value", v: 1e-2 }, "relative_l2", 1e-6, 1e-2, true)).toBe(1);
    expect(intensity({ kind: "value", v: 0 }, "relative_l2", 1e-6, 1e-2, true)).toBe(0);
    expect(intensity({ kind: "value", v: 0.5 }, "cosine", 0, 1, false)).toBe(0.5);
  });
});
