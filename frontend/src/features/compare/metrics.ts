// Values of the change report as the Compare screen shows them, and the layer x role cells.
//
// Cell aggregation reproduces the CLI's Rollup (weightlab/analysis/accumulate.py) from the per-tensor records of
// exactly the tensors in that layer and role. It never combines the separate layer and role marginals:
//   delta_l2      = sqrt(sum l2_i^2)                       (L2 of the concatenated deltas)
//   delta_rms     = sqrt(sum l2_i^2 / sum finite_pairs_i)
//   relative_l2   = sqrt(sum l2_i^2) / max(sqrt(sum norm_a_i^2), epsilon_norm)
//   max_abs_delta = max_i max_abs_i
//   any member with withheld deltas (float64 overflow) -> the cell is withheld; no finite pairs -> undefined.
// Cosine has no exact rollup from these values: a cell shows it only when it holds exactly one tensor.
// Sums of squares are scaled by the largest member, so values up to float64's range do not overflow.
import type { Metric, RollupMetric, RollupRow, TensorMetric, TensorRow } from "../../lib/api/compare";

export type Val =
  | { kind: "value"; v: number }
  | { kind: "withheld" } // exists but overflowed float64 (precision "overflowed"); never shown as a number
  | { kind: "undefined" } // mathematically undefined (no finite pairs, cosine with a zero vector)
  | { kind: "not_aggregable" } // no exact rollup from per-tensor values (cosine of several tensors)
  | { kind: "unrepresentable" }; // the aggregate cannot be formed from the reported values (an input was null)

export const value = (v: number): Val => ({ kind: "value", v });

export const METRIC_INFO: Record<TensorMetric, { label: string; short: string; definition: string; more: "more" | "less" }> = {
  relative_l2: {
    label: "Relative L2",
    short: "rel. L2",
    definition: "‖B − A‖₂ / max(‖A‖₂, ε): size of the change relative to the base weights.",
    more: "more",
  },
  delta_l2: { label: "Delta L2", short: "ΔL2", definition: "‖B − A‖₂ over finite pairs: total size of the change.", more: "more" },
  delta_rms: {
    label: "Delta RMS",
    short: "ΔRMS",
    definition: "√(Σ(B − A)² / finite pairs): typical per-element change.",
    more: "more",
  },
  max_abs_delta: { label: "Max |Δ|", short: "max|Δ|", definition: "max |B − A| over finite pairs: the largest single change.", more: "more" },
  cosine: {
    label: "Cosine similarity",
    short: "cos",
    definition: "⟨A, B⟩ / (‖A‖ ‖B‖) over finite pairs: 1 means same direction; lower means more change. Not defined for rollups.",
    more: "less",
  },
};

/** A tensor metric as a Val: the number, or why there is none. */
export function metricVal(m: Metric | undefined): Val {
  if (!m) return { kind: "undefined" };
  if (typeof m.value === "number" && Number.isFinite(m.value)) return value(m.value);
  if (m.numeric_precision === "overflowed") return { kind: "withheld" };
  return { kind: "undefined" };
}

/** A rollup (layer / role / total) metric from the report itself. */
export function rollupVal(r: RollupRow, k: TensorMetric): Val {
  if (k === "cosine") return { kind: "not_aggregable" };
  const v = r[k];
  if (typeof v === "number" && Number.isFinite(v)) return value(v);
  return r.withheld_metrics.includes(k) || r.withheld_tensors > 0 ? { kind: "withheld" } : { kind: "undefined" };
}

const num = (x: number | string): number => (typeof x === "number" ? x : Number(x));

/** sqrt(sum x_i^2) with the largest |x| factored out (no overflow for any finite inputs). */
export function rootSumSquares(xs: readonly number[]): number {
  let m = 0;
  for (const x of xs) m = Math.max(m, Math.abs(x));
  if (m === 0) return 0;
  if (!Number.isFinite(m)) return NaN;
  let s = 0;
  for (const x of xs) s += (x / m) * (x / m);
  return m * Math.sqrt(s);
}

export interface Cell {
  tensors: TensorRow[];
  finitePairs: number;
  withheld: number;
  metrics: Record<TensorMetric, Val>;
}

export function aggregate(rows: readonly TensorRow[], epsilon: number): Cell {
  const finitePairs = rows.reduce((s, r) => s + num(r.counts.finite_pairs ?? 0), 0);
  const withheldRows = rows.filter((r) => r.metrics.delta_l2?.numeric_precision === "overflowed");
  const withheld = withheldRows.length;
  const ok = rows.filter((r) => !withheldRows.includes(r));
  const l2s = ok.map((r) => metricVal(r.metrics.delta_l2));
  const blocked: Val | null = withheld > 0 ? { kind: "withheld" } : l2s.some((v) => v.kind !== "value") ? { kind: "unrepresentable" } : null;
  const dl2 = blocked ? NaN : rootSumSquares(l2s.map((v) => (v as { v: number }).v));
  // A non-finite aggregate of finite inputs left float64's range: the CLI lists it as withheld too.
  const fin = (x: number): Val => (Number.isFinite(x) ? value(x) : { kind: "withheld" });

  const metrics = {} as Record<TensorMetric, Val>;
  metrics.delta_l2 = blocked ?? fin(dl2);
  if (blocked) {
    metrics.delta_rms = blocked;
    metrics.relative_l2 = blocked;
    metrics.max_abs_delta = blocked;
  } else if (finitePairs === 0) {
    metrics.delta_rms = { kind: "undefined" };
    metrics.relative_l2 = { kind: "undefined" };
    metrics.max_abs_delta = { kind: "undefined" };
  } else {
    metrics.delta_rms = fin(dl2 / Math.sqrt(finitePairs));
    const norms = rows.map((r) => metricVal(r.metrics.norm_a));
    if (norms.some((v) => v.kind !== "value")) metrics.relative_l2 = { kind: "unrepresentable" };
    else {
      const normA = rootSumSquares(norms.map((v) => (v as { v: number }).v));
      metrics.relative_l2 = fin(normA < epsilon ? dl2 / epsilon : dl2 / normA);
    }
    let mx = 0;
    for (const r of ok) {
      const v = metricVal(r.metrics.max_abs_delta);
      if (v.kind === "value") mx = Math.max(mx, v.v);
    }
    metrics.max_abs_delta = value(mx);
  }
  const only = rows.length === 1 ? rows[0] : undefined;
  metrics.cosine = only ? metricVal(only.metrics.cosine) : { kind: "not_aggregable" };
  return { tensors: [...rows], finitePairs, withheld, metrics };
}

/** Canonical role order (weightlab/analysis/roles.py); roles the report has beyond these are appended. */
export const ROLE_ORDER = [
  "embedding",
  "attention.q",
  "attention.k",
  "attention.v",
  "attention.qkv",
  "attention.o",
  "mlp.gate",
  "mlp.up",
  "mlp.down",
  "norm",
  "lm_head",
  "other",
  "unknown",
];

export interface Grid {
  layers: (number | null)[]; // null = tensors with no layer index (embeddings, final norm, head, ...)
  roles: string[];
  cells: Map<string, Cell>;
}

export const cellKey = (layer: number | null, role: string): string => `${layer ?? "none"}|${role}`;

export function buildGrid(rows: readonly TensorRow[], epsilon: number): Grid {
  const groups = new Map<string, TensorRow[]>();
  const layerSet = new Set<number | null>();
  const roleSet = new Set<string>();
  for (const r of rows) {
    layerSet.add(r.layer);
    roleSet.add(r.role);
    const k = cellKey(r.layer, r.role);
    const g = groups.get(k);
    if (g) g.push(r);
    else groups.set(k, [r]);
  }
  const layers = [...layerSet].sort((a, b) => (a === null ? 1 : b === null ? -1 : a - b));
  const extra = [...roleSet].filter((r) => !ROLE_ORDER.includes(r)).sort();
  const roles = [...ROLE_ORDER.filter((r) => roleSet.has(r)), ...extra];
  const cells = new Map<string, Cell>();
  for (const [k, g] of groups) cells.set(k, aggregate(g, epsilon));
  return { layers, roles, cells };
}

/** Readable scientific notation: 4 significant digits; exponent form outside [1e-3, 1e4). */
export function formatSci(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e4) return String(Number(v.toPrecision(4)));
  const [m, e] = v.toExponential(3).split("e");
  return `${Number(m)}e${(e ?? "").replace("+", "").replace("-", "−")}`;
}

export const UNAVAILABLE: Record<Exclude<Val["kind"], "value">, { text: string; hint: string }> = {
  withheld: { text: "withheld", hint: "The value exists but overflowed float64 arithmetic; it is withheld, never shown as a number." },
  undefined: { text: "undefined", hint: "Mathematically undefined here (for example no finite pairs, or a zero vector)." },
  not_aggregable: { text: "n/a", hint: "Not defined for several tensors: cosine has no exact rollup from per-tensor values." },
  unrepresentable: {
    text: "unavailable",
    hint: "Cannot be formed exactly from the reported values (an input metric is missing or not representable).",
  },
};

export function valText(v: Val): string {
  return v.kind === "value" ? formatSci(v.v) : UNAVAILABLE[v.kind].text;
}

export function valTitle(v: Val): string {
  return v.kind === "value" ? String(v.v) : UNAVAILABLE[v.kind].hint;
}

/** Sort key: numbers by magnitude in the metric's "more change" direction; every unavailable kind last. */
export function changeOrder(v: Val, k: TensorMetric): number {
  if (v.kind !== "value") return Number.NEGATIVE_INFINITY;
  return METRIC_INFO[k].more === "more" ? v.v : -v.v;
}

/** 0..1 colour intensity of `v` on a scale whose top is `max` (log scale: over [min, max] positive values). */
export function intensity(v: Val, k: TensorMetric, lo: number, hi: number, log: boolean): number | null {
  if (v.kind !== "value") return null;
  const x = k === "cosine" ? 1 - v.v : v.v;
  if (!(hi > lo)) return x > 0 ? 1 : 0;
  if (log) {
    if (x <= 0) return 0;
    const l = Math.log10(Math.max(x, lo));
    return Math.min(1, Math.max(0, (l - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))));
  }
  return Math.min(1, Math.max(0, (x - lo) / (hi - lo)));
}

/** The colour-scale domain of a set of values (change direction: 1 - cosine for cosine). */
export function domain(vals: readonly Val[], k: TensorMetric, log: boolean): { lo: number; hi: number } {
  let lo = Infinity;
  let hi = -Infinity;
  let any = false;
  for (const v of vals) {
    if (v.kind !== "value") continue;
    const x = k === "cosine" ? 1 - v.v : v.v;
    if (log && !(x > 0)) continue;
    any = true;
    lo = Math.min(lo, x);
    hi = Math.max(hi, x);
  }
  if (!any) return log ? { lo: 1, hi: 1 } : { lo: 0, hi: 0 };
  return log ? { lo, hi } : { lo: Math.min(0, lo), hi };
}

export type { RollupMetric };
