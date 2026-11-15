// Pure geometry and labels for the histogram charts. Positions come from the backend's float64 edges (bins are
// drawn at their true widths, so the wider last integer bin or merged float bins stay visible as such); counts are
// never rescaled, smoothed or re-binned here.
import type { HistogramValue, MagnitudeBucket } from "../../lib/api/histogram";
import { toNumber } from "../../lib/api/histogram";

export type YScale = "linear" | "log";

/** Fraction (0..1) of the plot height for a count. Log scale is log10(count) over [1, max]; 0 draws nothing. */
export function yFraction(count: number, max: number, scale: YScale): number {
  if (count <= 0 || max <= 0) return 0;
  if (scale === "linear") return count / max;
  if (max <= 1) return 1;
  return Math.max(Math.log10(count) / Math.log10(max), 0.02); // a count of 1 stays visible
}

/** Gridline values: about four round steps for linear, powers of ten for log (both within [0, max]). */
export function yTicks(max: number, scale: YScale): number[] {
  if (max <= 0) return [];
  if (scale === "log") {
    const out: number[] = [];
    for (let p = 1; p <= max; p *= 10) out.push(p);
    return out;
  }
  const raw = max / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const mults = mag >= 10 ? [1, 2, 2.5, 5, 10] : [1, 2, 5, 10]; // counts are integers: no 2.5 steps below 25
  const step = Math.max(1, mults.map((m) => m * mag).find((s) => s >= raw) ?? raw);
  const out: number[] = [];
  for (let v = step; v <= max + step * 1e-9; v += step) out.push(Math.round(v));
  return out;
}

/** Each edge's horizontal position as a fraction of the plot width. A constant tensor's single bin [v, v] spans it. */
export function edgeFractions(edges: readonly HistogramValue[]): number[] {
  const e = edges.map(toNumber);
  const lo = e[0] ?? 0;
  const span = (e[e.length - 1] ?? lo) - lo;
  if (!(span > 0)) return e.map((_, i) => i / (e.length - 1));
  return e.map((x) => (x - lo) / span);
}

/** Short axis label of a value (four significant digits); the exact value is in the readout and the table. */
export function shortValue(v: HistogramValue): string {
  if (typeof v === "string") return v;
  if (Number.isInteger(v) && Math.abs(v) < 1e6) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2).replace("e+", "e");
  return String(Number(v.toPrecision(4)));
}

/** The exact value: shortest round-trip text of the float64, or the integer as sent. -0 is shown as 0. */
export function exactValue(v: HistogramValue): string {
  return typeof v === "string" ? v : String(v === 0 ? 0 : v);
}

/** The interval a bin holds, in the backend's rule: half-open, except the last float bin when it is closed. */
export function binInterval(edges: readonly HistogramValue[], i: number, domain: "float" | "integer", lastClosed: boolean): string {
  const a = edges[i] ?? 0;
  const b = edges[i + 1] ?? a;
  if (domain === "integer") {
    const hiIncl = typeof b === "string" ? `${b} − 1` : String(b - 1);
    return toNumber(b) - toNumber(a) === 1 ? exactValue(a) : `${exactValue(a)} … ${hiIncl}`;
  }
  if (toNumber(a) === toNumber(b)) return `[${exactValue(a)}, ${exactValue(b)}]`;
  const closed = lastClosed && i === edges.length - 2;
  return `[${exactValue(a)}, ${exactValue(b)}${closed ? "]" : ")"}`;
}

export interface MagnitudeColumn {
  exponent: number;
  positive: number;
  negative: number;
}

/** Contiguous exponents from the smallest to the largest present, empty ones included (so gaps are visible). */
export function magnitudeColumns(buckets: readonly MagnitudeBucket[]): MagnitudeColumn[] {
  const first = buckets[0];
  const last = buckets[buckets.length - 1];
  if (!first || !last) return [];
  const byExp = new Map(buckets.map((b) => [b.exponent, b]));
  const lo = first.exponent;
  const hi = last.exponent;
  const out: MagnitudeColumn[] = [];
  for (let e = lo; e <= hi; e++) out.push(byExp.get(e) ?? { exponent: e, positive: 0, negative: 0 });
  return out;
}

export function percent(count: number, total: number): string {
  if (total <= 0) return "—";
  const p = (100 * count) / total;
  if (p === 0) return "0%";
  if (p < 0.01) return "<0.01%";
  return `${p < 10 ? p.toFixed(2) : p.toFixed(1)}%`;
}
