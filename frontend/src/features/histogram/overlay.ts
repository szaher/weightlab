// A/B histogram overlay rules (ROADMAP P0.2). Two histograms are drawn on one axis only when their bins are the
// same bins: both computed by the backend over one explicit [lo, hi] with the same bin count and value domain, and
// their returned edges equal exactly. Nothing is re-binned or interpolated in the browser.
import type { HistogramValue, TensorHistogram } from "../../lib/api/histogram";
import { exactValue } from "./chart";

export type RangeResult = { lo: number; hi: number } | { error: string };

const asSafe = (v: HistogramValue): number | null => (typeof v === "number" ? v : null);

/** The union of both tensors' finite [min, max] (the range shared by the overlay), or why there is none. */
export function sharedRange(a: TensorHistogram, b: TensorHistogram): RangeResult {
  if (a.value_domain !== b.value_domain) return { error: "A and B have different value domains (float vs integer), so their bins cannot be the same" };
  const ends = [a.min, a.max, b.min, b.max].filter((v): v is HistogramValue => v !== null);
  if (ends.length === 0) return { error: "neither A nor B has a finite value, so there is no range to bin" };
  const nums = ends.map(asSafe);
  if (nums.some((v) => v === null)) return { error: "an extreme value is beyond ±(2^53 − 1); enter an explicit range instead" };
  const n = nums as number[];
  return { lo: Math.min(...n), hi: Math.max(...n) };
}

/** True when A's and B's bins are exactly the same intervals (so their counts can share one axis). */
export function sameBins(a: TensorHistogram, b: TensorHistogram): boolean {
  if (!a.bins || !b.bins || a.value_domain !== b.value_domain) return false;
  const ea = a.bins.edges.map(exactValue);
  const eb = b.bins.edges.map(exactValue);
  return ea.length === eb.length && ea.every((x, i) => x === eb[i]) && a.bins.last_bin_closed === b.bins.last_bin_closed;
}

/** B − A of a count, as signed text. */
export function signed(n: number): string {
  return n > 0 ? `+${n.toLocaleString("en-US")}` : n < 0 ? `−${(-n).toLocaleString("en-US")}` : "0";
}
