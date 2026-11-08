// Wording of every numerical disclosure on the overview (doc 05, doc 08 section 9). One place, so tests can pin it.
import { HAS_NAN, HAS_NEG_INF, HAS_POS_INF, NO_FINITE, NOT_REPRESENTABLE } from "../../lib/api/tiles";
import type { Aggregation, SamplingPolicy, TileQuality } from "../../lib/api/types";

export const AGG_LABEL: Record<Aggregation, string> = {
  mean: "mean",
  mean_abs: "mean |value|",
  rms: "RMS",
  min: "minimum",
  max: "maximum",
};

/** What one heatmap cell shows under each aggregation (finite elements of its block only). */
export const AGG_MEANING: Record<Aggregation, string> = {
  mean: "Each cell is the signed mean of the finite elements in its block.",
  mean_abs: "Each cell is the mean absolute value of the finite elements in its block (magnitudes, not signed weights).",
  rms: "Each cell is the root mean square √(mean of squares) of the finite elements in its block. It is not a standard deviation: the mean is not subtracted.",
  min: "Each cell is the smallest finite element in its block.",
  max: "Each cell is the largest finite element in its block.",
};

export const QUALITY_BADGE: Record<TileQuality, string> = {
  exact: "EXACT coverage",
  row_sampled: "ROW-SAMPLED preview",
};

export const PRECISION_TEXT =
  "Displayed values are float32 visualizations of the aggregates. Exact numbers come only from Exact ROI.";

export function qualityText(q: TileQuality, sampledRows: number | null, policy: SamplingPolicy | null | undefined, seed: number | null | undefined): string {
  if (q === "exact") return "Exact coverage: every element of every block was aggregated.";
  const how =
    policy === "splitmix64_rows"
      ? `rows chosen by seeded SplitMix64 (seed ${seed ?? "?"})`
      : policy === "stride"
        ? "evenly strided rows"
        : "rows chosen by the server's sampling policy";
  return `Row-sampled preview: only ${sampledRows ?? "?"} row(s) of each block were read (${how}). Blocks are not fully covered; this is not exact.`;
}

export function blockText(level: number): string {
  const b = 2 ** level;
  return level === 0 ? "L0: one cell per element" : `L${level}: each cell covers up to ${b.toLocaleString("en-US")} × ${b.toLocaleString("en-US")} elements`;
}

/** Text for one cell's status bits (hover, selection, screen readers). Never used as a value. */
export function statusText(s: number): string {
  const kinds: string[] = [];
  if (s & HAS_NAN) kinds.push("NaN");
  if (s & HAS_POS_INF) kinds.push("+Inf");
  if (s & HAS_NEG_INF) kinds.push("−Inf");
  if (s & NO_FINITE) return `no finite element in this block (${kinds.join(", ")}); nothing to aggregate`;
  if (s & NOT_REPRESENTABLE) return "a finite aggregate exists but cannot be shown as float32 (see Exact ROI)";
  if (kinds.length > 0) return `finite aggregate of the finite elements; the block also contains ${kinds.join(", ")}`;
  return "finite aggregate";
}
