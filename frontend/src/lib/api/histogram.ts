// The tensor_histogram job (contracts/openapi.yaml 0.5.0, ADR-021): an exact value histogram of one tensor,
// computed by the backend (weightlab/analysis/histogram.py) and returned as the completed job's result. The UI never
// bins values itself; it validates the result's structure and invariants here, then only draws it. A result that
// fails any check is a BAD_RESPONSE error, never a partial chart.
import { ApiRequestError, request } from "./client";
import { cancelJob, getJob, parseJob } from "./tiles";
import type { Job } from "./types";

export type HistogramRange = "min_max" | "symmetric" | "explicit";
export type Consistency = "best_effort" | "verified";
export const MAX_BINS = 1024;
export const DEFAULT_BINS = 64;
/** The job's fixed working-memory reservation (histogram.WORKING_SET); smaller budgets are refused with 422. */
export const HISTOGRAM_WORKING_SET_MIB = 64;

export interface HistogramOptions {
  bins: number;
  range: HistogramRange;
  lo?: number;
  hi?: number;
}

/** A float64 value, or an integer; integers beyond +-(2^53 - 1) arrive as decimal strings. */
export type HistogramValue = number | string;

export interface HistogramCounts {
  elements: number;
  finite: number;
  zeros: number;
  negative_zeros: number;
  positives: number;
  negatives: number;
  nan: number;
  pos_inf: number;
  neg_inf: number;
}

export interface HistogramBins {
  range: HistogramRange;
  requested_bins: number;
  lo: HistogramValue;
  hi: HistogramValue;
  edges: HistogramValue[];
  counts: number[];
  below: number;
  above: number;
  last_bin_closed: boolean;
  rule: string;
}

export interface MagnitudeBucket {
  exponent: number;
  positive: number;
  negative: number;
}

export interface TensorHistogram {
  histogram_version: 1;
  tensor_key: string;
  dtype: string;
  shape: number[];
  coverage: "complete";
  value_domain: "float" | "integer";
  counts: HistogramCounts;
  min: HistogramValue | null;
  max: HistogramValue | null;
  bins: HistogramBins | null;
  magnitude: { rule: string; buckets: MagnitudeBucket[] } | null;
  method: string;
}

export interface HistogramJobResult {
  kind: "tensor_histogram";
  model_id: string;
  tensor_id: string;
  consistency: Consistency;
  histogram: TensorHistogram;
}

const COUNT_KEYS = ["elements", "finite", "zeros", "negative_zeros", "positives", "negatives", "nan", "pos_inf", "neg_inf"] as const;
const BIG = /^-?[0-9]{16,20}$/;

function bad(message: string): ApiRequestError {
  return new ApiRequestError({ status: 200, code: "BAD_RESPONSE", message: `invalid histogram: ${message}` });
}
const isCount = (v: unknown): v is number => typeof v === "number" && Number.isSafeInteger(v) && v >= 0;
const isValue = (v: unknown): v is HistogramValue => (typeof v === "number" && Number.isFinite(v)) || (typeof v === "string" && BIG.test(v));
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** A HistogramValue as a JS number, for drawing only (decimal strings round to the nearest double). */
export const toNumber = (v: HistogramValue): number => (typeof v === "number" ? v : Number(v));

/** Checks a tensor_histogram job result against the request it answers and against ADR-021's invariants. */
export function parseHistogramResult(v: unknown, expect: { modelId: string; tensorId: string }): HistogramJobResult {
  if (!isObj(v) || v.kind !== "tensor_histogram") throw bad("not a tensor_histogram result");
  if (v.model_id !== expect.modelId || v.tensor_id !== expect.tensorId) throw bad("the result is for a different tensor");
  if (v.consistency !== "best_effort" && v.consistency !== "verified") throw bad("consistency");
  const h = v.histogram;
  if (!isObj(h) || h.histogram_version !== 1) throw bad("unsupported histogram_version");
  if (h.coverage !== "complete") throw bad("coverage must be complete");
  if (h.value_domain !== "float" && h.value_domain !== "integer") throw bad("value_domain");
  if (typeof h.dtype !== "string" || typeof h.tensor_key !== "string" || typeof h.method !== "string") throw bad("dtype, tensor_key or method");
  if (!Array.isArray(h.shape) || !h.shape.every(isCount)) throw bad("shape");
  const c = h.counts;
  if (!isObj(c) || !COUNT_KEYS.every((k) => isCount(c[k]))) throw bad("counts");
  const n = c as unknown as HistogramCounts;
  if (n.elements !== h.shape.reduce((a: number, d: number) => a * d, 1)) throw bad("counts.elements does not match the shape");
  if (n.finite + n.nan + n.pos_inf + n.neg_inf !== n.elements) throw bad("finite + nan + inf counts do not add up to elements");
  if (n.zeros + n.positives + n.negatives !== n.finite || n.negative_zeros > n.zeros) throw bad("sign counts do not add up to finite");
  if ((h.min === null) !== (h.max === null) || (h.min !== null && !(isValue(h.min) && isValue(h.max)))) throw bad("min / max");
  if ((h.min === null) !== (n.finite === 0)) throw bad("min / max must be present exactly when there are finite values");
  if (h.min !== null && toNumber(h.min) > toNumber(h.max as HistogramValue)) throw bad("min > max");

  const b = h.bins;
  if (b !== null) {
    if (!isObj(b)) throw bad("bins");
    if (b.range !== "min_max" && b.range !== "symmetric" && b.range !== "explicit") throw bad("bins.range");
    if (!Array.isArray(b.edges) || !Array.isArray(b.counts) || b.counts.length < 1 || b.counts.length > MAX_BINS)
      throw bad("bins.edges / bins.counts");
    if (b.edges.length !== b.counts.length + 1 || !b.edges.every(isValue) || !b.counts.every(isCount)) throw bad("bins.edges must have one more entry than bins.counts");
    if (!isValue(b.lo) || !isValue(b.hi) || !isCount(b.below) || !isCount(b.above) || typeof b.last_bin_closed !== "boolean" || typeof b.rule !== "string")
      throw bad("bins fields");
    if (!Number.isInteger(b.requested_bins) || b.counts.length > (b.requested_bins as number)) throw bad("more bins than requested");
    const e = (b.edges as HistogramValue[]).map(toNumber);
    const constant = e.length === 2 && e[0] === e[1];
    if (!constant && !e.every((x, i) => i === 0 || x > (e[i - 1] ?? -Infinity))) throw bad("bins.edges must increase");
    const sum = (b.counts as number[]).reduce((a, x) => a + x, 0);
    if (b.below + sum + b.above !== n.finite) throw bad("below + bins + above does not equal the finite count");
  } else if (n.finite > 0) {
    throw bad("bins are missing although the tensor has finite values");
  }

  const m = h.magnitude;
  if ((m === null) !== (h.value_domain === "integer")) throw bad("magnitude is present exactly for float tensors");
  if (m !== null) {
    if (!isObj(m) || typeof m.rule !== "string" || !Array.isArray(m.buckets)) throw bad("magnitude");
    let prev = -Infinity;
    let total = 0;
    for (const k of m.buckets as unknown[]) {
      if (!isObj(k) || !Number.isInteger(k.exponent) || !isCount(k.positive) || !isCount(k.negative)) throw bad("magnitude bucket");
      if ((k.exponent as number) <= prev) throw bad("magnitude buckets must be ordered by exponent");
      prev = k.exponent as number;
      total += k.positive + k.negative;
    }
    if (total !== n.positives + n.negatives) throw bad("magnitude buckets do not add up to the nonzero finite count");
  }
  return v as unknown as HistogramJobResult;
}

export interface HistogramBudget {
  max_working_set_mib: number;
  max_threads: number;
  max_wall_seconds: number;
}

/** The server's memory and wall-time caps from Settings; the histogram itself reserves 64 MiB of them. */
export function histogramBudget(settings: Record<string, number | boolean> | undefined): HistogramBudget {
  const n = (k: string, d: number) => (typeof settings?.[k] === "number" ? (settings[k] as number) : d);
  return { max_working_set_mib: n("memory_budget_mib", 1024), max_threads: 1, max_wall_seconds: n("max_job_seconds", 600) };
}

export const histogramApi = {
  submit: async (modelId: string, tensorId: string, histogram: HistogramOptions, consistency: Consistency, budget: HistogramBudget): Promise<Job> =>
    parseJob(
      await request<unknown>("/jobs", {
        method: "POST",
        body: { kind: "tensor_histogram", source: { kind: "tensor", model_id: modelId, tensor_id: tensorId }, histogram, consistency, budget },
      }),
    ),
  job: getJob,
  cancel: cancelJob,
};
