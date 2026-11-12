// Comparisons and comparison reports (contracts/openapi.yaml 0.4.0, ADR-020). The report body is the
// weightlab/schemas/change_report.schema.json document exactly as `weightlab diff` writes it; these types list
// only what the Compare screen reads.
import { request } from "./client";
import { getJob, cancelJob, parseJob } from "./tiles";
import type { Job } from "./types";

const id = (v: string): string => encodeURIComponent(v);
type Sig = AbortSignal | undefined;

export type CompatClass = "V" | "L" | "S" | "X";
export type Policy = "strict" | "descriptive_only";
export type Consistency = "best_effort" | "verified";

export interface Lineage {
  compatibility_class: CompatClass;
  base_file_identity_verified: boolean;
  base_identity_method: string;
  lineage_claim_source: string;
  lineage_trust_level: string;
  lineage_evidence: string;
  transfer_eligibility: string;
  reasons: string[];
}

export interface ComparisonCounts {
  matched: number;
  missing_in_a: number;
  missing_in_b: number;
  tied: number;
  shape_mismatch: number;
  dtype_mismatch: number;
}

export interface ReportSummary {
  job_id: string;
  complete: boolean;
  consistency: Consistency;
  created_at: string;
}

export interface Comparison {
  id: string;
  project_id: string;
  model_a_id: string;
  model_b_id: string;
  lineage: Lineage;
  counts: ComparisonCounts;
  policy?: Policy;
  report?: ReportSummary | null;
  latest_report_job_id?: string | null;
  created_at: string;
}

export interface ComparisonRequest {
  project_id: string;
  model_a_id: string;
  model_b_id: string;
  policy: Policy;
  user_attestation: boolean;
}

export type Precision = "lossless" | "rounded" | "overflowed" | "unsupported";

export interface Metric {
  name: string;
  value: number | string | null;
  method: string;
  source_dtype: string;
  element_count: number | string;
  finite_count: number | string;
  numeric_precision: Precision;
  epsilon: number | null;
  numerical_warnings: string[];
}

export const TENSOR_METRICS = ["relative_l2", "delta_l2", "delta_rms", "max_abs_delta", "cosine"] as const;
export type TensorMetric = (typeof TENSOR_METRICS)[number];
export type RollupMetric = Exclude<TensorMetric, "cosine">;

export interface TensorRow {
  rank: number;
  key: string;
  layer: number | null;
  role: string;
  status: string;
  shape: number[];
  dtype_a: string;
  dtype_b: string;
  counts: Record<string, number | string>;
  metrics: Record<TensorMetric | "norm_a" | "norm_b", Metric>;
  withheld_metrics: string[];
  caveat: string | null;
}

export interface RollupRow {
  layer?: number | null;
  role?: string;
  scope?: string;
  tensors: number;
  elements: number | string;
  finite_pairs: number | string;
  invalid_pairs: number | string;
  withheld_tensors: number;
  delta_l2: number | null;
  delta_rms: number | null;
  relative_l2: number | null;
  max_abs_delta: number | null;
  near_zero_reference: boolean;
  withheld_metrics: string[];
}

export interface NotCompared {
  raw_key: string;
  status: string;
  shape_a: number[] | null;
  shape_b: number[] | null;
  dtype_a: string | null;
  dtype_b: string | null;
  tied_to: string | null;
  reasons: string[];
}

export interface ModelInfo {
  format: string;
  entry: string;
  files: { name: string; size: number; sha256: string | null }[];
  fingerprint_kind: string;
  strong_hash_verified: boolean;
  tensor_count: number;
}

export interface ChangeReport {
  schema_version: number;
  report_kind: "weightlab_change_report";
  mode: "checkpoint_pair";
  orientation: "B_minus_A";
  complete: boolean;
  completeness: {
    tensors_analyzed: number;
    tensors_withheld: number;
    rollups_withheld: number;
    entries_not_compared: number;
    tied_entries: number;
  };
  compatibility: { lineage: Lineage; counts: ComparisonCounts; not_compared: NotCompared[] };
  provenance: {
    created_at: string;
    tool: { version: string; numerics_version: number };
    inputs: { base: ModelInfo; derived: ModelInfo };
  };
  caveats: string[];
  settings: { sort: string; epsilon_norm: number; consistency: Consistency };
  tensors: TensorRow[];
  layers: RollupRow[];
  roles: RollupRow[];
  totals: RollupRow;
  warnings: string[];
}

export type ModelStatusNow = "ready" | "incomplete" | "changed" | "disconnected" | "unsupported";

export interface ComparisonReport {
  comparison_id: string;
  job_id: string;
  complete: boolean;
  consistency: Consistency;
  created_at: string;
  current_model_status: { model_a: ModelStatusNow; model_b: ModelStatusNow };
  report: ChangeReport;
}

export interface ReportJobResult {
  kind: "comparison_report";
  comparison_id: string;
  complete: boolean;
  published: boolean;
}

/** The smallest budget the server accepts for a report of `entries` entries (ADR-020), within its caps. */
export function reportBudget(settings: Record<string, number | boolean> | undefined): {
  max_working_set_mib: number;
  max_threads: number;
  max_wall_seconds: number;
} {
  const n = (k: string, d: number) => (typeof settings?.[k] === "number" ? (settings[k] as number) : d);
  return { max_working_set_mib: n("memory_budget_mib", 1024), max_threads: 1, max_wall_seconds: n("max_job_seconds", 600) };
}

export const compareApi = {
  list: (projectId: string, signal?: Sig) => request<Comparison[]>(`/projects/${id(projectId)}/comparisons`, { signal }),
  get: (comparisonId: string, signal?: Sig) => request<Comparison>(`/comparisons/${id(comparisonId)}`, { signal }),
  create: (body: ComparisonRequest) => request<Comparison>("/comparisons", { method: "POST", body }),
  report: (comparisonId: string, signal?: Sig) => request<ComparisonReport>(`/comparisons/${id(comparisonId)}/report`, { signal }),
  submit: async (comparisonId: string, consistency: Consistency, budget: ReturnType<typeof reportBudget>): Promise<Job> =>
    parseJob(
      await request<unknown>("/jobs", {
        method: "POST",
        body: { kind: "comparison_report", comparison_id: comparisonId, consistency, budget },
      }),
    ),
  job: getJob,
  cancel: cancelJob,
};
