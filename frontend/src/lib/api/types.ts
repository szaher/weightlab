// Response and request shapes of contracts/openapi.yaml 0.3.0 used by the UI shell (Phase 1 Task 3).
// Clients tolerate unknown optional response fields (doc 09), so these list only what the UI reads.

export const DTYPES = ["F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U64", "U32", "U16", "U8", "BOOL"] as const;
export type Dtype = (typeof DTYPES)[number];

export const ROLES = [
  "embedding",
  "attention.q",
  "attention.k",
  "attention.v",
  "attention.o",
  "mlp.up",
  "mlp.down",
  "mlp.gate",
  "norm",
  "lm_head",
  "other",
  "unknown",
] as const;
export type Role = (typeof ROLES)[number];

export const SORT_KEYS = ["raw_key", "numel", "byte_length", "layer_index"] as const;
export type SortKey = (typeof SORT_KEYS)[number];
export type SortOrder = "asc" | "desc";

export interface Health {
  status: "ok" | "degraded";
  version: string;
  offline: boolean;
  capabilities: string[];
}

export interface Project {
  id: string;
  name: string;
  created_at: string;
}

export type ModelStatus = "ready" | "incomplete" | "changed" | "disconnected" | "unsupported";

export interface Model {
  id: string;
  project_id: string;
  display_name: string;
  format: "safetensors" | "sharded_safetensors";
  status: ModelStatus;
  tensor_count: number;
  fingerprint_method: "size_mtime" | "sha256" | "hf_cache_blob_sha256";
  warnings?: string[];
}

export interface ModelImport {
  entry_path: string;
  selected_root: string;
  display_name?: string;
}

export interface ModelRelink {
  entry_path: string;
  selected_root: string;
}

export interface Tensor {
  id: string;
  model_id: string;
  raw_key: string;
  shape: number[];
  dtype: Dtype;
  rank: number;
  numel: number;
  byte_length: number;
  shard_relpath: string;
  file_offset_begin: number;
  file_offset_end: number;
  role: Role | null;
  role_is_heuristic: true;
  layer_index: number | null;
}

export interface TensorPage {
  total: number;
  items: Tensor[];
}

export type Settings = Record<string, number | boolean>;

export interface ErrorBody {
  code: string;
  message: string;
  retryable: boolean;
  recommended_action?: string;
  details?: Record<string, string | number | boolean | null>;
}

export interface ErrorEnvelope {
  error: ErrorBody;
  request_id: string;
}

// ---- Exact ROI (POST /tensors/roi; doc 09 section 2). Only the `tensor` source kind is used by the UI today.

export interface TensorSource {
  kind: "tensor";
  model_id: string;
  tensor_id: string;
}

export interface RoiView {
  row_axis: number | null;
  col_axis: number | null;
  fixed_indices: Record<string, number>;
  transpose: boolean;
}

export interface RoiBounds {
  r0: number;
  r1: number;
  c0: number;
  c1: number;
}

export interface RoiRequest {
  source: TensorSource;
  view: RoiView;
  roi: RoiBounds;
  value_encoding: "auto";
}

export type ValueEncoding = "float_json" | "int_json" | "int_decimal_string" | "bool_json" | "hex_bits";
export type NumericPrecision = "lossless" | "rounded" | "overflowed";
export type NonfiniteKind = "nan" | "pos_inf" | "neg_inf" | "overflow";

/** As received; `values` are still unchecked JSON. `parseRoi` validates every field before anything is drawn. */
export interface RoiResponse {
  schema_version: number;
  source: unknown;
  roi: RoiBounds;
  shape: number[];
  source_dtype: Dtype;
  value_encoding: ValueEncoding;
  coverage: string;
  numeric_precision: NumericPrecision;
  values: unknown[][];
  nonfinite_mask: (NonfiniteKind | null)[][] | null;
  warnings: string[];
}

// ---- Tiles and jobs (POST /tiles, GET /tiles/{cache_key}/metadata, /jobs; doc 09 sections 3 and 5).

export const AGGREGATIONS = ["mean", "mean_abs", "rms", "min", "max"] as const;
export type Aggregation = (typeof AGGREGATIONS)[number];
export type QualityMode = "auto" | "exact" | "row_sampled";
export type TileQuality = "exact" | "row_sampled";
export type SamplingPolicy = "stride" | "splitmix64_rows";

export interface TileRequest {
  source: TensorSource;
  view: RoiView;
  tile: { level: number; x: number; y: number; size: number };
  aggregation: Aggregation;
  quality: { mode: QualityMode; sampled_rows_per_block?: number; seed?: number };
}

/** GET /tiles/{cache_key}/metadata (TileMetadata). */
export interface TileMetadata {
  schema_version: 2;
  cache_key: string;
  rows: number;
  cols: number;
  row_start: number;
  row_end: number;
  col_start: number;
  col_end: number;
  level: number;
  quality: TileQuality;
  value_precision: "float32_visualization";
  sampled_rows?: number | null;
  sampling_policy?: SamplingPolicy | null;
  seed?: number | null;
  aggregation: Aggregation;
  finite_count: number;
  nonfinite_count: number;
  invalid_pair_count?: number;
  source: unknown;
  view: unknown;
  warnings?: string[];
}

export type JobState = "created" | "queued" | "running" | "completed" | "canceled" | "failed" | "interrupted";
export const PENDING_JOB_STATES: readonly JobState[] = ["created", "queued", "running"];

export interface Job {
  job_id: string;
  kind: "tensor_stats" | "pyramid_build" | "compare_stats" | "file_hash" | "export";
  state: JobState;
  progress: number;
  poll_after_ms?: number;
  estimated_bytes?: number;
  bytes_scanned?: number;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  deduplicated?: boolean;
  result?: { kind: string; pyramid_id?: string } | null;
  error?: ErrorBody | null;
}
