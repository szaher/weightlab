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
