-- WeightLab project database, migration 0001 (Phase 1 Task 1).
--
-- Source: docs/10_PERSISTENCE_DATA_MODEL.md section 3 ("Suggested SQLite schema (initial)"), taken as is,
-- with additions the frozen API contract (contracts/openapi.yaml 0.3.0) needs. Every addition is marked
-- "ADDED" with its reason. PRAGMAs (WAL, foreign_keys) are set per connection by weightlab.db.schema,
-- because journal_mode cannot change inside a transaction.
--
-- Projects store references, never copied checkpoints (doc 10 section 1). Deleting a project removes its
-- rows (ON DELETE CASCADE, ADDED: doc 10 section 7 requires project deletion to remove app metadata; the
-- suggested schema had no cascades) and never touches checkpoint files.

CREATE TABLE schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL,
  checksum TEXT NOT NULL  -- ADDED: SHA-256 of the migration file; an edited applied migration fails closed
);

CREATE TABLE projects (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL, settings_json TEXT NOT NULL
);

CREATE TABLE models (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  display_name TEXT NOT NULL, selected_root TEXT NOT NULL, entry_path TEXT NOT NULL,
  format TEXT NOT NULL CHECK (format IN ('safetensors', 'sharded_safetensors')),
  architecture_hint TEXT, lineage_json TEXT NOT NULL,
  quick_fingerprint_json TEXT NOT NULL,  -- derived summary only; per-file rows in model_files are authoritative
  sha256 TEXT, status TEXT NOT NULL CHECK (status IN ('ready', 'incomplete', 'changed', 'disconnected', 'unsupported')),
  imported_at TEXT NOT NULL,
  warnings_json TEXT NOT NULL DEFAULT '[]'  -- ADDED: Model.warnings (e.g. UNC network location, index notes)
);

CREATE TABLE model_files (
  id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  relpath TEXT NOT NULL, size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
  file_id TEXT, sha256 TEXT,
  fingerprint_method TEXT NOT NULL CHECK (fingerprint_method IN ('size_mtime', 'sha256', 'hf_cache_blob_sha256')),
  verified_at TEXT,
  UNIQUE(model_id, relpath)
);

CREATE TABLE tensors (
  id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  raw_key TEXT NOT NULL, rank INTEGER NOT NULL, shape_json TEXT NOT NULL,
  dtype TEXT NOT NULL, shard_relpath TEXT NOT NULL, byte_length INTEGER NOT NULL,
  role TEXT, layer_index INTEGER,
  numel INTEGER NOT NULL,              -- ADDED: Tensor.numel, catalog filter min_numel and sort by numel
  file_offset_begin INTEGER NOT NULL,  -- ADDED: Tensor.file_offset_begin/end (absolute byte extent in the shard)
  file_offset_end INTEGER NOT NULL,
  UNIQUE(model_id, raw_key)
);

CREATE TABLE adapters (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  base_model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE, config_json TEXT NOT NULL,
  adapter_path TEXT NOT NULL, mapping_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('ready', 'partial', 'unsupported')),
  lineage_json TEXT NOT NULL,  -- ADDED: Adapter.lineage (LineageAssessment)
  files_json TEXT NOT NULL,    -- ADDED: adapter file fingerprints (size, mtime_ns, sha256) for stale detection
  created_at TEXT NOT NULL     -- ADDED: stable listing order
);

CREATE TABLE comparisons (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  model_a_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  model_b_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  compatibility_json TEXT NOT NULL, created_at TEXT NOT NULL
);

-- ADDED: ComparisonEntry rows (GET /comparisons/{id}/entries is paged and filtered by status; the
-- delta source addresses one entry by entry_id). The suggested schema kept them only inside
-- compatibility_json, which cannot be paged or referenced.
CREATE TABLE comparison_entries (
  id TEXT PRIMARY KEY, comparison_id TEXT NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, raw_key TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('matched', 'missing_in_a', 'missing_in_b', 'tied', 'shape_mismatch',
                                         'dtype_mismatch')),
  tensor_id_a TEXT, tensor_id_b TEXT, detail_json TEXT NOT NULL,
  UNIQUE(comparison_id, raw_key)
);

CREATE TABLE bookmarks (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  view_json TEXT NOT NULL, note TEXT, created_at TEXT NOT NULL
);

CREATE TABLE jobs (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind TEXT NOT NULL, spec_json TEXT NOT NULL, state TEXT NOT NULL,
  progress REAL NOT NULL, result_json TEXT, error_json TEXT, created_at TEXT NOT NULL,
  started_at TEXT, finished_at TEXT
);

CREATE TABLE experiments (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  spec_json TEXT NOT NULL, status TEXT NOT NULL, manifest_path TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE exports (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  format TEXT NOT NULL, export_path TEXT NOT NULL, manifest_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- ADDED: GET/PATCH /settings are application-wide (not per project); one row, id = 1.
CREATE TABLE app_settings (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  settings_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX idx_tensors_model_layer ON tensors(model_id, layer_index);
CREATE INDEX idx_jobs_project_state ON jobs(project_id, state);
-- ADDED: ownership look-ups and paging
CREATE INDEX idx_models_project ON models(project_id);
CREATE INDEX idx_entries_comparison_status ON comparison_entries(comparison_id, status, ordinal);
CREATE INDEX idx_adapters_project ON adapters(project_id);
CREATE INDEX idx_bookmarks_project ON bookmarks(project_id);
CREATE INDEX idx_comparisons_project ON comparisons(project_id);
