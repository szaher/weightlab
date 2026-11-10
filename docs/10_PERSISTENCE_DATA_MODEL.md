# 10 — Persistence, Schemas and Provenance

## 1. Storage philosophy

Projects store **references**, not copied checkpoints. SQLite is local, single-user, schema migrated and backed up cautiously. Expensive derived tile cache is disposable; durable experiment records are not. No credential material is persisted in plaintext project database. Add content-addressed cache keys tied to file identity and method version.

## 2. Relational entities

```text
Project 1--* ImportedModel 1--* Tensor
       1--* Bookmark
       1--* Comparison -> Model A, Model B
       1--* Adapter -> ImportedModel base
       1--* Job
       1--* Experiment -> models, eval dataset, runs
       1--* Export
```

## 3. Suggested SQLite schema (initial)

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
CREATE TABLE projects (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL, settings_json TEXT NOT NULL
);
CREATE TABLE models (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  display_name TEXT NOT NULL, selected_root TEXT NOT NULL, entry_path TEXT NOT NULL,
  format TEXT NOT NULL, architecture_hint TEXT, lineage_json TEXT NOT NULL,
  quick_fingerprint_json TEXT NOT NULL, sha256 TEXT, status TEXT NOT NULL,
  imported_at TEXT NOT NULL
);
CREATE TABLE model_files (
  id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id),
  relpath TEXT NOT NULL, size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
  file_id TEXT, sha256 TEXT, fingerprint_method TEXT NOT NULL, verified_at TEXT,
  UNIQUE(model_id,relpath)
);
CREATE TABLE tensors (
  id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id),
  raw_key TEXT NOT NULL, rank INTEGER NOT NULL, shape_json TEXT NOT NULL,
  dtype TEXT NOT NULL, shard_relpath TEXT NOT NULL, byte_length INTEGER NOT NULL,
  role TEXT, layer_index INTEGER, UNIQUE(model_id,raw_key)
);
CREATE TABLE adapters (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  base_model_id TEXT NOT NULL REFERENCES models(id), config_json TEXT NOT NULL,
  adapter_path TEXT NOT NULL, mapping_json TEXT NOT NULL, status TEXT NOT NULL
);
CREATE TABLE comparisons (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  model_a_id TEXT NOT NULL REFERENCES models(id), model_b_id TEXT NOT NULL REFERENCES models(id),
  compatibility_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE comparison_reports (  -- migration 0003, ADR-020
  comparison_id TEXT PRIMARY KEY REFERENCES comparisons(id) ON DELETE CASCADE,
  job_id TEXT NOT NULL, complete INTEGER NOT NULL CHECK (complete IN (0,1)),
  report_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE bookmarks (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  view_json TEXT NOT NULL, note TEXT, created_at TEXT NOT NULL
);
CREATE TABLE jobs (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  kind TEXT NOT NULL, spec_json TEXT NOT NULL, state TEXT NOT NULL,
  progress REAL NOT NULL, result_json TEXT, error_json TEXT, created_at TEXT NOT NULL,
  started_at TEXT, finished_at TEXT
);
CREATE TABLE experiments (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  spec_json TEXT NOT NULL, status TEXT NOT NULL, manifest_path TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE exports (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  format TEXT NOT NULL, export_path TEXT NOT NULL, manifest_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_tensors_model_layer ON tensors(model_id,layer_index);
CREATE INDEX idx_jobs_project_state ON jobs(project_id,state);
```

Note: SQLite stores project state only; per-tensor metadata index may require large tens-of-thousands row insert. Use batched transactions and index only necessary queries. Each referenced shard has its own `model_files` row with its fingerprint; the model-level fingerprint is derived from those rows, never stored alone.

## 4. File fingerprinting

Quick fingerprint = `(normalized selected-root-relative path, size, mtime_ns, file ID, identity of each shard)`; fast stale-cache detection but NOT secure proof of equality. Strong fingerprint = SHA256 of all referenced files or separately signed manifests; full hash is O(file size) and may be an expensive job. When files live in a local Hugging Face hub cache, the LFS blob filename is the file's expected SHA-256 and the snapshot directory names the revision; a fresh hash that matches the blob name gives `fingerprint_method=hf_cache_blob_sha256` without any network access (doc 04 section 7). Persist `fingerprint_method`, completeness and last verification timestamp. Present model lineage separately from file identity (model card and HF base reference are claims, not content proof).

## 5. Artifact versioning

All exports/experiments have `schema_version=1`, `tool_version`, `reader_version`, `numerics_version`, `platform`, `project_id`, `sources`, `ops`, `approximation_details`, `warnings`. Add migration when changing semantics; never reinterpret old metrics silently. Validations are in `contracts/*.json` and test suite. Use canonical sorted-key JSON serialization where hashing a manifest.

## 6. Cache

Two independent LRU caches: memory tile cache (default <= 128 MiB backend), disk tile/pyramid/stat cache (default <= 1 GiB). Entries are keyed by the SHA-256 of a canonical sorted-key JSON identity (doc 03 section 5): source kind; every input file's fingerprint; tensor key(s); view (row_axis, col_axis, fixed_indices, transpose); operation parameters (transform, epsilon, LoRA native scale, experiment multiplier, module mask); `numerics_version`; `pyramid_record_format`; and, for tiles, level, x, y, size, aggregation, quality mode and sampling seed. Any fingerprint change of any input invalidates every entry that includes it. Cache may be safely deleted on shutdown or version mismatch. Browser cache <= 64 MiB decoded tile arrays as target; per-project tile prefetch bounded.

## 7. Data retention

User controls local data directory. Deleting a project removes app metadata and optional cache, **not external checkpoint originals**. Deleting experiment outputs requires confirmation. `weightlab doctor` checks DB version and referential integrity; WAL backup uses SQLite backup API rather than raw copy of active DB. Migration must be reversible where safe or backup first.

## 8. Example provenance object

```json
{
  "schema_version": 1,
  "source_model": {"id": "mod_a", "format": "safetensors", "fingerprint_kind": "size_mtime", "strong_hash_verified": false},
  "source_tensor": {"key": "model.layers.0.self_attn.q_proj.weight", "shape": [896,896], "dtype": "F16"},
  "operation": {"kind": "delta", "orientation": "B_minus_A", "compute_dtype": "float32"},
  "view": {"row_axis": 0, "col_axis": 1, "fixed_indices": {}, "roi": [0,128,0,128]},
  "quality": {"exact": true, "sampling_method": null},
  "warnings": ["Model lineage declared but not cryptographically verified"]
}
```

## 9. Privacy note

Absolute paths may include usernames and private project names. Hide by default in exported reports; allow user to include after explicit confirmation. Numeric tensor extracts can reveal proprietary model parameters; exports warn and require deliberate action to export raw values.
