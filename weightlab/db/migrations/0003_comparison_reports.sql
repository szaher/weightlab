-- WeightLab project database, migration 0003 (ADR-020, contract 0.4.0).
--
-- ADDED: comparison_reports, the change report of one comparison computed by a `comparison_report` job and served
-- by GET /comparisons/{id}/report. One row per comparison: a newer completed job replaces the row in the same
-- transaction that publishes it, so a reader never sees a partial or mixed report. report_json is the
-- weightlab/schemas/change_report.schema.json document exactly as `weightlab diff` writes it (paths redacted).
-- Deleting the comparison (or its project or either model) deletes its report.

CREATE TABLE comparison_reports (
  comparison_id TEXT PRIMARY KEY REFERENCES comparisons(id) ON DELETE CASCADE,
  job_id TEXT NOT NULL,
  complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
  report_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
