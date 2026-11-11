-- WeightLab project database, migration 0003 (ADR-020, contract 0.4.0).
--
-- ADDED: comparison_reports, the change report of one comparison computed by a `comparison_report` job and served
-- by GET /comparisons/{id}/report. One row per comparison. The row is written only inside the transaction that marks
-- its job `completed` (weightlab/jobs.py commit hook), so a visible report always belongs to a completed job.
-- report_json is the weightlab/schemas/change_report.schema.json document exactly as `weightlab diff` writes it
-- (paths redacted). Deleting the comparison (or its project or either model) deletes its report.
--
-- job_seq orders replacement: each comparison_report job gets a strictly increasing number at submission from
-- comparison_report_seq (AUTOINCREMENT: never reused, even after deletes). A completed job replaces the stored report
-- only if its job_seq is greater, so a job submitted earlier never overwrites a later job's report by finishing later.

CREATE TABLE comparison_report_seq (seq INTEGER PRIMARY KEY AUTOINCREMENT);

CREATE TABLE comparison_reports (
  comparison_id TEXT PRIMARY KEY REFERENCES comparisons(id) ON DELETE CASCADE,
  job_id TEXT NOT NULL,
  job_seq INTEGER NOT NULL,
  consistency TEXT NOT NULL CHECK (consistency IN ('best_effort', 'verified')),
  complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
  report_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
