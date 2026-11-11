// Run / re-run the analysis, its progress, cancel, and every failure explained with the action that resolves it.
import { useState } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { ApiRequestError } from "../../lib/api/client";
import type { Consistency, ReportJobResult } from "../../lib/api/compare";
import { PENDING_JOB_STATES, type ErrorBody, type Job } from "../../lib/api/types";
import { formatBytes } from "../../lib/format";
import { MAX_POLL_FAILURES, type ReportJob } from "./useReportJob";

/** What resolves each failure of a comparison report (submit errors and failed jobs). */
export const REPORT_ADVICE: Record<string, string> = {
  FILE_CHANGED:
    "A model's files changed since they were imported. Re-import the changed model (or relink it in the Workspace if it was only moved) and create a new comparison.",
  CONFLICT:
    "This comparison no longer matches its models, or was created before comparison reports existed. Create a new comparison of the same two models.",
  LINEAGE_UNVERIFIED:
    "This pair has no lineage evidence (Class S or X) and the comparison uses the strict policy. Create the comparison again with “Descriptive only” to acknowledge that the numbers describe raw, possibly unaligned coordinates.",
  RESOURCE_LIMIT:
    "The report is larger or slower than the server's limits allow. Raise memory_budget_mib or max_job_seconds in Settings if this computer has room, or compare smaller checkpoints.",
  NOT_FOUND: "The comparison or one of its models was deleted. Go back to the list and create the comparison again.",
  RATE_LIMITED: "The job queue is full. Wait for running jobs to finish, then try again.",
};

const STATE_TEXT: Record<Job["state"], string> = {
  created: "Queued",
  queued: "Queued",
  running: "Analyzing",
  completed: "Completed",
  failed: "Failed",
  canceled: "Canceled",
  interrupted: "Interrupted",
};

interface Props {
  rj: ReportJob;
  hasReport: boolean;
  reportCreated: string | null;
}

export function ReportJobPanel({ rj, hasReport, reportCreated }: Props) {
  const [consistency, setConsistency] = useState<Consistency>("best_effort");
  const { job } = rj;
  const pending = !!job && PENDING_JOB_STATES.includes(job.state);
  const result = job?.state === "completed" ? (job.result as unknown as ReportJobResult | null) : null;
  const jobError: ErrorBody | null = job?.state === "failed" ? (job.error ?? null) : null;
  const pct = job ? Math.round(job.progress * 100) : 0;

  return (
    <section className="job-panel cmp-job stack-sm" aria-labelledby="cmp-job-h">
      <h3 id="cmp-job-h" className="h3">
        Analysis
      </h3>
      {pending && job ? (
        <div role="status" aria-live="polite" className="stack-sm">
          <p>
            <strong>{STATE_TEXT[job.state]}</strong>
            {job.state === "running" ? ` · ${pct}%` : ""}
            {job.estimated_bytes ? ` · ${formatBytes(job.bytes_scanned ?? 0)} of up to ${formatBytes(job.estimated_bytes)} read` : ""}
            {job.deduplicated ? " · joined the analysis that was already running" : ""}
          </p>
          <progress max={1} value={job.progress} aria-label="Analysis progress" />
          <p className="muted small">
            Reading both checkpoints in bounded chunks. The report appears only once the job has completed and stored it.
          </p>
          <button type="button" className="btn btn-small" disabled={rj.busy} onClick={() => void rj.cancel()}>
            Cancel analysis
          </button>
        </div>
      ) : (
        <div className="row">
          <fieldset className="radio-row" aria-label="Consistency">
            <label className="radio">
              <input type="radio" name="cmp-consistency" checked={consistency === "best_effort"} onChange={() => setConsistency("best_effort")} />
              Best effort
            </label>
            <label className="radio" title="Hashes every input file before and after the scan (two extra full reads)">
              <input type="radio" name="cmp-consistency" checked={consistency === "verified"} onChange={() => setConsistency("verified")} />
              Verified (SHA-256 before and after)
            </label>
          </fieldset>
          <button type="button" className="btn btn-primary" disabled={rj.busy} onClick={() => void rj.submit(consistency)}>
            {hasReport ? "Re-run analysis" : "Run analysis"}
          </button>
        </div>
      )}

      {result && !result.published ? (
        <p className="notice notice-info" role="status">
          This run finished after a later-submitted run had already stored its report, so the later report is shown.
        </p>
      ) : null}
      {job && !pending && job.state !== "completed" ? (
        <div className={`notice ${job.state === "failed" ? "notice-error" : "notice-warn"}`} role="alert">
          <p className="notice-title">
            <span aria-hidden="true">⚠ </span>
            {job.state === "failed"
              ? "The last analysis failed"
              : job.state === "canceled"
                ? "The last analysis was canceled"
                : "The last analysis was interrupted (WeightLab stopped while it ran)"}
          </p>
          {jobError ? (
            <>
              <p>{jobError.message}</p>
              {REPORT_ADVICE[jobError.code] ? <p className="notice-action">{REPORT_ADVICE[jobError.code]}</p> : null}
              <p className="notice-meta">
                Code <code>{jobError.code}</code> · Job <code>{job.job_id}</code>
              </p>
            </>
          ) : null}
          <p>
            Nothing from that run was published.
            {hasReport && reportCreated ? ` The report below is the previous successful one (${new Date(reportCreated).toLocaleString()}).` : ""}
          </p>
        </div>
      ) : null}
      {rj.error ? (
        <>
          <ErrorPanel error={rj.error} />
          {rj.error instanceof ApiRequestError && REPORT_ADVICE[rj.error.code] ? (
            <p className="notice-action">{REPORT_ADVICE[rj.error.code]}</p>
          ) : null}
        </>
      ) : null}
      {pending && rj.pollFailures >= MAX_POLL_FAILURES ? (
        <button type="button" className="btn btn-small" onClick={rj.checkAgain}>
          Check again
        </button>
      ) : null}
    </section>
  );
}
