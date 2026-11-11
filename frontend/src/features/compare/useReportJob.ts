// The comparison_report job lifecycle: rediscover the latest job after a reload (Comparison.latest_report_job_id),
// submit, poll with poll_after_ms (bounded backoff on failed polls, as the Overview does), cancel. Navigating away
// stops polling but never cancels a job. `onSettled` runs once when a job seen pending reaches a terminal state;
// the caller then reloads the comparison and its report: the backend writes the report in the same transaction
// that marks the job completed, so a completed job's report is already the stored one (or a newer one).
import { useCallback, useEffect, useRef, useState } from "react";
import { isAbort } from "../../lib/api/client";
import { compareApi, reportBudget, type Comparison, type Consistency } from "../../lib/api/compare";
import { PENDING_JOB_STATES, type Job } from "../../lib/api/types";

export const MAX_POLL_FAILURES = 5;
const isPending = (j: Job | null) => !!j && PENDING_JOB_STATES.includes(j.state);
const delay = (j: Job, failures: number) =>
  failures === 0 ? Math.min(10_000, Math.max(100, j.poll_after_ms ?? 500)) : Math.min(8_000, 1_000 * 2 ** (failures - 1));

export interface ReportJob {
  job: Job | null;
  /** Error of the last submit / cancel / status request (not the job's own error). */
  error: unknown;
  busy: boolean;
  pollFailures: number;
  submit: (consistency: Consistency) => Promise<void>;
  cancel: () => Promise<void>;
  checkAgain: () => void;
}

export function useReportJob(
  comparison: Comparison | undefined,
  settings: Record<string, number | boolean> | undefined,
  onSettled: (job: Job) => void,
): ReportJob {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [failures, setFailures] = useState(0);
  const [nonce, setNonce] = useState(0);
  const settledRef = useRef(onSettled);
  useEffect(() => {
    settledRef.current = onSettled;
  });
  const cid = comparison?.id ?? null;
  const latest = comparison?.latest_report_job_id ?? null;

  // Rediscover the comparison's latest job (after a reload, or when another tab submitted one).
  // The hook lives as long as one comparison's dashboard (keyed by comparison id), so no job means none yet.
  useEffect(() => {
    if (!latest) return;
    const ctl = new AbortController();
    compareApi.job(latest, ctl.signal).then(
      (j) => {
        setError(null);
        setFailures(0);
        setJob(j);
      },
      (e: unknown) => {
        if (!isAbort(e)) setError(e);
      },
    );
    return () => ctl.abort();
  }, [cid, latest]);

  // Poll while pending.
  const pendingId = job && isPending(job) ? job.job_id : null;
  useEffect(() => {
    if (!job || !pendingId || failures >= MAX_POLL_FAILURES) return;
    const ctl = new AbortController();
    const t = window.setTimeout(() => {
      compareApi.job(pendingId, ctl.signal).then(
        (j) => {
          setFailures(0);
          setJob(j);
          if (!isPending(j)) settledRef.current(j);
        },
        (e: unknown) => {
          if (isAbort(e)) return;
          setError(e);
          setFailures((f) => f + 1);
        },
      );
    }, delay(job, failures));
    return () => {
      ctl.abort();
      window.clearTimeout(t);
    };
    // job identity and progress drive the next poll
  }, [job, pendingId, failures, nonce]);

  const submit = useCallback(
    async (consistency: Consistency) => {
      if (!cid) return;
      setBusy(true);
      setError(null);
      try {
        const j = await compareApi.submit(cid, consistency, reportBudget(settings));
        setFailures(0);
        setJob(j);
        if (!isPending(j)) settledRef.current(j);
      } catch (e) {
        setError(e);
      } finally {
        setBusy(false);
      }
    },
    [cid, settings],
  );

  const cancel = useCallback(async () => {
    if (!job) return;
    setBusy(true);
    try {
      const j = await compareApi.cancel(job.job_id);
      setJob(j);
      if (!isPending(j)) settledRef.current(j);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }, [job]);

  const checkAgain = useCallback(() => {
    setFailures(0);
    setNonce((n) => n + 1);
  }, []);

  return { job, error, busy, pollFailures: failures, submit, cancel, checkAgain };
}
