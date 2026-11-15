// The tensor_histogram job lifecycle for one tensor: submit, poll with poll_after_ms (bounded backoff on failed
// polls, as the Compare report and the Overview do), cancel. The histogram is the completed job's result; it is
// validated (parseHistogramResult) before anything is drawn. Leaving the tensor stops polling but never cancels the
// job. A newer submission always replaces an older one, and a slow status answer never regresses a finished job.
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiRequestError, isAbort } from "../../lib/api/client";
import { histogramApi, histogramBudget, parseHistogramResult, type Consistency, type HistogramJobResult, type HistogramOptions } from "../../lib/api/histogram";
import { PENDING_JOB_STATES, type Job } from "../../lib/api/types";

export const MAX_POLL_FAILURES = 5;
const isPending = (j: Job | null) => !!j && PENDING_JOB_STATES.includes(j.state);
const delay = (j: Job, failures: number) =>
  failures === 0 ? Math.min(10_000, Math.max(100, j.poll_after_ms ?? 500)) : Math.min(8_000, 1_000 * 2 ** (failures - 1));

export interface HistogramRun {
  options: HistogramOptions;
  consistency: Consistency;
}

export interface HistogramJob {
  job: Job | null;
  /** What the current job was submitted with. */
  run: HistogramRun | null;
  /** The validated histogram of the last completed job, with the options it was computed with. */
  result: { data: HistogramJobResult; run: HistogramRun; jobId: string } | null;
  /** The last submit / cancel / status error, or a result that failed validation (not the job's own error). */
  error: unknown;
  busy: boolean;
  pollFailures: number;
  submit: (run: HistogramRun) => Promise<void>;
  cancel: () => Promise<void>;
  checkAgain: () => void;
}

export function useHistogramJob(modelId: string, tensorId: string, settings: Record<string, number | boolean> | undefined): HistogramJob {
  const [job, setJob] = useState<Job | null>(null);
  const [run, setRun] = useState<HistogramRun | null>(null);
  const [result, setResult] = useState<{ data: HistogramJobResult; run: HistogramRun; jobId: string } | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [failures, setFailures] = useState(0);
  const [nonce, setNonce] = useState(0);
  const currentRef = useRef<Job | null>(null);
  const runRef = useRef<HistogramRun | null>(null);

  /** Accepts a job answer unless it is stale; a completed job's result is validated before it is kept. */
  const accept = useCallback(
    (next: Job): boolean => {
      const prev = currentRef.current;
      if (prev && next.job_id !== prev.job_id) return false;
      if (prev && !isPending(prev) && isPending(next)) return false;
      currentRef.current = next;
      setJob(next);
      if (next.state === "failed" && next.error?.code === "FILE_CHANGED") setResult(null); // it described the old bytes
      if (next.state === "completed") {
        try {
          const data = parseHistogramResult(next.result, { modelId, tensorId });
          if (runRef.current) setResult({ data, run: runRef.current, jobId: next.job_id });
        } catch (e) {
          setError(e);
        }
      }
      return true;
    },
    [modelId, tensorId],
  );

  const pendingId = job && isPending(job) ? job.job_id : null;
  useEffect(() => {
    if (!job || !pendingId || failures >= MAX_POLL_FAILURES) return;
    const ctl = new AbortController();
    const t = window.setTimeout(() => {
      histogramApi.job(pendingId, ctl.signal).then(
        (j) => {
          if (ctl.signal.aborted || !accept(j)) return;
          setFailures(0);
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
  }, [accept, job, pendingId, failures, nonce]);

  const submit = useCallback(
    async (r: HistogramRun) => {
      setBusy(true);
      setError(null);
      try {
        const j = await histogramApi.submit(modelId, tensorId, r.options, r.consistency, histogramBudget(settings));
        currentRef.current = null; // a new submission replaces the job shown before (its histogram stays until then)
        runRef.current = r;
        setRun(r);
        setFailures(0);
        accept(j);
      } catch (e) {
        if (e instanceof ApiRequestError && e.code === "FILE_CHANGED") setResult(null);
        setError(e);
      } finally {
        setBusy(false);
      }
    },
    [accept, modelId, tensorId, settings],
  );

  const cancel = useCallback(async () => {
    if (!job) return;
    setBusy(true);
    try {
      accept(await histogramApi.cancel(job.job_id));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }, [accept, job]);

  const checkAgain = useCallback(() => {
    setFailures(0);
    setNonce((n) => n + 1);
  }, []);

  return { job, run, result, error, busy, pollFailures: failures, submit, cancel, checkAgain };
}
