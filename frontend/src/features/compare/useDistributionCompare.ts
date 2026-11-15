// Runs the A/B histogram overlay of one report tensor as tensor_histogram jobs (ADR-021), so the backend counts
// every element and the browser never re-bins:
//   shared range: 1) min_max histograms of A and B (their finite ranges), 2) both binned over the union [lo, hi];
//   explicit range: step 2 only, over the user's [lo, hi].
// The overlay is shown only when the two returned bin edge lists are identical. Any failure cancels the other
// side's job and shows no overlay; so does Cancel. Both abort with CANCEL_JOB, which also cancels a job whose
// submission was still in flight. Leaving the drill-down stops waiting but never cancels the jobs.
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiRequestError, isAbort } from "../../lib/api/client";
import {
  CANCEL_JOB,
  histogramBudget,
  runHistogramJob,
  type Consistency,
  type HistogramJobResult,
  type HistogramOptions,
  type TensorHistogram,
} from "../../lib/api/histogram";
import type { Job, Tensor } from "../../lib/api/types";
import { sameBins, sharedRange } from "../histogram/overlay";
import { findTensor } from "./findTensor";

export type RangeMode = "shared" | "explicit";
export type Phase = "idle" | "resolving" | "range" | "bins" | "done" | "failed" | "canceled";

export interface OverlayRequest {
  bins: number;
  mode: RangeMode;
  lo?: number;
  hi?: number;
  consistency: Consistency;
}

export interface OverlayResult {
  request: OverlayRequest;
  lo: number;
  hi: number;
  a: TensorHistogram;
  b: TensorHistogram;
  consistency: Consistency;
}

export interface DistributionCompare {
  phase: Phase;
  jobs: { a: Job | null; b: Job | null };
  result: OverlayResult | null;
  error: unknown;
  run: (r: OverlayRequest) => Promise<void>;
  cancel: () => Promise<void>;
}

export function useDistributionCompare(
  rawKey: string,
  modelA: string | null,
  modelB: string | null,
  settings: Record<string, number | boolean> | undefined,
): DistributionCompare {
  const [phase, setPhase] = useState<Phase>("idle");
  const [jobs, setJobs] = useState<{ a: Job | null; b: Job | null }>({ a: null, b: null });
  const [result, setResult] = useState<OverlayResult | null>(null);
  const [error, setError] = useState<unknown>(null);
  const ctlRef = useRef<AbortController | null>(null);
  const jobsRef = useRef<{ a: Job | null; b: Job | null }>({ a: null, b: null });

  useEffect(() => () => ctlRef.current?.abort(), []); // stop waiting on unmount (jobs keep running)

  const track = useCallback((side: "a" | "b", j: Job) => {
    jobsRef.current = { ...jobsRef.current, [side]: j };
    setJobs(jobsRef.current);
  }, []);

  const run = useCallback(
    async (r: OverlayRequest) => {
      if (!modelA || !modelB) return;
      ctlRef.current?.abort(CANCEL_JOB); // a new run supersedes the previous one, whose jobs would never be shown
      const ctl = new AbortController();
      ctlRef.current = ctl;
      const signal = ctl.signal;
      jobsRef.current = { a: null, b: null };
      setJobs(jobsRef.current);
      setError(null);
      setResult(null);
      const budget = histogramBudget(settings);
      const both = async (ta: Tensor, tb: Tensor, options: HistogramOptions): Promise<[HistogramJobResult, HistogramJobResult]> => {
        const one = (t: Tensor, side: "a" | "b") =>
          runHistogramJob({ modelId: t.model_id, tensorId: t.id, options, consistency: r.consistency, budget, signal, onJob: (j) => track(side, j) });
        return Promise.all([one(ta, "a"), one(tb, "b")]);
      };
      try {
        setPhase("resolving");
        const [ta, tb] = await Promise.all([findTensor(modelA, rawKey, signal), findTensor(modelB, rawKey, signal)]);
        let lo = r.lo ?? 0;
        let hi = r.hi ?? 0;
        if (r.mode === "shared") {
          setPhase("range");
          const [ra, rb] = await both(ta, tb, { bins: r.bins, range: "min_max" });
          const range = sharedRange(ra.histogram, rb.histogram);
          if ("error" in range) throw new ApiRequestError({ status: 200, code: "NO_SHARED_RANGE", message: range.error });
          ({ lo, hi } = range);
        }
        setPhase("bins");
        const [ra, rb] = await both(ta, tb, { bins: r.bins, range: "explicit", lo, hi });
        if (!sameBins(ra.histogram, rb.histogram))
          throw new ApiRequestError({ status: 200, code: "BAD_RESPONSE", message: "A's and B's bins differ, so they cannot share one axis" });
        setResult({ request: r, lo, hi, a: ra.histogram, b: rb.histogram, consistency: r.consistency });
        setPhase("done");
      } catch (e) {
        if (signal.aborted && isAbort(e)) return; // canceled or replaced; whoever aborted set the phase
        ctl.abort(CANCEL_JOB); // the other side stops waiting and cancels its job, even one still being submitted
        setError(e);
        setPhase("failed");
      }
    },
    [modelA, modelB, rawKey, settings, track],
  );

  // Each side's runHistogramJob cancels its own job on this abort (pending, polling or still being submitted).
  const cancel = useCallback(() => {
    ctlRef.current?.abort(CANCEL_JOB);
    setPhase("canceled");
    return Promise.resolve();
  }, []);

  return { phase, jobs, result, error, run, cancel };
}
