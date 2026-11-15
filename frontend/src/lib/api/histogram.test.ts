// parseHistogramResult against real backend histograms (features/histogram/fixtures, written by
// weightlab/analysis/histogram.py) and against each invariant broken one at a time.
import { describe, expect, it } from "vitest";
import explicitJson from "../../features/histogram/fixtures/f32_explicit.json";
import minmaxJson from "../../features/histogram/fixtures/f32_minmax.json";
import nonfiniteJson from "../../features/histogram/fixtures/f32_nonfinite.json";
import i8Json from "../../features/histogram/fixtures/i8_symmetric.json";
import { ApiRequestError } from "./client";
import { fakeApi, json, withToken } from "../../test/fakeApi";
import { CANCEL_JOB, histogramBudget, parseHistogramResult, runHistogramJob, toNumber } from "./histogram";

const EXPECT = { modelId: "mod_1", tensorId: "ten_1" };
const wrap = (h: unknown, extra: object = {}) => ({ kind: "tensor_histogram", model_id: "mod_1", tensor_id: "ten_1", consistency: "best_effort", histogram: h, ...extra });
const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T;
type H = typeof minmaxJson & Record<string, unknown>;

function rejects(mutate: (h: H) => void, extra: object = {}): void {
  const h = clone(minmaxJson) as H;
  mutate(h);
  let err: unknown;
  try {
    parseHistogramResult(wrap(h, extra), EXPECT);
  } catch (e) {
    err = e;
  }
  expect(err).toBeInstanceOf(ApiRequestError);
  expect((err as ApiRequestError).code).toBe("BAD_RESPONSE");
}

describe("parseHistogramResult", () => {
  it.each([
    ["f32 min_max", minmaxJson],
    ["f32 explicit", explicitJson],
    ["i8 symmetric", i8Json],
    ["no finite values", nonfiniteJson],
  ])("accepts the backend's %s histogram unchanged", (_, h) => {
    const r = parseHistogramResult(wrap(h), EXPECT);
    expect(r.histogram).toEqual(h);
  });

  it("refuses a result for another tensor or kind", () => {
    expect(() => parseHistogramResult(wrap(minmaxJson, { tensor_id: "ten_2" }), EXPECT)).toThrow(/different tensor/);
    expect(() => parseHistogramResult(wrap(minmaxJson, { kind: "tensor_stats" }), EXPECT)).toThrow(ApiRequestError);
    expect(() => parseHistogramResult(null, EXPECT)).toThrow(ApiRequestError);
  });

  it("refuses unknown versions and sampled coverage", () => {
    rejects((h) => (h.histogram_version = 2 as 1));
    rejects((h) => (h.coverage = "sampled"));
    rejects((h) => (h.value_domain = "complex"));
  });

  it("refuses counts that do not add up", () => {
    rejects((h) => (h.counts.elements += 1));
    rejects((h) => (h.counts.nan += 1));
    rejects((h) => (h.counts.zeros += 1));
    rejects((h) => (h.counts.negative_zeros = h.counts.zeros + 1));
    rejects((h) => (h.counts.finite = -1));
    rejects((h) => (h.counts.positives = 1.5));
  });

  it("refuses bins that do not cover the finite values", () => {
    rejects((h) => (h.bins.counts[0]! += 1));
    rejects((h) => (h.bins.below += 1));
    rejects((h) => h.bins.edges.pop());
    rejects((h) => h.bins.edges.reverse());
    rejects((h) => (h.bins.requested_bins = 4));
    rejects((h) => ((h as Record<string, unknown>).bins = null));
  });

  it("refuses min / max that contradict the counts", () => {
    rejects((h) => ((h as Record<string, unknown>).min = null));
    rejects((h) => {
      const t = h.min;
      h.min = h.max;
      h.max = t;
    });
  });

  it("refuses a magnitude view that is missing, misordered or does not add up", () => {
    rejects((h) => ((h as Record<string, unknown>).magnitude = null));
    rejects((h) => h.magnitude.buckets.reverse());
    rejects((h) => (h.magnitude.buckets[0]!.positive += 1));
    const i8 = clone(i8Json) as Record<string, unknown>;
    i8.magnitude = { rule: "x", buckets: [] };
    expect(() => parseHistogramResult(wrap(i8), EXPECT)).toThrow(/magnitude/);
  });

  it("keeps 64-bit integers that arrive as decimal strings", () => {
    const h = clone(i8Json) as Record<string, unknown> & typeof i8Json;
    (h as Record<string, unknown>).max = "9007199254740993";
    const r = parseHistogramResult(wrap(h), EXPECT);
    expect(r.histogram.max).toBe("9007199254740993");
    expect(toNumber("9007199254740993")).toBe(9007199254740992);
    (h as Record<string, unknown>).max = "12abc";
    expect(() => parseHistogramResult(wrap(h), EXPECT)).toThrow(ApiRequestError);
  });
});

describe("histogramBudget", () => {
  it("uses the server's caps from Settings", () => {
    expect(histogramBudget({ memory_budget_mib: 512, max_job_seconds: 90 })).toEqual({ max_working_set_mib: 512, max_threads: 1, max_wall_seconds: 90 });
    expect(histogramBudget(undefined)).toEqual({ max_working_set_mib: 1024, max_threads: 1, max_wall_seconds: 600 });
  });
});

describe("runHistogramJob", () => {
  const job = (state: string, extra: object = {}) => ({ job_id: "job_1", kind: "tensor_histogram", state, progress: state === "completed" ? 1 : 0.5, poll_after_ms: 100, created_at: "2026-11-15T00:00:00Z", ...extra });
  const args = (onJob: (j: unknown) => void = () => undefined) => ({
    modelId: "mod_1",
    tensorId: "ten_1",
    options: { bins: 32, range: "min_max" as const },
    consistency: "best_effort" as const,
    budget: histogramBudget(undefined),
    signal: new AbortController().signal,
    onJob,
  });

  it("submits, polls until completed and returns the validated result", async () => {
    withToken();
    const states: string[] = [];
    const polls = [job("running"), job("completed", { result: wrap(minmaxJson) })];
    fakeApi((c) => (c.method === "POST" ? json(job("queued")) : json(polls.shift())));
    const r = await runHistogramJob(args((j) => states.push((j as { state: string }).state)));
    expect(r.histogram).toEqual(minmaxJson);
    expect(states).toEqual(["queued", "running", "completed"]);
  });

  it("turns a failed, canceled or interrupted job into its error", async () => {
    withToken();
    for (const [end, code] of [
      [job("failed", { error: { code: "FILE_CHANGED", message: "changed", retryable: false } }), "FILE_CHANGED"],
      [job("canceled"), "CANCELED"],
      [job("interrupted"), "INTERRUPTED"],
    ] as const) {
      fakeApi((c) => (c.method === "POST" ? json(job("queued")) : json(end)));
      await expect(runHistogramJob(args())).rejects.toMatchObject({ code });
    }
  });

  it("a CANCEL_JOB abort during submission cancels the job as soon as its id arrives", async () => {
    withToken();
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const calls = fakeApi(async (c) => {
      if (c.url.pathname === "/api/v1/jobs") {
        await gate; // the submission is still in flight when the caller cancels
        return json(job("queued"));
      }
      return json(job("canceled"));
    });
    const ctl = new AbortController();
    const p = runHistogramJob({ ...args(), signal: ctl.signal });
    ctl.abort(CANCEL_JOB);
    release();
    await expect(p).rejects.toMatchObject({ name: "AbortError" });
    expect(calls.map((c) => `${c.method} ${c.url.pathname}`)).toEqual(["POST /api/v1/jobs", "POST /api/v1/jobs/job_1/cancel"]);
  });

  it("a CANCEL_JOB abort while polling cancels the job; any other abort leaves it running", async () => {
    withToken();
    for (const reason of [CANCEL_JOB, undefined]) {
      const calls = fakeApi((c) => json(job(c.url.pathname.endsWith("/cancel") ? "canceled" : "running", { poll_after_ms: 5000 })));
      const ctl = new AbortController();
      const p = runHistogramJob({ ...args(), signal: ctl.signal, onJob: () => ctl.abort(reason) });
      await expect(p).rejects.toMatchObject({ name: "AbortError" });
      const cancels = calls.filter((c) => c.url.pathname.endsWith("/cancel"));
      expect(cancels).toHaveLength(reason === CANCEL_JOB ? 1 : 0);
    }
  });

  it("refuses a completed result that is not valid", async () => {
    withToken();
    fakeApi(() => json(job("completed", { result: wrap({ ...minmaxJson, coverage: "sampled" }) })));
    await expect(runHistogramJob(args())).rejects.toMatchObject({ code: "BAD_RESPONSE" });
  });
});
