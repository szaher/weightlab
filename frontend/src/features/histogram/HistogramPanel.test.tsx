// The histogram panel against a scripted backend that answers with real histograms (fixtures/*.json, written by
// weightlab/analysis/histogram.py from small synthetic tensors).
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { Job, Tensor } from "../../lib/api/types";
import { apiError, fakeApi, json, withToken, type Call } from "../../test/fakeApi";
import { TensorWorkbench } from "../catalog/TensorWorkbench";
import explicitJson from "./fixtures/f32_explicit.json";
import f64Json from "./fixtures/f64_extreme.json";
import minmaxJson from "./fixtures/f32_minmax.json";
import nonfiniteJson from "./fixtures/f32_nonfinite.json";
import i8Json from "./fixtures/i8_symmetric.json";
import { HistogramPanel } from "./HistogramPanel";

const T0 = "2026-11-15T10:00:00Z";
const SETTINGS = { memory_budget_mib: 512, max_job_seconds: 120, max_threads: 2 };

function tensor(extra: Partial<Tensor> = {}): Tensor {
  return {
    id: "ten_1",
    model_id: "mod_1",
    raw_key: "w",
    shape: [4096],
    dtype: "F32",
    rank: 1,
    numel: 4096,
    byte_length: 16384,
    shard_relpath: "model.safetensors",
    file_offset_begin: 0,
    file_offset_end: 16384,
    role: null,
    role_is_heuristic: true,
    layer_index: null,
    ...extra,
  };
}

const job = (state: Job["state"], extra: Partial<Job> = {}): Job => ({
  job_id: "job_1",
  kind: "tensor_histogram",
  state,
  progress: state === "completed" ? 1 : 0.5,
  poll_after_ms: 100,
  estimated_bytes: 32768,
  bytes_scanned: 16384,
  created_at: T0,
  ...extra,
});
const done = (h: unknown) =>
  job("completed", { result: { kind: "tensor_histogram", model_id: "mod_1", tensor_id: "ten_1", consistency: "best_effort", histogram: h } as unknown as NonNullable<Job["result"]> });

/** POST /jobs answers `submitted`; each GET /jobs/job_1 answers the next of `polls` (the last one repeats). */
function backend(submitted: Job | Response, polls: (Job | Response)[] = []) {
  let i = 0;
  const answer = (v: Job | Response) => (v instanceof Response ? v : json(v));
  return fakeApi((c: Call) => {
    const k = `${c.method} ${c.url.pathname}`;
    if (k === "POST /api/v1/jobs") return answer(submitted);
    if (k === "GET /api/v1/jobs/job_1") return answer(polls[Math.min(i++, polls.length - 1)] ?? job("running"));
    if (k === "POST /api/v1/jobs/job_1/cancel") return json(job("canceled"));
    if (k === "GET /api/v1/settings") return json(SETTINGS);
    if (k === "POST /api/v1/tensors/roi") return apiError(503, "UNAVAILABLE", "not in this test");
    throw new Error(`unexpected request ${k}`);
  });
}

const panel = (t = tensor()) => render(<HistogramPanel modelId="mod_1" tensor={t} settings={SETTINGS} />);

describe("HistogramPanel", () => {
  it("submits an exact histogram job, shows progress, then the validated result", async () => {
    withToken();
    const calls = backend(job("queued"), [job("running"), done(minmaxJson)]);
    panel();
    expect(screen.getByTestId("hist-cost")).toHaveTextContent("Two passes over the tensor (32.0 KiB read)");
    await userEvent.selectOptions(screen.getByTestId("hist-bins"), "32");
    await userEvent.click(screen.getByTestId("hist-compute"));

    const post = calls.find((c) => c.method === "POST");
    expect(post?.body).toEqual({
      kind: "tensor_histogram",
      source: { kind: "tensor", model_id: "mod_1", tensor_id: "ten_1" },
      histogram: { bins: 32, range: "min_max" },
      consistency: "best_effort",
      budget: { max_working_set_mib: 512, max_threads: 1, max_wall_seconds: 120 },
    });
    expect(await screen.findByTestId("hist-progress")).toHaveTextContent(/Queued|Counting/);
    const result = await screen.findByTestId("hist-result", {}, { timeout: 3000 });
    expect(screen.queryByTestId("hist-progress")).not.toBeInTheDocument();

    const counts = within(screen.getByTestId("hist-counts"));
    expect(counts.getByText("NaN").nextSibling).toHaveTextContent("1");
    expect(counts.getByText("+Inf").nextSibling).toHaveTextContent("1");
    expect(counts.getByText("−Inf").nextSibling).toHaveTextContent("1");
    expect(counts.getByText("Zeros").nextSibling).toHaveTextContent("2 (incl. 1 −0)");
    expect(counts.getByText("Min").nextSibling).toHaveTextContent(String(minmaxJson.min));
    expect(within(result).getByText(/32 bins · best effort · exact, all 4,096 elements/)).toBeInTheDocument();

    // One bar per nonzero bin; the readout states the bin's exact interval and count.
    const plot = screen.getByTestId("hist-values");
    const nonzero = minmaxJson.bins.counts.filter((n) => n > 0).length;
    expect(plot.querySelectorAll("rect.hist-bar")).toHaveLength(nonzero);
    fireEvent.mouseEnter(plot.querySelectorAll("rect.hist-hit")[0]!);
    const e = minmaxJson.bins.edges;
    expect(screen.getByTestId("hist-values-readout")).toHaveTextContent(`[${e[0]}, ${e[1]}): ${minmaxJson.bins.counts[0]}`);

    // Keyboard: End reads the last bin, which is closed.
    plot.focus();
    await userEvent.keyboard("{End}");
    expect(screen.getByTestId("hist-values-readout")).toHaveTextContent(`[${e[31]}, ${e[32]}]`);

    // The table is the accessible equivalent: one row per bin.
    await userEvent.click(screen.getByTestId("hist-table-toggle"));
    expect(within(screen.getByTestId("hist-values-table")).getAllByRole("row")).toHaveLength(1 + 32);
  });

  it("log counts change bar heights; the magnitude view mirrors positive and negative values", async () => {
    withToken();
    backend(done(minmaxJson));
    panel();
    await userEvent.click(screen.getByTestId("hist-compute"));
    await screen.findByTestId("hist-result");
    const heights = () => [...screen.getByTestId("hist-values").querySelectorAll("rect.hist-bar")].map((r) => Number(r.getAttribute("height")));
    const linear = heights();
    await userEvent.click(screen.getByLabelText("Log counts"));
    const log = heights();
    expect(log).toHaveLength(linear.length);
    expect(Math.min(...log) / Math.max(...log)).toBeGreaterThan(Math.min(...linear) / Math.max(...linear));

    await userEvent.click(screen.getByLabelText("Magnitude (log₂ |x|)"));
    const mag = screen.getByTestId("hist-magnitude");
    expect(mag.querySelectorAll("rect.hist-bar-neg").length).toBeGreaterThan(0);
    await userEvent.click(screen.getByTestId("hist-table-toggle"));
    expect(within(screen.getByTestId("hist-magnitude-table")).getAllByRole("row")).toHaveLength(1 + minmaxJson.magnitude.buckets.length);
    fireEvent.mouseEnter(mag.querySelectorAll("rect.hist-hit")[0]!);
    const b0 = minmaxJson.magnitude.buckets[0]!;
    expect(screen.getByTestId("hist-magnitude-readout")).toHaveTextContent(`2^${b0.exponent} ≤ |x| < 2^${b0.exponent + 1}: ${b0.positive} positive, ${b0.negative} negative`);
  });

  it("explicit range: integer tensors need whole-number bounds, lo ≤ hi; out-of-range values are counted", async () => {
    withToken();
    const calls = backend(done(i8Json));
    panel(tensor({ dtype: "I8", byte_length: 1000, numel: 1000, shape: [1000] }));
    await userEvent.selectOptions(screen.getByTestId("hist-range"), "explicit");
    expect(screen.getByTestId("hist-form-error")).toHaveTextContent("lo: required");
    await userEvent.type(screen.getByLabelText("Lower bound (lo)"), "1.5");
    await userEvent.type(screen.getByLabelText("Upper bound (hi)"), "0");
    expect(screen.getByTestId("hist-form-error")).toHaveTextContent("lo: must be a whole number for an integer tensor");
    expect(screen.getByTestId("hist-compute")).toBeDisabled();
    await userEvent.clear(screen.getByLabelText("Lower bound (lo)"));
    await userEvent.type(screen.getByLabelText("Lower bound (lo)"), "5");
    expect(screen.getByTestId("hist-form-error")).toHaveTextContent("lo must not exceed hi");
    await userEvent.clear(screen.getByLabelText("Upper bound (hi)"));
    await userEvent.type(screen.getByLabelText("Upper bound (hi)"), "100");
    expect(screen.queryByTestId("hist-form-error")).not.toBeInTheDocument();
    expect(screen.getByTestId("hist-cost")).toHaveTextContent("One pass over the tensor");
    await userEvent.click(screen.getByTestId("hist-compute"));
    expect(calls.find((c) => c.method === "POST")?.body).toMatchObject({ histogram: { bins: 64, range: "explicit", lo: 5, hi: 100 } });
    await screen.findByTestId("hist-result");
    // Integer tensors: no NaN/Inf rows and no magnitude view.
    expect(within(screen.getByTestId("hist-counts")).queryByText("NaN")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Magnitude (log₂ |x|)")).not.toBeInTheDocument();
  });

  it("shows below / above counts of an explicit float range", async () => {
    withToken();
    backend(done(explicitJson));
    panel();
    await userEvent.selectOptions(screen.getByTestId("hist-range"), "explicit");
    await userEvent.type(screen.getByLabelText("Lower bound (lo)"), "-0.02");
    await userEvent.type(screen.getByLabelText("Upper bound (hi)"), "0.02");
    await userEvent.click(screen.getByTestId("hist-compute"));
    await screen.findByTestId("hist-result");
    const counts = within(screen.getByTestId("hist-counts"));
    expect(counts.getByText("Below -0.02").nextSibling).toHaveTextContent(String(explicitJson.bins.below));
    expect(counts.getByText("Above 0.02").nextSibling).toHaveTextContent(String(explicitJson.bins.above));
    await userEvent.click(screen.getByTestId("hist-table-toggle"));
    expect(within(screen.getByTestId("hist-values-table")).getAllByRole("row")).toHaveLength(1 + 1 + explicitJson.bins.counts.length + 1);
  });

  it("draws an F64 tensor spanning ±1.8e308 with finite coordinates", async () => {
    withToken();
    backend(done(f64Json));
    panel(tensor({ dtype: "F64", shape: [8], numel: 8, byte_length: 64 }));
    await userEvent.click(screen.getByTestId("hist-compute"));
    await screen.findByTestId("hist-result");
    const svg = screen.getByTestId("hist-values").querySelector("svg")!;
    const coords = [...svg.querySelectorAll("rect, line, text")].flatMap((el) => ["x", "y", "width", "height", "x1", "x2", "y1", "y2"].map((a) => el.getAttribute(a)).filter((v) => v !== null));
    expect(coords.length).toBeGreaterThan(0);
    expect(coords.every((v) => Number.isFinite(Number(v)))).toBe(true);
    const bars = [...svg.querySelectorAll("rect.hist-bar")].map((r) => Number(r.getAttribute("width")));
    expect(bars).toHaveLength(f64Json.bins.counts.filter((n) => n > 0).length);
    expect(svg.textContent).toContain("0"); // the zero label is placed mid-axis
  });

  it("a tensor without finite values has counts but no bins", async () => {
    withToken();
    backend(done(nonfiniteJson));
    panel(tensor({ shape: [2], numel: 2, byte_length: 8 }));
    await userEvent.click(screen.getByTestId("hist-compute"));
    expect(await screen.findByTestId("hist-no-bins")).toBeInTheDocument();
    expect(screen.queryByTestId("hist-values")).not.toBeInTheDocument();
  });

  it("a failed job explains how to resolve it and shows no histogram", async () => {
    withToken();
    backend(job("queued"), [job("failed", { error: { code: "FILE_CHANGED", message: "the tensor's file changed after the job was queued", retryable: false } })]);
    panel();
    await userEvent.click(screen.getByTestId("hist-compute"));
    const ended = await screen.findByTestId("hist-job-ended");
    expect(ended).toHaveTextContent("The histogram failed");
    expect(ended).toHaveTextContent("Re-import the model");
    expect(ended).toHaveTextContent("No partial histogram is ever shown.");
    expect(screen.queryByTestId("hist-result")).not.toBeInTheDocument();
  });

  it("drops the shown histogram when the tensor's file changed (it described the old bytes)", async () => {
    withToken();
    let first = true;
    fakeApi((c: Call) => {
      if (c.method === "POST" && c.url.pathname === "/api/v1/jobs") {
        if (first) {
          first = false;
          return json(done(minmaxJson));
        }
        return apiError(409, "FILE_CHANGED", "model files changed since import");
      }
      throw new Error(`unexpected request ${c.method} ${c.url.pathname}`);
    });
    panel();
    await userEvent.click(screen.getByTestId("hist-compute"));
    await screen.findByTestId("hist-result");
    await userEvent.click(screen.getByTestId("hist-compute"));
    expect(await screen.findByText("Model files changed on disk")).toBeInTheDocument();
    expect(screen.getAllByText(/Re-import the model/)).toHaveLength(1); // advice is given once
    expect(screen.queryByTestId("hist-result")).not.toBeInTheDocument();
  });

  it("refuses a completed result that breaks the histogram invariants", async () => {
    withToken();
    const broken = JSON.parse(JSON.stringify(minmaxJson)) as typeof minmaxJson;
    broken.bins.counts[3]! += 1;
    backend(done(broken));
    panel();
    await userEvent.click(screen.getByTestId("hist-compute"));
    expect(await screen.findByText(/below \+ bins \+ above does not equal the finite count/)).toBeInTheDocument();
    expect(screen.queryByTestId("hist-result")).not.toBeInTheDocument();
  });

  it("cancels a running job", async () => {
    withToken();
    const calls = backend(job("running", { poll_after_ms: 5000 }));
    panel();
    await userEvent.click(screen.getByTestId("hist-compute"));
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(calls.some((c) => c.method === "POST" && c.url.pathname === "/api/v1/jobs/job_1/cancel")).toBe(true);
    expect(await screen.findByTestId("hist-job-ended")).toHaveTextContent("The histogram was canceled");
  });

  it("explains a submit refused for its memory budget", async () => {
    withToken();
    backend(apiError(422, "RESOURCE_LIMIT", "a histogram needs 64 MiB of working memory"));
    panel();
    await userEvent.click(screen.getByTestId("hist-compute"));
    expect(await screen.findByText(/Raise memory_budget_mib or max_job_seconds/)).toBeInTheDocument();
  });

  it("is disabled up front when Settings allow less memory than a histogram needs", () => {
    withToken();
    backend(job("queued"));
    render(<HistogramPanel modelId="mod_1" tensor={tensor()} settings={{ memory_budget_mib: 32 }} />);
    expect(screen.getByTestId("hist-compute")).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("below the 64 MiB a histogram needs");
  });
});

describe("TensorWorkbench histogram tab", () => {
  it("is offered for 1-D tensors and keeps its result while another tab is shown", async () => {
    withToken();
    backend(done(minmaxJson));
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    expect(screen.getByRole("tab", { name: "Exact ROI" })).toHaveAttribute("aria-selected", "true");
    await userEvent.click(screen.getByRole("tab", { name: "Histogram" }));
    await waitFor(() => expect(screen.getByTestId("hist-compute")).toBeEnabled());
    await userEvent.click(screen.getByTestId("hist-compute"));
    await screen.findByTestId("hist-result");
    screen.getByRole("tab", { name: "Histogram" }).focus();
    await userEvent.keyboard("{ArrowLeft}");
    expect(screen.getByRole("tab", { name: "Exact ROI" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByTestId("histogram-panel").closest("[role=tabpanel]")).not.toBeVisible();
    await userEvent.click(screen.getByRole("tab", { name: "Histogram" }));
    expect(screen.getByTestId("hist-result")).toBeVisible();
  });

  it("is not offered for empty tensors", () => {
    withToken();
    backend(job("queued"));
    render(<TensorWorkbench modelId="mod_1" tensor={tensor({ shape: [0], numel: 0, byte_length: 0 })} />);
    expect(screen.queryByRole("tab", { name: "Histogram" })).not.toBeInTheDocument();
  });
});
