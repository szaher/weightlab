// The A/B distribution overlay against a scripted backend that answers with real histograms (histogram/fixtures/
// ab_f32.json: the backend's min-max and shared-range histograms of two synthetic tensors).
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { TensorRow } from "../../lib/api/compare";
import type { Job, Tensor } from "../../lib/api/types";
import { apiError, fakeApi, json, withToken, type Call } from "../../test/fakeApi";
import ab from "../histogram/fixtures/ab_f32.json";
import { DistributionCompare } from "./DistributionCompare";

const KEY = "model.layers.0.mlp.up_proj.weight";
const ROW = { key: KEY, dtype_a: "F32", dtype_b: "F32", counts: { elements: 2048 }, shape: [2048] } as unknown as TensorRow;
const A = { id: "mod_a", name: "base.safetensors" };
const B = { id: "mod_b", name: "ft.safetensors" };
const tensor = (model: string): Tensor =>
  ({ id: `ten_${model}`, model_id: model, raw_key: KEY, shape: [2048], dtype: "F32", rank: 1, numel: 2048 }) as unknown as Tensor;

let seq = 0;
const job = (state: Job["state"], extra: Partial<Job> = {}): Job => ({
  job_id: `job_${++seq}`,
  kind: "tensor_histogram",
  state,
  progress: state === "completed" ? 1 : 0.3,
  poll_after_ms: 100,
  created_at: "2026-11-15T00:00:00Z",
  ...extra,
});
const result = (model: string, h: unknown) => ({ kind: "tensor_histogram", model_id: model, tensor_id: `ten_${model}`, consistency: "best_effort", histogram: h }) as unknown as NonNullable<Job["result"]>;

interface Body {
  source: { model_id: string };
  histogram: { range: string; lo?: number; hi?: number; bins: number };
}

/** Each POST /jobs is answered by `answer(model, body)`; GET/cancel of a job id answer from `jobs`. */
function backend(answer: (model: string, body: Body) => Job | Response) {
  const jobs: Record<string, Job> = {};
  const calls = fakeApi((c: Call) => {
    const k = `${c.method} ${c.url.pathname}`;
    if (k === "GET /api/v1/settings") return json({ memory_budget_mib: 1024, max_job_seconds: 600 });
    const m = /^GET \/api\/v1\/models\/(mod_[ab])\/tensors$/.exec(k);
    if (m) return json({ total: 1, items: [tensor(m[1]!)] });
    if (k === "POST /api/v1/jobs") {
      const body = c.body as Body;
      const r = answer(body.source.model_id, body);
      if (r instanceof Response) return r;
      jobs[r.job_id] = r;
      return json(r);
    }
    const j = /^(GET|POST) \/api\/v1\/jobs\/([^/]+)(\/cancel)?$/.exec(k);
    if (j && jobs[j[2]!]) {
      if (j[3]) jobs[j[2]!] = { ...jobs[j[2]!]!, state: "canceled" };
      return json(jobs[j[2]!]);
    }
    throw new Error(`unexpected request ${k}`);
  });
  return { calls, jobs };
}

const ok = (model: string, body: Body) => {
  const side = model === "mod_a" ? "a" : "b";
  const h = body.histogram.range === "min_max" ? ab[`${side}_minmax`] : ab[`${side}_shared`];
  return job("completed", { result: result(model, h) });
};

const view = () => render(<DistributionCompare row={ROW} modelA={A} modelB={B} />);

describe("DistributionCompare", () => {
  it("finds the shared range, bins both sides over it and overlays the backend's counts", async () => {
    withToken();
    const { calls } = backend(ok);
    view();
    expect(screen.getByTestId("dist-cost")).toHaveTextContent("Three passes over each tensor");
    await userEvent.selectOptions(screen.getByTestId("dist-bins"), "16");
    await userEvent.click(screen.getByTestId("dist-run"));
    await screen.findByTestId("dist-result");

    const submits = calls.filter((c) => c.method === "POST" && c.url.pathname === "/api/v1/jobs").map((c) => c.body as Body);
    expect(submits.map((b) => [b.source.model_id, b.histogram])).toEqual([
      ["mod_a", { bins: 16, range: "min_max" }],
      ["mod_b", { bins: 16, range: "min_max" }],
      ["mod_a", { bins: 16, range: "explicit", lo: ab.lo, hi: ab.hi }],
      ["mod_b", { bins: 16, range: "explicit", lo: ab.lo, hi: ab.hi }],
    ]);

    const counts = within(screen.getByTestId("dist-counts"));
    expect(counts.getByText("NaN").parentElement).toHaveTextContent("NaN01");
    expect(counts.getByText("Min").parentElement).toHaveTextContent(`Min${ab.a_shared.min}${ab.b_shared.min}`);

    const chart = screen.getByTestId("overlay-chart");
    expect(chart.querySelectorAll("rect.hist-bar-a")).toHaveLength(ab.a_shared.bins.counts.filter((n) => n > 0).length);
    expect(screen.getByTestId("overlay-b")).toBeInTheDocument();
    expect(screen.getByText(/A · base.safetensors/)).toBeInTheDocument();
    fireEvent.mouseEnter(chart.querySelectorAll("rect.hist-hit")[5]!);
    const a5 = ab.a_shared.bins.counts[5]!;
    const b5 = ab.b_shared.bins.counts[5]!;
    expect(screen.getByTestId("overlay-readout")).toHaveTextContent(`A ${a5.toLocaleString("en-US")}`);
    expect(screen.getByTestId("overlay-readout")).toHaveTextContent(`B ${b5.toLocaleString("en-US")}`);

    await userEvent.click(screen.getByTestId("dist-table-toggle"));
    const rows = within(screen.getByTestId("dist-bins-table")).getAllByRole("row");
    expect(rows).toHaveLength(1 + ab.a_shared.bins.counts.length);
    const d = b5 - a5;
    expect(rows[6]).toHaveTextContent(d > 0 ? `+${d}` : d < 0 ? `−${-d}` : "0");
  });

  it("explicit range: one job per side over the user's bounds", async () => {
    withToken();
    const { calls } = backend(ok);
    view();
    await userEvent.selectOptions(screen.getByTestId("dist-range"), "explicit");
    await userEvent.type(screen.getByLabelText("Lower bound (lo)"), String(ab.lo));
    await userEvent.type(screen.getByLabelText("Upper bound (hi)"), String(ab.hi));
    expect(screen.getByTestId("dist-cost")).toHaveTextContent("One pass over each tensor");
    await userEvent.click(screen.getByTestId("dist-run"));
    await screen.findByTestId("dist-result");
    const submits = calls.filter((c) => c.method === "POST" && c.url.pathname === "/api/v1/jobs").map((c) => (c.body as Body).histogram.range);
    expect(submits).toEqual(["explicit", "explicit"]);
    expect(within(screen.getByTestId("dist-counts")).getByText("Below lo")).toBeInTheDocument();
  });

  it("refuses to overlay bins that are not identical", async () => {
    withToken();
    backend((model, body) => {
      const r = ok(model, body);
      if (model === "mod_b" && body.histogram.range === "explicit") {
        const h = JSON.parse(JSON.stringify(ab.b_shared)) as typeof ab.b_shared;
        h.bins.edges[2] = h.bins.edges[2]! + 1e-9;
        return job("completed", { result: result(model, h) });
      }
      return r;
    });
    view();
    await userEvent.click(screen.getByTestId("dist-run"));
    expect(await screen.findByTestId("dist-error")).toHaveTextContent("A's and B's bins differ");
    expect(screen.queryByTestId("dist-result")).not.toBeInTheDocument();
  });

  it("when one side fails, cancels the other side's job and shows no overlay", async () => {
    withToken();
    const { calls } = backend((model) =>
      model === "mod_a" ? job("running", { poll_after_ms: 5000 }) : apiError(409, "FILE_CHANGED", "model files changed since import"),
    );
    view();
    await userEvent.click(screen.getByTestId("dist-run"));
    expect(await screen.findByTestId("dist-error")).toHaveTextContent("Model files changed on disk");
    expect(calls.some((c) => c.method === "POST" && /\/jobs\/job_\d+\/cancel$/.test(c.url.pathname))).toBe(true);
    expect(screen.queryByTestId("dist-result")).not.toBeInTheDocument();
  });

  it("cancel asks both jobs to stop and shows no partial overlay", async () => {
    withToken();
    const { calls } = backend(() => job("running", { poll_after_ms: 5000 }));
    view();
    await userEvent.click(screen.getByTestId("dist-run"));
    expect(await screen.findByTestId("dist-progress")).toHaveTextContent("Step 1 of 2");
    await screen.findByText(/^B: counting/);
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(await screen.findByTestId("dist-canceled")).toBeInTheDocument();
    expect(calls.filter((c) => c.method === "POST" && /\/cancel$/.test(c.url.pathname))).toHaveLength(2);
  });

  it("is not offered when A and B have different value domains", () => {
    withToken();
    backend(ok);
    render(<DistributionCompare row={{ ...ROW, dtype_b: "I8" }} modelA={A} modelB={B} />);
    expect(screen.getByText(/cannot share bins/)).toBeInTheDocument();
    expect(screen.queryByTestId("dist-run")).not.toBeInTheDocument();
  });
});
