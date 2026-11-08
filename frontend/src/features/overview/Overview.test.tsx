// Component tests of the overview and its drill-down, against a fake server (src/test/fakeApi.ts). jsdom has no
// canvas, so what is drawn is observed through the text the component derives from the same tiles: coverage
// badge, legend range (computed from the drawn tiles' values), selection bounds and the requests it sends.
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { Job, RoiRequest, Tensor, TileRequest } from "../../lib/api/types";
import { fakeApi, json, withToken, type Call } from "../../test/fakeApi";
import { KEY, tileResponse } from "../../test/tiles";
import { TensorWorkbench } from "../catalog/TensorWorkbench";
import { sharedTileCache } from "./tileStore";

const SHAPE = [1024, 2304];
function tensor(id = "tns_1", shape = SHAPE): Tensor {
  const numel = shape.reduce((a, b) => a * b, 1);
  return { id, model_id: "mod_1", raw_key: `w.${id}`, shape, dtype: "F32", rank: shape.length, numel, byte_length: numel * 4, shard_relpath: "m.safetensors", file_offset_begin: 0, file_offset_end: numel * 4, role: null, role_is_heuristic: true, layer_index: null };
}

const SETTINGS = { roi_max_cells: 4096, pyramid_exact_max_mib: 64, max_active_tiles: 2, direct_tile_max_mib: 8 };
const job = (state: Job["state"], extra: Partial<Job> = {}): Job => ({ job_id: "job_1", kind: "pyramid_build", state, progress: state === "completed" ? 1 : 0.25, poll_after_ms: 100, estimated_bytes: 9437184, bytes_scanned: 1024, created_at: "2026-01-01T00:00:00Z", ...extra });
const meta = (t: TileRequest, extra: Record<string, unknown> = {}, dims = { rows: 1024, cols: 2304 }) => {
  const b = 2 ** t.tile.level;
  return { schema_version: 2, cache_key: KEY, rows: Math.ceil(dims.rows / b), cols: Math.ceil(dims.cols / b), row_start: 0, row_end: dims.rows, col_start: 0, col_end: dims.cols, level: t.tile.level, quality: "exact", value_precision: "float32_visualization", sampled_rows: null, sampling_policy: null, seed: null, aggregation: t.aggregation, finite_count: 1, nonfinite_count: 0, source: t.source, view: t.view, warnings: [], ...extra };
};

interface Server {
  tile?: (body: TileRequest, call: Call) => Response | Promise<Response>;
  job?: (call: Call) => Response;
  meta?: (call: Call) => Response;
  roi?: (body: RoiRequest) => Response;
  settings?: Record<string, number | boolean>;
}
let lastTile: TileRequest | null = null;
function server(s: Server): Call[] {
  sharedTileCache.purge(() => true);
  withToken();
  return fakeApi((c) => {
    const p = c.url.pathname;
    if (p === "/api/v1/settings") return json(s.settings ?? SETTINGS);
    if (p === "/api/v1/tiles") {
      lastTile = c.body as TileRequest;
      const t = c.body as TileRequest;
      return s.tile ? s.tile(t, c) : tileResponse(t, t.view.transpose ? { rows: 2304, cols: 1024 } : { rows: 1024, cols: 2304 });
    }
    if (p.startsWith("/api/v1/tiles/")) return s.meta ? s.meta(c) : json(meta(lastTile!));
    if (p.startsWith("/api/v1/jobs/")) return s.job ? s.job(c) : json(job("running"));
    if (p === "/api/v1/tensors/roi") {
      const b = c.body as RoiRequest;
      if (s.roi) return s.roi(b);
      const { r0, r1, c0, c1 } = b.roi;
      const values = Array.from({ length: r1 - r0 }, (_, i) => Array.from({ length: c1 - c0 }, (_, j) => (r0 + i) * 10000 + c0 + j));
      return json({ schema_version: 2, source: b.source, roi: b.roi, shape: [r1 - r0, c1 - c0], source_dtype: "F32", value_encoding: "float_json", coverage: "complete", numeric_precision: "lossless", values, nonfinite_mask: null, warnings: [] });
    }
    throw new Error(`unexpected ${c.method} ${p}`);
  });
}
const tiles = (calls: Call[]) => calls.filter((c) => c.url.pathname === "/api/v1/tiles");
const jobs = (calls: Call[]) => calls.filter((c) => c.url.pathname.startsWith("/api/v1/jobs/"));

describe("whole-tensor overview", () => {
  it("shows a 200 exact tile with its coverage, precision and aggregation disclosures", async () => {
    const calls = server({});
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    expect(await screen.findByTestId("coverage-badge")).toHaveTextContent("EXACT coverage");
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByTestId("overview-badges")).toHaveTextContent("float32 visualization");
    expect(screen.getByTestId("coverage-text")).toHaveTextContent("every element of every block was aggregated");
    expect(screen.getByTestId("overview-facts")).toHaveTextContent("mean absolute value");
    expect(screen.getByTestId("overview-legend")).toHaveTextContent("visible-region scale");
    const body = tiles(calls)[0]!.body as TileRequest;
    expect(body).toMatchObject({ source: { kind: "tensor", model_id: "mod_1", tensor_id: "tns_1" }, tile: { level: 4, x: 0, y: 0, size: 256 }, aggregation: "mean_abs", quality: { mode: "auto" } });
    expect(tiles(calls)[0]!.headers.Accept).toBe("application/vnd.weightlab.tile.v2");
  });

  it("labels a row-sampled tile with rows, policy and seed, and changes the seed on request", async () => {
    const calls = server({
      tile: (b) => tileResponse(b, { rows: 1024, cols: 18432 }, { quality: "row_sampled", sampledRows: 4 }),
      meta: () => json(meta(lastTile!, { quality: "row_sampled", sampled_rows: 4, sampling_policy: lastTile!.quality.seed === undefined ? "stride" : "splitmix64_rows", seed: lastTile!.quality.seed ?? null }, { rows: 1024, cols: 18432 })),
    });
    render(<TensorWorkbench modelId="mod_1" tensor={tensor("tns_big", [1024, 2304 * 8])} />);
    expect(await screen.findByTestId("coverage-badge")).toHaveTextContent("ROW-SAMPLED preview");
    await waitFor(() => expect(screen.getByTestId("coverage-text")).toHaveTextContent("only 4 row(s) of each block were read (evenly strided rows)"));
    expect(screen.getByTestId("sampling-panel")).toHaveTextContent("not exact");
    await userEvent.type(screen.getByTestId("seed-input"), "7");
    await userEvent.click(screen.getByRole("button", { name: "Apply seed" }));
    await waitFor(() => expect(screen.getByTestId("coverage-text")).toHaveTextContent("seeded SplitMix64 (seed 7)"));
    expect(tiles(calls).at(-1)!.body).toMatchObject({ quality: { mode: "auto", seed: 7 } });
    // Asking for the exact overview shows the I/O cost first; nothing exact is requested before confirming.
    await userEvent.click(screen.getByTestId("request-exact"));
    expect(screen.getByTestId("confirm-exact")).toHaveTextContent("Estimated read: 72.0 MiB");
    expect(tiles(calls).filter((c) => (c.body as TileRequest).quality.mode === "exact")).toEqual([]);
  }, 20_000);

  it("polls a 202 job by poll_after_ms, then requests the tile again when it completes", async () => {
    let built = false;
    let polls = 0;
    const calls = server({
      tile: (b) => (built ? tileResponse(b, { rows: 1024, cols: 2304 }) : json(job("queued"), 202)),
      job: () => {
        polls++;
        if (polls >= 2) {
          built = true;
          return json(job("completed", { result: { kind: "pyramid_build", pyramid_id: "f".repeat(64) } }));
        }
        return json(job("running", { progress: 0.5, bytes_scanned: 4718592 }));
      },
    });
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    expect(await screen.findByTestId("job-panel")).toHaveTextContent("Building the exact overview (queued)");
    expect(screen.getByTestId("building-badge")).toBeInTheDocument();
    expect(screen.queryByTestId("coverage-badge")).toBeNull(); // never shown as exact while building
    await waitFor(() => expect(screen.getByTestId("job-panel")).toHaveTextContent("50%"));
    expect(screen.getByTestId("job-panel")).toHaveTextContent("read 4.50 MiB of ~9.00 MiB");
    expect(await screen.findByTestId("coverage-badge")).toHaveTextContent("EXACT coverage");
    expect(screen.queryByTestId("job-panel")).toBeNull();
    expect(tiles(calls).length).toBeGreaterThanOrEqual(2);
    const before = jobs(calls).length;
    await new Promise((r) => setTimeout(r, 300));
    expect(jobs(calls).length).toBe(before); // a terminal job is not polled again
  });

  it("cancels the job once on explicit request, and never shows the canceled build as exact", async () => {
    const calls = server({
      tile: () => json(job("running"), 202),
      job: (c) => (c.method === "POST" ? json(job("canceled", { progress: 0.25 })) : json(job("running"))),
    });
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    const cancel = await screen.findByTestId("job-cancel");
    fireEvent.click(cancel);
    fireEvent.click(cancel);
    await waitFor(() => expect(screen.getByTestId("job-panel")).toHaveAttribute("data-state", "canceled"));
    expect(screen.getByTestId("job-panel")).toHaveTextContent("Nothing was published");
    expect(calls.filter((c) => c.method === "POST" && c.url.pathname === "/api/v1/jobs/job_1/cancel").length).toBe(1);
    expect(screen.queryByTestId("coverage-badge")).toBeNull();
  });

  it("stops polling on unmount without canceling the shared job", async () => {
    const calls = server({ tile: () => json(job("running", { deduplicated: true }), 202) });
    const { unmount } = render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    expect(await screen.findByTestId("job-panel")).toHaveTextContent("shared with another request");
    unmount();
    const n = jobs(calls).length;
    await new Promise((r) => setTimeout(r, 350));
    expect(jobs(calls).length).toBe(n);
    expect(calls.some((c) => c.url.pathname.endsWith("/cancel"))).toBe(false);
  });

  it("keeps polling after a temporary status failure, with backoff, and completes", async () => {
    let built = false;
    let polls = 0;
    const calls = server({
      tile: (b) => (built ? tileResponse(b, { rows: 1024, cols: 2304 }) : json(job("queued"), 202)),
      job: () => {
        polls++;
        if (polls === 1) throw new TypeError("connection reset"); // the client turns this into NETWORK
        built = true;
        return json(job("completed", { result: { kind: "pyramid_build", pyramid_id: "f".repeat(64) } }));
      },
    });
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    expect(await screen.findByTestId("job-poll-error")).toHaveTextContent("checking again shortly");
    expect(screen.getByTestId("job-panel")).toHaveAttribute("data-state", "queued");
    expect(await screen.findByTestId("coverage-badge", {}, { timeout: 4000 })).toHaveTextContent("EXACT coverage");
    expect(polls).toBe(2);
    expect(calls.some((c) => c.url.pathname.endsWith("/cancel"))).toBe(false);
  });

  it("offers a manual status check that restarts polling at once", async () => {
    let fail = true;
    let polls = 0;
    server({
      tile: () => json(job("running"), 202),
      job: () => {
        polls++;
        if (fail) throw new TypeError("connection reset");
        return json(job("canceled"));
      },
    });
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    const again = await screen.findByTestId("job-check-again");
    fail = false;
    fireEvent.click(again);
    await waitFor(() => expect(screen.getByTestId("job-panel")).toHaveAttribute("data-state", "canceled"), { timeout: 1500 });
    expect(polls).toBe(2);
  });

  it("explains failed and interrupted builds", async () => {
    server({ tile: () => json(job("running"), 202), job: () => json(job("interrupted")) });
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    expect(await screen.findByText(/The build did not complete: the WeightLab server stopped or restarted/)).toBeInTheDocument();
    expect(screen.getByTestId("job-retry")).toBeInTheDocument();
  });

  it("never lets a late answer of the previous aggregation replace the current one", async () => {
    let releaseLate: (() => void) | null = null;
    server({
      tile: (b) =>
        b.aggregation === "mean_abs"
          ? new Promise<Response>((ok) => (releaseLate = () => ok(tileResponse(b, { rows: 1024, cols: 2304 }, {}, () => 1000))))
          : tileResponse(b, { rows: 1024, cols: 2304 }, {}, (i, j) => (i === 0 && j === 0 ? 1 : 2)),
    });
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    await screen.findByTestId("agg-select");
    await waitFor(() => expect(releaseLate).not.toBeNull());
    fireEvent.change(screen.getByTestId("agg-select"), { target: { value: "max" } });
    await waitFor(() => expect(screen.getByTestId("overview-legend-hi")).toHaveTextContent("2"));
    await act(async () => {
      releaseLate?.();
      await new Promise((r) => setTimeout(r, 50));
    });
    expect(screen.getByTestId("overview-legend-lo")).toHaveTextContent("1");
    expect(screen.getByTestId("overview-legend-hi")).toHaveTextContent("2");
    expect(screen.getByTestId("overview-legend")).not.toHaveTextContent("1000");
  });

  it("maps a selected block to the existing Exact ROI explorer on the same view and raw bounds", async () => {
    const calls = server({});
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    await screen.findByTestId("coverage-badge");
    const canvas = screen.getByTestId("overview-canvas");
    canvas.focus();
    fireEvent.keyDown(canvas, { key: "ArrowDown" }); // first block of the visible region
    fireEvent.keyDown(canvas, { key: "ArrowRight" });
    fireEvent.keyDown(canvas, { key: "ArrowDown" });
    expect(screen.getByTestId("selection-bounds")).toHaveTextContent("rows [16, 32) · columns [16, 32)");
    await userEvent.click(screen.getByTestId("inspect-exact"));
    expect(screen.getByRole("tab", { name: "Exact ROI" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByTestId("roi-from-overview")).toHaveTextContent("rows [16, 32) × columns [16, 32)");
    await waitFor(() => expect(calls.some((c) => c.url.pathname === "/api/v1/tensors/roi")).toBe(true));
    const roi = calls.find((c) => c.url.pathname === "/api/v1/tensors/roi")!.body as RoiRequest;
    expect(roi.roi).toEqual({ r0: 16, r1: 32, c0: 16, c1: 32 });
    expect(roi.view).toEqual({ row_axis: 0, col_axis: 1, fixed_indices: {}, transpose: false });
    expect(await screen.findByTestId("roi-bounds")).toHaveTextContent("Rows [16, 32) of 1,024");
  });

  it("keeps the transposed view when drilling down", async () => {
    const calls = server({});
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    await screen.findByTestId("coverage-badge");
    await userEvent.click(within(screen.getByTestId("overview")).getByLabelText("Transpose"));
    await screen.findByTestId("coverage-badge");
    const canvas = screen.getByTestId("overview-canvas");
    expect(canvas.getAttribute("aria-label")).toContain("2,304 × 1,024 view");
    fireEvent.keyDown(canvas, { key: "ArrowDown" });
    await userEvent.click(screen.getByTestId("inspect-exact"));
    await waitFor(() => expect(calls.some((c) => c.url.pathname === "/api/v1/tensors/roi")).toBe(true));
    const roi = calls.find((c) => c.url.pathname === "/api/v1/tensors/roi")!.body as RoiRequest;
    expect(roi.view.transpose).toBe(true);
    expect(roi.roi).toEqual({ r0: 0, r1: 8, c0: 0, c1: 8 }); // the taller transposed view fits at L3
  });

  it("does not silently truncate a region above roi_max_cells", async () => {
    const calls = server({ settings: { ...SETTINGS, roi_max_cells: 100 } });
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    await screen.findByTestId("coverage-badge");
    fireEvent.keyDown(screen.getByTestId("overview-canvas"), { key: "ArrowDown" });
    expect(screen.getByTestId("roi-too-big")).toHaveTextContent("This region has 256 elements; Exact ROI reads at most 100");
    expect(screen.getByTestId("inspect-exact")).toBeDisabled();
    await userEvent.click(screen.getByTestId("shrink-selection"));
    expect(screen.getByTestId("selection-bounds")).toHaveTextContent("rows [0, 8) · columns [0, 8)");
    expect(screen.getByTestId("inspect-exact")).toBeEnabled();
    expect(calls.some((c) => c.url.pathname === "/api/v1/tensors/roi")).toBe(false);
  });

  it("shows a corrupted tile as an error, never as a picture", async () => {
    server({ tile: (b) => {
      const r = tileResponse(b, { rows: 1024, cols: 2304 });
      return new Response(new ArrayBuffer(10), { status: 200, headers: r.headers });
    } });
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("BAD_RESPONSE");
    expect(screen.queryByTestId("coverage-badge")).toBeNull();
  });

  it("shows the server's error for a changed file and draws nothing", async () => {
    server({ tile: () => json({ error: { code: "FILE_CHANGED", message: "model.safetensors changed", retryable: false }, request_id: "req_1" }, 409) });
    render(<TensorWorkbench modelId="mod_1" tensor={tensor()} />);
    expect(await screen.findByText("Model files changed on disk")).toBeInTheDocument();
    expect(screen.queryByTestId("coverage-badge")).toBeNull();
  });

  it("keeps rank-0/1 and empty tensors on the exact representation only", async () => {
    server({});
    render(<TensorWorkbench modelId="mod_1" tensor={tensor("tns_v", [64])} />);
    expect(screen.queryByRole("tab", { name: "Overview" })).toBeNull();
    render(<TensorWorkbench modelId="mod_1" tensor={tensor("tns_e", [0, 4])} />);
    expect(await screen.findByTestId("roi-empty")).toBeInTheDocument();
  });
});

describe("job status polling backoff", () => {
  it("follows poll_after_ms, backs off after failures and is bounded", async () => {
    const { pollBackoff, MAX_POLL_FAILURES } = await import("./Overview");
    const j = job("running", { poll_after_ms: 500 });
    expect([0, 1, 2, 3, 4, 5].map((f) => pollBackoff(j, f))).toEqual([500, 1000, 2000, 4000, 8000, 8000]);
    expect(pollBackoff(job("running", { poll_after_ms: 50_000 }), 0)).toBe(10_000);
    expect(MAX_POLL_FAILURES).toBe(5);
  });
});
