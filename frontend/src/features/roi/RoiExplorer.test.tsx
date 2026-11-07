// Interaction tests of the exact ROI explorer against a fake server. The fake's values come from an oracle
// (value = C-order flat index / 4 - 1, mapped through the view here, independently of view.ts), so a wrong
// orientation, slice or window shows up as a wrong number.
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { RoiRequest, Tensor } from "../../lib/api/types";
import { apiError, fakeApi, json, withToken, type Call } from "../../test/fakeApi";
import { RoiExplorer, ROI_QUIET_MS } from "./RoiExplorer";

function tensor(id: string, shape: number[], dtype: Tensor["dtype"] = "F32"): Tensor {
  const numel = shape.reduce((a, b) => a * b, 1);
  return {
    id,
    model_id: "mod_1",
    raw_key: `t.${id}`,
    shape,
    dtype,
    rank: shape.length,
    numel,
    byte_length: numel * 4,
    shard_relpath: "m.safetensors",
    file_offset_begin: 0,
    file_offset_end: numel * 4,
    role: null,
    role_is_heuristic: true,
    layer_index: null,
  };
}

/** Value of tensor element `idx` (C order). */
function oracle(shape: number[], idx: number[]): number {
  let flat = 0;
  for (let a = 0; a < shape.length; a++) flat = flat * shape[a]! + idx[a]!;
  return flat / 4 - 1;
}

/** Tensor index of view cell (r, c): rows/cols axes swap under transpose; other axes are fixed. */
function indexOf(shape: number[], v: RoiRequest["view"], r: number, c: number): number[] {
  if (shape.length === 0) return [];
  if (shape.length === 1) return [c];
  const rowAxis = v.transpose ? v.col_axis! : v.row_axis!;
  const colAxis = v.transpose ? v.row_axis! : v.col_axis!;
  return shape.map((_, a) => (a === rowAxis ? r : a === colAxis ? c : v.fixed_indices[String(a)]!));
}

function answer(t: Tensor, body: RoiRequest) {
  const { r0, r1, c0, c1 } = body.roi;
  const values: number[][] = [];
  for (let r = r0; r < r1; r++) {
    const row: number[] = [];
    for (let c = c0; c < c1; c++) row.push(oracle(t.shape, indexOf(t.shape, body.view, r, c)));
    values.push(row);
  }
  return {
    schema_version: 2,
    source: body.source,
    roi: body.roi,
    shape: [r1 - r0, c1 - c0],
    source_dtype: t.dtype,
    value_encoding: "float_json",
    coverage: "complete",
    numeric_precision: "lossless",
    values,
    nonfinite_mask: null,
    warnings: [],
  };
}

type RoiHandler = (body: RoiRequest, call: Call) => Response | Promise<Response>;

function serve(tensors: Tensor[], roi?: RoiHandler): Call[] {
  return fakeApi((call) => {
    if (call.url.pathname === "/api/v1/settings") return json({ roi_max_cells: 4096 });
    if (call.url.pathname === "/api/v1/tensors/roi") {
      const body = call.body as RoiRequest;
      if (roi) return roi(body, call);
      const t = tensors.find((x) => x.id === body.source.tensor_id)!;
      return json(answer(t, body));
    }
    return apiError(404, "NOT_FOUND", "unexpected route in test");
  });
}

const rois = (calls: Call[]) => calls.filter((c) => c.url.pathname === "/api/v1/tensors/roi");
const last = (calls: Call[]) => rois(calls).at(-1)!.body as RoiRequest;

async function clickCell(i: number, j: number) {
  const canvas = await screen.findByTestId("heatmap-canvas");
  const cell = Number(canvas.dataset.cellPx);
  const left = Number(canvas.dataset.left);
  const top = Number(canvas.dataset.top);
  fireEvent.pointerDown(canvas, { clientX: left + (j + 0.5) * cell, clientY: top + (i + 0.5) * cell });
}

const valueText = () => screen.getByTestId("selected-value").textContent;
const num = (shape: number[], idx: number[]) => String(oracle(shape, idx)).replace(/^(-?\d+)$/, "$1.0");

describe("RoiExplorer", () => {
  it("selecting a tensor loads the 64 x 64 origin window with the contract's request body", async () => {
    withToken();
    const t = tensor("tns_q", [257, 263]);
    const calls = serve([t]);
    render(<RoiExplorer modelId="mod_1" tensor={t} />);
    await screen.findByTestId("heatmap-canvas");
    expect(rois(calls)).toHaveLength(1);
    expect(rois(calls)[0]!.body).toEqual({
      source: { kind: "tensor", model_id: "mod_1", tensor_id: "tns_q" },
      view: { row_axis: 0, col_axis: 1, fixed_indices: {}, transpose: false },
      roi: { r0: 0, r1: 64, c0: 0, c1: 64 },
      value_encoding: "auto",
    });
    expect(rois(calls)[0]!.headers.Authorization).toMatch(/^Bearer /);
    expect(screen.getByTestId("roi-bounds")).toHaveTextContent("Rows [0, 64) of 257");
    expect(screen.getByTestId("roi-bounds")).toHaveTextContent("Columns [0, 64) of 263");
    expect(screen.getByTestId("roi-coverage-note")).toHaveTextContent("this is a window, not the whole tensor");
    expect(screen.getByTestId("roi-precision")).toHaveTextContent("coverage: complete");
    expect(screen.getByTestId("roi-precision")).toHaveTextContent("numeric precision: lossless");
  });

  it("clicking a cell shows its exact value and coordinates; pan and zoom keep coordinates meaningful", async () => {
    withToken();
    const t = tensor("tns_q", [257, 263]);
    const calls = serve([t]);
    render(<RoiExplorer modelId="mod_1" tensor={t} />);
    await clickCell(5, 7);
    expect(screen.getByTestId("selected-coords")).toHaveTextContent("5, 7");
    expect(valueText()).toBe(num(t.shape, [5, 7]));

    await userEvent.click(screen.getByRole("button", { name: "Pan right by 32 columns" }));
    await waitFor(() => expect(last(calls).roi).toEqual({ r0: 0, r1: 64, c0: 32, c1: 96 }));
    await screen.findByTestId("heatmap-canvas");
    expect(screen.getByTestId("selected-coords")).toHaveTextContent("5, 32"); // clamped into the new window
    await clickCell(1, 2); // view (1, 34)
    expect(valueText()).toBe(num(t.shape, [1, 34]));

    await userEvent.click(screen.getByRole("button", { name: "Zoom in to 32 × 32 cells" }));
    await waitFor(() => expect(last(calls).roi.r1 - last(calls).roi.r0).toBe(32));
    await screen.findByTestId("heatmap-canvas");
    const b = last(calls).roi;
    expect(1 >= b.r0 && 1 < b.r1 && 34 >= b.c0 && 34 < b.c1).toBe(true);
    expect(screen.getByTestId("selected-coords")).toHaveTextContent("1, 34");
    expect(valueText()).toBe(num(t.shape, [1, 34]));
  });

  it("transpose swaps the request's displayed axes and keeps the same element selected", async () => {
    withToken();
    const t = tensor("tns_q", [257, 263]);
    const calls = serve([t]);
    render(<RoiExplorer modelId="mod_1" tensor={t} />);
    await clickCell(3, 9);
    await userEvent.click(screen.getByLabelText("Transpose"));
    await waitFor(() => expect(last(calls).view.transpose).toBe(true));
    await screen.findByTestId("heatmap-canvas");
    expect(screen.getByTestId("selected-coords")).toHaveTextContent("9, 3");
    expect(screen.getByTestId("selected-index")).toHaveTextContent("[3, 9]");
    expect(valueText()).toBe(num(t.shape, [3, 9]));
    expect(screen.getByTestId("roi-bounds")).toHaveTextContent("of 263");
  });

  it("rank-3: changing a fixed index or an axis selects the right slice", async () => {
    withToken();
    const t = tensor("tns_c", [2, 4, 8]);
    const calls = serve([t]);
    render(<RoiExplorer modelId="mod_1" tensor={t} />);
    await screen.findByTestId("heatmap-canvas");
    expect(last(calls).view).toEqual({ row_axis: 1, col_axis: 2, fixed_indices: { "0": 0 }, transpose: false });
    fireEvent.change(screen.getByTestId("fixed-0"), { target: { value: "1" } });
    await waitFor(() => expect(last(calls).view.fixed_indices).toEqual({ "0": 1 }));
    await clickCell(1, 2);
    expect(screen.getByTestId("selected-index")).toHaveTextContent("[1, 1, 2]");
    expect(valueText()).toBe(num(t.shape, [1, 1, 2]));
    const before = rois(calls).length;
    fireEvent.change(screen.getByTestId("fixed-0"), { target: { value: "7" } }); // out of range
    expect(await screen.findByTestId("roi-view-error")).toHaveTextContent("outside 0–1");
    expect(screen.queryByTestId("heatmap-canvas")).toBeNull();
    await new Promise((r) => setTimeout(r, ROI_QUIET_MS * 2));
    expect(rois(calls)).toHaveLength(before); // nothing sent for an invalid view
    fireEvent.change(screen.getByTestId("fixed-0"), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("Row axis"), { target: { value: "0" } });
    await waitFor(() => expect(last(calls).view).toEqual({ row_axis: 0, col_axis: 2, fixed_indices: { "1": 0 }, transpose: false }));
    await clickCell(1, 5);
    expect(screen.getByTestId("selected-index")).toHaveTextContent("[1, 0, 5]");
    expect(valueText()).toBe(num(t.shape, [1, 0, 5]));
  });

  it("switching tensors aborts the old request and a late old answer is never shown", async () => {
    withToken();
    const a = tensor("tns_a", [8, 8]);
    const b = tensor("tns_b", [8, 8]);
    let releaseA: (() => void) | undefined;
    const calls = serve([a, b], (body) => {
      if (body.source.tensor_id === "tns_a") {
        return new Promise<Response>((ok) => {
          releaseA = () => ok(json(answer(a, body)));
        });
      }
      // b's values are distinguishable: every value + 1000
      const r = answer(b, body);
      r.values = r.values.map((row) => row.map((v) => v + 1000));
      return json(r);
    });
    const { rerender } = render(<RoiExplorer key="a" modelId="mod_1" tensor={a} />);
    await waitFor(() => expect(rois(calls)).toHaveLength(1));
    rerender(<RoiExplorer key="b" modelId="mod_1" tensor={b} />);
    await screen.findByTestId("heatmap-canvas");
    expect(rois(calls)[0]!.init.signal!.aborted).toBe(true);
    await act(async () => releaseA?.());
    await clickCell(0, 0);
    expect(valueText()).toBe(String(oracle(b.shape, [0, 0]) + 1000) + ".0");
  });

  it("rapid panning aborts in-flight reads and sends one request for where the user stops", async () => {
    withToken();
    const t = tensor("tns_q", [257, 263]);
    const pending: Call[] = [];
    const calls = serve([t], (body, call) => {
      if (body.roi.c0 === 0) return json(answer(t, body));
      pending.push(call);
      return new Promise<Response>(() => {}); // never answers unless the test says so
    });
    render(<RoiExplorer modelId="mod_1" tensor={t} />);
    await screen.findByTestId("heatmap-canvas");
    const right = screen.getByRole("button", { name: "Pan right by 32 columns" });
    for (let k = 0; k < 5; k++) fireEvent.click(right);
    await waitFor(() => expect(rois(calls)).toHaveLength(2));
    expect(last(calls).roi).toEqual({ r0: 0, r1: 64, c0: 160, c1: 224 });
    // While that read is pending, the old window's values are not shown under the new bounds.
    expect(screen.getByTestId("roi-bounds")).toHaveTextContent("Columns [160, 224)");
    expect(screen.queryByTestId("heatmap-canvas")).toBeNull();
    expect(screen.getByTestId("roi-loading")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reset to origin" }));
    await screen.findByTestId("heatmap-canvas");
    expect(pending[0]!.init.signal!.aborted).toBe(true);
  });

  it("a response for another region or a malformed one is an error, never drawn", async () => {
    withToken();
    const t = tensor("tns_q", [16, 16]);
    serve([t], (body) => json({ ...answer(t, body), roi: { r0: 1, r1: 17, c0: 0, c1: 16 } }));
    render(<RoiExplorer modelId="mod_1" tensor={t} />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Unexpected server response");
    expect(alert).toHaveTextContent("failed validation");
    expect(screen.queryByTestId("heatmap-canvas")).toBeNull();
  });

  it("FILE_CHANGED shows re-import/relink guidance and no values; returning to a window reads again", async () => {
    withToken();
    const t = tensor("tns_q", [257, 263]);
    let changed = false;
    const calls = serve([t], (body) =>
      changed ? apiError(409, "FILE_CHANGED", "tiny.safetensors changed since import (size, time or identity)") : json(answer(t, body)),
    );
    render(<RoiExplorer modelId="mod_1" tensor={t} />);
    await clickCell(0, 0);
    changed = true;
    await userEvent.click(screen.getByRole("button", { name: "Pan right by 32 columns" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Model files changed on disk");
    expect(alert).toHaveTextContent("Re-import the model if it was modified; relink it in the Workspace if it was only moved.");
    expect(alert).toHaveTextContent("FILE_CHANGED");
    expect(screen.queryByTestId("heatmap-canvas")).toBeNull();
    expect(screen.queryByTestId("selected-value")).toBeNull();
    const n = rois(calls).length;
    await userEvent.click(screen.getByRole("button", { name: "Pan left by 32 columns" })); // back to the first window
    await waitFor(() => expect(rois(calls).length).toBe(n + 1)); // no cached reuse
    expect(await screen.findByRole("alert")).toHaveTextContent("FILE_CHANGED");
    expect(screen.queryByTestId("heatmap-canvas")).toBeNull();
  });

  it("RESOURCE_LIMIT and INVALID_ROI from the server are explained", async () => {
    withToken();
    const t = tensor("tns_q", [16, 16]);
    serve([t], () => apiError(413, "RESOURCE_LIMIT", "ROI has more than 100 cells"));
    const { unmount } = render(<RoiExplorer modelId="mod_1" tensor={t} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Zoom in to a smaller window");
    unmount();
    serve([t], () => apiError(422, "INVALID_ROI", "ROI outside the 16x16 view"));
    render(<RoiExplorer modelId="mod_1" tensor={t} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Region or view not valid for this tensor");
  });

  it("is operable by keyboard alone: arrows select and move, tabs switch with arrow keys", async () => {
    withToken();
    const t = tensor("tns_q", [257, 263]);
    const calls = serve([t]);
    const user = userEvent.setup();
    render(<RoiExplorer modelId="mod_1" tensor={t} />);
    const canvas = await screen.findByTestId("heatmap-canvas");
    canvas.focus();
    await user.keyboard("{ArrowRight}"); // first key selects the window's first cell
    expect(screen.getByTestId("selected-coords")).toHaveTextContent("0, 0");
    await user.keyboard("{ArrowRight}{ArrowRight}{ArrowDown}");
    expect(screen.getByTestId("selected-coords")).toHaveTextContent("1, 2");
    expect(valueText()).toBe(num(t.shape, [1, 2]));
    await user.keyboard("{PageDown}"); // 64 rows down: the window follows
    await waitFor(() => expect(last(calls).roi.r1).toBe(66));
    await screen.findByTestId("heatmap-canvas");
    expect(screen.getByTestId("selected-coords")).toHaveTextContent("65, 2");
    expect(valueText()).toBe(num(t.shape, [65, 2]));
    const tab = screen.getByRole("tab", { name: "Heatmap" });
    tab.focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Values" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Values" })).toHaveFocus();
    expect(within(screen.getByTestId("value-65-2")).getByRole("button", { name: /^Select/ })).toHaveTextContent(num(t.shape, [65, 2]));
  });

  it("rank 0 shows the scalar; rank 1 is one row; zero-size tensors send no request", async () => {
    withToken();
    const s = tensor("tns_s", []);
    const v = tensor("tns_v", [6]);
    const e = tensor("tns_e", [0, 4]);
    const calls = serve([s, v, e]);
    const { unmount } = render(<RoiExplorer modelId="mod_1" tensor={s} />);
    expect(await screen.findByTestId("selected-value")).toHaveTextContent("-1.0");
    expect(last(calls).view).toEqual({ row_axis: null, col_axis: null, fixed_indices: {}, transpose: false });
    expect(screen.queryByTestId("heatmap-canvas")).toBeNull();
    unmount();
    const { unmount: u2 } = render(<RoiExplorer modelId="mod_1" tensor={v} />);
    await screen.findByTestId("heatmap-canvas");
    expect(last(calls).roi).toEqual({ r0: 0, r1: 1, c0: 0, c1: 6 });
    expect(screen.getByTestId("roi-bounds")).toHaveTextContent("Indices [0, 6) of 6");
    expect(screen.queryByLabelText("Transpose")).toBeNull();
    await clickCell(0, 4);
    expect(valueText()).toBe(num(v.shape, [4]));
    u2();
    const n = rois(calls).length;
    render(<RoiExplorer modelId="mod_1" tensor={e} />);
    expect(await screen.findByTestId("roi-empty")).toHaveTextContent("no elements");
    await new Promise((r) => setTimeout(r, ROI_QUIET_MS * 2));
    expect(rois(calls)).toHaveLength(n);
  });

  it("I64 values stay exact decimal strings in the table and the copy action", async () => {
    withToken();
    const t = tensor("tns_i", [3], "I64");
    const big = ["-9223372036854775808", "9007199254740993", "9223372036854775807"];
    serve([t], (body) => json({ ...answer(t, body), value_encoding: "int_decimal_string", values: [big] }));
    const writes: string[] = [];
    Object.defineProperty(navigator, "clipboard", { value: { writeText: async (s: string) => void writes.push(s) }, configurable: true });
    render(<RoiExplorer modelId="mod_1" tensor={t} />);
    await screen.findByTestId("heatmap-canvas");
    await userEvent.click(screen.getByRole("tab", { name: "Values" }));
    big.forEach((s, k) => expect(within(screen.getByTestId(`value-0-${k}`)).getByRole("button", { name: /^Select/ })).toHaveTextContent(s));
    await userEvent.click(within(screen.getByTestId("value-0-1")).getByRole("button", { name: /^Copy/ }));
    expect(writes).toEqual(["9007199254740993"]);
    expect(screen.getByTestId("value-0-1")).toHaveTextContent("colour withheld");
  });
});
