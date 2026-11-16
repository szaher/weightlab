// The delta heatmap of the Compare drill-down (ROADMAP P1.1) against a fake server: what it requests (a `delta`
// source for tiles and exact values, both checkpoints' sizes in the cost), how it starts (zero-centred diverging
// scale for signed B − A) and what it refuses (invalid ε, a derived answer claiming to be lossless, a missing entry).
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { TensorRow } from "../../lib/api/compare";
import type { RoiRequest, Tensor, TileRequest } from "../../lib/api/types";
import { apiError, fakeApi, json, withToken, type Call } from "../../test/fakeApi";
import { tileResponse } from "../../test/tiles";
import { sharedTileCache } from "../overview/tileStore";
import { DeltaWorkbench, epsilonError } from "./DeltaWorkbench";

const KEY = "model.layers.0.mlp.down_proj.weight";
const ROW = { key: KEY, status: "matched", dtype_a: "F32", dtype_b: "BF16", shape: [64, 96] } as unknown as TensorRow;
const A = { id: "mod_a", name: "base" };
const B = { id: "mod_b", name: "ft" };
const SETTINGS = { roi_max_cells: 4096, pyramid_exact_max_mib: 64, max_active_tiles: 2, direct_tile_max_mib: 8 };

function tensor(model: string, shape = [64, 96], dtype = model === "mod_a" ? "F32" : "BF16"): Tensor {
  const numel = shape.reduce((a, b) => a * b, 1);
  const size = dtype === "F32" ? 4 : 2;
  return { id: `tns_${model}`, model_id: model, raw_key: KEY, shape, dtype, rank: shape.length, numel, byte_length: numel * size,
           shard_relpath: "m.safetensors", file_offset_begin: 0, file_offset_end: numel * size, role: null, role_is_heuristic: true,
           layer_index: 0 } as Tensor; // prettier-ignore
}

interface Opts {
  shape?: number[];
  entries?: (offset: number) => { total: number; items: { entry_id: string; raw_key: string; status: string; reasons: string[] }[] };
  roi?: (b: RoiRequest) => Response;
}

function server(o: Opts = {}): Call[] {
  sharedTileCache.purge(() => true);
  withToken();
  const shape = o.shape ?? [64, 96];
  return fakeApi((c) => {
    const p = c.url.pathname;
    if (p === "/api/v1/settings") return json(SETTINGS);
    if (p === "/api/v1/comparisons/cmp_1/entries") {
      const off = Number(c.url.searchParams.get("offset"));
      return json(o.entries ? o.entries(off) : { total: 1, items: [{ entry_id: "ent_7", raw_key: KEY, status: "matched", reasons: [] }] });
    }
    const m = /^\/api\/v1\/models\/(mod_[ab])\/tensors$/.exec(p);
    if (m) return json({ total: 1, items: [tensor(m[1]!, shape)] });
    if (p === "/api/v1/tiles") return tileResponse(c.body as TileRequest, { rows: shape[0]!, cols: shape[1] ?? 1 }, {}, (i, j) => i - j);
    if (p.startsWith("/api/v1/tiles/")) return apiError(404, "NOT_FOUND", "no metadata in this test");
    if (p === "/api/v1/tensors/roi") {
      const b = c.body as RoiRequest;
      if (o.roi) return o.roi(b);
      const { r0, r1, c0, c1 } = b.roi;
      const values = Array.from({ length: r1 - r0 }, (_, i) => Array.from({ length: c1 - c0 }, (_, j) => (i === 0 && j === 0 ? null : (r0 + i) - (c0 + j) + 0.5)));
      const mask = values.map((row) => row.map((v) => (v === null ? "overflow" : null)));
      return json({ schema_version: 2, source: b.source, roi: b.roi, shape: [r1 - r0, c1 - c0], source_dtype: "F32", value_encoding: "float_json", coverage: "complete",
                    numeric_precision: "overflowed", values, nonfinite_mask: mask, warnings: ["dtypes differ: A F32, B BF16; both promoted to float64"] }); // prettier-ignore
    }
    throw new Error(`unexpected ${c.method} ${p}`);
  });
}

const tiles = (calls: Call[]) => calls.filter((c) => c.url.pathname === "/api/v1/tiles").map((c) => c.body as TileRequest);
const view = () => render(<DeltaWorkbench comparisonId="cmp_1" row={ROW} modelA={A} modelB={B} />);

describe("DeltaWorkbench", () => {
  it("reads nothing until opened, then draws B − A from a delta source on a zero-centred diverging scale", async () => {
    const user = userEvent.setup();
    const calls = server();
    view();
    expect(screen.getByTestId("delta-toggle")).toHaveAttribute("aria-expanded", "false");
    expect(calls).toHaveLength(0);
    await user.click(screen.getByTestId("delta-toggle"));
    await screen.findByTestId("coverage-badge");
    const entries = calls.find((c) => c.url.pathname.endsWith("/entries"))!;
    expect(entries.url.searchParams.getAll("status")).toEqual(["matched", "dtype_mismatch"]);
    expect(tiles(calls)[0]!.source).toEqual({ kind: "delta", comparison_id: "cmp_1", entry_id: "ent_7", transform: "b_minus_a" });
    expect(screen.getByRole("heading", { name: "Delta heatmap, B − A" })).toBeInTheDocument();
    expect(screen.getByTestId("overview-badges")).toHaveTextContent("B − A of F32 → BF16 inputs, float64");
    expect(screen.getByLabelText("Scale")).toHaveValue("symmetric");
    expect(screen.getByLabelText("Palette")).toHaveValue("diverging");
    expect(screen.getByTestId("delta-explainer")).toHaveTextContent("Colour is centred on zero");
    expect(screen.getByTestId("delta-explainer")).toHaveTextContent("One full read is 36.0 KiB (both checkpoints)"); // 64 × 96 × (4 + 2) bytes
  });

  it("pages the comparison's entries until it finds the tensor's entry", async () => {
    const user = userEvent.setup();
    const other = (n: number) => ({ entry_id: `ent_${n}`, raw_key: `k.${n}`, status: "matched", reasons: [] });
    const calls = server({
      entries: (off) =>
        off === 0
          ? { total: 1500, items: Array.from({ length: 1000 }, (_, i) => other(i)) }
          : { total: 1500, items: [...Array.from({ length: 499 }, (_, i) => other(1000 + i)), { entry_id: "ent_last", raw_key: KEY, status: "matched", reasons: [] }] },
    });
    view();
    await user.click(screen.getByTestId("delta-toggle"));
    await screen.findByTestId("coverage-badge");
    expect(calls.filter((c) => c.url.pathname.endsWith("/entries")).map((c) => c.url.searchParams.get("offset"))).toEqual(["0", "1000"]);
    expect(tiles(calls).at(-1)!.source).toMatchObject({ entry_id: "ent_last" });
  });

  it("switches transforms, starting magnitudes on a sequential scale, and applies only a valid ε", async () => {
    const user = userEvent.setup();
    const calls = server();
    view();
    await user.click(screen.getByTestId("delta-toggle"));
    await screen.findByTestId("coverage-badge");
    await user.selectOptions(screen.getByTestId("delta-transform"), "abs_b_minus_a");
    await waitFor(() => expect(tiles(calls).at(-1)!.source).toMatchObject({ transform: "abs_b_minus_a" }));
    expect(screen.getByLabelText("Palette")).toHaveValue("sequential");
    expect(screen.getByLabelText("Scale")).toHaveValue("auto");

    await user.selectOptions(screen.getByTestId("delta-transform"), "relative_abs");
    await waitFor(() => expect(tiles(calls).at(-1)!.source).toEqual({ kind: "delta", comparison_id: "cmp_1", entry_id: "ent_7", transform: "relative_abs", epsilon: 1e-8 }));
    const eps = screen.getByTestId("delta-epsilon");
    await user.clear(eps);
    await user.type(eps, "0");
    expect(screen.getByText("ε must be greater than 0 and at most 1")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply ε" })).toBeDisabled();
    await user.clear(eps);
    await user.type(eps, "0.5");
    await user.click(screen.getByRole("button", { name: "Apply ε" }));
    await waitFor(() => expect(tiles(calls).at(-1)!.source).toMatchObject({ transform: "relative_abs", epsilon: 0.5 }));
    expect(screen.getByRole("heading", { name: "Delta heatmap, |B − A| / max(|A|, 0.5)" })).toBeInTheDocument();
  });

  it("Exact ROI reads paired delta values and shows overflow as withheld, never as a number", async () => {
    const user = userEvent.setup();
    const calls = server();
    view();
    await user.click(screen.getByTestId("delta-toggle"));
    await user.click(await screen.findByRole("tab", { name: "Exact ROI" }));
    const prec = await screen.findByTestId("roi-precision");
    expect(prec).toHaveTextContent("numeric precision: overflowed");
    expect(prec).toHaveTextContent("B − A · float64 (F32 → BF16 inputs)");
    const roi = calls.find((c) => c.url.pathname === "/api/v1/tensors/roi")!.body as RoiRequest;
    expect(roi.source).toEqual({ kind: "delta", comparison_id: "cmp_1", entry_id: "ent_7", transform: "b_minus_a" });
    expect(screen.getByText(/dtypes differ: A F32, B BF16/)).toBeInTheDocument();
    expect(screen.getByLabelText("Palette")).toHaveValue("diverging");
  });

  it("refuses a derived answer that claims to be lossless", async () => {
    const user = userEvent.setup();
    server({
      roi: (b) =>
        json({ schema_version: 2, source: b.source, roi: b.roi, shape: [b.roi.r1 - b.roi.r0, b.roi.c1 - b.roi.c0], source_dtype: "F32", value_encoding: "float_json",
               coverage: "complete", numeric_precision: "lossless", values: Array.from({ length: b.roi.r1 - b.roi.r0 }, () => Array.from({ length: b.roi.c1 - b.roi.c0 }, () => 1)),
               nonfinite_mask: null, warnings: [] }), // prettier-ignore
    });
    view();
    await user.click(screen.getByTestId("delta-toggle"));
    await user.click(await screen.findByRole("tab", { name: "Exact ROI" }));
    expect(await screen.findByText(/never lossless/)).toBeInTheDocument();
    expect(screen.queryByTestId("roi-precision")).not.toBeInTheDocument();
  });

  it("explains a key without an entry, and offers only exact values for a vector", async () => {
    const user = userEvent.setup();
    server({ entries: () => ({ total: 0, items: [] }) });
    const { unmount } = view();
    await user.click(screen.getByTestId("delta-toggle"));
    expect(await screen.findByText(/has no matched entry with an elementwise B − A/)).toBeInTheDocument();
    unmount();

    const calls = server({ shape: [96] });
    view();
    await user.click(screen.getByTestId("delta-toggle"));
    await screen.findByTestId("roi-precision");
    expect(screen.queryByRole("tablist", { name: "Delta views" })).not.toBeInTheDocument(); // no 2D heatmap of a vector
    expect(tiles(calls)).toHaveLength(0);
  });
});

describe("epsilonError", () => {
  it("accepts 0 < ε <= 1 only", () => {
    expect(epsilonError("1e-8")).toBeNull();
    expect(epsilonError("1")).toBeNull();
    for (const bad of ["", "0", "-1", "1.5", "abc", "Infinity"]) expect(epsilonError(bad)).not.toBeNull();
  });
});

describe("rendered inside the drill-down", () => {
  it("names the checkpoints it subtracts", () => {
    server();
    view();
    expect(within(screen.getByTestId("delta-heatmap")).getByText(/B = ft, A = base/)).toBeInTheDocument();
  });
});
