import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Tensor } from "../../lib/api/types";
import { apiError, fakeApi, json, withToken, type Call } from "../../test/fakeApi";
import { SEARCH_DEBOUNCE_MS, TensorCatalog } from "./TensorCatalog";

const LONG = `model.layers.${"9".repeat(3)}.${"very_long_segment_".repeat(40)}weight`;
const XSS = `<img src=x onerror="window.__pwned=1">.weight`;

function tensor(i: number, key = `layers.${i}.mlp.up_proj.weight`): Tensor {
  return {
    id: `tns_${i}`,
    model_id: "mod_1",
    raw_key: key,
    shape: [8, 8],
    dtype: "F32",
    rank: 2,
    numel: 64,
    byte_length: 256,
    shard_relpath: "tiny.safetensors",
    file_offset_begin: 100,
    file_offset_end: 356,
    role: "mlp.up",
    role_is_heuristic: true,
    layer_index: i,
  };
}

function serve(total: number, items: (call: Call) => Tensor[]) {
  return fakeApi((c) => json({ total, items: items(c) }));
}

describe("TensorCatalog", () => {
  it("renders a bounded page from the server and the global position", async () => {
    withToken();
    const calls = serve(1234, (c) => Array.from({ length: Number(c.url.searchParams.get("limit")) }, (_, i) => tensor(i)));
    render(<TensorCatalog modelId="mod_1" selectedId={null} onSelect={() => {}} />);
    expect(await screen.findByText("Rows 1–50 of 1,234 tensors, sorted by name (ascending) across the whole catalog.")).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(51);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url.searchParams.get("limit")).toBe("50");
    expect(calls[0]!.url.searchParams.get("offset")).toBe("0");
  });

  it("renders hostile and very long names as plain text, in full", async () => {
    withToken();
    serve(2, () => [tensor(0, XSS), tensor(1, LONG)]);
    const onSelect = vi.fn();
    render(<TensorCatalog modelId="mod_1" selectedId={null} onSelect={onSelect} />);
    const xss = await screen.findByRole("button", { name: XSS });
    expect(xss.textContent).toBe(XSS);
    expect(document.querySelector("img")).toBeNull();
    expect((window as unknown as { __pwned?: number }).__pwned).toBeUndefined();
    const long = screen.getByRole("button", { name: LONG });
    expect(long.textContent).toBe(LONG); // not truncated in the DOM
    await userEvent.click(long);
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ raw_key: LONG }));
  });

  it("debounces search and resets to the first page", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      withToken();
      const calls = serve(500, () => [tensor(0)]);
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(<TensorCatalog modelId="mod_1" selectedId={null} onSelect={() => {}} />);
      await screen.findByText(/Rows 1–1 of 500/);
      await user.click(screen.getByRole("button", { name: "Next" }));
      await waitFor(() => expect(calls.at(-1)!.url.searchParams.get("offset")).toBe("50"));
      const before = calls.length;
      await user.type(screen.getByLabelText("Search tensor names"), "up_proj");
      await act(async () => {
        vi.advanceTimersByTime(SEARCH_DEBOUNCE_MS + 50);
      });
      await waitFor(() => expect(calls.at(-1)!.url.searchParams.get("q")).toBe("up_proj"));
      const searches = calls.slice(before).filter((c) => c.url.searchParams.has("q"));
      expect(searches).toHaveLength(1); // one request for the whole word, not one per keystroke
      expect(calls.at(-1)!.url.searchParams.get("offset")).toBe("0");
    } finally {
      vi.useRealTimers();
    }
  });

  it("sorts on the server and marks the sorted column", async () => {
    withToken();
    const calls = serve(10, () => [tensor(0)]);
    render(<TensorCatalog modelId="mod_1" selectedId={null} onSelect={() => {}} />);
    await screen.findByText(/Rows 1–1 of 10/);
    const params = within(screen.getByRole("table")).getByRole("button", { name: /Params/ });
    await userEvent.click(params);
    await waitFor(() => expect(calls.at(-1)!.url.searchParams.get("sort")).toBe("numel"));
    expect(calls.at(-1)!.url.searchParams.get("order")).toBe("asc");
    expect(params.closest("th")).toHaveAttribute("aria-sort", "ascending");
    await userEvent.click(params);
    await waitFor(() => expect(calls.at(-1)!.url.searchParams.get("order")).toBe("desc"));
    expect(params.closest("th")).toHaveAttribute("aria-sort", "descending");
    expect(screen.getByRole("columnheader", { name: /Shape/ })).not.toHaveAttribute("aria-sort");
  });

  it("filters by dtype, role, layer, rank and minimum size", async () => {
    withToken();
    const calls = serve(3, () => [tensor(0)]);
    render(<TensorCatalog modelId="mod_1" selectedId={null} onSelect={() => {}} />);
    await screen.findByText(/Rows 1–1 of 3/);
    await userEvent.selectOptions(screen.getByLabelText("Dtype"), "BF16");
    await userEvent.selectOptions(screen.getByLabelText("Role (heuristic)"), "attention.q");
    await userEvent.type(screen.getByLabelText("Layer"), "2");
    await userEvent.type(screen.getByLabelText("Rank"), "2");
    await userEvent.type(screen.getByLabelText("Min. params"), "64");
    await waitFor(() => expect(calls.at(-1)!.url.searchParams.get("min_numel")).toBe("64"));
    const p = calls.at(-1)!.url.searchParams;
    expect([p.get("dtype"), p.get("role"), p.get("layer"), p.get("rank")]).toEqual(["BF16", "attention.q", "2", "2"]);
    await userEvent.type(screen.getByLabelText("Rank"), "x");
    expect(screen.getByLabelText("Rank")).toHaveAttribute("aria-invalid", "true");
  });

  it("pages forward and backward", async () => {
    withToken();
    const calls = serve(120, (c) => Array.from({ length: Math.min(50, 120 - Number(c.url.searchParams.get("offset"))) }, (_, i) => tensor(i)));
    render(<TensorCatalog modelId="mod_1" selectedId={null} onSelect={() => {}} />);
    await screen.findByText(/Rows 1–50 of 120/);
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Last" }));
    expect(await screen.findByText(/Rows 101–120 of 120/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Previous" }));
    expect(await screen.findByText(/Rows 51–100 of 120/)).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText(/Rows per page/), "100");
    await waitFor(() => expect(calls.at(-1)!.url.searchParams.get("offset")).toBe("0"));
    expect(calls.at(-1)!.url.searchParams.get("limit")).toBe("100");
  });

  it("shows empty and error states", async () => {
    withToken();
    let fail = false;
    fakeApi(() => (fail ? apiError(429, "RATE_LIMITED", "too many requests in flight") : json({ total: 0, items: [] })));
    render(<TensorCatalog modelId="mod_1" selectedId={null} onSelect={() => {}} />);
    expect(await screen.findByText("No tensors match these filters.")).toBeInTheDocument();
    fail = true;
    await userEvent.type(screen.getByLabelText("Layer"), "1");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Too many requests at once");
    expect(alert).toHaveTextContent("RATE_LIMITED");
    expect(alert).toHaveTextContent("req_0123456789abcdef");
  });
});
