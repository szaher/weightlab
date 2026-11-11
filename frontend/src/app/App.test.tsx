import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { clearToken, TOKEN_KEY } from "../lib/auth/token";
import { apiError, fakeApi, json, TEST_TOKEN, withToken, type Call } from "../test/fakeApi";
import { App } from "./App";

const HEALTH = { status: "ok", version: "0.0.1.dev0", offline: true, capabilities: [] };
const PROJECT = { id: "prj_1", name: "Research", created_at: "2026-11-07T10:00:00Z" };
const MODEL = {
  id: "mod_1",
  project_id: "prj_1",
  display_name: "tiny_base.safetensors",
  format: "safetensors",
  status: "ready",
  tensor_count: 5,
  fingerprint_method: "size_mtime",
};

function backend(routes: Record<string, (c: Call) => Response>) {
  return fakeApi((c) => {
    const key = `${c.method} ${c.url.pathname}`;
    if (key === "GET /api/v1/health") return json(HEALTH);
    const h = routes[key];
    if (!h) throw new Error(`unexpected request ${key}`);
    return h(c);
  });
}

describe("App", () => {
  it("asks for the launch link when the tab has no token", async () => {
    clearToken();
    backend({});
    render(<App />);
    expect(screen.getByRole("heading", { name: "Connect to WeightLab" })).toBeInTheDocument();
    expect(screen.getByText(/#token=…/)).toBeInTheDocument();
  });

  it("picks up the token from the fragment and strips it", async () => {
    clearToken();
    window.history.replaceState(null, "", "/");
    backend({ "GET /api/v1/projects": () => json([PROJECT]), "GET /api/v1/projects/prj_1/models": () => json([]) });
    render(<App />);
    window.history.replaceState(null, "", `/#token=${TEST_TOKEN}`);
    fireEvent(window, new HashChangeEvent("hashchange"));
    expect(await screen.findByRole("heading", { name: "Research" })).toBeInTheDocument();
    expect(window.location.href).not.toContain(TEST_TOKEN);
    expect(sessionStorage.getItem(TOKEN_KEY)).toBe(TEST_TOKEN);
  });

  it("shows the expired-session screen on 401 and forgets the token", async () => {
    withToken();
    backend({ "GET /api/v1/projects": () => apiError(401, "UNAUTHORIZED", "missing or invalid bearer token") });
    render(<App />);
    expect(await screen.findByRole("heading", { name: "This WeightLab session has ended" })).toBeInTheDocument();
    expect(sessionStorage.getItem(TOKEN_KEY)).toBeNull();
    expect(screen.getByRole("status", { name: /Backend: Not authorized/ })).toBeInTheDocument();
  });

  it("welcomes a first-time user and creates a project", async () => {
    withToken();
    let projects: (typeof PROJECT)[] = [];
    const calls = backend({
      "GET /api/v1/projects": () => json(projects),
      "POST /api/v1/projects": (c) => {
        projects = [{ ...PROJECT, name: (c.body as { name: string }).name }];
        return json(projects[0], 201);
      },
      "GET /api/v1/projects/prj_1/models": () => json([]),
    });
    render(<App />);
    expect(await screen.findByRole("heading", { name: "Explore LLM weights on your computer." })).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("New project name"), "Research");
    await userEvent.click(screen.getByRole("button", { name: "Create project" }));
    expect(await screen.findByRole("heading", { name: "Research" })).toBeInTheDocument();
    expect(await screen.findByText(/No models in this project yet/)).toBeInTheDocument();
    expect(calls.find((c) => c.method === "POST")!.body).toEqual({ name: "Research" });
  });

  it("confirms project deletion and says source files are untouched", async () => {
    withToken();
    let projects = [PROJECT];
    const calls = backend({
      "GET /api/v1/projects": () => json(projects),
      "GET /api/v1/projects/prj_1/models": () => json([MODEL]),
      "DELETE /api/v1/projects/prj_1": () => {
        projects = [];
        return new Response(null, { status: 204 });
      },
    });
    render(<App />);
    const open = await screen.findByRole("button", { name: "Delete project…" });
    await userEvent.click(open);
    const dialog = screen.getByRole("dialog", { name: /Delete project “Research”/ });
    expect(within(dialog).getByText(/checkpoint files are not deleted or modified/)).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Cancel" })).toHaveFocus();
    await userEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(calls.some((c) => c.method === "DELETE")).toBe(false);
    await userEvent.click(open);
    await userEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Delete project" }));
    await waitFor(() => expect(calls.some((c) => c.method === "DELETE")).toBe(true));
    expect(await screen.findByRole("heading", { name: "Explore LLM weights on your computer." })).toBeInTheDocument();
  });

  it("shows changed and disconnected models distinctly", async () => {
    withToken();
    backend({
      "GET /api/v1/projects": () => json([PROJECT]),
      "GET /api/v1/projects/prj_1/models": () =>
        json([
          { ...MODEL, id: "mod_a", display_name: "A", status: "changed" },
          { ...MODEL, id: "mod_b", display_name: "B", status: "disconnected" },
          { ...MODEL, id: "mod_c", display_name: "C", status: "ready" },
        ]),
    });
    render(<App />);
    expect(await screen.findByText("▲")).toBeInTheDocument();
    expect(screen.getByText(/Files changed since import: import the model again/)).toBeInTheDocument();
    expect(screen.getByText(/Files not found at the recorded location/)).toBeInTheDocument();
    const badges = [...document.querySelectorAll(".badge")].map((e) => e.textContent);
    expect(badges).toEqual(["▲ Changed", "○ Disconnected", "● Ready"]); // a word and a symbol, not colour alone
  });

  it("reports an unreachable backend", async () => {
    withToken();
    fakeApi(() => {
      throw new TypeError("Failed to fetch");
    });
    render(<App />);
    expect((await screen.findAllByText("WeightLab server unavailable")).length).toBeGreaterThan(0);
    expect(screen.getByRole("status", { name: /Backend: Server unavailable/ })).toBeInTheDocument();
  });

  it("explains that dropped files cannot be imported, and never reads them", async () => {
    withToken();
    backend({ "GET /api/v1/projects": () => json([PROJECT]), "GET /api/v1/projects/prj_1/models": () => json([]) });
    render(<App />);
    await screen.findByRole("heading", { name: "Research" });
    const file = new File(["weights"], "model.safetensors");
    const ev = new Event("drop", { bubbles: true, cancelable: true });
    Object.defineProperty(ev, "dataTransfer", { value: { types: ["Files"], files: [file] } });
    window.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
    expect(await screen.findByText("Dropped files are not imported")).toBeInTheDocument();
  });

  it("offers Compare as a working screen; Adapters stays planned", async () => {
    withToken();
    backend({ "GET /api/v1/projects": () => json([]) });
    render(<App />);
    const nav = screen.getByRole("navigation", { name: "Main" });
    expect(within(nav).getByRole("button", { name: "Compare" })).toBeEnabled();
    expect(within(nav).queryByRole("button", { name: /Adapters/ })).toBeNull();
    expect(within(nav).getByText("Adapters").closest("[aria-disabled]")).toHaveAttribute("aria-disabled", "true");
  });

  it("explores a model: catalog, selection and inspector", async () => {
    withToken();
    const T = {
      id: "tns_up",
      model_id: "mod_1",
      raw_key: "demo.layers.0.mlp.up_proj.weight",
      shape: [8, 8],
      dtype: "F32",
      rank: 2,
      numel: 64,
      byte_length: 256,
      shard_relpath: "tiny_base.safetensors",
      file_offset_begin: 1000,
      file_offset_end: 1256,
      role: "mlp.up",
      role_is_heuristic: true,
      layer_index: 0,
    };
    backend({
      "GET /api/v1/projects": () => json([PROJECT]),
      "GET /api/v1/projects/prj_1/models": () => json([MODEL]),
      "GET /api/v1/models/mod_1/tensors": () => json({ total: 1, items: [T] }),
      "GET /api/v1/models/mod_1/tensors/tns_up": () => json(T),
    });
    render(<App />);
    await userEvent.click(await screen.findByRole("button", { name: "Explore tensors" }));
    await userEvent.click(await screen.findByRole("button", { name: T.raw_key }));
    expect(await screen.findByTestId("inspector-key")).toHaveTextContent(T.raw_key);
    expect(screen.getByTestId("inspector-shape")).toHaveTextContent("8 × 8");
    expect(screen.getByTestId("inspector-dtype")).toHaveTextContent("F32");
    expect(screen.getByText("[1,000, 1,256)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy raw tensor key" })).toBeInTheDocument();
  });
});
