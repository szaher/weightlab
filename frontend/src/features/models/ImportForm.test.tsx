import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { apiError, fakeApi, json, withToken } from "../../test/fakeApi";
import { ImportForm } from "./ImportForm";

const MODEL = {
  id: "mod_1",
  project_id: "prj_1",
  display_name: "tiny_base.safetensors",
  format: "safetensors",
  status: "ready",
  tensor_count: 5,
  fingerprint_method: "size_mtime",
};

describe("ImportForm", () => {
  it("validates before sending anything", async () => {
    withToken();
    const calls = fakeApi(() => json(MODEL, 201));
    render(<ImportForm projectId="prj_1" onImported={() => {}} />);
    await userEvent.click(screen.getByRole("button", { name: "Import model" }));
    expect(screen.getAllByText("Required.")).toHaveLength(2);
    await userEvent.type(screen.getByLabelText(/Model path/), "models/tiny.safetensors");
    await userEvent.click(screen.getByRole("button", { name: "Import model" }));
    expect(screen.getByLabelText(/Model path/)).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByText(/Use an absolute path/)).toBeInTheDocument();
    expect(calls).toHaveLength(0);
  });

  it("imports by path only, with an explicit selected root", async () => {
    withToken();
    const calls = fakeApi(() => json(MODEL, 201));
    const onImported = vi.fn();
    render(<ImportForm projectId="prj_1" onImported={onImported} />);
    await userEvent.type(screen.getByLabelText(/Model path/), "/data/models/tiny_base.safetensors");
    expect(screen.getByLabelText(/Selected root/)).toHaveValue(""); // never filled silently
    await userEvent.click(screen.getByRole("button", { name: /Use the folder containing/ }));
    expect(screen.getByLabelText(/Selected root/)).toHaveValue("/data/models");
    await userEvent.click(screen.getByRole("button", { name: "Import model" }));
    expect(onImported).toHaveBeenCalledWith(MODEL);
    expect(calls[0]!.method).toBe("POST");
    expect(calls[0]!.url.pathname).toBe("/api/v1/projects/prj_1/models");
    expect(calls[0]!.body).toEqual({ entry_path: "/data/models/tiny_base.safetensors", selected_root: "/data/models" });
  });

  it("shows a busy state while the server validates", async () => {
    withToken();
    let release: (r: Response) => void = () => {};
    fakeApi(() => new Promise<Response>((r) => (release = r)));
    render(<ImportForm projectId="prj_1" onImported={() => {}} />);
    await userEvent.type(screen.getByLabelText(/Model path/), "/m/a.safetensors");
    await userEvent.type(screen.getByLabelText(/Selected root/), "/m");
    await userEvent.click(screen.getByRole("button", { name: "Import model" }));
    expect(screen.getByRole("button", { name: /Importing/ })).toBeDisabled();
    expect(screen.getByText(/No tensor values are read/)).toBeInTheDocument();
    release(json(MODEL, 201));
    expect(await screen.findByRole("button", { name: "Import model" })).toBeEnabled();
  });

  it("explains server-side failures", async () => {
    withToken();
    fakeApi(() => apiError(404, "NOT_FOUND", "entry path does not exist"));
    render(<ImportForm projectId="prj_1" onImported={() => {}} />);
    await userEvent.type(screen.getByLabelText(/Model path/), "/nope/a.safetensors");
    await userEvent.type(screen.getByLabelText(/Selected root/), "/nope");
    await userEvent.click(screen.getByRole("button", { name: "Import model" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Path not found");
    expect(alert).toHaveTextContent("entry path does not exist");
    expect(alert).toHaveTextContent("NOT_FOUND");
  });
});
