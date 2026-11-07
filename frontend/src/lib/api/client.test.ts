import { describe, expect, it, vi } from "vitest";
import { apiError, fakeApi, json, TEST_TOKEN, withToken } from "../../test/fakeApi";
import { ApiRequestError, onUnauthorized, request } from "./client";
import { api } from "./endpoints";

describe("API client", () => {
  it("sends the bearer token, same-origin, without credentials", async () => {
    withToken();
    const calls = fakeApi(() => json([]));
    await api.listProjects();
    const c = calls[0]!;
    expect(c.url.pathname).toBe("/api/v1/projects");
    expect(c.headers.Authorization).toBe(`Bearer ${TEST_TOKEN}`);
    expect(c.init.credentials).toBe("omit");
    expect(c.init.cache).toBe("no-store");
    expect(c.url.search).toBe("");
    expect(c.url.href).not.toContain(TEST_TOKEN);
  });

  it("does not send the token to the public health route", async () => {
    withToken();
    const calls = fakeApi(() => json({ status: "ok", version: "1", offline: true, capabilities: [] }));
    await api.health();
    expect(calls[0]!.headers.Authorization).toBeUndefined();
  });

  it("refuses to send without a token", async () => {
    fakeApi(() => json([]));
    await expect(api.listProjects()).rejects.toMatchObject({ code: "NO_TOKEN", status: 401 });
  });

  it("parses the error envelope", async () => {
    withToken();
    fakeApi(() => apiError(409, "FILE_CHANGED", "model.safetensors changed since import", { recommended_action: "Re-import it." }));
    const err = await api.listProjects().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiRequestError);
    expect(err).toMatchObject({
      status: 409,
      code: "FILE_CHANGED",
      message: "model.safetensors changed since import",
      recommendedAction: "Re-import it.",
      requestId: "req_0123456789abcdef",
    });
  });

  it("handles non-envelope errors and network failures", async () => {
    withToken();
    fakeApi(() => new Response("<html>oops</html>", { status: 502, headers: { "x-request-id": "req_ffff" } }));
    await expect(api.listProjects()).rejects.toMatchObject({ code: "BAD_RESPONSE", status: 502, requestId: "req_ffff" });
    fakeApi(() => {
      throw new TypeError("Failed to fetch");
    });
    await expect(api.listProjects()).rejects.toMatchObject({ code: "NETWORK", status: 0, retryable: true });
  });

  it("notifies listeners on 401", async () => {
    withToken();
    fakeApi(() => apiError(401, "UNAUTHORIZED", "missing or invalid bearer token"));
    const spy = vi.fn();
    const off = onUnauthorized(spy);
    await expect(api.listProjects()).rejects.toMatchObject({ code: "UNAUTHORIZED" });
    off();
    expect(spy).toHaveBeenCalledOnce();
  });

  it("encodes ids and never puts tensor names into the path", async () => {
    withToken();
    const calls = fakeApi(() => json({}));
    await api.getTensor("mod_a", "tns_b");
    expect(calls[0]!.url.pathname).toBe("/api/v1/models/mod_a/tensors/tns_b");
    await api.getTensor("mod_a", "../../projects");
    expect(calls[1]!.url.pathname).toBe("/api/v1/models/mod_a/tensors/..%2F..%2Fprojects");
  });

  it("only builds same-origin paths", async () => {
    withToken();
    fakeApi(() => json({}));
    await expect(request("//evil.example/x")).rejects.toThrow(/same-origin/);
    await expect(request("https://evil.example/x")).rejects.toThrow(/same-origin/);
  });

  it("returns undefined for 204", async () => {
    withToken();
    const calls = fakeApi(() => new Response(null, { status: 204 }));
    await expect(api.deleteProject("prj_1")).resolves.toBeUndefined();
    expect(calls[0]!.method).toBe("DELETE");
  });
});
