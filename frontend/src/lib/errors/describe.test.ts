import { describe, expect, it } from "vitest";
import { ApiRequestError } from "../api/client";
import { describeError } from "./describe";

const err = (status: number, code: string, message = "m", extra: Partial<ConstructorParameters<typeof ApiRequestError>[0]> = {}) =>
  new ApiRequestError({ status, code, message, requestId: "req_1", ...extra });

describe("describeError", () => {
  it("explains authentication failures with the reconnect instruction", () => {
    const v = describeError(err(401, "UNAUTHORIZED"));
    expect(v.kind).toBe("auth");
    expect(v.action).toContain("weightlab start");
    expect(v.code).toBe("UNAUTHORIZED");
    expect(v.requestId).toBe("req_1");
  });

  it("explains an unavailable backend", () => {
    const v = describeError(err(0, "NETWORK"));
    expect(v.kind).toBe("offline");
    expect(v.title).toMatch(/unavailable/);
  });

  it("recognizes permission denied (403 VALIDATION_ERROR) without suggesting elevation", () => {
    const v = describeError(err(403, "VALIDATION_ERROR", "permission denied: entry path (Permission denied)"), "import");
    expect(v.title).toBe("Permission denied");
    expect(v.action).toMatch(/never needs administrator/);
    expect(v.action).not.toMatch(/sudo|run as administrator/i);
  });

  it("distinguishes not-found paths during import", () => {
    expect(describeError(err(404, "NOT_FOUND", "entry path does not exist"), "import").title).toBe("Path not found");
    expect(describeError(err(404, "NOT_FOUND", "project not found")).title).toBe("Not found");
  });

  it("explains unsupported formats", () => {
    const v = describeError(err(422, "INVALID_FORMAT", "not a SafeTensors file"), "import");
    expect(v.action).toMatch(/\.safetensors/);
    expect(v.explanation).toBe("not a SafeTensors file");
  });

  it("gives relink-specific guidance for FILE_CHANGED", () => {
    expect(describeError(err(409, "FILE_CHANGED"), "relink").action).toMatch(/import it again/);
    expect(describeError(err(409, "FILE_CHANGED", "x", { recommendedAction: "server says" })).action).toBe("server says");
  });

  it("handles limits and unknown codes with the server's own text", () => {
    expect(describeError(err(429, "RATE_LIMITED", "too many requests in flight")).title).toMatch(/Too many/);
    expect(describeError(err(413, "RESOURCE_LIMIT", "request body above 1048576 bytes")).explanation).toContain("1048576");
    const unknown = describeError(err(409, "SOMETHING_NEW", "new thing"));
    expect(unknown.explanation).toBe("new thing");
    expect(unknown.action).toBeNull();
  });

  it("does not leak non-API exceptions", () => {
    const v = describeError(new Error("/Users/me/secret/path stack trace"));
    expect(JSON.stringify(v)).not.toContain("/Users/me");
  });
});
