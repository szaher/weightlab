import { describe, expect, it } from "vitest";
import { validatePathFields } from "../features/models/pathForm";
import { formatBytes, formatInt, formatShape, isAbsolutePath, parentDir, shapeText } from "./format";

describe("formatting", () => {
  it("formats integers, shapes and bytes", () => {
    expect(formatInt(67591)).toBe("67,591");
    expect(formatShape([257, 263])).toBe("257 × 263");
    expect(formatShape([])).toBe("scalar");
    expect(shapeText([8, 8])).toBe("8x8");
    expect(formatBytes(256)).toBe("256 B");
    expect(formatBytes(2048)).toBe("2.00 KiB");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3.00 GiB");
  });
});

describe("paths", () => {
  it("recognizes absolute POSIX, Windows drive, long and UNC paths", () => {
    for (const p of ["/models/a.safetensors", "C:\\models\\a", "c:/models", "\\\\?\\C:\\models", "\\\\server\\share\\m"]) {
      expect(isAbsolutePath(p)).toBe(true);
    }
    for (const p of ["models/a", "./a", "~/models", "C:models", "\\models"]) {
      expect(isAbsolutePath(p)).toBe(false);
    }
  });

  it("computes the containing folder", () => {
    expect(parentDir("/models/tiny/model.safetensors")).toBe("/models/tiny");
    expect(parentDir("/models/tiny/")).toBe("/models");
    expect(parentDir("/model.safetensors")).toBe("/");
    expect(parentDir("C:\\models\\m.safetensors")).toBe("C:\\models");
    expect(parentDir("C:\\m.safetensors")).toBe("C:\\");
  });

  it("validates the import form like the contract", () => {
    expect(validatePathFields({ entry_path: "", selected_root: "" })).toEqual({ entry_path: "Required.", selected_root: "Required." });
    expect(validatePathFields({ entry_path: "models/a", selected_root: "/m" }).entry_path).toMatch(/absolute/);
    expect(validatePathFields({ entry_path: " /a", selected_root: "/m" }).entry_path).toMatch(/spaces/);
    expect(validatePathFields({ entry_path: `/${"a".repeat(4096)}`, selected_root: "/m" }).entry_path).toMatch(/4096/);
    expect(validatePathFields({ entry_path: "/a", selected_root: "/m", display_name: "x".repeat(121) }).display_name).toMatch(/120/);
    expect(validatePathFields({ entry_path: "/a/b.safetensors", selected_root: "/a" })).toEqual({});
  });
});
