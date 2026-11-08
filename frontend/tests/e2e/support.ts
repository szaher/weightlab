// E2E support: real `weightlab start` processes, synthetic SafeTensors written here (independent of the
// backend's reader), a loopback-only network guard, and file hashing for "source files untouched" checks.
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createHash } from "node:crypto";
import { closeSync, copyFileSync, ftruncateSync, mkdtempSync, openSync, readFileSync, writeFileSync, writeSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type BrowserContext, type Page } from "@playwright/test";

export const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
export const FIXTURES = join(REPO, "examples", "fixtures");
const URL_RE = /http:\/\/127\.0\.0\.1:(\d+)\/#token=([A-Za-z0-9_-]{43,})/;

export interface Server {
  url: string;
  origin: string;
  port: number;
  token: string;
  stdout: () => string;
  stderr: () => string;
  stop: () => Promise<void>;
}

function command(): string[] {
  const raw = process.env.WEIGHTLAB_CMD;
  if (raw) return JSON.parse(raw) as string[];
  return ["uv", "run", "--locked", "--project", REPO, "weightlab"];
}

export async function startServer(dataDir: string): Promise<Server> {
  const [cmd, ...args] = command();
  const proc: ChildProcessWithoutNullStreams = spawn(cmd!, [...args, "start", "--port", "0", "--data-dir", dataDir], {
    cwd: REPO,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    detached: process.platform !== "win32",
  });
  let out = "";
  let err = "";
  proc.stdout.on("data", (d: Buffer) => (out += d.toString()));
  proc.stderr.on("data", (d: Buffer) => (err += d.toString()));
  const exited = new Promise<void>((r) => proc.once("exit", () => r()));
  const deadline = Date.now() + 90_000;
  while (!URL_RE.test(out)) {
    if (Date.now() > deadline || proc.exitCode !== null) throw new Error(`weightlab start failed:\n${out}\n${err}`);
    await new Promise((r) => setTimeout(r, 100));
  }
  const m = URL_RE.exec(out)!;
  const port = Number(m[1]);
  const origin = `http://127.0.0.1:${port}`;
  for (let i = 0; ; i++) {
    try {
      const r = await raw(port, "GET", "/api/v1/health");
      if (r.status === 200) break;
    } catch {
      if (i > 200) throw new Error("server never accepted connections");
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  return {
    url: m[0],
    origin,
    port,
    token: m[2]!,
    stdout: () => out,
    stderr: () => err,
    stop: async () => {
      if (proc.exitCode === null) {
        // POSIX: the whole process group (uv and the server it runs); Windows: the process itself.
        if (process.platform !== "win32" && proc.pid) process.kill(-proc.pid, "SIGINT");
        else proc.kill("SIGINT");
      }
      const t = setTimeout(() => proc.kill("SIGKILL"), 20_000);
      await exited;
      clearTimeout(t);
    },
  };
}

/** A raw HTTP request with full control over Host/Origin headers (fetch would rewrite them). */
export function raw(
  port: number,
  method: string,
  path: string,
  headers: Record<string, string> = {},
  body?: string,
): Promise<{ status: number; body: string; headers: Record<string, string | string[] | undefined> }> {
  return new Promise((ok, fail) => {
    const req = httpRequest(
      { host: "127.0.0.1", port, method, path, headers: { Host: `127.0.0.1:${port}`, ...headers } },
      (res) => {
        let data = "";
        res.on("data", (d: Buffer) => (data += d.toString()));
        res.on("end", () => ok({ status: res.statusCode ?? 0, body: data, headers: res.headers }));
      },
    );
    req.on("error", fail);
    if (body !== undefined) req.write(body);
    req.end();
  });
}

export function tempDir(prefix: string): string {
  return mkdtempSync(join(tmpdir(), `weightlab-e2e-${prefix}-`));
}

export interface HeaderEntry {
  dtype: string;
  shape: number[];
}

/** Reads a SafeTensors header directly (8-byte little-endian length + JSON): the test's own expectation. */
export function readHeader(path: string): Record<string, HeaderEntry> {
  const b = readFileSync(path);
  const n = Number(b.readBigUInt64LE(0));
  const h = JSON.parse(b.subarray(8, 8 + n).toString("utf8")) as Record<string, HeaderEntry & { data_offsets: number[] }>;
  delete (h as Record<string, unknown>).__metadata__;
  return h;
}

const DTYPE_BYTES: Record<string, number> = { F32: 4, F16: 2, BF16: 2, I8: 1, F64: 8 };

/** Writes a valid SafeTensors file with the given tensors (zero-filled data, contiguous offsets). */
export function writeSafetensors(path: string, tensors: { key: string; dtype: string; shape: number[] }[]): void {
  const header: Record<string, unknown> = { __metadata__: { source: "WeightLab E2E synthetic, not a model" } };
  let offset = 0;
  for (const t of tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1) * DTYPE_BYTES[t.dtype]!;
    header[t.key] = { dtype: t.dtype, shape: t.shape, data_offsets: [offset, offset + n] };
    offset += n;
  }
  let json = Buffer.from(JSON.stringify(header), "utf8");
  const pad = (8 - (json.length % 8)) % 8;
  json = Buffer.concat([json, Buffer.alloc(pad, 0x20)]);
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(json.length));
  writeFileSync(path, Buffer.concat([len, json, Buffer.alloc(offset)]));
}

/** A catalog of `layers` x 6 tensors with varied sizes, dtypes and roles. */
export function bigModel(path: string, layers: number): { key: string; dtype: string; shape: number[] }[] {
  const tensors: { key: string; dtype: string; shape: number[] }[] = [{ key: "model.embed_tokens.weight", dtype: "F16", shape: [64, 16] }];
  const parts = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm"];
  for (let l = 0; l < layers; l++) {
    parts.forEach((p, i) => {
      const shape = p.endsWith("layernorm") ? [4 + (l % 5)] : [2 + (l % 7), 2 + i];
      tensors.push({ key: `model.layers.${l}.${p}.weight`, dtype: i % 2 === 0 ? "F32" : "BF16", shape });
    });
  }
  writeSafetensors(path, tensors);
  return tensors;
}

export function sha256(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

export function copyFixture(name: string, destDir: string): string {
  const dest = join(destDir, name);
  copyFileSync(join(FIXTURES, name), dest);
  return dest;
}

/** Aborts every request that does not go to the local server, and records it. */
export async function loopbackOnly(context: BrowserContext): Promise<string[]> {
  const blocked: string[] = [];
  await context.route("**/*", (route) => {
    const u = new URL(route.request().url());
    if (u.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]"].includes(u.hostname)) return route.continue();
    if (u.protocol === "data:" || u.protocol === "blob:") return route.continue();
    blocked.push(u.href);
    return route.abort("internetdisconnected");
  });
  return blocked;
}

/** Console errors (CSP violations included) and uncaught page errors. */
export function watchConsole(page: Page): string[] {
  const problems: string[] = [];
  page.on("console", (m) => {
    // Expected: API errors the UI handles (401/404/409/413/422) and refused connections while a test stops the server.
    const expected = /the server responded with a status of (401|404|409|413|422)|net::ERR_CONNECTION_REFUSED/;
    if (m.type() === "error" && !expected.test(m.text())) problems.push(m.text());
  });
  page.on("pageerror", (e) => problems.push(String(e)));
  page.on("dialog", (d) => {
    problems.push(`unexpected dialog: ${d.message()}`);
    void d.dismiss();
  });
  return problems;
}

export async function expectNoToken(page: Page, token: string): Promise<void> {
  expect(page.url()).not.toContain(token);
  expect(page.url()).not.toContain("token=");
  const local = await page.evaluate(() => JSON.stringify({ ...localStorage }));
  expect(local).not.toContain(token);
  expect(await page.context().cookies()).toEqual([]);
  const html = await page.content();
  expect(html).not.toContain(token);
}

/** A decoded element: a float64 (exact for every float dtype), a bigint (64-bit ints) or a boolean. */
export type Elem = number | bigint | boolean;

function f16(bits: number): number {
  const s = bits & 0x8000 ? -1 : 1;
  const e = (bits >> 10) & 0x1f;
  const m = bits & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** Decodes one tensor straight from the file's bytes (little-endian, C order): the E2E's own expectation,
 * independent of the backend's reader and of the API's serialization. */
export function readTensor(path: string, key: string): { dtype: string; shape: number[]; values: Elem[] } {
  const b = readFileSync(path);
  const n = Number(b.readBigUInt64LE(0));
  const header = JSON.parse(b.subarray(8, 8 + n).toString("utf8")) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const t = header[key];
  if (!t) throw new Error(`no tensor ${key}`);
  const data = b.subarray(8 + n + t.data_offsets[0], 8 + n + t.data_offsets[1]);
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const count = t.shape.reduce((a, c) => a * c, 1);
  const values: Elem[] = [];
  for (let i = 0; i < count; i++) {
    switch (t.dtype) {
      case "F64": values.push(dv.getFloat64(i * 8, true)); break;
      case "F32": values.push(dv.getFloat32(i * 4, true)); break;
      case "F16": values.push(f16(dv.getUint16(i * 2, true))); break;
      case "BF16": {
        const tmp = new DataView(new ArrayBuffer(4));
        tmp.setUint32(0, dv.getUint16(i * 2, true) << 16, false);
        values.push(tmp.getFloat32(0, false));
        break;
      }
      case "I64": values.push(dv.getBigInt64(i * 8, true)); break;
      case "U64": values.push(dv.getBigUint64(i * 8, true)); break;
      case "I32": values.push(dv.getInt32(i * 4, true)); break;
      case "BOOL": values.push(dv.getUint8(i) !== 0); break;
      default: throw new Error(`readTensor: ${t.dtype} not needed by the tests`);
    }
  }
  return { dtype: t.dtype, shape: t.shape, values };
}

/** True when the UI's text is exactly the expected element (numbers compared as float64, keeping -0 and NaN). */
export function sameValue(text: string, expected: Elem): boolean {
  if (typeof expected === "bigint") return text === expected.toString();
  if (typeof expected === "boolean") return text === String(expected);
  if (Number.isNaN(expected)) return text === "NaN";
  if (expected === Infinity) return text === "+Infinity";
  if (expected === -Infinity) return text === "−Infinity";
  return Object.is(Number(text), expected);
}

/** Writes a SafeTensors file of 2D F32 tensors whose element (r, c) is `fill(r, c)` (rounded to float32). With
 * `sparseRows`, only those rows are written and the rest of the file is a hole (zeros), so very large tensors cost
 * almost no disk. The test computes its expectations from the same `fill`, never from the API. */
export function writeF32(path: string, tensors: { key: string; shape: [number, number]; fill: (r: number, c: number) => number; sparseRows?: (r: number) => number }[]): void {
  const header: Record<string, unknown> = { __metadata__: { source: "WeightLab E2E synthetic, not a model" } };
  let offset = 0;
  for (const t of tensors) {
    const n = t.shape[0] * t.shape[1] * 4;
    header[t.key] = { dtype: "F32", shape: t.shape, data_offsets: [offset, offset + n] };
    offset += n;
  }
  let json = Buffer.from(JSON.stringify(header), "utf8");
  json = Buffer.concat([json, Buffer.alloc((8 - (json.length % 8)) % 8, 0x20)]);
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(json.length));
  const fd = openSync(path, "w");
  try {
    writeSync(fd, Buffer.concat([len, json]));
    const base = 8 + json.length;
    ftruncateSync(fd, base + offset);
    let at = base;
    for (const t of tensors) {
      const [rows, cols] = t.shape;
      const row = Buffer.alloc(cols * 4);
      for (let r = 0; r < rows; r++) {
        const width = t.sparseRows ? t.sparseRows(r) : cols;
        if (width > 0) {
          for (let c = 0; c < width; c++) row.writeFloatLE(t.fill(r, c), c * 4);
          writeSync(fd, row, 0, width * 4, at + r * cols * 4);
        }
      }
      at += rows * cols * 4;
    }
  } finally {
    closeSync(fd);
  }
}

/** float32 rounding, as the server's float32 display values. */
export const f32 = (x: number): number => Math.fround(x);
