// PR #9: the browser workflow on real checkpoints. Runs only when WEIGHTLAB_REAL_MODELS points at the directory
// written by evidence/phase1-real-models/fetch_models.py (weights are never in the repository). Offline: every
// non-loopback request is blocked.
//
//   caffeinate -dims env WEIGHTLAB_REAL_MODELS=~/weightlab-models WEIGHTLAB_REAL_OUT=../evidence/phase1-real-models/results/browser.json \
//   WEIGHTLAB_SCREENSHOTS=../evidence/phase1-real-models/screenshots WEIGHTLAB_SOAK_MIN=20 npx playwright test real
//
// Workflow: import Qwen2.5-0.5B and -Instruct through the UI; browse layers by role; open real attention and MLP
// matrices; overview at several levels with every aggregation; zoom and pan; select a block; Exact ROI values
// equal the file's bytes (this suite's own BF16 decoder, independent of the backend and of the validation script's
// PyTorch reader); the embedding's real pyramid job; a row-sampled preview of a Math-1.5B embedding. Then a soak
// session (WEIGHTLAB_SOAK_MIN minutes, default 20) of pan / zoom / tensor and model switches, sampling the JS heap,
// the backend RSS, the decoded-tile cache and request concurrency once a minute.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { loopbackOnly, raw, readTensor, sameValue, startServer, tempDir, watchConsole } from "./support";

const ROOT = (process.env.WEIGHTLAB_REAL_MODELS ?? "").replace(/^~/, homedir());
const OUT = process.env.WEIGHTLAB_REAL_OUT;
const SHOTS = process.env.WEIGHTLAB_SCREENSHOTS;
const SOAK_MIN = Number(process.env.WEIGHTLAB_SOAK_MIN ?? "20");
// No trace: ~30k soak actions would make a multi-GB trace.
test.use({ trace: "off" });
test.skip(!ROOT || !existsSync(join(ROOT, "Qwen2.5-0.5B", "model.safetensors")), "set WEIGHTLAB_REAL_MODELS to the fetched model directory");

const BASE = "Qwen2.5-0.5B";
const INSTRUCT = "Qwen2.5-0.5B-Instruct";
const MATH = "Qwen2.5-Math-1.5B";

async function shot(page: Page, name: string) {
  if (!SHOTS) return;
  mkdirSync(SHOTS, { recursive: true });
  await page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage: true });
}

async function openTensor(page: Page, key: string) {
  await page.getByLabel("Search tensor names").fill(key);
  await page.getByRole("button", { name: key, exact: true }).click();
  await expect(page.getByTestId("inspector-key")).toHaveText(key);
}

async function geom(page: Page) {
  const cv = page.getByTestId("overview-canvas");
  await cv.scrollIntoViewIfNeeded();
  const box = (await cv.boundingBox())!;
  const n = async (a: string) => Number(await cv.getAttribute(a));
  return { box, ppe: await n("data-ppe"), top: await n("data-top"), left: await n("data-left"), ax: await n("data-axis-left"), ay: await n("data-axis-top"), level: await n("data-level"),
           cacheBytes: await n("data-cache-bytes"), cacheEntries: await n("data-cache-entries"), peak: await n("data-peak-inflight"), requests: await n("data-requests") }; // prettier-ignore
}

async function clickRaw(page: Page, r: number, c: number) {
  const g = await geom(page);
  await page.mouse.click(g.box.x + g.ax + (c + 0.5 - g.left) * g.ppe, g.box.y + g.ay + (r + 0.5 - g.top) * g.ppe);
}

async function settled(page: Page, timeout = 120_000) {
  await expect(page.getByTestId("overview-loading")).toBeHidden({ timeout });
  await expect(page.getByTestId("coverage-badge")).toBeVisible({ timeout });
}

/** Backend RSS (MiB) of the `weightlab start` process serving `dataDir` (largest matching process). */
function backendRss(dataDir: string): number {
  if (process.platform === "win32") return NaN;
  const out = execFileSync("ps", ["-axo", "rss=,command="], { encoding: "utf8" });
  let best = 0;
  for (const line of out.split("\n")) if (line.includes(dataDir) && line.includes(" start")) best = Math.max(best, Number(line.trim().split(/\s+/)[0]) / 1024);
  return Math.round(best * 10) / 10;
}

test("real checkpoints: workflow, exact drill-down and a soak session", async ({ browser }) => {
  test.setTimeout((SOAK_MIN + 15) * 60_000);
  const dataDir = tempDir("real-data");
  const server = await startServer(dataDir);
  const context = await browser.newContext();
  const blocked = await loopbackOnly(context);
  const page = await context.newPage();
  page.setDefaultTimeout(60_000); // fail fast on a wrong locator rather than at the test timeout
  const problems = watchConsole(page);
  const cdp = await context.newCDPSession(page);
  await cdp.send("Performance.enable");
  const heap = async () => Math.round((((await cdp.send("Performance.getMetrics")).metrics.find((m) => m.name === "JSHeapUsedSize")?.value ?? 0) / 2 ** 20) * 10) / 10;
  const out: Record<string, unknown> = { soak_minutes: SOAK_MIN };
  const auth = { Authorization: `Bearer ${server.token}`, "Content-Type": "application/json" };
  try {
    // Import through the UI (path form), the second model too.
    await page.goto(server.url);
    await page.getByLabel("New project name").fill("Qwen validation");
    await page.getByRole("button", { name: "Create project" }).click();
    for (const name of [BASE, INSTRUCT, MATH]) {
      if (!existsSync(join(ROOT, name, "model.safetensors"))) continue;
      await page.getByLabel(/Model path/).fill(join(ROOT, name, "model.safetensors"));
      await page.getByLabel(/Selected root/).fill(join(ROOT, name));
      const dn = page.getByLabel(/Display name/);
      if ((await dn.count()) > 0) await dn.fill(name);
      await page.getByRole("button", { name: "Import model" }).click();
      await expect(page.getByText(new RegExp(`Imported “(${name}|model\\.safetensors)”`))).toBeVisible();
    }
    await shot(page, "real-01-workspace");
    await page.getByRole("listitem").filter({ hasText: BASE }).first().getByRole("button", { name: "Explore tensors" }).click();

    // Browse layers by role.
    await page.getByLabel("Role (heuristic)").selectOption("attention.q");
    await expect(page.getByText(/Rows 1–\d+ of 48 matching tensors/)).toBeVisible();
    await page.getByLabel("Role (heuristic)").selectOption("");
    await openTensor(page, "model.layers.12.mlp.gate_proj.weight");
    await settled(page);
    await shot(page, "real-02-mlp-gate-overview");

    // Every aggregation and several levels on a real MLP matrix (4864 x 896 BF16).
    const levels: number[] = [];
    for (const agg of ["mean", "mean_abs", "rms", "min", "max"]) {
      await page.getByTestId("agg-select").selectOption(agg);
      await settled(page);
      await expect(page.getByTestId("coverage-badge")).toHaveText("EXACT coverage");
    }
    // L0 of the whole matrix would need 76 tiles: refused with an explanation, not loaded.
    await page.getByTestId("level-select").selectOption("0");
    await expect(page.getByText(/Level L0 would need \d+ tiles for this screen \(limit 48\)/)).toBeVisible();
    for (const level of ["1", "2", "4", "5"]) {
      await page.getByTestId("level-select").selectOption(level);
      await settled(page);
      levels.push((await geom(page)).level);
    }
    out.levels_rendered = levels;
    await page.getByTestId("level-select").selectOption("auto");
    await page.getByTestId("agg-select").selectOption("mean_abs");

    // Real attention matrix: zoom, pan, select a block, exact values against the file's bytes.
    await openTensor(page, "model.layers.0.self_attn.q_proj.weight");
    await settled(page);
    await page.getByRole("button", { name: "Zoom in" }).click();
    await settled(page);
    const g = await geom(page);
    await page.mouse.move(g.box.x + g.ax + 200, g.box.y + g.ay + 150);
    await page.mouse.down();
    await page.mouse.move(g.box.x + g.ax + 120, g.box.y + g.ay + 90, { steps: 6 });
    await page.mouse.up();
    await settled(page);
    await page.getByRole("button", { name: "Fit" }).click();
    await page.getByTestId("level-select").selectOption("2"); // 896 x 896: L2 (4 x 4 blocks) is the top level
    await settled(page);
    await clickRaw(page, 301, 502);
    await expect(page.getByTestId("selection-bounds")).toHaveText("rows [300, 304) · columns [500, 504)");
    await shot(page, "real-03-attention-selected");
    await page.getByTestId("inspect-exact").click();
    await expect(page.getByTestId("roi-bounds")).toContainText("Rows [300, 304) of 896");
    const q = readTensor(join(ROOT, BASE, "model.safetensors"), "model.layers.0.self_attn.q_proj.weight");
    const heat = page.getByTestId("heatmap-canvas");
    await heat.scrollIntoViewIfNeeded();
    const hb = (await heat.boundingBox())!;
    const cell = Number(await heat.getAttribute("data-cell-px"));
    const hl = Number(await heat.getAttribute("data-left"));
    const ht = Number(await heat.getAttribute("data-top"));
    let compared = 0;
    for (const [i, j] of [[0, 0], [1, 2], [3, 3], [2, 1]] as const) {
      await page.mouse.click(hb.x + hl + (j + 0.5) * cell, hb.y + ht + (i + 0.5) * cell);
      await expect(page.getByTestId("selected-coords")).toHaveText(`${300 + i}, ${500 + j}`);
      const v = (await page.getByTestId("selected-value").textContent()) ?? "";
      expect(sameValue(v, q.values[(300 + i) * 896 + 500 + j]!), `q_proj (${300 + i}, ${500 + j})`).toBe(true);
      compared++;
    }
    out.browser_roi_values_compared = compared;
    await shot(page, "real-04-exact-roi");

    // The embedding (151936 x 896 BF16, 260 MiB): a real pyramid job, then the exact overview.
    await page.getByRole("tab", { name: "Overview" }).click();
    await openTensor(page, "model.embed_tokens.weight");
    const t0 = Date.now();
    await page.getByTestId("level-select").selectOption("9");
    await expect(page.getByTestId("job-panel")).toBeVisible({ timeout: 60_000 });
    await shot(page, "real-05-embedding-building");
    // The edge tile (rows 131072-151936, 36 MiB) is read directly and is exact at once; the rest waits for the job.
    // "Done" means the job panel is gone and the whole view is exact, not merely the first exact tile.
    await expect(page.getByTestId("job-panel")).toBeHidden({ timeout: 300_000 });
    await expect(page.getByTestId("coverage-badge")).toHaveText("EXACT coverage", { timeout: 60_000 });
    await settled(page);
    out.embedding_exact_ms_incl_build = Date.now() - t0;
    await shot(page, "real-06-embedding-exact");

    // Row-sampled preview of a larger embedding (pyramid_exact_max_mib lowered to its minimum, 64 MiB).
    if (existsSync(join(ROOT, MATH, "model.safetensors"))) {
      expect((await raw(server.port, "PATCH", "/api/v1/settings", auth, JSON.stringify({ pyramid_exact_max_mib: 64 }))).status).toBe(200);
      await page.reload(); // lands on the workspace
      await page.getByRole("navigation", { name: "Main" }).getByRole("button", { name: "Explore" }).click();
      await page.getByLabel("Model", { exact: true }).selectOption({ label: MATH });
      await openTensor(page, "model.embed_tokens.weight");
      // Auto L9: tile y=0 spans 131072 x 1536 BF16 rows (384 MiB > the 64 MiB direct limit) and is row-sampled; the
      // edge tile y=1 spans 20864 rows (61 MiB) and is computed exactly. The overview says so instead of claiming one.
      await expect(page.getByTestId("coverage-badge")).toHaveText("MIXED: exact and ROW-SAMPLED tiles", { timeout: 120_000 });
      await expect(page.getByTestId("coverage-text")).toContainText("Some visible tiles are exact, others are row-sampled");
      await shot(page, "real-07-math-embedding-mixed");
      // L10 is a single 151936 x 1536 tile (445 MiB): a whole row-sampled preview.
      await page.getByTestId("level-select").selectOption("10");
      await expect(page.getByTestId("coverage-badge")).toHaveText("ROW-SAMPLED preview", { timeout: 120_000 });
      await expect(page.getByTestId("coverage-text")).toContainText("evenly strided rows");
      await shot(page, "real-08-math-embedding-sampled");
      await page.getByTestId("level-select").selectOption("auto");
      expect((await raw(server.port, "PATCH", "/api/v1/settings", auth, JSON.stringify({ pyramid_exact_max_mib: 1024 }))).status).toBe(200);
      await page.reload();
      await page.getByRole("navigation", { name: "Main" }).getByRole("button", { name: "Explore" }).click();
    }

    // Soak: pan / zoom / tensor and model switches for SOAK_MIN minutes; sample once a minute.
    const tensors = ["model.layers.0.self_attn.q_proj.weight", "model.layers.12.mlp.gate_proj.weight", "model.layers.23.mlp.down_proj.weight", "model.embed_tokens.weight", "model.layers.5.self_attn.k_proj.weight", "model.layers.17.mlp.up_proj.weight"];
    const models = [BASE, INSTRUCT];
    const samples: Record<string, number>[] = [];
    // Mean ms per soak iteration by phase since the previous sample, to tell where any slowdown comes from.
    const ph = { switch: 0, agg: 0, keys: 0, settle: 0, n: 0 };
    const metric = async (name: string) => (await cdp.send("Performance.getMetrics")).metrics.find((m) => m.name === name)?.value ?? NaN;
    const roundTrip = async () => {
      const t0 = performance.now();
      for (let i = 0; i < 20; i++) await page.evaluate(() => 1);
      return Math.round(((performance.now() - t0) / 20) * 100) / 100;
    };
    const sample = async (minute: number, actions: number) => {
      await cdp.send("HeapProfiler.collectGarbage");
      const gg = (await page.getByTestId("overview-canvas").count()) > 0 ? await geom(page) : null;
      const per = (v: number) => (ph.n ? Math.round(v / ph.n) : NaN);
      samples.push({ minute, actions, js_heap_mib: await heap(), backend_rss_mib: backendRss(dataDir), decoded_cache_mib: gg ? Math.round((gg.cacheBytes / 2 ** 20) * 10) / 10 : NaN, decoded_cache_tiles: gg?.cacheEntries ?? NaN, peak_inflight: gg?.peak ?? NaN,
        noop_roundtrip_ms: await roundTrip(), node_heap_mib: Math.round(process.memoryUsage().heapUsed / 2 ** 20),
        dom_nodes: await metric("Nodes"), js_listeners: await metric("JSEventListeners"), ms_switch: per(ph.switch), ms_agg_settle: per(ph.agg), ms_keys: per(ph.keys), ms_keys_settle: per(ph.settle) }); // prettier-ignore
      Object.assign(ph, { switch: 0, agg: 0, keys: 0, settle: 0, n: 0 });
    };
    const keys = ["+", "+", "Shift+ArrowRight", "Shift+ArrowDown", "Shift+ArrowLeft", "-", "Shift+ArrowUp", "+", "Shift+ArrowRight", "0"];
    const start = Date.now();
    let actions = 0;
    let nextSample = 0;
    let k = 0;
    // A suspended host (laptop idle sleep) freezes the browser while the wall clock keeps running; such a run is not
    // a soak. Record the longest gap between iterations and refuse runs with one over 60 s (run under `caffeinate`).
    let lastIter = Date.now();
    let maxGap = 0;
    while (Date.now() - start < SOAK_MIN * 60_000) {
      maxGap = Math.max(maxGap, Date.now() - lastIter);
      lastIter = Date.now();
      const minute = Math.floor((Date.now() - start) / 60_000);
      if (minute >= nextSample) {
        await sample(minute, actions);
        nextSample = minute + 1;
      }
      const model = models[k % models.length]!;
      let t = Date.now();
      const lap = (key: "switch" | "agg" | "keys" | "settle") => {
        const now = Date.now();
        ph[key] += now - t;
        t = now;
      };
      await page.getByLabel("Model", { exact: true }).selectOption({ label: model });
      await openTensor(page, tensors[k % tensors.length]!);
      lap("switch");
      const agg = ["mean", "mean_abs", "rms", "min", "max"][k % 5]!;
      await page.getByTestId("agg-select").selectOption(agg);
      await settled(page, 300_000);
      lap("agg");
      const cv = page.getByTestId("overview-canvas");
      await cv.focus();
      for (const key of keys) {
        await page.keyboard.press(key);
        actions++;
      }
      lap("keys");
      await settled(page, 300_000);
      lap("settle");
      ph.n++;
      actions += 3;
      k++;
    }
    maxGap = Math.max(maxGap, Date.now() - lastIter);
    await sample(SOAK_MIN, actions);
    out.soak_max_iteration_gap_ms = maxGap;
    expect(maxGap, "soak iterations must be continuous (host suspended?)").toBeLessThan(60_000);
    out.soak_samples = samples;
    out.soak_actions = actions;
    out.soak_tensor_switches = k;
    const first = samples[1] ?? samples[0]!;
    const last = samples.at(-1)!;
    out.js_heap_growth_mib = Math.round((last.js_heap_mib! - first.js_heap_mib!) * 10) / 10;
    out.backend_rss_growth_mib = Math.round((last.backend_rss_mib! - first.backend_rss_mib!) * 10) / 10;
    out.max_js_heap_mib = Math.max(...samples.map((s) => s.js_heap_mib!));
    out.max_backend_rss_mib_sampled = Math.max(...samples.map((s) => s.backend_rss_mib!));
    out.max_decoded_cache_mib = Math.max(...samples.map((s) => s.decoded_cache_mib!).filter((v) => !Number.isNaN(v)));
    out.max_peak_inflight = Math.max(...samples.map((s) => s.peak_inflight!).filter((v) => !Number.isNaN(v)));
    expect(Number(out.max_decoded_cache_mib)).toBeLessThanOrEqual(64);
    expect(Number(out.max_peak_inflight)).toBeLessThanOrEqual(6);
    expect(blocked).toEqual([]);
    expect(problems).toEqual([]);
    out.verdict = "PASS";
  } finally {
    if (OUT) writeFileSync(OUT, JSON.stringify(out, null, 2));
    await server.stop();
  }
});
