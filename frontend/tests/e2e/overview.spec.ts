// PR #8: the whole-tensor overview against a real `weightlab start`, offline (every non-loopback request blocked).
// Expectations come from this test's own generator (`fill`) and the independent byte decoder (support.readTensor),
// never from the API. Covers: a real pyramid job with progress and completion, exact aggregates checked against
// an independent block computation, zoom and pan, block selection opening Exact ROI on the same raw bounds with the
// exact values of the file, an over-budget region that is not truncated, a row-sampled preview with its seed and an
// explicit, cancelable exact build, the transposed-sampling limitation, a stale response that must not replace the
// current tensor, network concurrency and browser memory, FILE_CHANGED after a source mutation, and axe.
import AxeBuilder from "@axe-core/playwright";
import { utimesSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page, type Request } from "@playwright/test";
import { f32, loopbackOnly, raw, readTensor, sameValue, startServer, tempDir, watchConsole, writeF32, type Server } from "./support";

const SHOTS = process.env.WEIGHTLAB_SCREENSHOTS;
const EVIDENCE = process.env.WEIGHTLAB_UI_EVIDENCE; // optional JSON file for measured browser numbers
async function shot(page: Page, name: string) {
  if (SHOTS) await page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage: true });
}

const MID: [number, number] = [1024, 2304]; // 9 MiB F32: below pyramid_exact_max_mib, needs a pyramid from L4
const BIG: [number, number] = [32768, 4096]; // 512 MiB F32, sparse: above pyramid_exact_max_mib (64)
const midFill = (r: number, c: number): number => {
  if (r === 3 && c === 5) return NaN;
  if (r === 700 && c === 2000) return Infinity;
  return (((r * 7 + c * 13) % 101) - 50) / 8;
};
const bigFill = (r: number, c: number): number => (c < 32 ? r / 64 : 0);

test.describe.configure({ mode: "serial" });
const cleanup: (() => Promise<void>)[] = [];
test.afterEach(async () => {
  for (const stop of cleanup.splice(0)) await stop();
});

async function openTensor(page: Page, key: string) {
  await page.getByLabel("Search tensor names").fill(key);
  await page.getByRole("button", { name: key, exact: true }).click();
  await expect(page.getByTestId("inspector-key")).toHaveText(key);
}

async function canvasGeom(page: Page) {
  const cv = page.getByTestId("overview-canvas");
  await cv.scrollIntoViewIfNeeded();
  const box = (await cv.boundingBox())!;
  const n = async (a: string) => Number(await cv.getAttribute(a));
  return { box, ppe: await n("data-ppe"), top: await n("data-top"), left: await n("data-left"), ax: await n("data-axis-left"), ay: await n("data-axis-top"), level: await n("data-level") };
}

/** Clicks the overview at raw view element (r, c). */
async function clickRaw(page: Page, r: number, c: number) {
  const g = await canvasGeom(page);
  await page.mouse.click(g.box.x + g.ax + (c + 0.5 - g.left) * g.ppe, g.box.y + g.ay + (r + 0.5 - g.top) * g.ppe);
}

/** Mean |value| of the finite elements of raw block [r0, r1) x [c0, c1): the reference for a mean_abs cell. */
function meanAbs(r0: number, r1: number, c0: number, c1: number): number {
  let s = 0;
  let n = 0;
  for (let r = r0; r < r1; r++) {
    for (let c = c0; c < c1; c++) {
      const v = f32(midFill(r, c));
      if (Number.isFinite(v)) {
        s += Math.abs(v);
        n++;
      }
    }
  }
  return s / n;
}

test("whole-tensor overview: pyramid job, exact aggregates, drill-down, sampling, cancellation and stale data", async ({ browser }) => {
  test.setTimeout(420_000);
  const dataDir = tempDir("ov-data");
  const models = tempDir("ov-models");
  const file = join(models, "model.safetensors");
  writeF32(file, [
    { key: "mid.weight", shape: MID, fill: midFill },
    { key: "big.weight", shape: BIG, fill: bigFill, sparseRows: () => 32 },
  ]);
  const server: Server = await startServer(dataDir);
  cleanup.push(() => server.stop());
  const context = await browser.newContext();
  const blocked = await loopbackOnly(context);
  const page = await context.newPage();
  const problems = watchConsole(page);

  // Network bookkeeping: every tile request body, and (inside the page) the number of /tiles fetches that have
  // been issued and not yet settled, aborts included: the page's own request concurrency.
  const tileBodies: { tensor_id: string; tile: { level: number; x: number; y: number }; quality: { mode: string } }[] = [];
  page.on("request", (r: Request) => {
    if (new URL(r.url()).pathname === "/api/v1/tiles") tileBodies.push(JSON.parse(r.postData() ?? "{}"));
  });
  await page.addInitScript(() => {
    const w = window as unknown as { __tiles: { now: number; peak: number } };
    w.__tiles = { now: 0, peak: 0 };
    const orig = window.fetch.bind(window);
    window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
      const tile = String(input).endsWith("/api/v1/tiles");
      if (!tile) return orig(input, init);
      w.__tiles.now++;
      w.__tiles.peak = Math.max(w.__tiles.peak, w.__tiles.now);
      const p = orig(input, init);
      const settle = () => {
        w.__tiles.now--;
      };
      p.then(settle, settle);
      return p;
    };
  });
  const tilesPeak = () => page.evaluate(() => (window as unknown as { __tiles: { peak: number } }).__tiles.peak);

  // 1-3. Start, create a project in the UI, import the synthetic checkpoint; small settings so every branch runs.
  await page.goto(server.url);
  await page.getByLabel("New project name").fill("Overview");
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name: "Overview", level: 2 })).toBeVisible();
  await page.getByLabel(/Model path/).fill(file);
  await page.getByLabel(/Selected root/).fill(models);
  await page.getByRole("button", { name: "Import model" }).click();
  await expect(page.getByText(`Imported “model.safetensors”`)).toBeVisible();
  const auth = { Authorization: `Bearer ${server.token}`, "Content-Type": "application/json" };
  const patched = await raw(server.port, "PATCH", "/api/v1/settings", auth, JSON.stringify({ direct_tile_max_mib: 8, pyramid_exact_max_mib: 64 }));
  expect(patched.status).toBe(200);
  await page.reload();

  // 4-5. Explore the 1024 x 2304 tensor: the overview is the default view; the fit level is a direct exact read.
  await page.getByRole("listitem").filter({ hasText: "model.safetensors" }).getByRole("button", { name: "Explore tensors" }).click();
  await openTensor(page, "mid.weight");
  await expect(page.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByTestId("coverage-badge")).toHaveText("EXACT coverage");
  await expect(page.getByTestId("overview-badges")).toContainText("float32 visualization");
  await expect(page.getByTestId("overview-badges")).toContainText("F32 source");
  await expect(page.getByTestId("overview-facts")).toContainText("mean absolute value of the finite elements in its block");

  // 6-8. Level L4 needs the exact pyramid: a real job, its progress, then the exact overview.
  await page.getByTestId("level-select").selectOption("4");
  await expect(page.getByTestId("job-panel")).toBeVisible();
  await expect(page.getByTestId("building-badge")).toBeVisible();
  await expect(page.getByTestId("job-progress")).toBeVisible();
  await expect(page.getByTestId("job-panel")).toContainText(/read .* of ~9\.00 MiB/);
  await shot(page, "overview-01-building");
  await expect(page.getByTestId("job-panel")).toBeHidden({ timeout: 60_000 });
  await expect(page.getByTestId("coverage-badge")).toHaveText("EXACT coverage");
  await expect(page.getByTestId("coverage-text")).toContainText("every element of every block");
  await expect(page.getByTestId("overview-legend")).toContainText("visible-region scale");
  await expect(page.getByTestId("overview-legend")).toContainText("finite aggregate, block also holds NaN/±Inf (corner mark): 2");
  await shot(page, "overview-02-exact");

  // Exact aggregates against an independent computation (the NaN block keeps its finite aggregate).
  for (const [r, c] of [
    [165, 325],
    [3, 5],
    [1020, 2300],
  ] as const) {
    await clickRaw(page, r, c);
    const r0 = Math.floor(r / 16) * 16;
    const c0 = Math.floor(c / 16) * 16;
    const r1 = Math.min(MID[0], r0 + 16);
    const c1 = Math.min(MID[1], c0 + 16);
    await expect(page.getByTestId("selection-bounds")).toHaveText(`rows [${r0}, ${r1}) · columns [${c0}, ${c1})`);
    const text = (await page.getByTestId("overview-selection").textContent()) ?? "";
    const shown = Number(/≈ (\S+) \(float32 visualization\)/.exec(text)?.[1]);
    const ref = f32(meanAbs(r0, r1, c0, c1));
    expect(Math.abs(shown - ref) / ref, `block at (${r0}, ${c0}): shown ${shown}, reference ${ref}`).toBeLessThan(1e-5);
    if (r === 3) expect(text).toContain("the block also contains NaN");
  }

  // 9. Zoom (toward the selected block) and pan (drag): raw bounds move, the level follows in Auto.
  await page.getByTestId("level-select").selectOption("auto");
  const g0 = await canvasGeom(page);
  await page.getByRole("button", { name: "Zoom in" }).click();
  await expect.poll(async () => (await canvasGeom(page)).ppe).toBeCloseTo(g0.ppe * 2, 6);
  const g1 = await canvasGeom(page);
  await page.mouse.move(g1.box.x + g1.ax + 200, g1.box.y + g1.ay + 200);
  await page.mouse.down();
  await page.mouse.move(g1.box.x + g1.ax + 100, g1.box.y + g1.ay + 150, { steps: 5 });
  await page.mouse.up();
  await expect.poll(async () => (await canvasGeom(page)).left).toBeGreaterThan(g1.left);
  await expect(page.getByTestId("overview-visible")).not.toContainText("columns [0, 2304)");
  await page.getByRole("button", { name: "Fit" }).click();
  await expect(page.getByTestId("overview-visible")).toContainText("rows [0, 1024) × columns [0, 2304)");

  // 10-12. A block opens Exact ROI on exactly its raw bounds; values equal the file's bytes.
  await page.getByTestId("level-select").selectOption("4");
  await expect(page.getByTestId("coverage-badge")).toHaveText("EXACT coverage");
  await clickRaw(page, 165, 325);
  await page.getByTestId("inspect-exact").click();
  await expect(page.getByRole("tab", { name: "Exact ROI" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByTestId("roi-from-overview")).toContainText("rows [160, 176) × columns [320, 336)");
  await expect(page.getByTestId("roi-bounds")).toContainText("Rows [160, 176) of 1,024");
  await expect(page.getByTestId("roi-bounds")).toContainText("Columns [320, 336) of 2,304");
  const mid = readTensor(file, "mid.weight");
  const heat = page.getByTestId("heatmap-canvas");
  await heat.scrollIntoViewIfNeeded();
  const hb = (await heat.boundingBox())!;
  const cell = Number(await heat.getAttribute("data-cell-px"));
  const hl = Number(await heat.getAttribute("data-left"));
  const ht = Number(await heat.getAttribute("data-top"));
  for (const [i, j] of [
    [0, 0],
    [2, 3],
    [15, 15],
  ] as const) {
    await page.mouse.click(hb.x + hl + (j + 0.5) * cell, hb.y + ht + (i + 0.5) * cell);
    await expect(page.getByTestId("selected-coords")).toHaveText(`${160 + i}, ${320 + j}`);
    const v = (await page.getByTestId("selected-value").textContent()) ?? "";
    expect(sameValue(v, mid.values[(160 + i) * MID[1] + 320 + j]!), `ROI (${160 + i}, ${320 + j})`).toBe(true);
  }
  await shot(page, "overview-03-roi-drilldown");

  // Over-budget region (Shift + drag over 5 x 5 L4 blocks: 6,400 elements > roi_max_cells 4,096): not truncated,
  // a smaller window is offered instead.
  await page.getByRole("tab", { name: "Overview" }).click();
  await page.getByTestId("level-select").selectOption("4");
  await expect(page.getByTestId("coverage-badge")).toHaveText("EXACT coverage");
  const gd = await canvasGeom(page);
  const at = (r: number, c: number) => [gd.box.x + gd.ax + (c + 0.5 - gd.left) * gd.ppe, gd.box.y + gd.ay + (r + 0.5 - gd.top) * gd.ppe] as const;
  await page.keyboard.down("Shift");
  await page.mouse.move(...at(2, 2));
  await page.mouse.down();
  await page.mouse.move(...at(70, 70), { steps: 8 });
  await page.mouse.up();
  await page.keyboard.up("Shift");
  await expect(page.getByTestId("selection-bounds")).toHaveText("rows [0, 80) · columns [0, 80)");
  await expect(page.getByTestId("roi-too-big")).toContainText("6,400 elements");
  await expect(page.getByTestId("inspect-exact")).toBeDisabled();
  await page.getByTestId("shrink-selection").click();
  await expect(page.getByTestId("selection-bounds")).toHaveText("rows [0, 80) · columns [0, 40)");
  await expect(page.getByTestId("inspect-exact")).toBeEnabled();

  // 14a. Stale responses: the previous tensor's slow tile answers after the switch and must not be drawn.
  let delayed = 0;
  await page.route("**/api/v1/tiles", async (route) => {
    const body = JSON.parse(route.request().postData() ?? "{}") as { aggregation?: string };
    if (body.aggregation === "rms") {
      delayed++;
      await new Promise((r) => setTimeout(r, 2500));
    }
    await route.continue().catch(() => undefined); // aborted by the page meanwhile
  });
  await page.getByTestId("agg-select").selectOption("rms"); // not cached yet: real requests, held back 2.5 s
  await expect.poll(() => delayed).toBeGreaterThan(0); // the slow answers are on their way
  await openTensor(page, "big.weight");
  await expect(page.getByTestId("coverage-badge")).toHaveText("ROW-SAMPLED preview", { timeout: 30_000 });
  await page.waitForTimeout(3000); // the delayed answers have arrived (or been aborted) by now
  expect(delayed).toBeGreaterThan(0);
  await expect(page.getByTestId("overview-canvas")).toHaveAttribute("aria-label", /32,768 × 4,096 view/);
  await expect(page.getByTestId("coverage-badge")).toHaveText("ROW-SAMPLED preview");
  await page.unroute("**/api/v1/tiles");

  // 13. Row-sampled preview: labelled with rows, policy and seed; the seed can be changed.
  await expect(page.getByTestId("sampling-panel")).toContainText("This is a row-sampled preview, not exact");
  await expect(page.getByTestId("coverage-text")).toContainText(/only \d+ row\(s\) of each block were read \(evenly strided rows\)/);
  await shot(page, "overview-04-sampled");
  await page.getByTestId("seed-input").fill("7");
  await page.getByRole("button", { name: "Apply seed" }).click();
  await expect(page.getByTestId("coverage-text")).toContainText("seeded SplitMix64 (seed 7)");
  const seeded = tileBodies.filter((b) => (b as { quality: { seed?: number } }).quality.seed === 7);
  expect(seeded.length).toBeGreaterThan(0);
  // No exact scan was issued for the large tensor by zooming or changing settings.
  expect(tileBodies.filter((b) => b.quality.mode === "exact")).toEqual([]);

  // Transposed views cannot be row-sampled: the server's limitation is shown, with real alternatives.
  await page.getByLabel("Transpose").first().check();
  await expect(page.getByTestId("sampling-unsupported")).toContainText("request the exact pyramid or the untransposed view");
  await page.getByRole("button", { name: "Show the untransposed view" }).click();
  await expect(page.getByTestId("coverage-badge")).toHaveText("ROW-SAMPLED preview");

  // Explicit exact build: the I/O cost first, then a cancelable job; canceling marks nothing exact.
  await page.getByTestId("request-exact").click();
  await expect(page.getByTestId("confirm-exact")).toContainText("Estimated read: 512.0 MiB");
  await page.getByTestId("confirm-exact-build").click();
  await expect(page.getByTestId("job-panel")).toBeVisible();
  await expect(page.getByTestId("building-badge")).toBeVisible();
  const cancels: string[] = [];
  page.on("request", (r) => {
    if (/\/api\/v1\/jobs\/[^/]+\/cancel$/.test(new URL(r.url()).pathname)) cancels.push(r.url());
  });
  await page.getByTestId("job-cancel").click();
  await expect(page.getByTestId("job-panel")).toHaveAttribute("data-state", "canceled", { timeout: 30_000 });
  await expect(page.getByTestId("job-panel")).toContainText("Nothing was published");
  expect(cancels.length).toBe(1);
  await expect(page.getByTestId("coverage-badge")).toBeHidden();
  await page.getByRole("button", { name: "Back to the sampled preview" }).click();
  await expect(page.getByTestId("coverage-badge")).toHaveText("ROW-SAMPLED preview");

  // Bounded browser: at most 6 concurrent tile requests, and the JS heap after much panning and zooming.
  const cdp = await context.newCDPSession(page);
  await cdp.send("Performance.enable");
  const heap = async () => ((await cdp.send("Performance.getMetrics")).metrics.find((m) => m.name === "JSHeapUsedSize")?.value ?? 0) / 2 ** 20;
  const heapBefore = await heap();
  const t0 = Date.now();
  const cv = page.getByTestId("overview-canvas");
  await cv.focus();
  for (let k = 0; k < 4; k++) await page.keyboard.press("+");
  for (let k = 0; k < 24; k++) await page.keyboard.press(k % 6 < 3 ? "Shift+ArrowDown" : "Shift+ArrowRight");
  await page.waitForTimeout(1500);
  await page.evaluate(() => (globalThis as { gc?: () => void }).gc?.());
  const heapAfter = await heap();
  const panMs = Date.now() - t0;
  const peak = await tilesPeak();
  expect(peak).toBeLessThanOrEqual(6);
  expect(heapAfter).toBeLessThan(256);
  if (EVIDENCE) writeFileSync(EVIDENCE, JSON.stringify({ peak_concurrent_tile_requests: peak, tile_requests: tileBodies.length, js_heap_mib_before: heapBefore, js_heap_mib_after_pan_zoom: heapAfter, pan_zoom_sequence_ms: panMs }, null, 2));

  // 14b. Source mutation: the next tile request is refused (FILE_CHANGED); nothing stale is drawn.
  await openTensor(page, "mid.weight");
  await expect(page.getByTestId("coverage-badge")).toBeVisible();
  const later = new Date(Date.now() + 10_000);
  utimesSync(file, later, later);
  await page.getByTestId("agg-select").selectOption("max");
  await expect(page.getByText("Model files changed on disk")).toBeVisible();
  await expect(page.getByTestId("coverage-badge")).toBeHidden();

  // Accessibility of the overview.
  await page.getByRole("tab", { name: "Overview" }).focus();
  const a11y = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
  expect(a11y.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);

  expect(blocked).toEqual([]);
  expect(problems).toEqual([]);
});
