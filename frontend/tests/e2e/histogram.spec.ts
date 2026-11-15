// ROADMAP P0.2 / ADR-021: the Histogram tab against a real `weightlab start`, offline. Expectations come from the
// fixture's own bytes (support.readTensor): counts, extrema and special values are recomputed here, and every bin
// count is recounted from the file using the edges the job returned, so the chart and table are checked against
// the checkpoint, not against the API. Also: the 1-D tensor tab, NaN/±Inf counted but never binned, keyboard
// readout, table view, light and dark axe, and FILE_CHANGED with no stale histogram after the file changes.
import AxeBuilder from "@axe-core/playwright";
import { appendFileSync, utimesSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { copyFixture, expectNoToken, loopbackOnly, raw, readTensor, startServer, tempDir, watchConsole } from "./support";

const SHOTS = process.env.WEIGHTLAB_SCREENSHOTS;
async function shot(page: Page, name: string) {
  if (SHOTS) await page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage: true });
}
async function axe(page: Page, label: string) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
  expect(r.violations.map((v) => `${label}: ${v.id} (${v.impact}) ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
}

const Q = "demo.layers.0.self_attn.q_proj.weight"; // 257 x 263 F32
const NF = "demo.special_nonfinite.weight"; // 2 x 4 F32 with NaN / ±Inf
const VEC = "demo.vector.weight"; // 6 F32

interface Hist {
  counts: { elements: number; finite: number; zeros: number; negatives: number; nan: number; pos_inf: number; neg_inf: number };
  min: number | null;
  max: number | null;
  bins: { edges: number[]; counts: number[]; below: number; above: number } | null;
}

test.describe.configure({ mode: "serial" });
const cleanup: (() => Promise<void>)[] = [];
test.afterEach(async () => {
  for (const stop of cleanup.splice(0)) await stop();
});

async function openHistogram(page: Page, key: string) {
  await page.getByLabel("Search tensor names").fill(key);
  await page.getByRole("button", { name: key, exact: true }).click();
  await expect(page.getByTestId("inspector-key")).toHaveText(key);
  await page.getByRole("tab", { name: "Histogram" }).click();
}

/** Clicks Compute and returns the histogram of the job it started, read back from the completed job. */
async function compute(page: Page, port: number, token: string): Promise<Hist> {
  const submitted = page.waitForResponse((r) => r.url().endsWith("/api/v1/jobs") && r.request().method() === "POST");
  await page.getByTestId("hist-compute").click();
  const job = (await (await submitted).json()) as { job_id: string };
  await expect(page.getByTestId("hist-result")).toBeVisible();
  const r = await raw(port, "GET", `/api/v1/jobs/${job.job_id}`, { Authorization: `Bearer ${token}` });
  const body = JSON.parse(r.body) as { state: string; result: { histogram: Hist } };
  expect(body.state).toBe("completed");
  return body.result.histogram;
}

/** Recounts the file's values into `edges` with ADR-021's rule: [e_i, e_i+1), the last bin closed. */
function recount(values: number[], edges: number[]): { counts: number[]; below: number; above: number } {
  const n = edges.length - 1;
  const counts = new Array<number>(n).fill(0);
  let below = 0;
  let above = 0;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < edges[0]!) below++;
    else if (v > edges[n]!) above++;
    else {
      let i = 0;
      while (i < n - 1 && v >= edges[i + 1]!) i++;
      counts[i]!++;
    }
  }
  return { counts, below, above };
}

test("exact histogram: counts recomputed from the file, special values, table, keyboard, axe, FILE_CHANGED", async ({ browser }) => {
  test.setTimeout(240_000);
  const models = tempDir("hist-models");
  const base = copyFixture("tiny_base.safetensors", models);
  const server = await startServer(tempDir("hist-data"));
  cleanup.push(() => server.stop());
  const context = await browser.newContext();
  const blocked = await loopbackOnly(context);
  const page = await context.newPage();
  const problems = watchConsole(page);

  await page.goto(server.url);
  await expectNoToken(page, server.token);
  await page.getByLabel("New project name").fill("Histograms");
  await page.getByRole("button", { name: "Create project" }).click();
  await page.getByLabel(/Model path/).fill(base);
  await page.getByLabel(/Selected root/).fill(models);
  await page.getByRole("button", { name: "Import model" }).click();
  await expect(page.getByText(`Imported “tiny_base.safetensors”`)).toBeVisible();
  await page.getByRole("listitem").filter({ hasText: "tiny_base.safetensors" }).getByRole("button", { name: "Explore tensors" }).click();

  // 1. A 2-D tensor: min-max bins. Everything checked against the file's bytes.
  const q = readTensor(base, Q).values as number[];
  await openHistogram(page, Q);
  await page.getByTestId("hist-bins").selectOption("32");
  const h = await compute(page, server.port, server.token);
  const finite = q.filter(Number.isFinite);
  expect(h.counts.elements).toBe(257 * 263);
  expect(h.counts.finite).toBe(finite.length);
  expect(h.counts.zeros).toBe(finite.filter((v) => v === 0).length);
  expect(h.counts.negatives).toBe(finite.filter((v) => v < 0).length);
  expect(h.min).toBe(Math.min(...finite));
  expect(h.max).toBe(Math.max(...finite));
  expect(h.bins!.edges[0]).toBe(h.min);
  expect(h.bins!.edges.at(-1)).toBe(h.max);
  expect(recount(q, h.bins!.edges)).toEqual({ counts: h.bins!.counts, below: 0, above: 0 });
  await expect(page.getByTestId("hist-counts")).toContainText(String(h.min));
  await expect(page.getByTestId("hist-result")).toContainText(`exact, all 67,591 elements`);

  // The chart draws one bar per nonzero bin; the keyboard readout and the table state the same exact counts.
  const nonzero = h.bins!.counts.filter((n) => n > 0).length;
  await expect(page.getByTestId("hist-values").locator("rect.hist-bar")).toHaveCount(nonzero);
  await page.getByTestId("hist-values").focus();
  await page.keyboard.press("Home");
  await expect(page.getByTestId("hist-values-readout")).toContainText(`[${h.bins!.edges[0]}, ${h.bins!.edges[1]}): ${h.bins!.counts[0]!.toLocaleString("en-US")}`);
  await page.getByTestId("hist-table-toggle").check();
  const rows = page.getByTestId("hist-values-table").locator("tbody tr");
  await expect(rows).toHaveCount(h.bins!.counts.length);
  await expect(rows.nth(5).locator("td").nth(1)).toHaveText(h.bins!.counts[5]!.toLocaleString("en-US"));
  await page.getByLabel("Log counts").check();
  await page.getByLabel("Magnitude (log₂ |x|)").check();
  await expect(page.getByTestId("hist-magnitude")).toBeVisible();
  await shot(page, "hist-01-magnitude");
  await page.getByLabel("Values").check();
  await shot(page, "hist-02-values-log");
  await axe(page, "histogram light");
  await page.emulateMedia({ colorScheme: "dark" });
  await axe(page, "histogram dark");
  await shot(page, "hist-03-values-dark");
  await page.emulateMedia({ colorScheme: "light" });

  // 2. Explicit range: values outside [lo, hi] are counted as below / above, never dropped.
  await page.getByTestId("hist-range").selectOption("explicit");
  await page.getByLabel("Lower bound (lo)").fill("-0.01");
  await page.getByLabel("Upper bound (hi)").fill("0.01");
  const e = await compute(page, server.port, server.token);
  expect(e.bins!.edges[0]).toBe(-0.01);
  expect(e.bins!.edges.at(-1)).toBe(0.01);
  expect(recount(q, e.bins!.edges)).toEqual({ counts: e.bins!.counts, below: e.bins!.below, above: e.bins!.above });
  expect(e.bins!.below + e.bins!.above).toBeGreaterThan(0);
  await expect(page.getByTestId("hist-counts")).toContainText(`Below -0.01${e.bins!.below.toLocaleString("en-US")}`);
  await expect(page.getByTestId("hist-counts")).toContainText(`Above 0.01${e.bins!.above.toLocaleString("en-US")}`);

  // 3. Non-finite values are counted and never binned.
  const nf = readTensor(base, NF).values as number[];
  await openHistogram(page, NF);
  const s = await compute(page, server.port, server.token);
  expect(s.counts.nan).toBe(nf.filter(Number.isNaN).length);
  expect(s.counts.pos_inf).toBe(nf.filter((v) => v === Infinity).length);
  expect(s.counts.neg_inf).toBe(nf.filter((v) => v === -Infinity).length);
  expect(s.counts.nan + s.counts.pos_inf + s.counts.neg_inf).toBeGreaterThan(0);
  if (s.bins) expect(s.bins.counts.reduce((a, b) => a + b, 0) + s.bins.below + s.bins.above).toBe(s.counts.finite);
  await expect(page.getByTestId("hist-counts").getByText("NaN")).toBeVisible();
  await shot(page, "hist-04-nonfinite");

  // 4. A 1-D tensor has the Histogram tab too (no overview); then its file changes: FILE_CHANGED, no stale chart.
  await openHistogram(page, VEC);
  await expect(page.getByRole("tab", { name: "Overview" })).toHaveCount(0);
  await compute(page, server.port, server.token);
  appendFileSync(base, Buffer.alloc(8));
  utimesSync(base, new Date(), new Date(Date.now() + 5_000));
  await page.getByRole("tab", { name: "Exact ROI" }).click();
  await page.getByRole("tab", { name: "Histogram" }).click();
  await page.getByTestId("hist-compute").click();
  await expect(page.getByText("Model files changed on disk")).toBeVisible();
  await expect(page.getByTestId("hist-result")).toHaveCount(0); // the earlier histogram described the old bytes
  await shot(page, "hist-05-file-changed");

  expect(problems.filter((p) => !/409|FILE_CHANGED|status of 4\d\d/.test(p))).toEqual([]);
  expect(blocked).toEqual([]);
  await context.close();
});
