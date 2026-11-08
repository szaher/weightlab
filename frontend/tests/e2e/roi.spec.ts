// Phase 1 Task 4A: the exact ROI explorer against a real `weightlab start`, offline (every non-loopback request
// blocked). Expected values are decoded by this test straight from the fixture bytes (support.readTensor), never
// taken from the API. Covers: heatmap of a 257 x 263 tensor, cell selection by pointer and keyboard, pan, zoom,
// jump, Values tab and exact copy, non-finite cells, rank-0/1/3 tensors, 64-bit integers, -0.0, empty tensors,
// request budget, no deferred endpoints, axe, CSP and FILE_CHANGED (no stale values after the file changes).
import AxeBuilder from "@axe-core/playwright";
import { appendFileSync, utimesSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page, type Request } from "@playwright/test";
import { copyFixture, expectNoToken, loopbackOnly, raw, readTensor, sameValue, startServer, tempDir, watchConsole, type Server } from "./support";

const SHOTS = process.env.WEIGHTLAB_SCREENSHOTS;
async function shot(page: Page, name: string) {
  if (SHOTS) await page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage: true });
}

async function axe(page: Page, label: string) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
  expect(r.violations.map((v) => `${label}: ${v.id} (${v.impact}) ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
}

const Q = "demo.layers.0.self_attn.q_proj.weight"; // 257 x 263 F32

test.describe.configure({ mode: "serial" });
const cleanup: (() => Promise<void>)[] = [];
test.afterEach(async () => {
  for (const stop of cleanup.splice(0)) await stop();
});

async function openTensor(page: Page, key: string) {
  await page.getByLabel("Search tensor names").fill(key);
  await page.getByRole("button", { name: key, exact: true }).click();
  await expect(page.getByTestId("inspector-key")).toHaveText(key);
  // 2D tensors open on the whole-tensor Overview (PR #8); this suite exercises the Exact ROI tab.
  const tab = page.getByRole("tab", { name: "Exact ROI" });
  if ((await tab.count()) > 0) await tab.click();
}

async function clickCell(page: Page, i: number, j: number) {
  const canvas = page.getByTestId("heatmap-canvas");
  await canvas.scrollIntoViewIfNeeded();
  const box = (await canvas.boundingBox())!;
  const cell = Number(await canvas.getAttribute("data-cell-px"));
  const left = Number(await canvas.getAttribute("data-left"));
  const top = Number(await canvas.getAttribute("data-top"));
  await page.mouse.click(box.x + left + (j + 0.5) * cell, box.y + top + (i + 0.5) * cell);
}

async function selectedValue(page: Page): Promise<string> {
  return (await page.getByTestId("selected-value").textContent()) ?? "";
}

test("exact ROI explorer: heatmap, navigation, values, slicing, precision and failure states", async ({ browser }) => {
  test.setTimeout(300_000);
  const dataDir = tempDir("roi-data");
  const models = tempDir("roi-models");
  const base = copyFixture("tiny_base.safetensors", models);
  const precision = copyFixture("tiny_precision_a.safetensors", models);
  const edge = copyFixture("tiny_edge.safetensors", models);
  const dtypes = copyFixture("tiny_dtypes.safetensors", models);
  const q = readTensor(base, Q);
  const qAt = (r: number, c: number) => q.values[r * 263 + c]!;

  const server: Server = await startServer(dataDir);
  cleanup.push(() => server.stop());
  const context = await browser.newContext({ permissions: ["clipboard-read", "clipboard-write"] });
  const blocked = await loopbackOnly(context);
  const page = await context.newPage();
  const problems = watchConsole(page);
  const apiCalls: { path: string; body: unknown }[] = [];
  page.on("request", (r: Request) => {
    const u = new URL(r.url());
    if (u.pathname.startsWith("/api/")) apiCalls.push({ path: u.pathname, body: r.postData() ? JSON.parse(r.postData()!) : null });
  });

  // 1-3. Launch, create a project, import the synthetic fixture in place (through the UI).
  await page.goto(server.url);
  await expectNoToken(page, server.token);
  await page.getByLabel("New project name").fill("ROI");
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name: "ROI", level: 2 })).toBeVisible();
  await page.getByLabel(/Model path/).fill(base);
  await page.getByLabel(/Selected root/).fill(models);
  await page.getByRole("button", { name: "Import model" }).click();
  await expect(page.getByText(`Imported “tiny_base.safetensors”`)).toBeVisible();
  // The other fixtures through the API (same server, same project).
  const auth = { Authorization: `Bearer ${server.token}`, "Content-Type": "application/json" };
  const project = (JSON.parse((await raw(server.port, "GET", "/api/v1/projects", auth)).body) as { id: string }[])[0]!.id;
  for (const f of [precision, edge, dtypes]) {
    const r = await raw(server.port, "POST", `/api/v1/projects/${project}/models`, auth, JSON.stringify({ entry_path: f, selected_root: models }));
    expect(r.status).toBe(201);
  }
  await page.reload(); // pick up the models imported through the API (the token stays in this tab's session)

  // 4-5. Select the 257 x 263 tensor: an exact 64 x 64 window, labelled as a window.
  await page.getByRole("listitem").filter({ hasText: "tiny_base.safetensors" }).getByRole("button", { name: "Explore tensors" }).click();
  await openTensor(page, Q);
  await expect(page.getByTestId("roi-bounds")).toContainText("Rows [0, 64) of 257");
  await expect(page.getByTestId("roi-bounds")).toContainText("Columns [0, 64) of 263");
  await expect(page.getByTestId("roi-coverage-note")).toContainText("4,096 of 67,591 cells");
  await expect(page.getByTestId("roi-coverage-note")).toContainText("not the whole tensor");
  await expect(page.getByTestId("roi-precision")).toContainText("coverage: complete");
  await expect(page.getByTestId("roi-precision")).toContainText("numeric precision: lossless");
  await expect(page.getByTestId("heatmap-canvas")).toBeVisible();

  // 6-7. Select a known coordinate by pointer; the value equals the fixture's bytes.
  await clickCell(page, 5, 7);
  await expect(page.getByTestId("selected-coords")).toHaveText("5, 7");
  expect(sameValue(await selectedValue(page), qAt(5, 7))).toBe(true);
  await expect(page.getByTestId("selected-dtype")).toHaveText("F32");
  await shot(page, "roi-01-heatmap-selected");

  // Keyboard: arrows move the selection; moving past the edge pans the exact window.
  await page.getByTestId("heatmap-canvas").focus();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowDown");
  await expect(page.getByTestId("selected-coords")).toHaveText("6, 8");
  expect(sameValue(await selectedValue(page), qAt(6, 8))).toBe(true);
  await page.keyboard.press("End"); // last column of the view (262): the window follows
  await expect(page.getByTestId("roi-bounds")).toContainText("Columns [199, 263) of 263");
  await expect(page.getByTestId("selected-coords")).toHaveText("6, 262");
  expect(sameValue(await selectedValue(page), qAt(6, 262))).toBe(true);

  // 8. Pan and zoom: bounds change by whole cells; zoom keeps the selected cell in view.
  await page.getByRole("button", { name: "Reset to origin" }).click();
  await expect(page.getByTestId("roi-bounds")).toContainText("Columns [0, 64)");
  await page.getByRole("button", { name: "Pan right by 32 columns" }).click();
  await expect(page.getByTestId("roi-bounds")).toContainText("Columns [32, 96)");
  await page.getByRole("button", { name: "Pan down by 32 rows" }).click();
  await expect(page.getByTestId("roi-bounds")).toContainText("Rows [32, 96)");
  await clickCell(page, 10, 20); // view (42, 52)
  await expect(page.getByTestId("selected-coords")).toHaveText("42, 52");
  await page.getByRole("button", { name: "Zoom in to 32 × 32 cells" }).click();
  await expect(page.getByTestId("roi-bounds")).toContainText("Rows [37, 69)");
  await expect(page.getByTestId("roi-bounds")).toContainText("Columns [42, 74)");
  await expect(page.getByTestId("selected-coords")).toHaveText("42, 52");
  expect(sameValue(await selectedValue(page), qAt(42, 52))).toBe(true);
  await page.getByRole("button", { name: "Zoom out to 64 × 64 cells" }).click();
  await expect(page.getByTestId("budget-note")).toContainText("4,096 cells (the server's roi_max_cells setting");
  await expect(page.getByRole("button", { name: "Zoom out" })).toBeDisabled();
  // Jump to the last cell.
  await page.getByLabel("Row", { exact: true }).fill("256");
  await page.getByLabel("Column", { exact: true }).fill("262");
  await page.getByRole("button", { name: "Go to cell" }).click();
  await expect(page.getByTestId("roi-bounds")).toContainText("Rows [193, 257)");
  await expect(page.getByTestId("selected-coords")).toHaveText("256, 262");
  expect(sameValue(await selectedValue(page), qAt(256, 262))).toBe(true);
  await page.getByLabel("Row", { exact: true }).fill("257");
  await page.getByRole("button", { name: "Go to cell" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Row 257 is outside 0–256" })).toBeVisible();

  // Transpose: rows and columns swap, the same element stays selected.
  await page.getByLabel("Transpose").check();
  await expect(page.getByTestId("roi-bounds")).toContainText("of 263");
  await expect(page.getByTestId("selected-coords")).toHaveText("262, 256");
  await expect(page.getByTestId("selected-index")).toHaveText("[256, 262]");
  expect(sameValue(await selectedValue(page), qAt(256, 262))).toBe(true);
  await page.getByLabel("Transpose").uncheck();
  await axe(page, "heatmap");

  // 9-10. Values tab: exact text and an exact copy.
  await page.getByRole("tab", { name: "Values" }).click();
  const row = page.getByTestId("value-256-262");
  await expect(row).toBeVisible();
  await row.getByRole("button", { name: "Copy exact value at row 256, column 262", exact: true }).click();
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(sameValue(copied, qAt(256, 262))).toBe(true);
  await expect(row).toContainText(copied);
  await shot(page, "roi-02-values-table");
  await axe(page, "values");
  await page.getByRole("tab", { name: "Heatmap" }).click();

  // Non-finite cells: a pattern and a text state, never a fabricated value.
  const nf = readTensor(base, "demo.special_nonfinite.weight");
  await openTensor(page, "demo.special_nonfinite.weight");
  await expect(page.getByTestId("legend")).toContainText("NaN: 1");
  await expect(page.getByTestId("legend")).toContainText("+Infinity: 1");
  await expect(page.getByTestId("legend")).toContainText("−Infinity: 1");
  for (let k = 0; k < nf.values.length; k++) {
    const [r, c] = [Math.floor(k / 4), k % 4];
    await clickCell(page, r, c);
    const text = await selectedValue(page);
    expect(sameValue(text, nf.values[k]!), `cell ${r},${c}: ${text}`).toBe(true);
  }
  await clickCell(page, 0, 1);
  await shot(page, "roi-03-nonfinite");
  await page.getByRole("tab", { name: "Values" }).click();
  await page.getByLabel(/Only non-finite cells/).check();
  await expect(page.getByTestId("values-table").locator("tbody tr")).toHaveCount(3);
  await shot(page, "roi-04-nonfinite-values");
  await page.getByRole("tab", { name: "Heatmap" }).click();

  // Rank 3: axis selectors and fixed index; fixture expectation (axis 0 = 1, rows 1..3, cols 2..5) = [[42,43,44],[50,51,52]].
  const cube = readTensor(base, "demo.cube.weight");
  await openTensor(page, "demo.cube.weight");
  await expect(page.getByTestId("slice-controls")).toBeVisible();
  await expect(page.getByTestId("roi-bounds")).toContainText("Rows [0, 4) of 4");
  await page.getByTestId("fixed-0").fill("1");
  await expect(page.getByTestId("roi-coverage-note")).toContainText("Slice: axis 0 = 1");
  await clickCell(page, 1, 2);
  await expect(page.getByTestId("selected-value")).toHaveText("42.0");
  await expect(page.getByTestId("selected-index")).toHaveText("[1, 1, 2]");
  expect(sameValue(await selectedValue(page), cube.values[1 * 32 + 1 * 8 + 2]!)).toBe(true);
  await clickCell(page, 2, 4);
  await expect(page.getByTestId("selected-value")).toHaveText("52.0");
  await page.getByTestId("fixed-0").fill("2"); // out of range: nothing is read
  await expect(page.getByTestId("roi-view-error")).toContainText("index 2 is outside 0–1");
  await expect(page.getByTestId("heatmap-canvas")).toHaveCount(0);
  await page.getByTestId("fixed-0").fill("0");
  await page.getByLabel("Row axis").selectOption("0"); // rows = axis 0, columns = axis 2, axis 1 fixed
  await expect(page.getByTestId("fixed-1")).toHaveValue("0");
  await expect(page.getByTestId("roi-bounds")).toContainText("Rows [0, 2) of 2");
  await clickCell(page, 1, 3);
  await expect(page.getByTestId("selected-index")).toHaveText("[1, 0, 3]");
  expect(sameValue(await selectedValue(page), cube.values[1 * 32 + 0 * 8 + 3]!)).toBe(true);
  await shot(page, "roi-05-slice-controls");
  await axe(page, "slice");

  // Rank 0 and rank 1 and zero-size; constant region.
  await openTensor(page, "demo.scalar");
  await expect(page.getByTestId("roi-bounds")).toHaveText("Scalar (rank 0): the single exact value");
  await expect(page.getByTestId("selected-value")).toHaveText("3.5");
  const vec = readTensor(base, "demo.vector.weight");
  await openTensor(page, "demo.vector.weight");
  await expect(page.getByTestId("roi-bounds")).toContainText("Indices [0, 6) of 6");
  await clickCell(page, 0, 5);
  expect(sameValue(await selectedValue(page), vec.values[5]!)).toBe(true);
  await openTensor(page, "demo.zeros.weight");
  await expect(page.getByTestId("legend")).toContainText("Constant region: every coloured value here is 0.0");

  // 64-bit integers and signed zero (tiny_precision_a).
  await page.getByLabel("Model", { exact: true }).selectOption({ label: "tiny_precision_a.safetensors" });
  const i64 = readTensor(precision, "p.i64");
  await openTensor(page, "p.i64");
  await page.getByRole("tab", { name: "Values" }).click();
  for (let k = 0; k < i64.values.length; k++) {
    await expect(page.getByTestId(`value-0-${k}`).getByRole("button", { name: /^Select/ })).toHaveText(String(i64.values[k]));
  }
  await page.getByTestId("value-0-0").getByRole("button", { name: /^Copy/ }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe("-9223372036854775808");
  await expect(page.getByTestId("value-0-0")).toContainText("colour withheld");
  await page.getByRole("tab", { name: "Heatmap" }).click();
  await expect(page.getByTestId("legend")).toContainText("colour withheld: beyond ±(2^53 − 1), exact decimal text kept: 4");
  const u64 = readTensor(precision, "p.u64");
  await openTensor(page, "p.u64");
  await clickCell(page, 0, 2);
  await expect(page.getByTestId("selected-value")).toHaveText(String(u64.values[2])); // 18446744073709551615
  await shot(page, "roi-06-u64-exact");
  const f64 = readTensor(precision, "p.f64");
  await openTensor(page, "p.f64");
  await page.getByRole("tab", { name: "Values" }).click();
  for (let k = 0; k < f64.values.length; k++) {
    const t = (await page.getByTestId(`value-0-${k}`).getByRole("button", { name: /^Select/ }).textContent()) ?? "";
    expect(sameValue(t, f64.values[k]!), `p.f64[${k}] = ${t}`).toBe(true);
  }
  await expect(page.getByTestId("value-0-4").getByRole("button", { name: /^Select/ })).toHaveText("-0.0");
  await page.getByRole("tab", { name: "Heatmap" }).click();

  // BF16 (decoded from bits by this test).
  await page.getByLabel("Model", { exact: true }).selectOption({ label: "tiny_dtypes.safetensors" });
  const bf = readTensor(dtypes, "demo.bf16_specials");
  await openTensor(page, "demo.bf16_specials");
  for (let k = 0; k < bf.values.length; k++) {
    await clickCell(page, 0, k);
    expect(sameValue(await selectedValue(page), bf.values[k]!), `bf16[${k}]`).toBe(true);
  }

  // Zero-size tensor: an explanation, no request.
  await page.getByLabel("Model", { exact: true }).selectOption({ label: "tiny_edge.safetensors" });
  const before = apiCalls.filter((c) => c.path === "/api/v1/tensors/roi").length;
  await openTensor(page, "edge.empty_rows");
  await expect(page.getByTestId("roi-empty")).toContainText("no elements");
  expect(apiCalls.filter((c) => c.path === "/api/v1/tensors/roi").length).toBe(before);

  // Resource evidence: every ROI request within the budget; no deferred endpoint was called (tiles and jobs are
  // the overview's, implemented since PR #8 and covered by overview.spec.ts).
  const rois = apiCalls.filter((c) => c.path === "/api/v1/tensors/roi").map((c) => c.body as { roi: { r0: number; r1: number; c0: number; c1: number } });
  expect(rois.length).toBeGreaterThan(10);
  for (const r of rois) expect((r.roi.r1 - r.roi.r0) * (r.roi.c1 - r.roi.c0)).toBeLessThanOrEqual(4096);
  expect(apiCalls.filter((c) => /\/stats|\/exports/.test(c.path))).toEqual([]);

  // FILE_CHANGED: the checkpoint changes on disk; the next read is refused and no stale values remain visible.
  await page.getByLabel("Model", { exact: true }).selectOption({ label: "tiny_base.safetensors" });
  await openTensor(page, Q);
  await expect(page.getByTestId("heatmap-canvas")).toBeVisible();
  appendFileSync(base, Buffer.from([0]));
  utimesSync(base, new Date(), new Date(Date.now() + 5000));
  await page.getByRole("button", { name: "Pan right by 32 columns" }).click();
  const alert = page.getByRole("alert").filter({ hasText: "Model files changed on disk" });
  await expect(alert).toContainText("FILE_CHANGED");
  await expect(alert).toContainText("Re-import the model if it was modified; relink it in the Workspace if it was only moved.");
  await expect(page.getByTestId("heatmap-canvas")).toHaveCount(0);
  await expect(page.getByTestId("selected-value")).toHaveCount(0);
  await shot(page, "roi-07-file-changed");

  await server.stop();
  expect(server.stderr()).not.toContain(server.token);
  expect(blocked).toEqual([]);
  expect(problems).toEqual([]);
  await context.close();
});
