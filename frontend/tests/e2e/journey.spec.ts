// The Phase 1 Task 3 user journey against a real `weightlab start`, with every non-loopback request blocked:
// first launch, token bootstrap, project, import, search/filter/sort/page, inspector, restart persistence,
// error states (nonexistent path, changed, disconnected, relink) and project deletion leaving files untouched.
import { appendFileSync, mkdirSync, renameSync, utimesSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import {
  bigModel,
  copyFixture,
  expectNoToken,
  loopbackOnly,
  readHeader,
  sha256,
  startServer,
  tempDir,
  watchConsole,
  type Server,
} from "./support";

const SHOTS = process.env.WEIGHTLAB_SCREENSHOTS;
async function shot(page: Page, name: string) {
  if (SHOTS) await page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage: true });
}

const KNOWN = "demo.layers.0.self_attn.q_proj.weight";

test.describe.configure({ mode: "serial" });
const cleanup: (() => Promise<void>)[] = [];
test.afterEach(async () => {
  for (const stop of cleanup.splice(0)) await stop(); // never leave a server running after a failure
});

test("first launch to inspected tensor, restart, error states and deletion", async ({ browser }) => {
  const dataDir = tempDir("data");
  const models = tempDir("models");
  const tiny = copyFixture("tiny_base.safetensors", models);
  const bigDir = join(models, "big");
  mkdirSync(bigDir);
  const bigPath = join(bigDir, "big.safetensors");
  const bigTensors = bigModel(bigPath, 500); // 3,001 tensors
  const header = readHeader(tiny);
  const expected = header[KNOWN]!;

  let server: Server = await startServer(dataDir);
  cleanup.push(() => server.stop());
  const context = await browser.newContext();
  const blocked = await loopbackOnly(context);
  const page = await context.newPage();
  const problems = watchConsole(page);
  // Measure every catalog page the browser receives. The body is read inside the route handler, before it is
  // handed to the page (a `response` listener can lose bodies Chromium has already evicted).
  const tensorPages: number[] = [];
  await page.route(/\/api\/v1\/models\/[^/]+\/tensors\?/, async (route) => {
    const response = await route.fetch();
    const body = await response.text();
    if (response.ok()) tensorPages.push((JSON.parse(body) as { items: unknown[] }).items.length);
    await route.fulfill({ response, body });
  });

  // 1-3. Launch link -> token moved out of the address bar.
  await page.goto(server.url);
  await expect(page.getByRole("heading", { name: "Explore LLM weights on your computer." })).toBeVisible();
  await expectNoToken(page, server.token);
  expect(await page.evaluate(() => sessionStorage.getItem("weightlab.session.token"))).toBe(server.token);
  await expect(page.getByRole("status", { name: "Backend: Connected" })).toBeVisible();
  await shot(page, "01-welcome");

  // 4. Create a project.
  await page.getByLabel("New project name").fill("Research");
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name: "Research", level: 2 })).toBeVisible();

  // 5. Import tiny_base.safetensors in place.
  await page.getByLabel(/Model path/).fill(tiny);
  await page.getByRole("button", { name: /Use the folder containing/ }).click();
  await expect(page.getByLabel(/Selected root/)).toHaveValue(models);
  await page.getByRole("button", { name: "Import model" }).click();
  await expect(page.getByText(`Imported “tiny_base.safetensors”`)).toBeVisible();
  await expect(page.getByText(`${Object.keys(header).length} tensors, status ready`)).toBeVisible();
  await shot(page, "02-workspace-imported");

  // 6-7. Find the known tensor by search and check it against the header read by this test.
  await page.getByRole("button", { name: "Explore tensors" }).first().click();
  await page.getByLabel("Search tensor names").fill("q_proj");
  await expect(page.getByText(/Rows 1–1 of 1 matching tensors/)).toBeVisible();
  await page.getByRole("button", { name: KNOWN }).click();
  const numel = expected.shape.reduce((a, b) => a * b, 1);
  await expect(page.getByTestId("inspector-key")).toHaveText(KNOWN);
  await expect(page.getByTestId("inspector-shape")).toHaveText(expected.shape.join(" × "));
  await expect(page.getByTestId("inspector-dtype")).toHaveText(expected.dtype);
  await expect(page.getByTestId("inspector-rank")).toHaveText(String(expected.shape.length));
  await expect(page.getByTestId("inspector-numel")).toHaveText(numel.toLocaleString("en-US"));
  await shot(page, "03-explore-inspector");

  // 8. A catalog of 3,001 tensors: filters, server-side sorting and paging, bounded pages only.
  await page.getByRole("button", { name: "Workspace" }).click();
  await page.getByLabel(/Model path/).fill(bigPath);
  await page.getByLabel(/Selected root/).fill(bigDir);
  await page.getByLabel("Display name").fill("Synthetic 3k");
  await page.getByRole("button", { name: "Import model" }).click();
  await expect(page.getByText(`Imported “Synthetic 3k”`)).toBeVisible();
  await page.getByRole("listitem").filter({ hasText: "Synthetic 3k" }).getByRole("button", { name: "Explore tensors" }).click();
  await expect(page.getByText(`Rows 1–50 of ${bigTensors.length.toLocaleString("en-US")} tensors`)).toBeVisible();
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByText(/Rows 51–100 of 3,001/)).toBeVisible();
  await page.getByRole("columnheader", { name: /Params/ }).getByRole("button").click(); // ascending
  await page.getByRole("columnheader", { name: /Params/ }).getByRole("button").click(); // descending
  await expect(page.getByRole("columnheader", { name: /Params/ })).toHaveAttribute("aria-sort", "descending");
  await expect(page.getByText(/Rows 1–50 of 3,001 tensors, sorted by parameter count \(descending\)/)).toBeVisible();
  const largest = Math.max(...bigTensors.map((t) => t.shape.reduce((a, b) => a * b, 1)));
  await expect(page.locator("tbody tr").first().locator("td").nth(4)).toHaveText(largest.toLocaleString("en-US"));
  await page.getByLabel("Dtype").selectOption("BF16");
  await page.getByLabel("Layer").fill("7");
  const bf16InLayer7 = bigTensors.filter((t) => t.dtype === "BF16" && t.key.startsWith("model.layers.7.")).length;
  await expect(page.getByText(`Rows 1–${bf16InLayer7} of ${bf16InLayer7} matching tensors`)).toBeVisible();
  await page.getByRole("button", { name: "Clear filters" }).click();
  await page.getByLabel(/Rows per page/).selectOption("200");
  await expect(page.getByText(/Rows 1–200 of 3,001/)).toBeVisible();
  await page.getByRole("button", { name: "Last" }).click();
  await expect(page.getByText(/Rows 3,001–3,001 of 3,001/)).toBeVisible();
  await shot(page, "04-catalog-3k");
  expect(tensorPages.length).toBeGreaterThan(5);
  expect(Math.max(...tensorPages)).toBeLessThanOrEqual(200); // never the whole catalog

  // 9-10. Restart with the same data directory: a new token, the old one is refused, data restored.
  const oldToken = server.token;
  await server.stop();
  await expect(page.getByRole("status", { name: "Backend: Server unavailable" })).toBeVisible({ timeout: 30_000 });
  server = await startServer(dataDir);
  expect(server.token).not.toBe(oldToken);
  await page.goto(server.url.replace(server.token, oldToken)); // stale link from the previous launch
  await expect(page.getByRole("heading", { name: "This WeightLab session has ended" })).toBeVisible();
  await page.goto(server.url);
  await expectNoToken(page, server.token);
  await expect(page.getByRole("heading", { name: "Research", level: 2 })).toBeVisible();
  await expect(page.getByText("SafeTensors (single file) · 9 tensors")).toBeVisible();
  await expect(page.getByText("SafeTensors (single file) · 3,001 tensors")).toBeVisible();

  // 11a. Nonexistent path.
  await page.getByLabel(/Model path/).fill(join(models, "missing.safetensors"));
  await page.getByLabel(/Selected root/).fill(models);
  await page.getByRole("button", { name: "Import model" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Path not found" })).toContainText("NOT_FOUND");
  await expect(page.getByRole("alert").filter({ hasText: "Path not found" })).toContainText(/Request req_/);
  await shot(page, "05-error-not-found");

  // 11b. Changed model: tiny_base gets an appended byte and a new mtime.
  appendFileSync(tiny, Buffer.from([0]));
  utimesSync(tiny, new Date(), new Date(Date.now() + 5000));
  // 11c. Disconnected model: the big model's folder is moved away.
  const moved = join(models, "big-moved");
  renameSync(bigDir, moved);
  await page.reload();
  const tinyItem = page.getByRole("listitem").filter({ hasText: "tiny_base.safetensors" });
  const bigItem = page.getByRole("listitem").filter({ hasText: "Synthetic 3k" });
  await expect(tinyItem.locator(".badge")).toHaveText("▲ Changed");
  await expect(bigItem.locator(".badge")).toHaveText("○ Disconnected");
  await shot(page, "06-changed-disconnected");

  // Relink the changed file is refused; relinking the moved one restores it.
  await tinyItem.getByRole("button", { name: "Relink…" }).click();
  await tinyItem.getByLabel(/New model path/).fill(tiny);
  await tinyItem.getByLabel(/New selected root/).fill(models);
  await tinyItem.getByRole("button", { name: "Relink", exact: true }).click();
  await expect(tinyItem.getByRole("alert")).toContainText("FILE_CHANGED");
  await expect(tinyItem.getByRole("alert")).toContainText("These files are not the imported ones");
  await bigItem.getByRole("button", { name: "Relink…" }).click();
  await bigItem.getByLabel(/New model path/).fill(join(moved, "big.safetensors"));
  await bigItem.getByLabel(/New selected root/).fill(moved);
  await bigItem.getByRole("button", { name: "Relink", exact: true }).click();
  await expect(bigItem.locator(".badge")).toHaveText("● Ready");

  // Explore shows the changed-model notice.
  await tinyItem.getByRole("button", { name: "Explore tensors" }).click();
  await expect(page.locator(".notice-title", { hasText: "Files changed since import" })).toBeVisible();

  // 12. Delete the project: source checkpoint bytes untouched.
  const before = { tiny: sha256(tiny), big: sha256(join(moved, "big.safetensors")) };
  await page.getByRole("button", { name: "Workspace" }).click();
  await page.getByRole("button", { name: "Delete project…" }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toContainText("Your checkpoint files are not deleted or modified.");
  await shot(page, "07-delete-confirm");
  await dialog.getByRole("button", { name: "Delete project" }).click();
  await expect(page.getByRole("heading", { name: "Explore LLM weights on your computer." })).toBeVisible();
  expect({ tiny: sha256(tiny), big: sha256(join(moved, "big.safetensors")) }).toEqual(before);

  await server.stop();
  expect(server.stderr()).not.toContain(server.token);
  expect(blocked).toEqual([]); // the workflow made no non-loopback request
  expect(problems).toEqual([]); // no CSP violation, page error or browser dialog
  await context.close();
});
