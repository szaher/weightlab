// The Compare journey (ADR-020, Alpha 2) against a real `weightlab start`, every non-loopback request blocked:
// import two checkpoints, create a comparison, run the analysis, read the layer x role grid and rankings, drill into
// a tensor and open it in Explore; numerical parity with `weightlab diff`; Class S policy and incompleteness; file
// changes and stale reports; refresh; cancel and retry; a 3,001-tensor report; keyboard and axe checks.
import AxeBuilder from "@axe-core/playwright";
import { spawnSync } from "node:child_process";
import { appendFileSync, copyFileSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { bigModel, copyFixture, loopbackOnly, raw, REPO, startServer, tempDir, watchConsole, writeF32, type Server } from "./support";

const SHOTS = process.env.WEIGHTLAB_SCREENSHOTS;
async function shot(page: Page, name: string) {
  if (SHOTS) await page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage: true });
}

test.describe.configure({ mode: "serial" });
const cleanup: (() => Promise<void>)[] = [];
test.afterEach(async () => {
  for (const stop of cleanup.splice(0)) await stop();
});

type Json = Record<string, unknown>;
function client(server: Server) {
  const auth = { Authorization: `Bearer ${server.token}`, "Content-Type": "application/json" };
  return async (method: string, path: string, body?: unknown): Promise<{ status: number; json: Json & Json[] }> => {
    const r = await raw(server.port, method, `/api/v1${path}`, auth, body === undefined ? undefined : JSON.stringify(body));
    return { status: r.status, json: JSON.parse(r.body || "null") as Json & Json[] };
  };
}

/** `weightlab diff` (the CLI) on the same files: the expected numbers. */
function cliReport(a: string, b: string, root: string): { tensors: { key: string; metrics: Record<string, { value: number | null }> }[]; complete: boolean } {
  const raw = process.env.WEIGHTLAB_CMD;
  const [cmd, ...args] = raw ? (JSON.parse(raw) as string[]) : ["uv", "run", "--locked", "--project", REPO, "weightlab"];
  const out = tempDir("cli");
  const r = spawnSync(cmd!, [...args, "diff", a, b, "--selected-root", root, "--out", out, "--quiet"], { cwd: REPO, encoding: "utf8" });
  expect([0, 3]).toContain(r.status);
  return JSON.parse(readFileSync(join(out, "report.json"), "utf8")) as ReturnType<typeof cliReport>;
}

async function openCompare(page: Page) {
  await page.getByRole("navigation", { name: "Main" }).getByRole("button", { name: "Compare" }).click();
  await expect(page.getByRole("heading", { name: /^Comparisons/ })).toBeVisible();
}

async function createComparison(page: Page, a: string, b: string, descriptive = false) {
  await page.getByLabel("Base checkpoint (A)").selectOption({ label: a });
  await page.getByLabel("Derived checkpoint (B)").selectOption({ label: b });
  await page.getByRole("radio", { name: descriptive ? /Descriptive only/ : /Strict/ }).check();
  await page.getByRole("button", { name: "Create comparison" }).click();
  await expect(page.getByRole("heading", { name: new RegExp(`${a.replace(/\./g, "\\.")}.*${b.replace(/\./g, "\\.")}`) })).toBeVisible();
}

const tensorRows = (page: Page) => page.getByRole("tabpanel", { name: /Tensors/ }).locator("tbody tr");

async function axe(page: Page, label: string) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
  expect(r.violations.map((v) => `${label}: ${v.id} ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
}

test("import, compare, analyze, visualize, drill down; parity with the CLI; S policy; stale files; refresh", async ({ browser }) => {
  const models = tempDir("cmp-models");
  const base = copyFixture("tiny_base.safetensors", models);
  const ft = copyFixture("tiny_ft.safetensors", models);
  copyFixture("tiny_precision_a.safetensors", models);
  copyFixture("tiny_precision_b.safetensors", models);
  const expected = cliReport(base, ft, models);
  const server = await startServer(tempDir("cmp-data"));
  cleanup.push(() => server.stop());
  const context = await browser.newContext();
  const blocked = await loopbackOnly(context);
  const page = await context.newPage();
  const problems = watchConsole(page);
  await page.goto(server.url);

  // Project and two imports through the UI.
  await page.getByLabel("New project name").fill("Fine-tune study");
  await page.getByRole("button", { name: "Create project" }).click();
  for (const path of [base, ft]) {
    await page.getByLabel(/Model path/).fill(path);
    await page.getByRole("button", { name: /Use the folder containing/ }).click();
    await page.getByRole("button", { name: "Import model" }).click();
    await expect(page.getByText(/Imported “/).last()).toBeVisible();
  }
  const api = client(server);
  const pid = ((await api("GET", "/projects")).json[0] as Json).id as string;
  for (const n of ["tiny_precision_a.safetensors", "tiny_precision_b.safetensors"]) {
    expect((await api("POST", `/projects/${pid}/models`, { entry_path: join(models, n), selected_root: models })).status).toBe(201);
  }
  await page.reload();
  await page.getByRole("navigation", { name: "Main" }).getByRole("button", { name: "Workspace" }).click();

  // Compare: create, header first, run.
  await openCompare(page);
  await createComparison(page, "tiny_base.safetensors", "tiny_ft.safetensors");
  await expect(page.getByText("B − A", { exact: true })).toBeVisible();
  await expect(page.getByText(/No report yet/)).toBeVisible();
  await page.getByRole("button", { name: "Run analysis" }).click();
  const grid = page.getByRole("grid", { name: /Relative L2 by layer and role/ });
  await expect(grid).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/^● Complete$/)).toBeVisible();
  await shot(page, "compare-01-dashboard");

  // Parity: the ranking order and every relative L2 equal the CLI's report.
  await expect(tensorRows(page)).toHaveCount(expected.tensors.length);
  const names = await tensorRows(page).locator("td.key-cell").allTextContents();
  expect(names).toEqual(expected.tensors.map((t) => t.key));
  for (const [i, t] of expected.tensors.entries()) {
    const cell = tensorRows(page).nth(i).locator("td.num.mono").first(); // relative_l2 column
    const title = await cell.getAttribute("title");
    if (t.metrics.relative_l2!.value === null) expect(await cell.textContent()).not.toMatch(/^[0-9]/);
    else expect(Number(title)).toBe(t.metrics.relative_l2!.value);
  }

  // Grid by keyboard: move and filter.
  await grid.getByRole("gridcell").first().focus();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Enter");
  await expect(page.getByText(/^Showing/)).toBeVisible();
  await shot(page, "compare-02-grid-filtered");
  await page.getByRole("button", { name: /Clear filter/ }).click();

  // Layer and role rankings.
  await page.getByRole("tab", { name: /Roles/ }).click();
  await expect(page.getByRole("tabpanel", { name: /Roles/ }).getByRole("button", { name: "mlp.up" })).toBeVisible();
  await shot(page, "compare-03-role-rankings");
  await page.getByRole("tab", { name: /Tensors/ }).click();

  // Drill-down and inspection in Explore.
  const top = expected.tensors[0]!.key;
  await page.getByRole("button", { name: top, exact: true }).click();
  const detail = page.getByRole("region", { name: top });
  await expect(detail.getByRole("button", { name: /Delta heatmap/ })).toBeDisabled();
  await shot(page, "compare-04-tensor-detail");
  await detail.getByRole("button", { name: /Inspect in B/ }).click();
  // Explore opens on checkpoint B with exactly this tensor in the inspector (the existing single-tensor views).
  await expect(page.getByRole("navigation", { name: "Main" }).getByRole("button", { name: "Explore" })).toHaveAttribute("aria-current", "page");
  await expect(page.getByLabel("Model", { exact: true })).toHaveValue(/.+/);
  await expect(page.getByLabel("Model", { exact: true }).locator("option:checked")).toHaveText("tiny_ft.safetensors");
  const inspector = page.getByRole("complementary", { name: "Tensor details" });
  await expect(inspector.getByText(top, { exact: true })).toBeVisible();
  await expect(inspector.getByRole("tab", { name: "Overview" })).toBeVisible();
  await shot(page, "compare-05-inspect-in-explore");

  // Refresh: the stored report comes back without re-running.
  await page.reload();
  await openCompare(page);
  await expect(page.getByRole("grid")).toBeVisible();

  // Class S: strict refuses with the reason; descriptive-only runs and is labelled incomplete.
  await createComparison(page, "tiny_precision_a.safetensors", "tiny_precision_b.safetensors");
  await expect(page.getByRole("note", { name: /Class S/ })).toBeVisible();
  await page.getByRole("button", { name: "Run analysis" }).click();
  await expect(page.getByText(/acknowledge that the numbers describe raw/)).toBeVisible();
  await createComparison(page, "tiny_precision_a.safetensors", "tiny_precision_b.safetensors", true);
  // The newly created record must be selected before submitting; the previous strict
  // comparison has exactly the same model names and must never receive this job.
  await expect(page.getByText("descriptive only (acknowledged)")).toBeVisible();
  await page.getByRole("button", { name: "Run analysis" }).click();
  await expect(page.getByRole("note", { name: "Incomplete report" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/^◐ Incomplete$/)).toBeVisible();
  await expect(tensorRows(page).filter({ hasText: "p.f64" }).getByText("withheld")).toHaveCount(4);
  await shot(page, "compare-06-class-s-incomplete");

  // Files change after the report: the snapshot is flagged; a new run is refused with the fix.
  appendFileSync(ft, Buffer.alloc(8));
  await page.reload();
  await openCompare(page);
  await page.getByRole("list", { name: "Comparisons" }).getByRole("button", { name: /tiny_base.*tiny_ft/ }).click();
  await expect(page.getByRole("note", { name: "Files changed since the report" })).toBeVisible();
  await page.getByRole("button", { name: "Re-run analysis" }).click();
  await expect(page.getByText(/Re-import the changed model/)).toBeVisible();
  await expect(page.getByRole("grid")).toBeVisible(); // the previous report stays
  await shot(page, "compare-07-files-changed");

  expect(blocked).toEqual([]);
  expect(problems).toEqual([]);
  await context.close();
});

test("cancel a running analysis, then retry; previous state never shows a candidate report", async ({ browser }) => {
  const models = tempDir("cmp-big");
  // Two 1 GiB F32 tensors that are mostly file holes: cheap on disk, real work to read and hash.
  const shape: [number, number] = [16384, 16384];
  writeF32(join(models, "a.safetensors"), [{ key: "model.layers.0.mlp.up_proj.weight", shape, fill: (r, c) => (r + c) % 7, sparseRows: (r) => (r < 8 ? 16384 : 0) }]);
  writeF32(join(models, "b.safetensors"), [{ key: "model.layers.0.mlp.up_proj.weight", shape, fill: (r, c) => ((r + c) % 7) + 0.5, sparseRows: (r) => (r < 8 ? 16384 : 0) }]);
  const server = await startServer(tempDir("cmp-big-data"));
  cleanup.push(() => server.stop());
  const api = client(server);
  const pid = (await api("POST", "/projects", { name: "Big" })).json.id as string;
  const ids: string[] = [];
  for (const n of ["a.safetensors", "b.safetensors"]) ids.push((await api("POST", `/projects/${pid}/models`, { entry_path: join(models, n), selected_root: models })).json.id as string);
  expect((await api("POST", "/comparisons", { project_id: pid, model_a_id: ids[0], model_b_id: ids[1], policy: "descriptive_only" })).status).toBe(201);

  const context = await browser.newContext();
  await loopbackOnly(context);
  const page = await context.newPage();
  const problems = watchConsole(page);
  await page.goto(server.url);
  await openCompare(page);
  await page.getByRole("radio", { name: /Verified/ }).check();
  await page.getByRole("button", { name: "Run analysis" }).click();
  await expect(page.getByRole("progressbar", { name: "Analysis progress" })).toBeVisible();
  await shot(page, "compare-08-running");
  await page.getByRole("button", { name: "Cancel analysis" }).click();
  await expect(page.getByRole("alert")).toContainText(/canceled/);
  await expect(page.getByRole("grid")).toHaveCount(0);
  const cmp = ((await api("GET", `/projects/${pid}/comparisons`)).json[0] as Json);
  expect(cmp.report).toBeNull(); // nothing was published
  await page.getByRole("radio", { name: /Best effort/ }).check();
  await page.getByRole("button", { name: "Run analysis" }).click();
  await expect(page.getByRole("grid")).toBeVisible({ timeout: 110_000 });
  expect(problems).toEqual([]);
  await context.close();
});

test("a 3,001-tensor report renders with bounded tables; keyboard and axe (light and dark)", async ({ browser }) => {
  const models = tempDir("cmp-wide");
  const dirB = join(models, "b");
  mkdirSync(dirB);
  bigModel(join(models, "wide.safetensors"), 500); // 3,001 tensors
  copyFileSync(join(models, "wide.safetensors"), join(dirB, "wide.safetensors"));
  const server = await startServer(tempDir("cmp-wide-data"));
  cleanup.push(() => server.stop());
  const api = client(server);
  const pid = (await api("POST", "/projects", { name: "Wide" })).json.id as string;
  const a = (await api("POST", `/projects/${pid}/models`, { entry_path: join(models, "wide.safetensors"), selected_root: models })).json.id;
  const b = (await api("POST", `/projects/${pid}/models`, { entry_path: join(dirB, "wide.safetensors"), selected_root: dirB, display_name: "wide (copy)" })).json.id;
  expect((await api("POST", "/comparisons", { project_id: pid, model_a_id: a, model_b_id: b, policy: "descriptive_only" })).status).toBe(201);

  for (const colorScheme of ["light", "dark"] as const) {
    const context = await browser.newContext({ colorScheme });
    await loopbackOnly(context);
    const page = await context.newPage();
    const problems = watchConsole(page);
    await page.goto(server.url);
    await openCompare(page);
    if (colorScheme === "light") {
      await page.getByRole("button", { name: "Run analysis" }).click();
    }
    const t0 = Date.now();
    await expect(page.getByRole("grid")).toBeVisible({ timeout: 90_000 });
    await expect(page.getByText(/3,001 tensors · page 1 of 31/)).toBeVisible();
    await expect(tensorRows(page)).toHaveCount(100);
    if (colorScheme === "dark") expect(Date.now() - t0).toBeLessThan(10_000); // stored report: render only
    await page.getByRole("button", { name: "Next" }).click();
    await expect(page.getByText(/page 2 of 31/)).toBeVisible();
    await axe(page, `compare/${colorScheme}`);
    await shot(page, `compare-09-wide-${colorScheme}`);
    expect(problems).toEqual([]);
    await context.close();
  }
});
