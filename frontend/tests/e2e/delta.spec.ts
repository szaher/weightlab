// ROADMAP P1.1: the delta heatmap of the Compare drill-down against a real `weightlab start`, offline. Expectations
// come from the two fixtures' own bytes (support.readTensor): the heatmap's tile, fetched again as JSON exactly as
// the page requested it, equals float32(B − A) cell by cell, and every exact value in the Exact ROI tab equals the
// float64 difference of the two stored values. Also: nothing is read before the section is opened, the signed delta
// starts on a zero-centred diverging scale, light and dark axe, and FILE_CHANGED after a checkpoint changes.
import AxeBuilder from "@axe-core/playwright";
import { appendFileSync } from "node:fs";
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

/** The UI's exact text of a float64 (shortest round trip, "-0.0" kept, integral values marked as floats). */
function floatText(x: number): string {
  if (Object.is(x, -0)) return "-0.0";
  const s = String(x);
  return /^-?\d+$/.test(s) ? `${s}.0` : s;
}

const UP = "demo.layers.0.mlp.up_proj.weight"; // 8 x 8 F32, 10 elements changed between tiny_base and tiny_ft

test("delta heatmap: tile and exact values recomputed from both files, diverging scale, axe, FILE_CHANGED", async ({ browser }) => {
  test.setTimeout(240_000);
  const models = tempDir("delta-models");
  const base = copyFixture("tiny_base.safetensors", models);
  const ft = copyFixture("tiny_ft.safetensors", models);
  const server = await startServer(tempDir("delta-data"));
  const context = await browser.newContext();
  const blocked = await loopbackOnly(context);
  const page = await context.newPage();
  const problems = watchConsole(page);
  const tileBodies: string[] = [];
  const reads: string[] = [];
  page.on("request", (r) => {
    const u = r.url();
    if (/\/api\/v1\/(tiles|tensors\/roi)$|\/entries\?/.test(u)) reads.push(u);
    if (u.endsWith("/api/v1/tiles") && r.method() === "POST") tileBodies.push(r.postData() ?? "");
  });

  try {
    await page.goto(server.url);
    await page.getByLabel("New project name").fill("Delta");
    await page.getByRole("button", { name: "Create project" }).click();
    for (const path of [base, ft]) {
      await page.getByLabel(/Model path/).fill(path);
      await page.getByLabel(/Selected root/).fill(models);
      await page.getByRole("button", { name: "Import model" }).click();
      await expect(page.getByText(/Imported “/).last()).toBeVisible();
    }
    await page.getByRole("navigation", { name: "Main" }).getByRole("button", { name: "Compare" }).click();
    await page.getByLabel("Base checkpoint (A)").selectOption({ label: "tiny_base.safetensors" });
    await page.getByLabel("Derived checkpoint (B)").selectOption({ label: "tiny_ft.safetensors" });
    await page.getByRole("radio", { name: /Strict/ }).check();
    await page.getByRole("button", { name: "Create comparison" }).click();
    await page.getByRole("button", { name: "Run analysis" }).click();
    await expect(page.getByRole("grid", { name: /Relative L2 by layer and role/ })).toBeVisible({ timeout: 60_000 });

    await page.getByRole("button", { name: UP, exact: true }).click();
    const detail = page.getByRole("region", { name: UP });
    const delta = detail.getByTestId("delta-heatmap");
    expect(reads).toEqual([]); // the drill-down alone reads no delta
    await delta.getByRole("button", { name: "Show delta heatmap" }).click();
    await expect(delta.getByTestId("coverage-badge")).toHaveText("EXACT coverage", { timeout: 60_000 });
    await expect(delta.getByLabel("Scale")).toHaveValue("symmetric");
    await expect(delta.getByLabel("Palette")).toHaveValue("diverging");
    await expect(delta.getByTestId("overview-badges")).toContainText("B − A of F32 inputs, float64");

    // The tile the page drew, fetched again as JSON: float32(B − A) of the stored values, cell by cell.
    const a = readTensor(base, UP).values as number[];
    const b = readTensor(ft, UP).values as number[];
    const d = b.map((v, i) => v - a[i]!); // exact in float64 for these F32 inputs
    expect(d.filter((x) => x !== 0)).toHaveLength(10);
    const req = JSON.parse(tileBodies[0]!) as { source: { kind: string; transform: string }; tile: { level: number } };
    expect(req.source).toMatchObject({ kind: "delta", transform: "b_minus_a" });
    const auth = { Authorization: `Bearer ${server.token}`, "Content-Type": "application/json", Accept: "application/json" };
    const tile = JSON.parse((await raw(server.port, "POST", "/api/v1/tiles", auth, JSON.stringify({ ...req, tile: { ...req.tile, level: 0 } }))).body) as {
      values: (number | null)[][];
      status: number[][];
      metadata: { quality: string; invalid_pair_count: number };
    };
    expect(tile.metadata).toMatchObject({ quality: "exact", invalid_pair_count: 0 });
    expect(tile.values.flat()).toEqual(d.map((x) => Math.fround(x)));
    expect(tile.status.flat().every((s) => s === 0)).toBe(true);
    await shot(page, "delta-01-heatmap");
    await axe(page, "delta heatmap light");
    await page.emulateMedia({ colorScheme: "dark" });
    await axe(page, "delta heatmap dark");
    await shot(page, "delta-02-heatmap-dark");
    await page.emulateMedia({ colorScheme: "light" });

    // Exact ROI: every paired value is the float64 difference of the two stored values.
    await delta.getByRole("tab", { name: "Exact ROI" }).click();
    await expect(delta.getByTestId("roi-precision")).toContainText("numeric precision: rounded");
    await expect(delta.getByTestId("roi-precision")).toContainText("B − A · float64 (F32 inputs)");
    await delta.getByRole("tab", { name: "Values" }).click();
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        await expect(delta.getByTestId(`value-${i}-${j}`).getByRole("button", { name: /^Select row/ })).toHaveText(floatText(d[i * 8 + j]!));
      }
    }
    await expect(delta.getByTestId("value-0-0")).toContainText("float64 (derived)");
    await axe(page, "delta values light");
    await shot(page, "delta-03-exact-values");

    // A changed checkpoint: the next read is refused, never served from earlier results.
    appendFileSync(ft, Buffer.alloc(8));
    await delta.getByTestId("delta-transform").selectOption("abs_b_minus_a");
    await expect(delta.getByText("Model files changed on disk").first()).toBeVisible({ timeout: 30_000 });

    await expectNoToken(page, server.token);
    expect(problems.filter((p) => !/409|FILE_CHANGED|status of 4\d\d/.test(p))).toEqual([]);
    expect(blocked).toEqual([]);
  } finally {
    await context.close();
    await server.stop();
  }
});
