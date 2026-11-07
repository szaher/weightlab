// Automated WCAG 2.x A/AA checks (axe-core) on every implemented screen, in light and dark themes and at phone
// width, plus keyboard-only operation of the core flow. Automated checks do not replace a manual audit.
import AxeBuilder from "@axe-core/playwright";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { copyFixture, loopbackOnly, raw, startServer, tempDir, type Server } from "./support";

const SHOTS = process.env.WEIGHTLAB_SCREENSHOTS;
let server: Server;
let models: string;

test.describe.configure({ mode: "serial" });
test.beforeAll(async () => {
  server = await startServer(tempDir("a11y-data"));
  models = tempDir("a11y-models");
  copyFixture("tiny_base.safetensors", models);
});
test.afterAll(async () => server.stop());

/** One project with tiny_base imported, created through the API (idempotent). */
async function ensureModel(): Promise<void> {
  const auth = { Authorization: `Bearer ${server.token}`, "Content-Type": "application/json" };
  const list = JSON.parse((await raw(server.port, "GET", "/api/v1/projects", auth)).body) as { id: string }[];
  if (list.length > 0) return;
  const p = JSON.parse((await raw(server.port, "POST", "/api/v1/projects", auth, JSON.stringify({ name: "Accessibility" }))).body) as {
    id: string;
  };
  const body = JSON.stringify({ entry_path: join(models, "tiny_base.safetensors"), selected_root: models });
  const r = await raw(server.port, "POST", `/api/v1/projects/${p.id}/models`, auth, body);
  expect(r.status).toBe(201);
}

async function axe(page: Page, label: string) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
  const summary = r.violations.map((v) => `${label}: ${v.id} (${v.impact}) ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`);
  expect(summary).toEqual([]);
}

test("welcome screen has no axe violations (light and dark)", async ({ browser }) => {
  for (const colorScheme of ["light", "dark"] as const) {
    const context = await browser.newContext({ colorScheme });
    await loopbackOnly(context);
    const page = await context.newPage();
    await page.goto(server.url);
    await expect(page.getByRole("heading", { name: "Explore LLM weights on your computer." })).toBeVisible();
    await axe(page, `welcome/${colorScheme}`);
    await context.close();
  }
});

for (const scheme of ["light", "dark"] as const) {
  test(`no axe violations on implemented screens (${scheme})`, async ({ browser }) => {
    await ensureModel();
    const context = await browser.newContext({ colorScheme: scheme });
    await loopbackOnly(context);
    const page = await context.newPage();
    await page.goto(server.url);
    await expect(page.getByRole("button", { name: "Explore tensors" })).toBeVisible();
    await page.getByRole("button", { name: "Import model" }).click(); // empty submit: inline field errors
    await expect(page.getByText("Required.").first()).toBeVisible();
    await axe(page, `workspace/${scheme}`);
    await page.getByRole("button", { name: "Explore tensors" }).click();
    await page.getByRole("button", { name: "demo.layers.0.mlp.up_proj.weight" }).click();
    await expect(page.getByTestId("inspector-key")).toBeVisible();
    await axe(page, `explore/${scheme}`);
    if (SHOTS) await page.screenshot({ path: join(SHOTS, `08-explore-${scheme}.png`), fullPage: true });
    await page.getByRole("navigation", { name: "Main" }).getByRole("button", { name: "Settings" }).click();
    await expect(page.getByRole("heading", { name: "Resource and privacy settings" })).toBeVisible();
    await axe(page, `settings/${scheme}`);
    await page.getByRole("navigation", { name: "Main" }).getByRole("button", { name: "Workspace" }).click();
    await page.getByRole("button", { name: "Delete project…" }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await axe(page, `delete-dialog/${scheme}`);
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog")).toBeHidden();
    await expect(page.getByRole("button", { name: "Delete project…" })).toBeFocused(); // focus returns
    await context.close();
  });
}

test("phone width: no horizontal page overflow, all controls reachable", async ({ browser }) => {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  await loopbackOnly(context);
  const page = await context.newPage();
  await ensureModel();
  await page.goto(server.url);
  await page.getByRole("button", { name: "Explore tensors" }).first().click();
  await page.getByRole("button", { name: "demo.layers.0.self_attn.q_proj.weight" }).click();
  await expect(page.getByTestId("inspector-key")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  if (SHOTS) await page.screenshot({ path: join(SHOTS, "09-phone-explore.png"), fullPage: true });
  await axe(page, "phone/explore");
  await context.close();
});

test("keyboard only: search, select a tensor, read its details", async ({ browser }) => {
  const context = await browser.newContext();
  await loopbackOnly(context);
  const page = await context.newPage();
  await ensureModel();
  await page.goto(server.url);
  await expect(page.getByRole("button", { name: "Explore tensors" })).toBeVisible();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to main content" })).toBeFocused();
  await page.getByRole("button", { name: "Explore tensors" }).first().focus();
  await page.keyboard.press("Enter");
  await page.getByLabel("Search tensor names").focus();
  await page.keyboard.type("down_proj");
  const key = page.getByRole("button", { name: "demo.layers.0.mlp.down_proj.weight" });
  await expect(page.getByText(/Rows 1–1 of 1 matching/)).toBeVisible();
  await key.focus();
  await expect(key).toBeFocused();
  const outline = await key.evaluate((el) => getComputedStyle(el).outlineStyle);
  expect(outline).not.toBe("none"); // visible focus indicator
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("inspector-shape")).toHaveText("8 × 8");
  await context.close();
});
