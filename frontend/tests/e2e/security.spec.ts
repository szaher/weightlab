// Doc 12 section 8 security E2E on a real server: token handling, Host/Origin, the public UI shell not being an
// API bypass, hostile model metadata rendered as text, and no network access beyond loopback.
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { expectNoToken, loopbackOnly, raw, startServer, tempDir, watchConsole, writeSafetensors, type Server } from "./support";

const HOSTILE = [
  `<img src=x onerror="window.__pwned=1">.weight`,
  `"><script>window.__pwned=2</script>.bias`,
  `javascript:window.__pwned=3`,
  `model.layers.0.${"x".repeat(600)}.weight`,
  `naïve.ünïcødé.权重.weight`,
  `a/b\\c?d#e%2F..%00.weight`,
];

let server: Server;
let modelsDir: string;

test.beforeAll(async () => {
  server = await startServer(tempDir("sec-data"));
  modelsDir = tempDir("sec-models");
  writeSafetensors(
    join(modelsDir, "hostile.safetensors"),
    HOSTILE.map((key) => ({ key, dtype: "F32", shape: [2, 2] })),
  );
});

test.afterAll(async () => {
  await server.stop();
  expect(server.stderr()).not.toContain(server.token); // logs never carry the token
  const launchLines = server.stdout().split("\n").filter((l) => l.includes(server.token));
  expect(launchLines).toHaveLength(1); // only the printed launch URL, in the terminal
  expect(launchLines[0]!.trim()).toBe(server.url);
});

test("missing or wrong token cannot read project data", async () => {
  for (const auth of [undefined, "Bearer wrong", `Bearer ${server.token}x`, server.token, `bearer ${server.token}`]) {
    const r = await raw(server.port, "GET", "/api/v1/projects", auth ? { Authorization: auth } : {});
    expect(r.status).toBe(401);
    expect(JSON.parse(r.body).error.code).toBe("UNAUTHORIZED");
  }
  const q = await raw(server.port, "GET", `/api/v1/projects?token=${server.token}`);
  expect(q.status).toBe(401);
  const ok = await raw(server.port, "GET", "/api/v1/projects", { Authorization: `Bearer ${server.token}` });
  expect(ok.status).toBe(200);
});

test("Host and Origin protections hold for the API and the UI shell", async () => {
  const auth = { Authorization: `Bearer ${server.token}` };
  for (const path of ["/", "/api/v1/projects", "/api/v1/health"]) {
    expect((await raw(server.port, "GET", path, { ...auth, Host: "evil.example" })).status).toBe(403);
    expect((await raw(server.port, "GET", path, { ...auth, Host: `rebind.attacker.example:${server.port}` })).status).toBe(403);
    expect((await raw(server.port, "GET", path, { ...auth, Origin: "http://evil.example" })).status).toBe(403);
  }
  const r = await raw(server.port, "GET", "/api/v1/projects", { ...auth, Origin: server.origin });
  expect(r.status).toBe(200);
  expect(Object.keys(r.headers).filter((h) => h.startsWith("access-control-"))).toEqual([]);
});

test("the served shell holds no token and uses the strict CSP", async () => {
  const r = await raw(server.port, "GET", "/");
  expect(r.status).toBe(200);
  expect(r.body).not.toContain(server.token);
  expect(r.body).toContain('<div id="root">');
  expect(r.headers["content-security-policy"]).toContain("script-src 'self'");
  expect(r.headers["content-security-policy"]).not.toContain("unsafe-inline");
  expect(r.headers["set-cookie"]).toBeUndefined();
});

test("static routes are not an unauthenticated API bypass", async () => {
  for (const path of [
    "/assets/../api/v1/projects",
    "/assets/%2e%2e/api/v1/projects",
    "/api/v1/../assets/x.js",
    "/index.html/../api/v1/projects",
    "//api/v1/projects",
    "/api/v1/projects/../../",
  ]) {
    const r = await raw(server.port, "GET", path);
    expect([401, 404]).toContain(r.status);
    expect(r.body).not.toContain('"id":"prj_');
  }
  for (const method of ["POST", "PUT", "DELETE"]) {
    expect((await raw(server.port, method, "/")).status).toBe(404);
  }
  expect((await raw(server.port, "GET", "/assets/")).status).toBe(404); // no directory listing
});

test("hostile names render as text; the app works with only loopback reachable", async ({ browser }) => {
  const context = await browser.newContext();
  const blocked = await loopbackOnly(context);
  const page = await context.newPage();
  const problems = watchConsole(page);
  await page.goto(server.url);
  await expectNoToken(page, server.token);

  const projectName = `<img src=x onerror="window.__pwned=9">`;
  await page.getByLabel("New project name").fill(projectName);
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name: projectName, level: 2 })).toBeVisible();

  await page.getByLabel(/Model path/).fill(join(modelsDir, "hostile.safetensors"));
  await page.getByLabel(/Selected root/).fill(modelsDir);
  await page.getByLabel("Display name").fill(`<b onmouseover=alert(1)>model</b>`);
  await page.getByRole("button", { name: "Import model" }).click();
  await expect(page.getByText(`Imported “<b onmouseover=alert(1)>model</b>”`)).toBeVisible();
  await page.getByRole("button", { name: "Explore tensors" }).click();
  await expect(page.getByText(/Rows 1–6 of 6 tensors/)).toBeVisible();
  for (const key of HOSTILE) {
    const button = page.getByRole("button", { name: key, exact: true });
    await expect(button).toHaveCount(1);
    expect(await button.textContent()).toBe(key); // full raw name, exact characters
  }
  await page.getByRole("button", { name: HOSTILE[1]!, exact: true }).click();
  await expect(page.getByTestId("inspector-key")).toHaveText(HOSTILE[1]!);
  await page.getByLabel("Search tensor names").fill("<script>");
  await expect(page.getByText(/Rows 1–1 of 1 matching/)).toBeVisible();

  expect(await page.evaluate(() => (window as unknown as { __pwned?: number }).__pwned)).toBeUndefined();
  expect(await page.locator("main img, main script, main b").count()).toBe(0);
  // No horizontal page overflow even with a 600-character name.
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(blocked).toEqual([]);
  expect(problems).toEqual([]);
  await context.close();
});

test("an outside fetch from the page is blocked while the app keeps working", async ({ browser }) => {
  const context = await browser.newContext();
  await loopbackOnly(context);
  const page = await context.newPage();
  await page.goto(server.url);
  await expect(page.getByRole("status", { name: "Backend: Connected" })).toBeVisible();
  // The guard really blocks: a deliberate external request fails (CSP connect-src 'self' also refuses it).
  const outcome = await page.evaluate(() =>
    fetch("https://example.com/").then(
      () => "reached",
      () => "blocked",
    ),
  );
  expect(outcome).toBe("blocked");
  await page.getByRole("button", { name: "Settings" }).first().click();
  await expect(page.getByRole("heading", { name: "Resource and privacy settings" })).toBeVisible();
  await expect(page.getByText("memory_budget_mib")).toBeVisible();
  await context.close();
});
