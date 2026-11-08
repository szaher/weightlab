// Browser-side performance of the overview on a large, sparse checkpoint (runs only with WEIGHTLAB_UI_PERF=1;
// writes WEIGHTLAB_UI_PERF_OUT when set). 8192 x 28672 F32 = 896 MiB on disk (below the default
// pyramid_exact_max_mib of 1024, so the real exact pyramid is built), plus a 2.0 GiB 32768 x 16384 tensor above it
// (row-sampled previews). Measures time to first picture, the exact build, a pan/zoom sequence, request
// concurrency, the decoded-tile cache and the JS heap. The server's pyramid cache is a separate, server-side store.
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { loopbackOnly, raw, startServer, tempDir, writeF32 } from "./support";

test.skip(!process.env.WEIGHTLAB_UI_PERF, "set WEIGHTLAB_UI_PERF=1 to measure");

async function attrs(page: Page) {
  const cv = page.getByTestId("overview-canvas");
  const n = async (a: string) => Number(await cv.getAttribute(a));
  return { cacheBytes: await n("data-cache-bytes"), cacheEntries: await n("data-cache-entries"), requests: await n("data-requests"), peak: await n("data-peak-inflight"), level: await n("data-level") };
}

test("overview performance on a large sparse checkpoint", async ({ browser }) => {
  test.setTimeout(900_000);
  const models = tempDir("perf-models");
  const file = join(models, "model.safetensors");
  writeF32(file, [
    { key: "layers.0.mlp.up_proj.weight", shape: [8192, 28672], fill: (r, c) => Math.sin(r / 97) * Math.cos(c / 131), sparseRows: (r) => (r % 64 === 0 ? 28672 : 0) },
    { key: "model.embed_tokens.weight", shape: [32768, 16384], fill: (r, c) => ((r + c) % 17) / 17 - 0.5, sparseRows: (r) => (r % 512 === 0 ? 16384 : 0) },
  ]);
  const server = await startServer(tempDir("perf-data"));
  const context = await browser.newContext();
  await loopbackOnly(context);
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Performance.enable");
  const heap = async () => ((await cdp.send("Performance.getMetrics")).metrics.find((m) => m.name === "JSHeapUsedSize")?.value ?? 0) / 2 ** 20;
  const out: Record<string, number | string> = { checkpoint: "8192x28672 F32 (896 MiB) + 32768x16384 F32 (2 GiB), sparse" };
  try {
    await page.goto(server.url);
    await page.getByLabel("New project name").fill("Perf");
    await page.getByRole("button", { name: "Create project" }).click();
    const auth = { Authorization: `Bearer ${server.token}`, "Content-Type": "application/json" };
    const project = (JSON.parse((await raw(server.port, "GET", "/api/v1/projects", auth)).body) as { id: string }[])[0]!.id;
    expect((await raw(server.port, "POST", `/api/v1/projects/${project}/models`, auth, JSON.stringify({ entry_path: file, selected_root: models }))).status).toBe(201);
    await page.reload();
    await page.getByRole("listitem").filter({ hasText: "model.safetensors" }).getByRole("button", { name: "Explore tensors" }).click();

    // Exact path: the pyramid job, then the exact overview.
    let t0 = Date.now();
    await page.getByLabel("Search tensor names").fill("up_proj");
    await page.getByRole("button", { name: "layers.0.mlp.up_proj.weight", exact: true }).click();
    await expect(page.getByTestId("job-panel")).toBeVisible({ timeout: 60_000 });
    out.ms_to_job_panel = Date.now() - t0;
    await expect(page.getByTestId("coverage-badge")).toHaveText("EXACT coverage", { timeout: 600_000 });
    out.ms_to_exact_overview_incl_pyramid_build = Date.now() - t0;
    out.exact_fit_level = (await attrs(page)).level;
    const cv = page.getByTestId("overview-canvas");
    await cv.focus();
    t0 = Date.now();
    for (let k = 0; k < 5; k++) await page.keyboard.press("+");
    for (let k = 0; k < 40; k++) await page.keyboard.press(["Shift+ArrowRight", "Shift+ArrowDown", "Shift+ArrowLeft", "Shift+ArrowUp"][Math.floor(k / 10)]!);
    await expect(page.getByTestId("overview-loading")).toBeHidden({ timeout: 120_000 });
    out.ms_pan_zoom_exact_settled = Date.now() - t0;
    Object.assign(out, Object.fromEntries(Object.entries(await attrs(page)).map(([k, v]) => [`exact_${k}`, v])));

    // Sampled path: the 2 GiB tensor in auto mode.
    t0 = Date.now();
    await page.getByLabel("Search tensor names").fill("embed");
    await page.getByRole("button", { name: "model.embed_tokens.weight", exact: true }).click();
    await expect(page.getByTestId("coverage-badge")).toHaveText("ROW-SAMPLED preview", { timeout: 120_000 });
    out.ms_to_sampled_preview = Date.now() - t0;
    await page.getByTestId("overview-canvas").focus();
    t0 = Date.now();
    for (let k = 0; k < 4; k++) await page.keyboard.press("+");
    for (let k = 0; k < 24; k++) await page.keyboard.press(k % 2 ? "Shift+ArrowDown" : "Shift+ArrowRight");
    await expect(page.getByTestId("overview-loading")).toBeHidden({ timeout: 300_000 });
    out.ms_pan_zoom_sampled_settled = Date.now() - t0;
    Object.assign(out, Object.fromEntries(Object.entries(await attrs(page)).map(([k, v]) => [`final_${k}`, v])));
    await page.evaluate(() => (globalThis as { gc?: () => void }).gc?.());
    out.js_heap_used_mib = Math.round((await heap()) * 10) / 10;
    expect(Number(out.final_peak)).toBeLessThanOrEqual(6);
    expect(Number(out.final_cacheBytes)).toBeLessThanOrEqual(64 * 2 ** 20);
  } finally {
    if (process.env.WEIGHTLAB_UI_PERF_OUT) writeFileSync(process.env.WEIGHTLAB_UI_PERF_OUT, JSON.stringify(out, null, 2));
    console.log(JSON.stringify(out, null, 2));
    await server.stop();
  }
});
