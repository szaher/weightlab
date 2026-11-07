import { defineConfig, devices } from "@playwright/test";

// The E2E suite starts real `weightlab start` processes itself (fresh data directory, fresh token per test file),
// so there is no webServer entry. WEIGHTLAB_CMD selects the command (default: `uv run weightlab` in the repo).
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 10_000 },
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]],
  use: {
    ...devices["Desktop Chrome"],
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    serviceWorkers: "block",
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
