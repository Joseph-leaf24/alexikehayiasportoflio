import { defineConfig, devices } from "@playwright/test";

/**
 * Browser tests in e2e/, run against `vite preview` of the built dist/. Build first:
 * `npm run test:e2e` does both.
 */
const PORT = 4173;
const BASE = `http://localhost:${PORT}/alexikehayiasportoflio/`;
const CI = !!process.env.CI;

export default defineConfig({
  testDir: "e2e",
  // CI renders WebGL on a slow, shared CPU, so allow more time there: a slow run only delays
  // the deploy, a flaky failure blocks it
  timeout: CI ? 180_000 : 90_000,
  expect: { timeout: CI ? 45_000 : 20_000 },
  forbidOnly: CI,
  retries: CI ? 1 : 0,
  // Every page renders WebGL on the CPU, so run few browsers at once. CI runners for private
  // repos have 2 vCPUs
  workers: CI ? 1 : 2,
  reporter: CI ? [["github"], ["html", { open: "never" }]] : [["list"]],
  use: {
    baseURL: BASE,
    trace: "retain-on-failure",
    // CI runners have no GPU: render WebGL in software so the 3D scene still runs
    launchOptions: { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] },
    // Locally, reuse the installed Chrome instead of downloading Playwright's Chromium
    channel: CI ? undefined : "chrome",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: `npm run preview -- --port ${PORT} --strictPort`,
    url: BASE,
    reuseExistingServer: !CI,
    timeout: 60_000,
  },
});
