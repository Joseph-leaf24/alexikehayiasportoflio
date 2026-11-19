import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end checks of the built site in a real browser (e2e/): pages load without
 * errors, navigation works, the layout fits phones, and axe finds no accessibility
 * violations. Runs against `vite preview` of dist/, so build first (`npm run test:e2e`
 * does both).
 */
const PORT = 4173;
const BASE = `http://localhost:${PORT}/alexikehayiasportoflio/`;
const CI = !!process.env.CI;

export default defineConfig({
  testDir: "e2e",
  timeout: 90_000,
  expect: { timeout: 20_000 },
  forbidOnly: CI,
  retries: CI ? 1 : 0,
  // Every page renders WebGL on the CPU (no GPU), so run few browsers at once: one on
  // CI (2-vCPU runners for private repos), two locally
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
