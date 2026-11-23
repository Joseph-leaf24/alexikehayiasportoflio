import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end checks of the built site in a real browser (e2e/): pages load without
 * errors, navigation works, the layout fits phones, the pixel scene reacts, and axe finds
 * no accessibility violations. Runs against `vite preview` of dist/, so build first
 * (`npm run test:e2e` does both). Site 2 is served from the root path.
 */
const PORT = 4174;
const BASE = `http://localhost:${PORT}/`;
const CI = !!process.env.CI;

export default defineConfig({
  testDir: "e2e",
  timeout: CI ? 180_000 : 90_000,
  expect: { timeout: CI ? 45_000 : 20_000 },
  forbidOnly: CI,
  retries: CI ? 1 : 0,
  workers: CI ? 1 : 2,
  reporter: CI ? [["github"], ["html", { open: "never" }]] : [["list"]],
  use: {
    baseURL: BASE,
    trace: "retain-on-failure",
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
