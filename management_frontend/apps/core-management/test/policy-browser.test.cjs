const test = require("node:test");
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const path = require("node:path");
const { chromium } = require("playwright");

const appDirectory = path.resolve(__dirname, "..");
const workspaceDirectory = path.resolve(appDirectory, "../..");
const webpack = path.resolve(workspaceDirectory, "node_modules/.bin/webpack");
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function startServer() {
  const server = spawn(webpack, ["serve", "--mode", "development"], { cwd: appDirectory, stdio: ["ignore", "pipe", "pipe"] });
  let output = "";
  const address = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`core server did not start: ${output}`)), 20_000);
    const inspect = (chunk) => { output += chunk; const found = output.match(/Loopback: http:\/\/localhost:(\d+)\//); if (found) { clearTimeout(timer); resolve(`http://127.0.0.1:${found[1]}/management/`); } };
    server.stdout.on("data", inspect); server.stderr.on("data", inspect); server.once("exit", (code) => reject(new Error(`core server exited ${code}: ${output}`)));
  });
  return { server, address };
}
async function stopServer(server) { if (server.exitCode === null) await new Promise((resolve) => { server.once("exit", resolve); server.kill("SIGTERM"); setTimeout(() => server.kill("SIGKILL"), 2_000); }); }

test("policy workspace loads, edits, validates, reviews, handles state conflict, reloads, and applies", { skip: process.env.RUN_LIVE_BROWSER_TEST !== "1" }, async () => {
  const { server, address } = await startServer(); let browser; let policies = [{ name: "first", enabled: true, match: { models: ["model-a"] }, actions: { parameters: { temperature: 0.2, extra: { keep: true } }, model: "model-a", provider: "primary" } }]; let applyAttempts = 0;
  const operational = { "/_admin/v1/configuration": { revision: "r1", configuration: { interfaces: { openai: {} } } }, "/_admin/v1/providers": { providers: [] }, "/_admin/v1/models": { models: [] }, "/_admin/v1/extensions": { extensions: [], diagnostics: [] }, "/_admin/v1/status": { gateway_version: "test", active_revision: "r1", persisted_revision: "r1", restart_required: false, interfaces: ["openai"], management_bind: "loopback", last_successful_activation: "now", last_rejected_reload: null } };
  try {
    browser = await chromium.launch({ headless: true }); const page = await browser.newPage();
    await page.route("**/*", async (route) => {
      const request = route.request(); const pathname = new URL(request.url()).pathname;
      if (!pathname.startsWith("/_admin/v1/")) return route.continue();
      if (pathname === "/_admin/v1/session") return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ permissions: ["gateway.operations.view"] }) });
      if (!request.headers().authorization) return route.fulfill({ status: 401, contentType: "application/json", body: JSON.stringify({ error: { code: "admin_unauthorized", message: "unauthorized" } }) });
      if (pathname === "/_admin/v1/configuration/policies" && request.method() === "GET") return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ revision: applyAttempts ? "r2" : "r1", policies }) });
      if (pathname === "/_admin/v1/configuration/policies/validate") return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ valid: true, revision: "candidate", errors: [] }) });
      if (pathname === "/_admin/v1/configuration/policies" && request.method() === "PUT") { applyAttempts += 1; if (applyAttempts === 1) return route.fulfill({ status: 409, contentType: "application/json", body: JSON.stringify({ error: { code: "configuration_state_conflict", message: "state differs" } }) }); const body = request.postDataJSON(); assert.equal(body.expected_revision, "r2"); assert.equal(body.policies[0].name, "edited"); assert.deepEqual(body.policies[0].actions.parameters.extra, { keep: true }); policies = body.policies; return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ active_revision: "r3", persisted_revision: "r3", restart_required: false, policies }) }); }
      return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(operational[pathname] ?? {}) });
    });
    await page.goto(address); await page.getByLabel("Management token").fill("synthetic-token"); await page.getByRole("button", { name: "Establish session" }).click();
    await page.getByText("Loaded revision: r1").waitFor(); const rule = page.locator("#policies fieldset").first(); await rule.getByLabel("Name").fill("edited");
    await page.getByRole("button", { name: "Review changes" }).click(); await page.getByLabel("Policy change review").getByText("Rule 1: first → edited").waitFor();
    await page.getByRole("button", { name: "Validate draft" }).click(); await page.getByText("Draft is valid. Review before applying.").waitFor(); await page.getByRole("button", { name: "Apply parameter rules" }).click();
    await page.getByText("Active and persisted configuration differ.").waitFor(); await page.getByRole("button", { name: "Reload active rules" }).click(); await page.getByText("Your draft was retained").waitFor();
    await page.getByRole("button", { name: "Validate draft" }).click(); await page.getByRole("button", { name: "Apply parameter rules" }).click(); await page.getByText("Parameter rules applied.").waitFor(); await page.getByText("Loaded revision: r3").waitFor();
  } finally { await browser?.close(); await stopServer(server); await delay(50); }
});
