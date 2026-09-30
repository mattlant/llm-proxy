const test = require("node:test");
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const net = require("node:net");
const path = require("node:path");
const { chromium } = require("playwright");

const appDirectory = path.resolve(__dirname, "..");
const webpack = path.resolve(appDirectory, "../../node_modules/.bin/webpack");
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
const stripAnsi = (value) => value.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, "");

async function startServer() {
  const server = spawn(webpack, ["serve", "--mode", "development", "--host", "127.0.0.1", "--port", "0"], { cwd: appDirectory, stdio: ["ignore", "pipe", "pipe"] });
  let output = "";
  try {
    const address = await new Promise((resolve, reject) => {
      let settled = false;
      const finish = (error, value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (error) reject(error);
        else resolve(value);
      };
      const timer = setTimeout(() => finish(new Error(`core server did not start within 20000ms; webpack output:\n${output.trim() || "(no output captured)"}`)), 20_000);
      const inspect = (chunk) => {
        output += stripAnsi(chunk.toString());
        const found = output.match(/(?:Loopback|Local):\s*http:\/\/(?:localhost|127\.0\.0\.1):(\d+)\//);
        if (found) finish(null, `http://127.0.0.1:${found[1]}/management/`);
      };
      server.stdout.on("data", inspect);
      server.stderr.on("data", inspect);
      server.once("error", (error) => finish(new Error(`core server could not be started: ${error.message}\nwebpack output:\n${output.trim() || "(no output captured)"}`)));
      server.once("close", (code, signal) => finish(new Error(`core server exited before becoming ready (code ${code}, signal ${signal}):\n${output.trim() || "(no output captured)"}`)));
    });
    return { server, address };
  } catch (error) {
    await stopServer(server);
    throw error;
  }
}
async function stopServer(server) {
  if (!server || server.pid === undefined || server.exitCode !== null || server.signalCode !== null) return;
  await new Promise((resolve) => {
    let forceTimer;
    let finalTimer;
    const finish = () => {
      clearTimeout(forceTimer);
      clearTimeout(finalTimer);
      server.off("close", finish);
      resolve();
    };
    server.once("close", finish);
    forceTimer = setTimeout(() => {
      if (server.exitCode !== null || server.signalCode !== null) return finish();
      server.kill("SIGKILL");
      finalTimer = setTimeout(finish, 2_000);
    }, 2_000);
    if (!server.kill("SIGTERM")) finish();
  });
}
async function closeBrowser(browserServer) {
  if (!browserServer) return;
  const process = browserServer.process();
  let timer;
  try {
    await Promise.race([
      browserServer.close(),
      new Promise((resolve) => { timer = setTimeout(async () => { await browserServer.kill(); resolve(); }, 2_000); }),
    ]);
  } finally {
    clearTimeout(timer);
    if (process.exitCode === null && process.signalCode === null) await browserServer.kill();
  }
}
async function rejectsConnection(url) { const { port } = new URL(url); await new Promise((resolve, reject) => { const socket = net.connect(Number(port), "127.0.0.1"); socket.once("connect", () => { socket.destroy(); reject(new Error("management listener remained reachable")); }); socket.once("error", resolve); }); }

test("one core server proves administration and model workflow then cleans up", { skip: process.env.RUN_LIVE_BROWSER_TEST !== "1" }, async () => {
  let server; let address; let browserServer; let browser; const pageErrors = [];
  let revision = "r1"; let conflict = false;
  const models = { "model-a": { upstream_model: "upstream-a", provider: "primary", interfaces: ["openai"], aliases: { openai: ["alias-a"] }, parameters: { temperature: 0.2, extra: {} }, compatibility: { expose_thinking: false, native_tools: true, structured_output: "unsupported" } } };
  const administration = { revision: "r1", server: { lifecycle: "restart_required" }, management: { lifecycle: "restart_required" }, interfaces: [{ name: "openai", enabled: true, expose_thinking: false, authentication_configured: false, lifecycle: "mixed" }], providers: [{ name: "primary", extension_id: "ollama", enabled: true, active: true, configured: true, lifecycle: "restart_required", command_count: 0, capabilities: { completion: true }, health: { available: false, status: "unavailable", diagnostic: "health capability is unavailable" } }], extensions: [{ extension_id: "ollama", display_name: "Ollama", package_version: "1", capabilities: { completion: true } }], diagnostics: [{ source: "fixture", message: "safe diagnostic" }] };
  try {
    ({ server, address } = await startServer());
    browserServer = await chromium.launchServer({ headless: true });
    browser = await chromium.connect(browserServer.wsEndpoint());
    const page = await browser.newPage();
    page.on("pageerror", (error) => pageErrors.push(error));
    await page.route("**/*", async (route) => {
      const request = route.request(); const pathname = new URL(request.url()).pathname;
      if (!pathname.startsWith("/_admin/v1/")) return route.continue();
      const body = (value, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(value) });
      if (pathname === "/_admin/v1/session") return body({ permissions: ["gateway.operations.view", "gateway.reload", "gateway.diagnostics.view"] });
      if (!request.headers().authorization) return body({ error: { message: "unauthorized" } }, 401);
      if (pathname === "/_admin/v1/status") return body({ gateway_version: "4.0", active_revision: revision, persisted_revision: revision, restart_required: false, interfaces: ["openai"], management_bind: "loopback", last_successful_activation: "now", last_rejected_reload: null });
      if (pathname === "/_admin/v1/configuration") return body({ revision, configuration: { interfaces: { openai: {} } } });
      if (pathname === "/_admin/v1/configuration/policies") return body({ revision, policies: [] });
      if (pathname === "/_admin/v1/providers") return body({ providers: administration.providers });
      if (pathname === "/_admin/v1/models") return body({ models: [] });
      if (pathname === "/_admin/v1/extensions") return body({ extensions: administration.extensions, diagnostics: administration.diagnostics });
      if (pathname === "/_admin/v1/configuration/administration") return body(administration);
      if (pathname === "/_admin/v1/configuration/model-profiles/validate") return body({ valid: true, revision: "r2", errors: [] });
      if (pathname === "/_admin/v1/configuration/model-profiles" && request.method() === "PUT") { if (conflict) return body({ error: { code: "revision_conflict", message: "changed elsewhere" } }, 409); revision = "r2"; return body({ active_revision: revision, persisted_revision: revision, activation_class: "reload_safe", models }); }
      if (pathname === "/_admin/v1/configuration/model-profiles") return body({ revision, models });
      if (pathname === "/_admin/v1/configuration/reload") return body({ restart_required: false });
      return body({});
    });
    await page.goto(address); await page.getByLabel("Management token").fill("synthetic-token"); await page.getByRole("button", { name: "Establish session" }).click();
    await page.getByText("Gateway version", { exact: false }).waitFor(); await page.getByText("Providers (read-only)", { exact: false }).waitFor();
    assert.doesNotMatch(await page.locator("body").textContent(), /base_url|synthetic-token/);
    const profile = page.locator("#model-profiles > fieldset").first(); await profile.getByLabel("Upstream model").fill("updated-upstream"); await profile.locator("fieldset").nth(1).getByRole("textbox").fill("updated-alias");
    const workspace = page.locator("#model-profiles"); await workspace.getByRole("button", { name: "Validate model profiles" }).click(); await workspace.getByRole("button", { name: "Review changes" }).click(); await workspace.getByRole("button", { name: "Apply model profiles" }).click(); await page.getByText("Model profiles applied.", { exact: false }).waitFor();
    conflict = true; await profile.getByLabel("Upstream model").fill("conflicting-upstream"); await workspace.getByRole("button", { name: "Validate model profiles" }).click(); await workspace.getByRole("button", { name: "Review changes" }).click(); await workspace.getByRole("button", { name: "Apply model profiles" }).click(); await page.getByText("Profiles changed elsewhere", { exact: false }).waitFor(); assert.equal(await profile.getByLabel("Upstream model").inputValue(), "conflicting-upstream");
    assert.deepEqual(pageErrors, [], "browser page reported uncaught errors");
  } finally {
    const cleanupErrors = [];
    try { await closeBrowser(browserServer); } catch (error) { cleanupErrors.push(error); }
    try { await stopServer(server); } catch (error) { cleanupErrors.push(error); }
    if (address) {
      try { await delay(50); await rejectsConnection(address); } catch (error) { cleanupErrors.push(error); }
    }
    if (cleanupErrors.length) throw new AggregateError(cleanupErrors, `browser test cleanup failed: ${cleanupErrors.map((error) => error.stack ?? error).join("\n")}`);
  }
});
