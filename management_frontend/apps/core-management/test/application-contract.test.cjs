const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

test("core application owns session and imports the operational view locally", () => {
  const source = fs.readFileSync(path.join(__dirname, "../src/App.tsx"), "utf8");
  assert.match(source, /import GatewayOverview from "\.\/GatewayOverview"/);
  assert.match(source, /ManagementContext/);
  assert.match(source, /Object\.freeze/);
  assert.match(source, /getSession/);
  assert.doesNotMatch(source, /RemoteContext|gatewayManagement|token=\{token\}|localStorage/);
});

test("webpack has one /management application output without federation", () => {
  const config = fs.readFileSync(path.join(__dirname, "../webpack.config.cjs"), "utf8");
  assert.match(config, /publicPath: "\/management\/"/);
  assert.match(config, /AssetManifestPlugin/);
  assert.doesNotMatch(config, /ModuleFederation|remoteEntry|remotes|exposes|Access-Control-Allow-Origin/);
});

test("overview remains an API-only component", () => {
  const source = fs.readFileSync(path.join(__dirname, "../src/GatewayOverview.tsx"), "utf8");
  assert.match(source, /context\.adminApi\.getStatus/);
  assert.match(source, /AbortController/);
  assert.doesNotMatch(source, /llm_proxy|provider_extensions|HttpAdminApiClient/);
});

test("model administration uses constrained core-owned controls", () => {
  const source = fs.readFileSync(path.join(__dirname, "../src/ModelProfilesWorkspace.tsx"), "utf8");
  assert.match(source, /getConfigurationAdministration/);
  assert.match(source, /<select value=\{profile\.provider\}/);
  assert.match(source, /type="checkbox"/);
  assert.match(source, /Aliases by interface/);
  assert.match(source, /Sampling parameters/);
  assert.match(source, /Compatibility/);
  assert.doesNotMatch(source, /Provider configuration|Management token|redirect/);
});
