const test = require("node:test");
const assert = require("node:assert/strict");
const { AdminApiError, HttpAdminApiClient } = require("../dist");

test("reads status exclusively from the typed admin API route", async () => {
  let request;
  const client = new HttpAdminApiClient("/_admin/v1", async (url, options) => {
    request = { url, options };
    return new Response(JSON.stringify({ gateway_version: "0.4.0" }), { status: 200 });
  });
  assert.equal((await client.getStatus()).gateway_version, "0.4.0");
  assert.equal(request.url, "/_admin/v1/status");
  assert.equal(request.options.credentials, "same-origin");
});

test("uses shell-managed bearer authentication and exact operational routes", async () => {
  const requests = [];
  const client = new HttpAdminApiClient("/_admin/v1", async (url, options) => {
    requests.push({ url, options });
    return new Response(JSON.stringify({ revision: "r1", providers: [], models: [], extensions: [], diagnostics: [] }), { status: 200 });
  }, "shell-token");
  await client.getSession(); await client.getConfiguration(); await client.getProviders(); await client.getModels(); await client.getExtensions(); await client.reload("r1");
  assert.deepEqual(requests.map((request) => request.url), ["/_admin/v1/session", "/_admin/v1/configuration", "/_admin/v1/providers", "/_admin/v1/models", "/_admin/v1/extensions", "/_admin/v1/configuration/reload"]);
  assert.equal(requests[0].options.headers.get("Authorization"), "Bearer shell-token");
  assert.equal(requests[5].options.method, "POST");
  assert.equal(requests[5].options.body, '{"expected_revision":"r1"}');
});

test("notifies the shell when an authenticated request becomes unauthorized", async () => {
  let expired = 0;
  const client = new HttpAdminApiClient("/_admin/v1", async () => new Response(JSON.stringify({ error: { code: "admin_unauthorized", message: "unauthorized" } }), { status: 401 }), "token", () => { expired += 1; });
  await assert.rejects(() => client.getSession(), (error) => error instanceof AdminApiError && error.status === 401 && error.code === "admin_unauthorized");
  assert.equal(expired, 1);
});

test("normalizes failed admin responses", async () => {
  const client = new HttpAdminApiClient("/_admin/v1", async () => new Response(null, { status: 401 }));
  await assert.rejects(() => client.getStatus(), (error) => error instanceof AdminApiError && error.status === 401);
});

test("uses strict policy routes and expected revision", async () => {
  const requests = [];
  const client = new HttpAdminApiClient("/_admin/v1", async (url, options) => { requests.push({ url, options }); return new Response(JSON.stringify({ valid: true, revision: "r2", errors: [], policies: [] }), { status: 200 }); });
  await client.getPolicies(); await client.validatePolicies([]); await client.applyPolicies([], "r1");
  assert.deepEqual(requests.map((request) => request.url), ["/_admin/v1/configuration/policies", "/_admin/v1/configuration/policies/validate", "/_admin/v1/configuration/policies"]);
  assert.equal(requests[1].options.body, '{"policies":[]}');
  assert.equal(requests[2].options.body, '{"expected_revision":"r1","policies":[]}');
});

test("uses typed administration and model profile routes", async () => {
  const requests = [];
  const client = new HttpAdminApiClient("/_admin/v1", async (url, options) => { requests.push({ url, options }); return new Response(JSON.stringify({ revision: "r1", models: {}, valid: true, errors: [] }), { status: 200 }); });
  await client.getConfigurationAdministration(); await client.getModelProfiles(); await client.validateModelProfiles({}); await client.applyModelProfiles({}, "r1");
  assert.deepEqual(requests.map((request) => request.url), ["/_admin/v1/configuration/administration", "/_admin/v1/configuration/model-profiles", "/_admin/v1/configuration/model-profiles/validate", "/_admin/v1/configuration/model-profiles"]);
  assert.equal(requests[2].options.body, '{"models":{}}');
  assert.equal(requests[3].options.body, '{"expected_revision":"r1","models":{}}');
});
