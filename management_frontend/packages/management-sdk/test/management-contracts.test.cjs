const test = require("node:test");
const assert = require("node:assert/strict");
const sdk = require("../dist");

test("exports durable permission and typed management contracts", () => {
  assert.equal(sdk.GatewayPermission.ViewOperations, "gateway.operations.view");
  assert.equal(typeof sdk.MANAGEMENT_SDK_VERSION, "undefined");
  assert.equal(typeof sdk.supportsRemote, "undefined");
});
