const path = require("path");

const root = path.resolve(__dirname, "..");
const aliases = {
  "@llm-proxy/management-sdk": path.join(root, "packages/management-sdk/src"),
  "@llm-proxy/admin-api-client": path.join(root, "packages/admin-api-client/src"),
  "@llm-proxy/ui-primitives": path.join(root, "packages/ui-primitives/src")
};

module.exports = { aliases };
