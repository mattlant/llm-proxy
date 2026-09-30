const fs = require("node:fs");
const path = require("node:path");

const workspace = path.resolve(__dirname, "..");
const source = path.join(workspace, "apps/core-management/dist");
const destination = path.resolve(workspace, "../llm_proxy/static/management");
const invalid = (value) => path.isAbsolute(value) || value.includes("..") || /:|remoteEntry/i.test(value);

function validate() {
  const manifestPath = path.join(source, "asset-manifest.json");
  if (!fs.existsSync(path.join(source, "index.html")) || !fs.existsSync(manifestPath)) throw new Error("core-management dist is incomplete");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  if (!Array.isArray(manifest.assets) || !manifest.assets.includes("index.html") || !manifest.assets.some((asset) => /\.[a-f0-9]{8,}\.js$/i.test(asset))) throw new Error("asset manifest must contain index.html and a content-hashed JavaScript asset");
  for (const asset of manifest.assets) {
    if (typeof asset !== "string" || invalid(asset) || !fs.statSync(path.join(source, asset)).isFile()) throw new Error(`invalid published asset: ${asset}`);
  }
}

validate();
fs.rmSync(destination, { recursive: true, force: true });
fs.mkdirSync(path.dirname(destination), { recursive: true });
fs.cpSync(source, destination, { recursive: true });
const count = fs.readdirSync(destination, { recursive: true }).length;
console.log(`published_management_assets files=${count} destination=llm_proxy/static/management`);
