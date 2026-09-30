const { spawnSync } = require("node:child_process");
const mode = process.argv[2];
if (!new Set(["focused", "warm", "clean"]).has(mode)) throw new Error("usage: verify.cjs focused|warm|clean");
const started = Date.now();
const run = (...args) => { const result = spawnSync(args[0], args.slice(1), { cwd: __dirname + "/..", stdio: "inherit", shell: process.platform === "win32" }); if (result.status !== 0) process.exit(result.status ?? 1); };
if (mode === "clean") { run("npm", "run", "clean"); run("npm", "ci", "--ignore-scripts"); }
if (mode === "focused") { run("npm", "run", "build", "--workspace=@llm-proxy/management-sdk"); run("npm", "run", "typecheck", "--workspace=@llm-proxy/core-management"); run("npm", "test", "--workspace=@llm-proxy/management-sdk"); run("npm", "test", "--workspace=@llm-proxy/admin-api-client"); run("npm", "test", "--workspace=@llm-proxy/core-management"); }
else { run("npm", "run", "typecheck"); run("npm", "test"); run("npm", "run", "lint"); run("npm", "run", "build"); run("npm", "run", "test:browser"); }
console.log(`verification_timing mode=${mode} duration_ms=${Date.now() - started}`);
