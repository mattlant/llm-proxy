const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

function workspaceState() {
  const source = fs.readFileSync(path.join(__dirname, "../src/modelProfilesWorkspaceState.ts"), "utf8");
  const output = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(output, { module, exports: module.exports, structuredClone, JSON, Set, Object });
  return module.exports;
}

const profile = (provider = "primary") => ({ upstream_model: "upstream", provider, interfaces: ["openai"], aliases: { openai: ["alias"] }, parameters: { temperature: 0.2, extra: {} }, compatibility: { expose_thinking: false, native_tools: true, structured_output: "unsupported" } });
const snapshot = (revision, models) => ({ revision, models });

test("model profile state invalidates validation and retains drafts through both conflict classes", () => {
  const { initialModelProfilesState, modelProfilesWorkspaceReducer } = workspaceState();
  let state = modelProfilesWorkspaceReducer(initialModelProfilesState, { type: "loaded", snapshot: snapshot("r1", { alpha: profile() }) });
  state = modelProfilesWorkspaceReducer(state, { type: "edited", draft: { alpha: profile("secondary") } });
  state = modelProfilesWorkspaceReducer(state, { type: "validated", result: { valid: true, revision: "r2", errors: [] } });
  state = modelProfilesWorkspaceReducer(state, { type: "conflict", message: "changed" });
  state = modelProfilesWorkspaceReducer(state, { type: "loaded", snapshot: snapshot("r3", { alpha: profile() }), preserveDraft: true });
  assert.equal(state.phase, "dirty"); assert.equal(state.draft.alpha.provider, "secondary");
  state = modelProfilesWorkspaceReducer(state, { type: "state_conflict", message: "restore" });
  assert.equal(state.phase, "state_conflict"); assert.equal(state.draft.alpha.provider, "secondary");
  state = modelProfilesWorkspaceReducer(state, { type: "discarded" });
  assert.equal(state.phase, "ready"); assert.equal(state.draft.alpha.provider, "primary");
});

test("model profile review is deterministic for adds removals and changes", () => {
  const { reviewModelProfileChanges } = workspaceState();
  assert.deepEqual(Array.from(reviewModelProfileChanges({ alpha: profile(), gone: profile() }, { alpha: profile("secondary"), added: profile() })), ["Profile added: added", "Profile alpha: changed", "Profile gone: removed"]);
});
