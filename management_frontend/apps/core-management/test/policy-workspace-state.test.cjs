const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

function workspaceState() {
  const source = fs.readFileSync(path.join(__dirname, "../src/policyWorkspaceState.ts"), "utf8");
  const output = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(output, { module, exports: module.exports, structuredClone, JSON });
  return module.exports;
}

const policy = (name, enabled = true) => ({ name, enabled, match: { model: name }, actions: { parameters: { temperature: 0.2 }, model: "preserved-model", provider: "preserved-provider" } });
const snapshot = (revision, policies) => ({ revision, policies });

test("policy state requires validation and retains a conflicting draft through reload", () => {
  const { initialPolicyState, policyWorkspaceReducer } = workspaceState();
  const initial = snapshot("r1", [policy("first"), policy("second")]);
  let state = policyWorkspaceReducer(initialPolicyState, { type: "loaded", snapshot: initial });
  state = policyWorkspaceReducer(state, { type: "edited", draft: [policy("first", false), policy("second")] });
  assert.equal(state.phase, "dirty");
  state = policyWorkspaceReducer(state, { type: "validated", result: { valid: true, revision: "r2", errors: [] } });
  assert.equal(state.phase, "validated");
  state = policyWorkspaceReducer(state, { type: "conflict", message: "changed elsewhere" });
  state = policyWorkspaceReducer(state, { type: "loaded", snapshot: snapshot("r3", [policy("first"), policy("second")]), preserveDraft: true });
  assert.equal(state.phase, "dirty");
  assert.equal(state.snapshot.revision, "r3");
  assert.equal(state.draft[0].enabled, false);
  state = policyWorkspaceReducer(state, { type: "discarded" });
  assert.equal(state.phase, "ready");
  assert.equal(state.draft[0].enabled, true);
});

test("policy review is deterministic and identifies rule-level adds, edits, and removals", () => {
  const { reviewPolicyChanges } = workspaceState();
  const changes = Array.from(reviewPolicyChanges([policy("first"), policy("second")], [policy("first", false), policy("third")]));
  assert.deepEqual(changes, ["Rule 1: first → first", "Rule 2: second → third"]);
  assert.deepEqual(Array.from(reviewPolicyChanges([], [policy("new")])), ["Rule 1: (new) → new"]);
  assert.deepEqual(Array.from(reviewPolicyChanges([policy("gone")], [])), ["Rule 1: gone → (removed)"]);
});

test("configuration state conflicts remain distinct from revision conflicts", () => {
  const { initialPolicyState, policyWorkspaceReducer } = workspaceState();
  let state = policyWorkspaceReducer(initialPolicyState, { type: "loaded", snapshot: snapshot("r1", [policy("first")]) });
  state = policyWorkspaceReducer(state, { type: "edited", draft: [policy("first", false)] });
  state = policyWorkspaceReducer(state, { type: "state_conflict", message: "restore or restart" });
  assert.equal(state.phase, "state_conflict");
  assert.equal(state.draft[0].enabled, false);
  assert.match(state.message, /restore or restart/);
});
