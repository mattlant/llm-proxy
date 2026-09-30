import { useEffect, useReducer, useState } from "react";
import { AdminApiError } from "@llm-proxy/admin-api-client";
import type { ManagementContext, PolicyDocument } from "@llm-proxy/management-sdk";
import { initialPolicyState, policyWorkspaceReducer, reviewPolicyChanges } from "./policyWorkspaceState";

const failureMessage = (error: unknown) => error instanceof Error ? error.message : "Unable to apply parameter rules";

export default function PolicyWorkspace({ context }: { context: ManagementContext }) {
  const [state, dispatch] = useReducer(policyWorkspaceReducer, initialPolicyState);
  const [review, setReview] = useState<string[]>([]);
  const { snapshot, draft, errors, message } = state;
  const load = async (preserveDraft = false) => {
    try { dispatch({ type: "loaded", snapshot: await context.adminApi.getPolicies(), preserveDraft }); }
    catch (error) { dispatch({ type: "failed", message: error instanceof Error ? error.message : "Unable to load parameter rules" }); }
  };
  useEffect(() => { void load(); }, []);
  const change = (next: PolicyDocument) => { setReview([]); dispatch({ type: "edited", draft: next }); };
  const validate = async () => {
    try { dispatch({ type: "validated", result: await context.adminApi.validatePolicies(draft) }); }
    catch (error) { dispatch({ type: "failed", message: failureMessage(error) }); }
  };
  const apply = async () => {
    if (!snapshot || state.phase !== "validated") return;
    try {
      dispatch({ type: "applying" });
      const result = await context.adminApi.applyPolicies(draft, snapshot.revision);
      dispatch({ type: "applied", snapshot: { revision: result.active_revision, policies: result.policies } });
      setReview([]); context.notify("Parameter rules applied.");
    } catch (error) {
      const code = error instanceof AdminApiError ? error.code : undefined;
      if (code === "revision_conflict") dispatch({ type: "conflict", message: "Rules changed elsewhere. Reload and then discard or validate and reapply your retained draft." });
      else if (code === "configuration_state_conflict") dispatch({ type: "state_conflict", message: "Active and persisted configuration differ. Restore or restart the configuration, then reload and reapply your retained draft." });
      else dispatch({ type: "failed", message: failureMessage(error) });
    }
  };
  return <section id="policies">
    <h2>Parameter rules</h2>
    <button onClick={() => void load(state.phase === "conflict" || state.phase === "state_conflict")}>Reload active rules</button>
    {!snapshot ? <p>Loading parameter rules…</p> : <>
      <p>Loaded revision: {snapshot.revision}</p>
      {draft.map((rule, index) => <fieldset key={`${rule.name}-${index}`}>
        <label>Name <input value={rule.name} onChange={(event) => change(draft.map((item, i) => i === index ? { ...item, name: event.target.value } : item))} /></label>
        <label>Enabled <input type="checkbox" checked={rule.enabled} onChange={(event) => change(draft.map((item, i) => i === index ? { ...item, enabled: event.target.checked } : item))} /></label>
        <label>Match <textarea value={JSON.stringify(rule.match, null, 2)} onChange={(event) => { try { change(draft.map((item, i) => i === index ? { ...item, match: JSON.parse(event.target.value) } : item)); } catch { dispatch({ type: "failed", message: "Match must be valid JSON." }); } }} /></label>
        <label>Parameter overrides <textarea value={JSON.stringify(rule.actions.parameters ?? {}, null, 2)} onChange={(event) => { try { change(draft.map((item, i) => i === index ? { ...item, actions: { ...item.actions, parameters: JSON.parse(event.target.value) } } : item)); } catch { dispatch({ type: "failed", message: "Parameter overrides must be valid JSON." }); } }} /></label>
        <p>Model: {rule.actions.model ?? "unchanged"}; Provider: {rule.actions.provider ?? "unchanged"}</p>
        <button onClick={() => change(draft.filter((_, i) => i !== index))}>Remove rule</button>
        <button disabled={!index} onClick={() => { const next = [...draft]; [next[index - 1], next[index]] = [next[index], next[index - 1]]; change(next); }}>Move up</button>
        <button disabled={index === draft.length - 1} onClick={() => { const next = [...draft]; [next[index], next[index + 1]] = [next[index + 1], next[index]]; change(next); }}>Move down</button>
      </fieldset>)}
      <button onClick={() => change([...draft, { name: "new-rule", enabled: true, match: {}, actions: { parameters: {} } }])}>Add rule</button>
      <button onClick={() => void validate()}>Validate draft</button>
      <button onClick={() => setReview(reviewPolicyChanges(snapshot.policies, draft))}>Review changes</button>
      <button onClick={() => void apply()} disabled={state.phase !== "validated"}>Apply parameter rules</button>
      {(state.phase === "conflict" || state.phase === "state_conflict") && <button onClick={() => dispatch({ type: "discarded" })}>Discard draft</button>}
      {review.length > 0 && <ol aria-label="Policy change review">{review.map((change) => <li key={change}>{change}</li>)}</ol>}
    </>}
    {message && <p role="status">{message}</p>}{errors.map((error) => <p role="alert" key={error}>{error}</p>)}
  </section>;
}
