import { useEffect, useReducer, useState } from "react";
import { AdminApiError } from "@llm-proxy/admin-api-client";
import type { ConfigurationAdministrationInspection, ManagementContext, ModelProfileDocument, ModelProfilesDocument } from "@llm-proxy/management-sdk";
import { initialModelProfilesState, modelProfilesWorkspaceReducer, reviewModelProfileChanges } from "./modelProfilesWorkspaceState";

const parameterFields = ["temperature", "top_p", "top_k", "min_p", "repeat_penalty", "repeat_last_n", "max_tokens"] as const;
const compatibilityFields = ["expose_thinking", "native_tools"] as const;

export default function ModelProfilesWorkspace({ context, onApplied }: { context: ManagementContext; onApplied: () => void }) {
  const [state, dispatch] = useReducer(modelProfilesWorkspaceReducer, initialModelProfilesState);
  const [inspection, setInspection] = useState<ConfigurationAdministrationInspection>();
  const [review, setReview] = useState<string[]>([]);
  const [reviewed, setReviewed] = useState(false);
  const load = async (preserveDraft = false) => {
    try {
      const [snapshot, administration] = await Promise.all([context.adminApi.getModelProfiles(), context.adminApi.getConfigurationAdministration()]);
      setInspection(administration); dispatch({ type: "loaded", snapshot, preserveDraft }); setReview([]); setReviewed(false);
    } catch (error) { dispatch({ type: "failed", message: error instanceof Error ? error.message : "Unable to load model profiles" }); }
  };
  useEffect(() => { void load(); }, []);
  const change = (draft: ModelProfilesDocument) => { setReview([]); setReviewed(false); dispatch({ type: "edited", draft }); };
  const changeProfile = (name: string, profile: ModelProfileDocument) => change({ ...state.draft, [name]: profile });
  const validate = async () => {
    try { dispatch({ type: "validated", result: await context.adminApi.validateModelProfiles(state.draft) }); }
    catch (error) { dispatch({ type: "failed", message: error instanceof Error ? error.message : "Unable to validate model profiles" }); }
  };
  const apply = async () => {
    if (!state.snapshot || state.phase !== "validated" || !reviewed) return;
    try {
      dispatch({ type: "applying" });
      const result = await context.adminApi.applyModelProfiles(state.draft, state.snapshot.revision);
      dispatch({ type: "applied", snapshot: { revision: result.active_revision, models: result.models } });
      setReview([]); setReviewed(false); onApplied(); context.notify("Model profiles applied.");
    } catch (error) {
      const code = error instanceof AdminApiError ? error.code : undefined;
      const type = code === "revision_conflict" ? "conflict" : code === "configuration_state_conflict" ? "state_conflict" : "failed";
      const message = code === "revision_conflict" ? "Profiles changed elsewhere. Reload and then discard or validate and reapply your retained draft." : code === "configuration_state_conflict" ? "Active and persisted configuration differ. Reload and reapply your retained draft after recovery." : error instanceof Error ? error.message : "Unable to apply model profiles";
      dispatch({ type, message });
    }
  };
  const enabledProviders = inspection?.providers.filter((provider) => provider.enabled).map((provider) => provider.name) ?? [];
  const enabledInterfaces = inspection?.interfaces.filter((item) => item.enabled).map((item) => item.name) ?? [];
  const add = () => {
    const name = `new-profile-${Object.keys(state.draft).length + 1}`;
    change({ ...state.draft, [name]: { upstream_model: "new-upstream-model", provider: enabledProviders[0] ?? "", interfaces: enabledInterfaces.slice(0, 1), aliases: {}, parameters: { extra: {} }, compatibility: { expose_thinking: false, native_tools: true, structured_output: "unsupported" } } });
  };
  return <section id="model-profiles">
    <h2>Model profiles</h2><button onClick={() => void load(state.phase === "conflict" || state.phase === "state_conflict")}>Reload model profiles</button>
    {state.snapshot && <><p>Loaded revision: {state.snapshot.revision}</p>
      {Object.entries(state.draft).sort(([a], [b]) => a.localeCompare(b)).map(([name, profile]) => <fieldset key={name}>
        <legend>{name} (stable name)</legend>
        <label>Upstream model <input value={profile.upstream_model} onChange={(event) => changeProfile(name, { ...profile, upstream_model: event.target.value })} /></label>
        <label>Provider <select value={profile.provider} onChange={(event) => changeProfile(name, { ...profile, provider: event.target.value })}>{enabledProviders.map((provider) => <option key={provider} value={provider}>{provider}</option>)}</select></label>
        <fieldset><legend>Exposed interfaces</legend>{enabledInterfaces.map((item) => <label key={item}><input type="checkbox" checked={profile.interfaces.includes(item)} onChange={(event) => changeProfile(name, { ...profile, interfaces: event.target.checked ? [...profile.interfaces, item] : profile.interfaces.filter((value) => value !== item) })} />{item}</label>)}</fieldset>
        <fieldset><legend>Aliases by interface</legend>{enabledInterfaces.filter((item) => profile.interfaces.includes(item)).map((item) => <label key={item}>{item}<input value={(profile.aliases[item] ?? []).join(", ")} onChange={(event) => changeProfile(name, { ...profile, aliases: { ...profile.aliases, [item]: event.target.value.split(",").map((alias) => alias.trim()).filter(Boolean) } })} /></label>)}</fieldset>
        <fieldset><legend>Sampling parameters</legend>{parameterFields.map((field) => <label key={field}>{field}<input type="number" value={typeof profile.parameters[field] === "number" ? profile.parameters[field] as number : ""} onChange={(event) => changeProfile(name, { ...profile, parameters: { ...profile.parameters, [field]: event.target.value === "" ? null : Number(event.target.value) } })} /></label>)}<label>Stop sequences<input value={Array.isArray(profile.parameters.stop_sequences) ? profile.parameters.stop_sequences.join(", ") : ""} onChange={(event) => changeProfile(name, { ...profile, parameters: { ...profile.parameters, stop_sequences: event.target.value.split(",").map((value) => value.trim()).filter(Boolean) } })} /></label><label>Parameter extensions (JSON)<textarea value={JSON.stringify(profile.parameters.extra ?? {})} onChange={(event) => { try { changeProfile(name, { ...profile, parameters: { ...profile.parameters, extra: JSON.parse(event.target.value) } }); } catch { dispatch({ type: "failed", message: "Parameter extensions must be valid JSON." }); } }} /></label></fieldset>
        <fieldset><legend>Compatibility</legend>{compatibilityFields.map((field) => <label key={field}><input type="checkbox" checked={Boolean(profile.compatibility[field])} onChange={(event) => changeProfile(name, { ...profile, compatibility: { ...profile.compatibility, [field]: event.target.checked } })} />{field}</label>)}<label>Structured output<select value={String(profile.compatibility.structured_output ?? "unsupported")} onChange={(event) => changeProfile(name, { ...profile, compatibility: { ...profile.compatibility, structured_output: event.target.value } })}><option value="unsupported">unsupported</option><option value="openai_json_schema">openai_json_schema</option><option value="ollama_format">ollama_format</option></select></label></fieldset>
        <button onClick={() => change(Object.fromEntries(Object.entries(state.draft).filter(([key]) => key !== name)))}>Remove profile</button>
      </fieldset>)}
      <button onClick={add}>Add profile</button><button onClick={() => void validate()}>Validate model profiles</button><button onClick={() => { setReview(reviewModelProfileChanges(state.snapshot?.models ?? {}, state.draft)); setReviewed(true); }}>Review changes</button><button disabled={state.phase !== "validated" || !reviewed} onClick={() => void apply()}>Apply model profiles</button>
      {(state.phase === "conflict" || state.phase === "state_conflict") && <button onClick={() => dispatch({ type: "discarded" })}>Discard draft</button>}{review.length > 0 && <ol aria-label="Model profile change review">{review.map((item) => <li key={item}>{item}</li>)}</ol>}
    </>}{state.message && <p role="status">{state.message}</p>}{state.errors.map((error) => <p role="alert" key={error}>{error}</p>)}
  </section>;
}
