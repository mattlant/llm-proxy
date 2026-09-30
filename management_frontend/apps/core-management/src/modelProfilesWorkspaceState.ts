import type { ModelProfilesDocument, ModelProfilesSnapshot, ModelProfilesValidationResult } from "@llm-proxy/management-sdk";

export type ModelProfilesPhase = "loading" | "ready" | "dirty" | "validated" | "validation_error" | "applying" | "conflict" | "state_conflict" | "error";
export type ModelProfilesWorkspaceState = { phase: ModelProfilesPhase; snapshot?: ModelProfilesSnapshot; draft: ModelProfilesDocument; errors: string[]; message?: string };
export type ModelProfilesWorkspaceEvent =
  | { type: "loaded"; snapshot: ModelProfilesSnapshot; preserveDraft?: boolean }
  | { type: "edited"; draft: ModelProfilesDocument }
  | { type: "validated"; result: ModelProfilesValidationResult }
  | { type: "applying" } | { type: "applied"; snapshot: ModelProfilesSnapshot }
  | { type: "conflict" | "state_conflict" | "failed"; message: string } | { type: "discarded" };

export const initialModelProfilesState: ModelProfilesWorkspaceState = { phase: "loading", draft: {}, errors: [] };
const loaded = (snapshot: ModelProfilesSnapshot): ModelProfilesWorkspaceState => ({ phase: "ready", snapshot, draft: structuredClone(snapshot.models), errors: [] });
export function modelProfilesWorkspaceReducer(state: ModelProfilesWorkspaceState, event: ModelProfilesWorkspaceEvent): ModelProfilesWorkspaceState {
  switch (event.type) {
    case "loaded": return event.preserveDraft ? { ...state, phase: "dirty", snapshot: event.snapshot, errors: [], message: "Profiles reloaded. Your draft was retained; validate it before reapplying." } : loaded(event.snapshot);
    case "edited": return { ...state, phase: "dirty", draft: event.draft, errors: [], message: undefined };
    case "validated": return { ...state, phase: event.result.valid ? "validated" : "validation_error", errors: event.result.errors, message: event.result.valid ? "Draft is valid. Review before applying." : "Correct validation errors before applying." };
    case "applying": return { ...state, phase: "applying", errors: [], message: undefined };
    case "applied": return loaded(event.snapshot);
    case "conflict": case "state_conflict": return { ...state, phase: event.type, errors: [], message: event.message };
    case "failed": return { ...state, phase: "error", errors: [], message: event.message };
    case "discarded": return state.snapshot ? loaded(state.snapshot) : state;
  }
}

export function reviewModelProfileChanges(before: ModelProfilesDocument, after: ModelProfilesDocument): string[] {
  return [...new Set([...Object.keys(before), ...Object.keys(after)])].sort().flatMap((name) => {
    if (!before[name]) return [`Profile ${name}: added`];
    if (!after[name]) return [`Profile ${name}: removed`];
    return JSON.stringify(before[name]) === JSON.stringify(after[name]) ? [] : [`Profile ${name}: changed`];
  });
}
