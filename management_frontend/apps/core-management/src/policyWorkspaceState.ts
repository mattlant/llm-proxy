import type { PolicyDocument, PolicySnapshot, PolicyValidationResult } from "@llm-proxy/management-sdk";

export type PolicyPhase = "loading" | "ready" | "dirty" | "validated" | "validation_error" | "applying" | "conflict" | "state_conflict" | "error";
export type PolicyWorkspaceState = { phase: PolicyPhase; snapshot?: PolicySnapshot; draft: PolicyDocument; errors: string[]; message?: string };
export type PolicyWorkspaceEvent =
  | { type: "loaded"; snapshot: PolicySnapshot; preserveDraft?: boolean }
  | { type: "edited"; draft: PolicyDocument }
  | { type: "validated"; result: PolicyValidationResult }
  | { type: "applying" }
  | { type: "applied"; snapshot: PolicySnapshot }
  | { type: "conflict"; message: string }
  | { type: "state_conflict"; message: string }
  | { type: "failed"; message: string }
  | { type: "discarded" };

export const initialPolicyState: PolicyWorkspaceState = { phase: "loading", draft: [], errors: [] };
export const loaded = (snapshot: PolicySnapshot): PolicyWorkspaceState => ({ phase: "ready", snapshot, draft: structuredClone(snapshot.policies), errors: [] });
export const edited = (state: PolicyWorkspaceState, draft: PolicyDocument): PolicyWorkspaceState => ({ ...state, phase: "dirty", draft, errors: [], message: undefined });
export const validated = (state: PolicyWorkspaceState, result: PolicyValidationResult): PolicyWorkspaceState => ({ ...state, phase: result.valid ? "validated" : "validation_error", errors: result.errors, message: result.valid ? "Draft is valid. Review before applying." : "Correct validation errors before applying." });

export function policyWorkspaceReducer(state: PolicyWorkspaceState, event: PolicyWorkspaceEvent): PolicyWorkspaceState {
  switch (event.type) {
    case "loaded": return event.preserveDraft ? { ...state, phase: "dirty", snapshot: event.snapshot, errors: [], message: "Rules reloaded. Your draft was retained; validate it before reapplying." } : loaded(event.snapshot);
    case "edited": return edited(state, event.draft);
    case "validated": return validated(state, event.result);
    case "applying": return { ...state, phase: "applying", errors: [], message: undefined };
    case "applied": return loaded(event.snapshot);
    case "conflict": return { ...state, phase: "conflict", errors: [], message: event.message };
    case "state_conflict": return { ...state, phase: "state_conflict", errors: [], message: event.message };
    case "failed": return { ...state, phase: "error", message: event.message };
    case "discarded": return state.snapshot ? loaded(state.snapshot) : state;
  }
}

function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value as Record<string, unknown>).sort().map((key) => `${JSON.stringify(key)}:${stable((value as Record<string, unknown>)[key])}`).join(",")}}`;
  return JSON.stringify(value);
}

export function reviewPolicyChanges(loadedPolicies: PolicyDocument, draft: PolicyDocument): string[] {
  const changes: string[] = [];
  const count = Math.max(loadedPolicies.length, draft.length);
  for (let index = 0; index < count; index += 1) {
    const before = loadedPolicies[index]; const after = draft[index];
    if (stable(before) !== stable(after)) changes.push(`Rule ${index + 1}: ${before?.name ?? "(new)"} → ${after?.name ?? "(removed)"}`);
  }
  return changes;
}
