import type { AdminApiClient, AdminStatus, ConfigurationAdministrationInspection, ConfigurationInspection, ExtensionsInspection, ManagementSession, ModelProfilesApplyResult, ModelProfilesDocument, ModelProfilesSnapshot, ModelProfilesValidationResult, ModelsInspection, PolicyApplyResult, PolicyDocument, PolicySnapshot, PolicyValidationResult, ProvidersInspection, ReloadResult } from "@llm-proxy/management-sdk";

export class AdminApiError extends Error {
  constructor(readonly status: number, message: string, readonly code?: string) { super(message); }
}

export class HttpAdminApiClient implements AdminApiClient {
  #token?: string;
  constructor(private readonly baseUrl = "/_admin/v1", private readonly fetchImpl: typeof fetch = fetch, token?: string, private readonly onUnauthorized?: () => void) { this.#token = token; }
  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    if (this.#token) headers.set("Authorization", `Bearer ${this.#token}`);
    const response = await this.fetchImpl.call(globalThis, `${this.baseUrl}${path}`, { credentials: "same-origin", ...init, headers });
    if (!response.ok) {
      const payload = await response.json().catch(() => undefined) as { error?: { code?: string; message?: string } } | undefined;
      if (response.status === 401) this.onUnauthorized?.();
      throw new AdminApiError(response.status, payload?.error?.message ?? `Admin request failed (${response.status})`, payload?.error?.code);
    }
    return response.json() as Promise<T>;
  }
  getSession(signal?: AbortSignal) { return this.request<ManagementSession>("/session", { signal }); }

  async getStatus(signal?: AbortSignal): Promise<AdminStatus> {
    return this.request<AdminStatus>("/status", { signal });
  }
  getConfiguration(signal?: AbortSignal) { return this.request<ConfigurationInspection>("/configuration", { signal }); }
  getConfigurationAdministration(signal?: AbortSignal) { return this.request<ConfigurationAdministrationInspection>("/configuration/administration", { signal }); }
  getProviders(signal?: AbortSignal) { return this.request<ProvidersInspection>("/providers", { signal }); }
  getModels(signal?: AbortSignal) { return this.request<ModelsInspection>("/models", { signal }); }
  getExtensions(signal?: AbortSignal) { return this.request<ExtensionsInspection>("/extensions", { signal }); }
  reload(expectedRevision: string, signal?: AbortSignal) { return this.request<ReloadResult>("/configuration/reload", { method: "POST", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ expected_revision: expectedRevision }) }); }
  getPolicies(signal?: AbortSignal) { return this.request<PolicySnapshot>("/configuration/policies", { signal }); }
  validatePolicies(policies: PolicyDocument, signal?: AbortSignal) { return this.request<PolicyValidationResult>("/configuration/policies/validate", { method: "POST", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ policies }) }); }
  applyPolicies(policies: PolicyDocument, expectedRevision: string, signal?: AbortSignal) { return this.request<PolicyApplyResult>("/configuration/policies", { method: "PUT", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ expected_revision: expectedRevision, policies }) }); }
  getModelProfiles(signal?: AbortSignal) { return this.request<ModelProfilesSnapshot>("/configuration/model-profiles", { signal }); }
  validateModelProfiles(models: ModelProfilesDocument, signal?: AbortSignal) { return this.request<ModelProfilesValidationResult>("/configuration/model-profiles/validate", { method: "POST", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ models }) }); }
  applyModelProfiles(models: ModelProfilesDocument, expectedRevision: string, signal?: AbortSignal) { return this.request<ModelProfilesApplyResult>("/configuration/model-profiles", { method: "PUT", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ expected_revision: expectedRevision, models }) }); }
}
