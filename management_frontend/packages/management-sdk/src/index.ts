export type Permission = string;
export const GatewayPermission = { ViewOperations: "gateway.operations.view", Reload: "gateway.reload", ViewDiagnostics: "gateway.diagnostics.view" } as const;

export interface AdminApiClient {
  getSession(signal?: AbortSignal): Promise<ManagementSession>;
  getStatus(signal?: AbortSignal): Promise<AdminStatus>;
  getConfiguration(signal?: AbortSignal): Promise<ConfigurationInspection>;
  getConfigurationAdministration(signal?: AbortSignal): Promise<ConfigurationAdministrationInspection>;
  getProviders(signal?: AbortSignal): Promise<ProvidersInspection>;
  getModels(signal?: AbortSignal): Promise<ModelsInspection>;
  getExtensions(signal?: AbortSignal): Promise<ExtensionsInspection>;
  reload(expectedRevision: string, signal?: AbortSignal): Promise<ReloadResult>;
  getPolicies(signal?: AbortSignal): Promise<PolicySnapshot>;
  validatePolicies(policies: PolicyDocument, signal?: AbortSignal): Promise<PolicyValidationResult>;
  applyPolicies(policies: PolicyDocument, expectedRevision: string, signal?: AbortSignal): Promise<PolicyApplyResult>;
  getModelProfiles(signal?: AbortSignal): Promise<ModelProfilesSnapshot>;
  validateModelProfiles(models: ModelProfilesDocument, signal?: AbortSignal): Promise<ModelProfilesValidationResult>;
  applyModelProfiles(models: ModelProfilesDocument, expectedRevision: string, signal?: AbortSignal): Promise<ModelProfilesApplyResult>;
}
export interface ManagementSession { permissions: Permission[]; }
export interface ConfigurationInspection { revision: string; configuration: { interfaces?: Record<string, unknown> }; }
export interface ProvidersInspection { providers: ProviderInspection[]; }
export interface ProviderInspection { name: string; extension_id: string; enabled: boolean; active: boolean; command_count: number; capabilities: Record<string, unknown>; }
export type LifecycleClassification = "reload_safe" | "restart_required" | "mixed" | "read_only" | "unsupported";
export interface ProviderHealthInspection { available: boolean; status: "healthy" | "degraded" | "unhealthy" | "unavailable"; diagnostic: string; }
export interface ConfigurationAdministrationInspection { revision: string; server: { host: string; port: number; timeout_seconds: number; config_reload_seconds: number; log_level: string; lifecycle: "restart_required" }; management: { enabled: boolean; allow_remote: boolean; command_timeout_seconds: number; lifecycle: "restart_required" }; interfaces: Array<{ name: string; enabled: boolean; expose_thinking: boolean; authentication_configured: boolean; lifecycle: "mixed" }>; providers: Array<ProviderInspection & { configured: boolean; lifecycle: "restart_required"; health: ProviderHealthInspection }>; extensions: ExtensionInspection[]; diagnostics: Diagnostic[]; }
export interface ModelsInspection { models: ModelInspection[]; }
export interface ModelInspection { name: string; upstream_model: string; provider: string; interfaces: string[]; aliases: Record<string, string[]>; }
export interface ExtensionsInspection { extensions: ExtensionInspection[]; diagnostics: Diagnostic[]; }
export interface ExtensionInspection { extension_id: string; display_name: string; package_version: string; capabilities: Record<string, unknown>; }
export interface Diagnostic { source: string; message: string; extension_id?: string | null; }
export interface ReloadResult { active_revision?: string; persisted_revision?: string; restart_required?: boolean; [key: string]: unknown; }
export type PolicyRule = { name: string; enabled: boolean; match: Record<string, unknown>; actions: { parameters?: Record<string, unknown> | null; model?: string | null; provider?: string | null } };
export type PolicyDocument = PolicyRule[];
export interface PolicySnapshot { revision: string; policies: PolicyDocument; }
export interface PolicyValidationResult { valid: boolean; revision: string | null; errors: string[]; }
export interface PolicyApplyResult { active_revision: string; persisted_revision: string; restart_required: false; policies: PolicyDocument; }
export type ModelProfileDocument = { name?: string; upstream_model: string; provider: string; interfaces: string[]; aliases: Record<string, string[]>; parameters: Record<string, unknown>; compatibility: Record<string, unknown> };
export type ModelProfilesDocument = Record<string, ModelProfileDocument>;
export interface ModelProfilesSnapshot { revision: string; models: ModelProfilesDocument; }
export interface ModelProfilesValidationResult { valid: boolean; revision: string | null; errors: string[]; }
export interface ModelProfilesApplyResult { active_revision: string; persisted_revision: string; activation_class: "reload_safe"; models: ModelProfilesDocument; }

export interface AdminStatus {
  gateway_version: string;
  active_revision: string;
  persisted_revision: string;
  restart_required: boolean;
  interfaces: string[];
  management_bind: "loopback" | "remote";
  last_successful_activation: string;
  last_rejected_reload: string | null;
}

export interface ManagementContext {
  readonly adminApi: AdminApiClient;
  readonly hasPermission: (permission: Permission) => boolean;
  readonly notify: (message: string) => void;
  readonly confirm: (message: string) => Promise<boolean>;
}
