import { Component, useMemo, useState, type FormEvent, type ReactNode } from "react";
import { AdminApiError, HttpAdminApiClient } from "@llm-proxy/admin-api-client";
import { GatewayPermission, type ManagementContext, type Permission } from "@llm-proxy/management-sdk";
import GatewayOverview from "./GatewayOverview";
import PolicyWorkspace from "./PolicyWorkspace";
import ConfigurationAdministration from "./ConfigurationAdministration";
import ModelProfilesWorkspace from "./ModelProfilesWorkspace";

type SessionStatus = "establishing" | "authenticated" | "expired" | "unavailable";
class ApplicationErrorBoundary extends Component<{ children: ReactNode }, { error?: Error }> {
  state: { error?: Error } = {};
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() { return this.state.error ? <p role="alert">Management application unavailable: {this.state.error.message}</p> : this.props.children; }
}

export default function App() {
  const [token, setToken] = useState(""); const [client, setClient] = useState<HttpAdminApiClient>(); const [permissions, setPermissions] = useState<Permission[]>([]); const [session, setSession] = useState<SessionStatus>("unavailable"); const [notice, setNotice] = useState<string>(); const [administrationRefresh, setAdministrationRefresh] = useState(0);
  const anonymousClient = useMemo(() => new HttpAdminApiClient("/_admin/v1", fetch, undefined, () => setSession("expired")), []);
  const activeClient = client ?? anonymousClient;
  const context = useMemo((): ManagementContext => Object.freeze({ adminApi: activeClient, hasPermission: (permission: Permission) => session === "authenticated" && permissions.includes(permission), notify: setNotice, confirm: async (message: string) => window.confirm(message) }), [activeClient, permissions, session]);
  const establish = async (event: FormEvent) => { event.preventDefault(); if (!token) return; setSession("establishing"); const candidate = new HttpAdminApiClient("/_admin/v1", fetch, token, () => { setClient(undefined); setPermissions([]); setSession("expired"); }); setToken(""); try { const result = await candidate.getSession(); setClient(candidate); setPermissions(result.permissions); setSession("authenticated"); setNotice("Management session established."); } catch (error) { setClient(undefined); setPermissions([]); if (error instanceof AdminApiError && error.status === 401) { setSession("expired"); setNotice("Session could not be established. Check the management token and try again."); } else { setSession("unavailable"); setNotice(`Management API is unavailable: ${error instanceof Error ? error.message : "unexpected error"}`); } } };
  const sessionMessage = session === "establishing" ? "Establishing management session…" : session === "expired" ? "Management session expired. Re-authenticate to continue." : session === "unavailable" ? "Establish a management session to access gateway operations." : undefined;
  return <main><h1>LLM Proxy management</h1>{context.hasPermission(GatewayPermission.ViewOperations) && <nav aria-label="Management navigation"><a href="#gateway">Gateway</a><a href="#administration">Administration</a><a href="#model-profiles">Model profiles</a><a href="#policies">Parameter rules</a></nav>}<form onSubmit={(event) => void establish(event)} aria-label="Management session"><label>Management token <input type="password" value={token} onChange={(event) => setToken(event.target.value)} autoComplete="off" /></label><button type="submit" disabled={session === "establishing"}>Establish session</button></form>{sessionMessage && <p role="status">{sessionMessage}</p>}{notice && <p role="status">{notice}</p>}{context.hasPermission(GatewayPermission.ViewOperations) ? <ApplicationErrorBoundary><GatewayOverview context={context} /><ConfigurationAdministration context={context} refreshed={administrationRefresh} /><ModelProfilesWorkspace context={context} onApplied={() => setAdministrationRefresh((value) => value + 1)} /><PolicyWorkspace context={context} /></ApplicationErrorBoundary> : <p role="status">Gateway operations are not permitted for this session.</p>}</main>;
}
