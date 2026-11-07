import { useCallback, useEffect, useRef, useState } from "react";
import { ExploreView } from "../features/catalog/ExploreView";
import { SettingsView } from "../features/settings/SettingsView";
import { WorkspaceView } from "../features/workspace/WorkspaceView";
import { ApiRequestError, onUnauthorized } from "../lib/api/client";
import { api } from "../lib/api/endpoints";
import type { Health } from "../lib/api/types";
import { clearToken, getToken, initToken } from "../lib/auth/token";
import { applyTheme, loadTheme, saveTheme, type Theme } from "../lib/prefs";
import { useResource } from "../lib/useResource";
import { AuthScreen } from "./AuthScreen";
import { SideNav, type Section } from "./SideNav";
import { TopBar, type Connection } from "./TopBar";

export const HEALTH_POLL_MS = 15000;
type Auth = "ok" | "missing" | "expired";

export function App() {
  const [auth, setAuth] = useState<Auth>(() => (getToken() ? "ok" : "missing"));
  const [session, setSession] = useState(0); // bumps when a new token arrives, reloading everything
  const [section, setSection] = useState<Section>("workspace");
  const [chosenProject, setProjectId] = useState<string | null>(null);
  const [chosenModel, setModelId] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(loadTheme);
  const [health, setHealth] = useState<Health | undefined>();
  const [reachable, setReachable] = useState<boolean | null>(null);
  const [dropNotice, setDropNotice] = useState(false);
  const mainRef = useRef<HTMLElement>(null);
  const firstSection = useRef(true);

  useEffect(() => applyTheme(theme), [theme]);

  // A 401 anywhere means this tab's token is not the running server's: forget it and say how to reconnect.
  useEffect(
    () =>
      onUnauthorized(() => {
        clearToken();
        setAuth("expired");
      }),
    [],
  );

  // Opening a new launch link in this tab only changes the fragment (no reload): pick the token up.
  const recheck = useCallback(() => {
    const t = initToken();
    if (t) {
      setAuth("ok");
      setSession((s) => s + 1);
    }
  }, []);
  useEffect(() => {
    window.addEventListener("hashchange", recheck);
    return () => window.removeEventListener("hashchange", recheck);
  }, [recheck]);

  // Health is public and constant-cost; it drives the connection indicator only.
  useEffect(() => {
    let stop = false;
    const ctl = new AbortController();
    const poll = async () => {
      try {
        const h = await api.health(ctl.signal);
        if (!stop) {
          setHealth(h);
          setReachable(true);
        }
      } catch (e) {
        if (!stop && !(e instanceof DOMException)) setReachable(false);
      }
    };
    void poll();
    const t = setInterval(() => void poll(), HEALTH_POLL_MS);
    return () => {
      stop = true;
      ctl.abort();
      clearInterval(t);
    };
  }, [session]);

  // Browsers do not give a page the filesystem path of a dropped file, so a drop cannot import in place.
  useEffect(() => {
    const over = (e: DragEvent) => {
      if (e.dataTransfer?.types.includes("Files")) e.preventDefault();
    };
    const drop = (e: DragEvent) => {
      if (!e.dataTransfer?.types.includes("Files")) return;
      e.preventDefault(); // never let the browser open (or upload) the dropped file
      setDropNotice(true);
    };
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragover", over);
      window.removeEventListener("drop", drop);
    };
  }, []);

  const authed = auth === "ok";
  const projects = useResource(authed ? `projects#${session}` : null, (signal) => api.listProjects(signal));
  const projectList = projects.data;
  // Selection falls back to the first project / model when none (or a deleted one) is selected.
  const project = projectList?.find((p) => p.id === chosenProject) ?? projectList?.[0] ?? null;
  const projectId = project?.id ?? null;
  const models = useResource(authed && project ? `models#${session}#${project.id}` : null, (signal) =>
    api.listModels(project?.id ?? "", signal),
  );
  const model = models.data?.find((m) => m.id === chosenModel) ?? models.data?.[0] ?? null;
  const modelId = model?.id ?? null;

  useEffect(() => {
    if (firstSection.current) {
      firstSection.current = false;
      return;
    }
    mainRef.current?.focus();
  }, [section]);

  const unauthorized = projects.error instanceof ApiRequestError && projects.error.status === 401;
  const connection: Connection =
    !authed || unauthorized ? "unauthorized" : reachable === null ? "checking" : reachable ? "connected" : "unavailable";

  let content;
  if (!authed) {
    content = <AuthScreen expired={auth === "expired"} onRetry={recheck} />;
  } else if (section === "workspace") {
    content = (
      <WorkspaceView
        projects={projects}
        models={models}
        projectId={projectId}
        modelId={modelId}
        onSelectProject={(id) => {
          setProjectId(id);
          setModelId(null);
        }}
        onExplore={(id) => {
          setModelId(id);
          setSection("explore");
        }}
      />
    );
  } else if (section === "explore") {
    content = (
      <ExploreView
        projectName={project?.name ?? null}
        models={models}
        modelId={modelId}
        onSelectModel={setModelId}
        onGoWorkspace={() => setSection("workspace")}
      />
    );
  } else {
    content = (
      <SettingsView
        theme={theme}
        health={health}
        onTheme={(t) => {
          setTheme(t);
          saveTheme(t);
        }}
      />
    );
  }

  return (
    <div className="app">
      <a href="#main" className="skip-link">
        Skip to main content
      </a>
      <TopBar
        projectName={project?.name ?? null}
        modelName={model?.display_name ?? null}
        connection={connection}
        health={health}
        onSettings={() => setSection("settings")}
      />
      <div className="app-body">
        <SideNav section={section} onSection={setSection} />
        <main id="main" ref={mainRef} tabIndex={-1} className="main">
          {reachable === false && authed ? (
            <div className="notice notice-error" role="alert">
              <p className="notice-title">WeightLab server unavailable</p>
              <p>The local server is not answering. Start it with `weightlab start` and open the new link it prints.</p>
            </div>
          ) : null}
          {dropNotice ? (
            <div className="notice notice-info" role="status">
              <p className="notice-title">Dropped files are not imported</p>
              <p>
                A web page cannot learn where a dropped file lives on disk, and WeightLab reads models in place instead of
                uploading them. Copy the file’s path instead (Finder: Option-⌘-C; Explorer: Shift + right-click → “Copy as
                path”; terminal: <code>realpath</code>) and paste it into the import form in the Workspace.
              </p>
              <button type="button" className="btn btn-small" onClick={() => setDropNotice(false)}>
                Dismiss
              </button>
            </div>
          ) : null}
          {content}
        </main>
      </div>
    </div>
  );
}
