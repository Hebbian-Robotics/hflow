import { useQuery } from "@tanstack/react-query";
import { CirclePlay, Film, Funnel, Waypoints } from "lucide-react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { fetchWorkspaceConfig } from "./api";
import { BrandMark } from "./components/BrandMark";
import { ThemeControl } from "./components/ThemeControl";

// Small workspace readout at the bottom of the rail: where the data lives and
// which versions are serving it. Quiet on purpose.
function WorkspaceSummary() {
  const configQuery = useQuery({ queryKey: ["config"], queryFn: fetchWorkspaceConfig });
  if (configQuery.isPending) return <p className="rail-note">connecting…</p>;
  if (configQuery.isError) return <p className="rail-note rail-note-error">server unreachable</p>;
  const config = configQuery.data;
  return (
    <div className="rail-meta">
      {config.read_only ? <span className="chip chip-muted">read-only</span> : null}
      <p className="rail-note" title={config.data_root}>
        {config.data_root}
      </p>
      <p className="rail-note">
        hflow {config.hflow_version} · ui {config.hflow_ui_version}
      </p>
    </div>
  );
}

export function AppShell() {
  const location = useLocation();
  const isEpisodesActive = location.pathname === "/" || location.pathname.startsWith("/episodes");
  const isRunsActive = location.pathname.startsWith("/runs");
  const isPipelineActive = location.pathname.startsWith("/pipeline");
  const isCurateActive = location.pathname.startsWith("/curate");
  return (
    <div className="app-shell">
      <nav className="nav-rail" aria-label="Primary">
        <div className="brand">
          <BrandMark className="brand-mark" />
          <span>HFlow</span>
        </div>
        <Link
          to="/"
          className={isEpisodesActive ? "nav-item is-active" : "nav-item"}
          aria-current={isEpisodesActive ? "page" : undefined}
        >
          <Film />
          <span>Episodes</span>
        </Link>
        <Link
          to="/runs"
          className={isRunsActive ? "nav-item is-active" : "nav-item"}
          aria-current={isRunsActive ? "page" : undefined}
        >
          <CirclePlay />
          <span>Runs</span>
        </Link>
        <Link
          to="/pipeline"
          className={isPipelineActive ? "nav-item is-active" : "nav-item"}
          aria-current={isPipelineActive ? "page" : undefined}
        >
          <Waypoints />
          <span>Pipeline</span>
        </Link>
        <Link
          to="/curate"
          className={isCurateActive ? "nav-item is-active" : "nav-item"}
          aria-current={isCurateActive ? "page" : undefined}
        >
          <Funnel />
          <span>Curate</span>
        </Link>
        <div className="rail-foot">
          <ThemeControl />
          <WorkspaceSummary />
        </div>
      </nav>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
