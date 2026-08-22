import { useQuery } from "@tanstack/react-query";
import {
  CirclePlay,
  Film,
  Funnel,
  Lock,
  PanelLeftClose,
  PanelLeftOpen,
  TriangleAlert,
  Waypoints,
} from "lucide-react";
import { Tooltip } from "radix-ui";
import { type ComponentType, type ReactElement, type ReactNode, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { fetchWorkspaceConfig } from "./api";
import { BrandMark } from "./components/BrandMark";
import { ThemeControl } from "./components/ThemeControl";
import { useRailLayout } from "./railLayout";

// The rail collapses to icons only (src/railLayout.ts owns the preference and
// the pre-paint attribute; styles.css owns the widths). Two rules follow from
// that and are enforced here rather than in CSS:
//
//   * every destination keeps an accessible name in both states — the label
//     stays in the DOM and only becomes visually-hidden, so the name can never
//     depend on JS and CSS agreeing about which state we are in;
//   * collapsed, an icon alone does not say where it goes, so each destination
//     grows a tooltip. Radix, not `title`: `title` is unreachable by keyboard,
//     appears after a browser-controlled delay nobody can tune, and cannot be
//     styled to match the rest of the chrome.
//
// The width itself is CSS. `isCollapsed` from useRailLayout() is the stored
// choice AND whether the stylesheet is currently honouring it, so below the
// bar-layout breakpoint this file agrees with what is on screen: labels shown,
// tooltips silent.

/** How long the pointer has to rest before a rail tooltip appears. */
const TOOLTIP_DELAY_MS = 250;

/**
 * A rail control with a tooltip that is live only while `when` is true.
 *
 * The Radix wrapper is mounted in BOTH states and the open state is fully
 * controlled, rather than the tooltip being conditionally wrapped around the
 * child. Wrapping conditionally would change the element tree at the moment of
 * collapse, so React would tear down the <a>/<button> inside and rebuild it —
 * and the keyboard focus sitting on it would fall to <body>. Same node either
 * way means focus stays exactly where the reader left it.
 */
function RailTooltip({
  when,
  label,
  children,
}: {
  when: boolean;
  label: string;
  children: ReactElement;
}) {
  const [isRequested, setIsRequested] = useState(false);
  return (
    <Tooltip.Root open={when && isRequested} onOpenChange={setIsRequested}>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="rail-tooltip" side="right" sideOffset={8} collisionPadding={8}>
          {label}
          <Tooltip.Arrow className="rail-tooltip-arrow" width={9} height={5} />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

function NavItem({
  to,
  icon: Icon,
  label,
  isActive,
  isCollapsed,
}: {
  to: string;
  icon: ComponentType;
  label: string;
  isActive: boolean;
  isCollapsed: boolean;
}) {
  return (
    <RailTooltip when={isCollapsed} label={label}>
      <Link
        to={to}
        className={isActive ? "nav-item is-active" : "nav-item"}
        aria-current={isActive ? "page" : undefined}
      >
        <Icon />
        {/* Present in both states. Collapsed, styles.css clips it rather than
            dropping it, so the link keeps its name without an aria-label that
            would have to be kept in sync with the visible word. */}
        <span className="nav-item-label">{label}</span>
      </Link>
    </RailTooltip>
  );
}

/**
 * A single glyph standing in for a footer readout that will not fit at 56px.
 *
 * The glyph is a plain span, not a button — there is nothing to press — so it
 * carries its sentence as visually-hidden text for the accessibility tree and
 * shows the same sentence on hover for everyone else. A focusable non-control
 * would put a stop in the tab order that leads nowhere.
 */
function RailGlyph({
  icon: Icon,
  label,
  className,
}: {
  icon: ComponentType;
  label: string;
  className?: string;
}) {
  return (
    <>
      <RailTooltip when={true} label={label}>
        <span className={className ? `rail-glyph ${className}` : "rail-glyph"} aria-hidden="true">
          <Icon />
        </span>
      </RailTooltip>
      <span className="visually-hidden">{label}</span>
    </>
  );
}

// Small workspace readout at the bottom of the rail: where the data lives and
// which versions are serving it. Quiet on purpose.
//
// Collapsed there is no room for a filesystem path or a version pair, so the
// readout drops to the two things that are still worth interrupting for — the
// server being unreachable, and the workspace being read-only — each as one
// tooltipped glyph. The prose comes back when the rail does.
function WorkspaceSummary({ isCollapsed }: { isCollapsed: boolean }) {
  const configQuery = useQuery({ queryKey: ["config"], queryFn: fetchWorkspaceConfig });

  if (isCollapsed) {
    if (configQuery.isPending) return null;
    if (configQuery.isError) {
      return (
        <RailGlyph icon={TriangleAlert} label="Server unreachable." className="rail-glyph-error" />
      );
    }
    return configQuery.data.read_only ? (
      <RailGlyph icon={Lock} label="This workspace is read-only." />
    ) : null;
  }

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
        hflow {config.hflow_version} · server {config.hflow_server_version}
      </p>
    </div>
  );
}

/**
 * The collapse/expand control.
 *
 * Its name says what pressing it will DO, not what state it is in — an icon
 * button whose name is "Sidebar" leaves a screen-reader user to guess — and
 * `aria-expanded` carries the state alongside it. The <button> element itself
 * is the same node in both states, so collapsing never moves focus off it.
 */
function RailToggle({ isCollapsed, onToggle }: { isCollapsed: boolean; onToggle: () => void }) {
  const label = isCollapsed
    ? "Expand the sidebar to show labels"
    : "Collapse the sidebar to icons only";
  const Icon = isCollapsed ? PanelLeftOpen : PanelLeftClose;
  return (
    <RailTooltip when={true} label={label}>
      <button
        type="button"
        className="rail-toggle"
        aria-label={label}
        aria-expanded={!isCollapsed}
        aria-controls="primary-nav"
        onClick={onToggle}
      >
        <Icon />
      </button>
    </RailTooltip>
  );
}

export function AppShell() {
  const location = useLocation();
  const { isCollapsed, toggle } = useRailLayout();
  const isEpisodesActive = location.pathname === "/" || location.pathname.startsWith("/episodes");
  const isRunsActive = location.pathname.startsWith("/runs");
  const isPipelineActive = location.pathname.startsWith("/pipeline");
  const isCurateActive = location.pathname.startsWith("/curate");
  const navItems: ReactNode = (
    <>
      <NavItem
        to="/"
        icon={Film}
        label="Episodes"
        isActive={isEpisodesActive}
        isCollapsed={isCollapsed}
      />
      <NavItem
        to="/runs"
        icon={CirclePlay}
        label="Runs"
        isActive={isRunsActive}
        isCollapsed={isCollapsed}
      />
      <NavItem
        to="/pipeline"
        icon={Waypoints}
        label="Pipeline"
        isActive={isPipelineActive}
        isCollapsed={isCollapsed}
      />
      <NavItem
        to="/curate"
        icon={Funnel}
        label="Curate"
        isActive={isCurateActive}
        isCollapsed={isCollapsed}
      />
    </>
  );
  return (
    <Tooltip.Provider delayDuration={TOOLTIP_DELAY_MS}>
      <div className="app-shell">
        <nav className="nav-rail" id="primary-nav" aria-label="Primary">
          <div className="brand">
            {/* The mark is the company's, the word beside it is the product's, so
                the pair reads "Hebbian Robotics HFlow" to a screen reader too.
                Collapsed, the mark stays and the word is clipped, not dropped. */}
            <BrandMark className="brand-mark" title="Hebbian Robotics" />
            <span className="brand-word">HFlow</span>
          </div>
          {navItems}
          <div className="rail-foot">
            <div className="rail-controls">
              <RailToggle isCollapsed={isCollapsed} onToggle={toggle} />
              <ThemeControl />
            </div>
            <WorkspaceSummary isCollapsed={isCollapsed} />
          </div>
        </nav>
        <main className="app-main">
          <Outlet />
        </main>
      </div>
    </Tooltip.Provider>
  );
}
