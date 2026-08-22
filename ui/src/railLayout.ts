import { useCallback, useEffect, useState } from "react";

// Nav-rail width preference — the sibling of theme.ts, and deliberately built
// the same way, because it has the same two hard requirements: it must survive
// a reload, and it must be in force BEFORE the first paint or the rail visibly
// jumps from 184px to 56px while the reader watches.
//
// Two states, not three. "Follow the system" is meaningful for a theme (the OS
// has an opinion about light and dark); no platform has an opinion about how
// wide this rail should be, so there is nothing for a third state to defer to.
//
// styles.css is authoritative for what each state LOOKS like — including the
// fact that the collapse is suspended below 780px, where the rail is already a
// horizontal bar and has no width to give back. This module only decides which
// of the two states is in force and stamps it on <html>.

export type RailLayout = "expanded" | "collapsed";

/**
 * The width at and below which styles.css turns the rail into a horizontal bar
 * and suspends the collapse — a bar has no width to give back.
 *
 * DUPLICATED from the 780px breakpoint at the bottom of styles.css, on purpose
 * and unavoidably: the stylesheet decides the layout, and this module only
 * needs to know when the stylesheet has stopped honouring the preference so
 * the markup does not contradict it (tooltips that repeat a label already on
 * screen, a footer glyph standing in for a readout that is showing anyway).
 * Change one, change the other.
 */
const RAIL_COLLAPSIBLE_QUERY = "(min-width: 781px)";

/**
 * localStorage key holding the RailLayout, as one of the two literal strings
 * above.
 *
 * Read in two places that must agree: this module, and the inline pre-paint
 * script at the top of index.html that applies the stored choice before React
 * mounts. Change one, change the other.
 */
export const RAIL_STORAGE_KEY = "hflow-ui-rail";

/**
 * The attribute styles.css keys off. Absent means "expanded" — the default is
 * spelled as the absence of the attribute so a reader who has never touched
 * the control, and a reader whose localStorage is blocked, land in the same
 * place with no JS having run.
 */
const RAIL_ATTRIBUTE = "data-rail";

function isRailLayout(value: unknown): value is RailLayout {
  return value === "expanded" || value === "collapsed";
}

/** localStorage throws in private modes and when site data is blocked. */
function readStoredLayout(): RailLayout {
  try {
    const stored = window.localStorage.getItem(RAIL_STORAGE_KEY);
    return isRailLayout(stored) ? stored : "expanded";
  } catch {
    return "expanded";
  }
}

function storeLayout(layout: RailLayout): void {
  try {
    window.localStorage.setItem(RAIL_STORAGE_KEY, layout);
  } catch {
    // A reader who cannot persist still gets the width for this session.
  }
}

function applyLayout(layout: RailLayout): void {
  const root = document.documentElement;
  if (layout === "expanded") root.removeAttribute(RAIL_ATTRIBUTE);
  else root.setAttribute(RAIL_ATTRIBUTE, layout);
}

function readIsCollapsible(): boolean {
  if (typeof window.matchMedia !== "function") return true;
  return window.matchMedia(RAIL_COLLAPSIBLE_QUERY).matches;
}

/**
 * Whether the rail is showing icons only RIGHT NOW, and a toggle that persists
 * the choice.
 *
 * The attribute on <html> — not React state — is what the stylesheet reads, so
 * the width is settled by the pre-paint script in index.html and this hook only
 * has to keep the two in step afterwards. React state exists so the rail's own
 * markup (labels, tooltips, the footer readout) can follow.
 *
 * `layout` is what the reader chose and what is stored; `isCollapsed` is
 * whether that choice is in force, which it is not while the viewport is
 * narrow enough for the bar layout. Keeping the two apart is what stops the
 * preference from being quietly rewritten by a window resize.
 */
export function useRailLayout(): {
  layout: RailLayout;
  isCollapsed: boolean;
  toggle: () => void;
} {
  const [layout, setLayout] = useState<RailLayout>(readStoredLayout);
  const [isCollapsible, setIsCollapsible] = useState<boolean>(readIsCollapsible);

  // The inline script in index.html already applied the stored layout, so the
  // first run of this is a no-op — it exists to follow later changes.
  useEffect(() => {
    applyLayout(layout);
  }, [layout]);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const query = window.matchMedia(RAIL_COLLAPSIBLE_QUERY);
    const handleChange = () => setIsCollapsible(query.matches);
    // Re-read on mount: the window may have been resized between module init
    // and here.
    handleChange();
    query.addEventListener("change", handleChange);
    return () => query.removeEventListener("change", handleChange);
  }, []);

  const toggle = useCallback(() => {
    setLayout((current) => {
      const next: RailLayout = current === "collapsed" ? "expanded" : "collapsed";
      storeLayout(next);
      return next;
    });
  }, []);

  return { layout, isCollapsed: isCollapsible && layout === "collapsed", toggle };
}
