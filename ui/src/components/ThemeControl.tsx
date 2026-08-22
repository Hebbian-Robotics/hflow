import { Monitor, Moon, Sun } from "lucide-react";
import { type ComponentType, useState } from "react";
import {
  type ResolvedTheme,
  THEME_PREFERENCES,
  type ThemePreference,
  useThemePreference,
} from "../theme";

// Theme control, in the nav rail's footer next to the rest of the workspace
// readout.
//
// ONE quiet icon button that cycles system -> light -> dark -> system, showing
// the state it is in. The rail is chrome, and a three-way segmented control
// down there was reading as loud as the four destinations above it; a theme is
// set once and then forgotten, so it earns one muted glyph, not three lit ones.
//
// What a cycling toggle costs, and how each cost is paid:
//   * the icon alone does not say what pressing will do — the accessible name
//     says both ("Theme: dark. Switch to following the system theme."), and it
//     is the button's tooltip as well, so sighted readers get the same sentence;
//   * the icon alone is not announced when it changes under a screen reader —
//     a visually-hidden polite status says the new state after every press;
//   * a state three presses away is slower than one click — acceptable for a
//     setting nobody touches twice a session.
//
// The three-state semantics, the persistence and the matchMedia tracking all
// stay in theme.ts, untouched: this file only decides how they are driven.

const THEME_ICONS: Record<ThemePreference, ComponentType> = {
  system: Monitor,
  light: Sun,
  dark: Moon,
};

/** The cycle order is THEME_PREFERENCES' order — system, light, dark. */
function nextPreference(current: ThemePreference): ThemePreference {
  const index = THEME_PREFERENCES.indexOf(current);
  return THEME_PREFERENCES[(index + 1) % THEME_PREFERENCES.length] ?? "system";
}

/**
 * The state the button is in, in words. "system" names what it is currently
 * resolving to, because that is the part a reader cannot see from the glyph.
 */
function describeState(preference: ThemePreference, resolvedTheme: ResolvedTheme): string {
  return preference === "system" ? `following system (${resolvedTheme})` : preference;
}

/** The state a press moves to, phrased as the end of "Switch to …". */
function describeTarget(preference: ThemePreference): string {
  return preference === "system" ? "following the system theme" : preference;
}

export function ThemeControl() {
  const { preference, resolvedTheme, setPreference } = useThemePreference();
  // The status region stays in the DOM from the first paint so the screen
  // reader has registered it, but stays EMPTY until the reader presses the
  // button — a live region that arrives with text already in it gets announced
  // on page load, which is not a change and not worth saying.
  const [hasCycled, setHasCycled] = useState(false);
  const Icon = THEME_ICONS[preference];
  const target = nextPreference(preference);
  const label = `Theme: ${describeState(preference, resolvedTheme)}. Switch to ${describeTarget(target)}.`;
  return (
    <div className="theme-control">
      <button
        type="button"
        className="theme-toggle"
        aria-label={label}
        title={label}
        onClick={() => {
          setPreference(target);
          setHasCycled(true);
        }}
      >
        <Icon />
      </button>
      {/* Derived from the live preference rather than from remembered text, so
          it cannot drift from what the button is showing — including when
          "system" is re-entered and resolves to the theme already on screen. */}
      <span className="visually-hidden" role="status" aria-live="polite">
        {hasCycled ? `Theme: ${describeState(preference, resolvedTheme)}.` : ""}
      </span>
    </div>
  );
}
