import { useCallback, useEffect, useState } from "react";

// Theme preference. Three states, because "follow the system" is a real
// choice and not the absence of one — a two-state toggle cannot express it,
// and a reader who has picked light on a dark laptop must keep light.
//
// styles.css is authoritative for what each state LOOKS like; this module only
// decides which of the three is in force and stamps it on <html>.

export type ThemePreference = "system" | "light" | "dark";

/** What "system" currently resolves to, and the only two things it can be. */
export type ResolvedTheme = "light" | "dark";

export const THEME_PREFERENCES: readonly ThemePreference[] = ["system", "light", "dark"];

/**
 * localStorage key holding the ThemePreference, as one of the three literal
 * strings above.
 *
 * Read in two places that must agree: this module, and the inline pre-paint
 * script at the top of index.html that applies the stored choice before React
 * mounts. Change one, change the other.
 */
export const THEME_STORAGE_KEY = "hflow-ui-theme";

/**
 * The attribute styles.css keys off. Absent means "system" — no attribute, so
 * the stylesheet's prefers-color-scheme block decides and keeps deciding as
 * the OS changes, with no JS in the loop.
 */
const THEME_ATTRIBUTE = "data-theme";

/** Present only while a theme change is being cross-faded. */
const THEME_CHANGING_ATTRIBUTE = "data-theme-changing";

/** Must outlast the transition in styles.css. */
const THEME_TRANSITION_MS = 160;

const DARK_QUERY = "(prefers-color-scheme: dark)";

function isThemePreference(value: unknown): value is ThemePreference {
  return value === "system" || value === "light" || value === "dark";
}

/** localStorage throws in private modes and when site data is blocked. */
function readStoredPreference(): ThemePreference {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isThemePreference(stored) ? stored : "system";
  } catch {
    return "system";
  }
}

function storePreference(preference: ThemePreference): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, preference);
  } catch {
    // A reader who cannot persist still gets the theme for this session.
  }
}

function readSystemTheme(): ResolvedTheme {
  if (typeof window.matchMedia !== "function") return "light";
  return window.matchMedia(DARK_QUERY).matches ? "dark" : "light";
}

function applyPreference(preference: ThemePreference): void {
  const root = document.documentElement;
  if (preference === "system") root.removeAttribute(THEME_ATTRIBUTE);
  else root.setAttribute(THEME_ATTRIBUTE, preference);
}

/**
 * The reader's theme preference, the theme it currently resolves to, and a
 * setter that persists the choice.
 *
 * "system" tracks the OS live in two independent ways, on purpose: the
 * stylesheet's prefers-color-scheme block repaints without any JS, and the
 * matchMedia listener here keeps `resolvedTheme` honest so the control can say
 * which way "system" is currently leaning while the reader is watching.
 */
export function useThemePreference(): {
  preference: ThemePreference;
  resolvedTheme: ResolvedTheme;
  setPreference: (preference: ThemePreference) => void;
} {
  const [preference, setPreferenceState] = useState<ThemePreference>(readStoredPreference);
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>(readSystemTheme);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const query = window.matchMedia(DARK_QUERY);
    const handleChange = () => setSystemTheme(query.matches ? "dark" : "light");
    // Re-read on mount: the OS may have flipped between module init and here.
    handleChange();
    query.addEventListener("change", handleChange);
    return () => query.removeEventListener("change", handleChange);
  }, []);

  // The inline script in index.html already applied the stored preference, so
  // the first run of this is a no-op — it exists to follow later changes.
  useEffect(() => {
    applyPreference(preference);
  }, [preference]);

  const setPreference = useCallback((next: ThemePreference) => {
    // Cross-fade the swap. The attribute is transient and the transition it
    // enables is disabled outright under prefers-reduced-motion in styles.css,
    // so there is one place to look for whether the UI animates.
    const root = document.documentElement;
    root.setAttribute(THEME_CHANGING_ATTRIBUTE, "");
    window.setTimeout(() => root.removeAttribute(THEME_CHANGING_ATTRIBUTE), THEME_TRANSITION_MS);
    setPreferenceState(next);
    storePreference(next);
  }, []);

  return {
    preference,
    resolvedTheme: preference === "system" ? systemTheme : preference,
    setPreference,
  };
}
