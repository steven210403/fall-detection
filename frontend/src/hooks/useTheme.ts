import { useCallback, useEffect, useState, useSyncExternalStore } from "react";

/** What the operator chose. `system` follows the OS setting and is the default. */
export type ThemePreference = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

/** Shared with the inline script in `index.html`, which applies the theme before first paint. */
export const THEME_STORAGE_KEY = "cctv-theme";

const DARK_QUERY = "(prefers-color-scheme: dark)";

function readPreference(): ThemePreference {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") {
      return stored;
    }
  } catch {
    // Storage can be unavailable (private mode, blocked site data); the default still works.
  }
  return "system";
}

function writePreference(preference: ThemePreference): void {
  try {
    if (preference === "system") {
      window.localStorage.removeItem(THEME_STORAGE_KEY);
    } else {
      window.localStorage.setItem(THEME_STORAGE_KEY, preference);
    }
  } catch {
    // Not persisting is acceptable: the choice still applies for this page.
  }
}

function subscribeToSystem(onChange: () => void): () => void {
  if (typeof window.matchMedia !== "function") {
    return () => {};
  }
  const query = window.matchMedia(DARK_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

function systemPrefersDark(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia(DARK_QUERY).matches;
}

/**
 * The colour theme: a stored preference, resolved against the OS setting when it is `system`.
 *
 * The OS setting is read through `useSyncExternalStore`, so switching the system theme while
 * the page is open re-renders immediately. The resolved theme is written to
 * `<html data-theme>`, which is what the CSS tokens in `index.css` key on.
 */
export function useTheme() {
  const [preference, setPreferenceState] = useState<ThemePreference>(readPreference);
  const systemDark = useSyncExternalStore(subscribeToSystem, systemPrefersDark, () => false);
  const resolved: ResolvedTheme =
    preference === "system" ? (systemDark ? "dark" : "light") : preference;

  useEffect(() => {
    document.documentElement.dataset.theme = resolved;
  }, [resolved]);

  const setPreference = useCallback((next: ThemePreference) => {
    setPreferenceState(next);
    writePreference(next);
  }, []);

  return { preference, resolved, setPreference };
}
