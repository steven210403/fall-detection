import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ThemeToggle } from "../src/components/ThemeToggle";
import { THEME_STORAGE_KEY, useTheme } from "../src/hooks/useTheme";

/** A controllable `prefers-color-scheme: dark` query, since jsdom has no `matchMedia`. */
function mockSystemTheme(initialDark: boolean) {
  const listeners = new Set<() => void>();
  const query = {
    matches: initialDark,
    addEventListener: (_: string, listener: () => void) => listeners.add(listener),
    removeEventListener: (_: string, listener: () => void) => listeners.delete(listener),
  };
  vi.stubGlobal("matchMedia", () => query);
  return (dark: boolean) => {
    query.matches = dark;
    for (const listener of listeners) {
      listener();
    }
  };
}

function Harness() {
  const theme = useTheme();
  return (
    <>
      <ThemeToggle preference={theme.preference} onChange={theme.setPreference} />
      <output>{theme.resolved}</output>
    </>
  );
}

describe("theme", () => {
  beforeEach(() => {
    window.localStorage.clear();
    delete document.documentElement.dataset.theme;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("follows the system theme by default", () => {
    mockSystemTheme(true);
    render(<Harness />);
    expect(screen.getByRole("radio", { name: "Ikuti tema sistem" })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("tracks a system change while following the system", () => {
    const setSystemDark = mockSystemTheme(false);
    render(<Harness />);
    expect(document.documentElement.dataset.theme).toBe("light");
    act(() => setSystemDark(true));
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("keeps an explicit choice over the system and persists it", () => {
    const setSystemDark = mockSystemTheme(true);
    render(<Harness />);
    fireEvent.click(screen.getByRole("radio", { name: "Tema terang" }));
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");
    act(() => setSystemDark(true));
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("forgets the stored choice when switched back to system", () => {
    mockSystemTheme(false);
    window.localStorage.setItem(THEME_STORAGE_KEY, "dark");
    render(<Harness />);
    expect(document.documentElement.dataset.theme).toBe("dark");
    fireEvent.click(screen.getByRole("radio", { name: "Ikuti tema sistem" }));
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBeNull();
    expect(document.documentElement.dataset.theme).toBe("light");
  });
});
