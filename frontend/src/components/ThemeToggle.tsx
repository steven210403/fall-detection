import type { ThemePreference } from "../hooks/useTheme";

interface ThemeToggleProps {
  preference: ThemePreference;
  onChange: (preference: ThemePreference) => void;
}

const OPTIONS: { value: ThemePreference; label: string; icon: string }[] = [
  {
    value: "system",
    label: "Ikuti tema sistem",
    icon: "M3 5h18v11H3zM8 20h8M12 16v4",
  },
  {
    value: "light",
    label: "Tema terang",
    icon: "M12 4V2M12 22v-2M4 12H2M22 12h-2M5.6 5.6 4.2 4.2M19.8 19.8l-1.4-1.4M5.6 18.4l-1.4 1.4M19.8 4.2l-1.4 1.4M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z",
  },
  {
    value: "dark",
    label: "Tema gelap",
    icon: "M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z",
  },
];

/**
 * A three-way theme switch: follow the system (the default), or force light or dark.
 *
 * It is a radio group rather than a single toggle so "follow the system" stays reachable
 * after an explicit choice; each option has an accessible name, not an icon alone.
 */
export function ThemeToggle({ preference, onChange }: ThemeToggleProps) {
  return (
    <div
      role="radiogroup"
      aria-label="Tema warna"
      className="inline-flex items-center border border-line bg-surface-2"
    >
      {OPTIONS.map((option) => {
        const selected = option.value === preference;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            aria-label={option.label}
            title={option.label}
            onClick={() => onChange(option.value)}
            className={`grid h-11 w-11 place-items-center transition-colors motion-reduce:transition-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:relative focus-visible:z-10 focus-visible:outline-focus active:translate-y-px ${
              selected
                ? "bg-fg text-canvas"
                : "text-fg-muted hover:bg-surface hover:text-fg hover:shadow-[inset_0_-2px_0_currentColor]"
            }`}
          >
            <svg
              aria-hidden="true"
              viewBox="0 0 24 24"
              className="h-4 w-4"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.8}
              strokeLinecap="square"
              strokeLinejoin="miter"
            >
              <path d={option.icon} />
            </svg>
          </button>
        );
      })}
    </div>
  );
}
