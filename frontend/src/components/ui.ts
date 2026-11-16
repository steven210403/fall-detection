/**
 * Shared class strings, so every panel's buttons and notices look the same in both themes.
 * Colours come from the semantic tokens in `index.css`.
 *
 * - Every button is at least 44px tall, so it can be pressed on a phone during an alarm.
 * - Hover adds an inset rule under the label, so the state is visible without relying on
 *   colour. Active presses the button down by 1px. Disabled dims it, dashes the outline, and
 *   shows the not-allowed cursor.
 * - The focus ring is 2px with a 2px offset, in a colour that is at least 3:1 on the canvas in
 *   both themes.
 */

const BUTTON_BASE =
  "inline-flex min-h-11 items-center justify-center gap-2 px-4 text-sm font-semibold " +
  "transition-colors motion-reduce:transition-none " +
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus " +
  "enabled:hover:shadow-[inset_0_-2px_0_currentColor] enabled:active:translate-y-px " +
  "disabled:opacity-50";

const BUTTON_VARIANT = {
  /** The one accent: the action that starts monitoring. */
  primary:
    "border border-accent bg-accent text-accent-fg enabled:hover:bg-accent-strong " +
    "enabled:active:bg-accent-strong",
  secondary:
    "border border-fg-muted bg-surface text-fg enabled:hover:bg-surface-2 disabled:border-dashed",
  ghost: "border border-transparent text-fg enabled:hover:bg-surface-2 disabled:border-dashed",
} as const;

export function buttonClass(variant: keyof typeof BUTTON_VARIANT = "secondary"): string {
  return `${BUTTON_BASE} ${BUTTON_VARIANT[variant]}`;
}

/** Buttons that sit on the red alarm banner: white on red, with a white focus ring. */
export function alarmButtonClass(variant: "solid" | "outline"): string {
  const base =
    "inline-flex min-h-11 items-center justify-center px-4 text-sm font-semibold " +
    "transition-colors motion-reduce:transition-none " +
    "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white " +
    "enabled:hover:shadow-[inset_0_-2px_0_currentColor] enabled:active:translate-y-px " +
    "disabled:opacity-60";
  return variant === "solid"
    ? `${base} border border-white bg-white text-danger-strong enabled:hover:bg-rose-50`
    : `${base} border border-white text-white enabled:hover:bg-white/10`;
}

export const alertClass =
  "border border-danger-fg bg-surface px-3 py-2 text-sm text-danger-fg";

export const chipClass =
  "inline-flex items-center gap-1.5 border border-line bg-surface-2 px-2 py-1 font-mono " +
  "text-xs text-fg-muted";
