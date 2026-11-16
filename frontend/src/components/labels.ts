/**
 * Presentation helpers: the single mapping from a fall state or a code to what a reader sees.
 *
 * Every state has a **word** and a **glyph**, and colour is a redundant third cue, never
 * the only one, so the dashboard stays readable in greyscale and for colour-blind viewers.
 * The dashboard's wording is Bahasa Indonesia.
 */

import type { FallState } from "../types";

export interface StatePresentation {
  /** The word shown to the reader. */
  text: string;
  /** A glyph that survives greyscale, for the same reason the text does. */
  glyph: string;
  /** What the state means, for a tooltip or a legend. */
  hint: string;
  /** Tailwind classes, used as a redundant cue rather than as the only one. */
  className: string;
}

const STATE_PRESENTATION: Record<FallState, StatePresentation> = {
  unknown: {
    text: "Tidak terukur",
    glyph: "?",
    hint: "Titik kunci tubuh belum terlihat cukup jelas untuk diukur.",
    className: "border border-dashed border-fg-muted text-fg-muted",
  },
  normal: {
    text: "Normal",
    glyph: "●",
    hint: "Berdiri atau bergerak wajar.",
    className: "border border-line-strong text-fg",
  },
  falling: {
    text: "Kemungkinan jatuh",
    glyph: "▼",
    hint: "Panggul turun mendadak. Sistem menunggu apakah tubuh berbaring.",
    className: "border border-danger-fg text-danger-fg",
  },
  impact: {
    text: "Tergeletak",
    glyph: "▬",
    hint: "Tubuh berbaring setelah turun mendadak. Alarm berbunyi jika ia tidak bangkit.",
    className: "border-2 border-danger-fg font-bold text-danger-fg",
  },
  fallen: {
    text: "Jatuh: alarm aktif",
    // U+FE0E keeps the warning sign as a text glyph instead of a colour emoji.
    glyph: "\u26A0\uFE0E",
    hint: "Orang ini tidak bangkit dalam waktu konfirmasi, jadi alarm berbunyi.",
    className: "border border-danger bg-danger text-white",
  },
};

/** States in which a person card frames itself in red: a fall is in progress or confirmed. */
export function isFallInProgress(state: FallState): boolean {
  return state === "falling" || state === "impact" || state === "fallen";
}

export function presentationFor(state: FallState): StatePresentation {
  return STATE_PRESENTATION[state] ?? STATE_PRESENTATION.unknown;
}

/** A measured value with a fixed number of decimals, or `—` when it was not measured. */
export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(digits);
}

/** A duration in seconds, rendered compactly. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0) {
    return "—";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(1)} dtk`;
  }
  if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.round(seconds - minutes * 60);
    return `${minutes} mnt ${remainder} dtk`;
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds - hours * 3600) / 60);
  return `${hours} jam ${minutes} mnt`;
}

/** A UTC epoch timestamp rendered as a local clock time. */
export function formatTime(epochSeconds: number | null | undefined): string {
  if (epochSeconds === null || epochSeconds === undefined || !Number.isFinite(epochSeconds) || epochSeconds <= 0) {
    return "—";
  }
  return new Date(epochSeconds * 1000).toLocaleTimeString();
}

/** A byte count rendered with a unit, for the memory and preview figures. */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === undefined || bytes === null || !Number.isFinite(bytes) || bytes <= 0) {
    return "—";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

const END_REASON_TEXT: Record<string, string> = {
  recovered: "bangkit kembali",
  track_lost: "orang tidak terlacak lagi",
  session_ended: "sesi kamera berganti",
  stopped: "kamera dihentikan",
  max_duration: "melewati durasi maksimum",
};

/** Why a fall event closed, in words. An unknown code is shown as-is, never hidden. */
export function describeEndReason(reason: string | null | undefined): string {
  if (!reason) {
    return "masih aktif";
  }
  return END_REASON_TEXT[reason] ?? reason.replaceAll("_", " ");
}

/** A person's name on screen. A missing display number is said in words, not as `#?`. */
export function personLabel(displayId: number | null | undefined): string {
  return displayId === null || displayId === undefined ? "Orang tanpa nomor" : `Orang #${displayId}`;
}

export interface StateWord {
  text: string;
  glyph: string;
}

const CAMERA_STATE_WORD: Record<string, StateWord> = {
  idle: { text: "belum dimulai", glyph: "○" },
  connecting: { text: "menyambung", glyph: "◌" },
  live: { text: "berjalan", glyph: "●" },
  reconnecting: { text: "menyambung ulang", glyph: "◌" },
  offline: { text: "terputus", glyph: "×" },
  ended: { text: "video selesai", glyph: "■" },
};

/** The camera state in words. An unknown code is shown as-is, never hidden. */
export function cameraStateWord(state: string | null | undefined): StateWord {
  if (!state) {
    return { text: "tidak diketahui", glyph: "?" };
  }
  return CAMERA_STATE_WORD[state] ?? { text: state, glyph: "?" };
}

const CONNECTION_WORD: Record<string, StateWord> = {
  connecting: { text: "menyambung", glyph: "◌" },
  live: { text: "tersambung", glyph: "●" },
  reconnecting: { text: "menyambung ulang", glyph: "◌" },
  offline: { text: "terputus", glyph: "×" },
  closed: { text: "ditutup", glyph: "■" },
};

/** The live-feed (WebSocket) state in words. */
export function connectionWord(connection: string | null | undefined): StateWord {
  if (!connection) {
    return { text: "tidak diketahui", glyph: "?" };
  }
  return CONNECTION_WORD[connection] ?? { text: connection, glyph: "?" };
}

/** Camera states in which the pipeline is running or trying to. */
export function isCameraRunning(cameraState: string | null | undefined): boolean {
  return cameraState === "live" || cameraState === "connecting" || cameraState === "reconnecting";
}
