import { useEffect, useRef } from "react";

import type { FallEvent } from "../types";
import { describeEndReason, formatDuration, formatTime, personLabel } from "./labels";
import { alarmButtonClass } from "./ui";

interface AlarmBannerProps {
  /** Active alarms (open fall events), newest first. */
  alarms: FallEvent[];
  muted: boolean;
  onToggleMute: () => void;
  onAcknowledge: (eventId: string) => void;
  busy?: boolean;
  /** Why the last acknowledgement failed, shown on the banner the operator is looking at. */
  error?: string | null;
}

/**
 * The fall alarm: the one element that lifts off the page, with text, a glyph, and a beep.
 *
 * Shown while at least one fall event is active. The beep repeats while any active alarm has
 * not been acknowledged, and stops when every one is acknowledged or the operator mutes it.
 * Acknowledging does not close the event: it stays active until the person gets up or
 * leaves the camera, so the history keeps the real duration of the fall.
 */
export function AlarmBanner({
  alarms,
  muted,
  onToggleMute,
  onAcknowledge,
  busy = false,
  error = null,
}: AlarmBannerProps) {
  const unacknowledged = alarms.filter((alarm) => alarm.acknowledged_utc === null);
  useAlarmSound(unacknowledged.length > 0 && !muted);

  if (alarms.length === 0) {
    return null;
  }
  return (
    <section
      role="alert"
      aria-live="assertive"
      className="border-2 border-danger-strong bg-danger px-4 py-4 text-white shadow-lg"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span aria-hidden="true" className="text-3xl leading-none">
            {"⚠︎"}
          </span>
          <div>
            <p className="text-lg font-bold">
              PERINGATAN: {alarms.length === 1 ? "terdeteksi orang jatuh" : `${alarms.length} orang jatuh`}
            </p>
            <p className="mt-0.5 text-sm text-white">
              Segera periksa lokasi, lalu tekan Konfirmasi. Konfirmasi menghentikan bunyi; kejadian
              tetap tercatat sampai orang tersebut bangkit.
            </p>
          </div>
        </div>
        <button type="button" className={alarmButtonClass("outline")} onClick={onToggleMute}>
          {muted ? "Nyalakan suara" : "Matikan suara"}
        </button>
      </div>
      {error ? (
        <p className="mt-3 border border-white px-3 py-2 text-sm font-semibold">
          Konfirmasi gagal: {error}. Coba tekan Konfirmasi lagi.
        </p>
      ) : null}
      <ul className="mt-3 space-y-2">
        {alarms.map((alarm) => (
          <li
            key={alarm.event_id}
            className="flex flex-wrap items-center justify-between gap-2 bg-danger-strong px-3 py-2 text-sm"
          >
            <span>
              <span className="font-semibold">{personLabel(alarm.display_id)}</span>
              <span className="font-mono">
                {" "}
                · alarm {formatTime(alarm.alert_utc)} · respons {formatDuration(alarm.response_time_s)}
              </span>{" "}
              setelah benturan
              {alarm.active ? "" : ` · ${describeEndReason(alarm.end_reason)}`}
            </span>
            {alarm.acknowledged_utc === null ? (
              <button
                type="button"
                className={alarmButtonClass("solid")}
                disabled={busy}
                onClick={() => onAcknowledge(alarm.event_id)}
              >
                Konfirmasi
              </button>
            ) : (
              <span className="text-sm">✓ dikonfirmasi {formatTime(alarm.acknowledged_utc)}</span>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

/** A repeating two-tone beep through the Web Audio API while `active` is true. */
function useAlarmSound(active: boolean) {
  const contextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    if (!active) {
      return;
    }
    const AudioContextClass =
      typeof window === "undefined"
        ? undefined
        : (window.AudioContext ??
          (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext);
    if (AudioContextClass === undefined) {
      return;
    }
    const context = contextRef.current ?? new AudioContextClass();
    contextRef.current = context;

    const beep = () => {
      try {
        void context.resume();
        for (const [offset, frequency] of [
          [0, 880],
          [0.25, 660],
        ] as const) {
          const oscillator = context.createOscillator();
          const gain = context.createGain();
          oscillator.type = "square";
          oscillator.frequency.value = frequency;
          gain.gain.value = 0.08;
          oscillator.connect(gain).connect(context.destination);
          const start = context.currentTime + offset;
          oscillator.start(start);
          oscillator.stop(start + 0.2);
        }
      } catch {
        // A browser that blocks audio until a user gesture must not break the banner.
      }
    };
    beep();
    const timer = window.setInterval(beep, 1500);
    return () => window.clearInterval(timer);
  }, [active]);
}
