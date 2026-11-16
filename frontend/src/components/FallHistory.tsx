import type { FallEvent } from "../types";
import { describeEndReason, formatDuration, formatNumber, formatTime, personLabel } from "./labels";
import { alertClass, buttonClass } from "./ui";

interface FallHistoryProps {
  events: FallEvent[];
  hasMore: boolean;
  onLoadMore: () => void;
  /** True until the first page of history has been read. */
  loading?: boolean;
  /** Why the history could not be read, shown in this panel rather than elsewhere. */
  error?: string | null;
  onRetry?: () => void;
}

/**
 * The stored fall events, newest alarm first, as a ruled log.
 *
 * Each row shows the observation time of the alarm, the response time (from the impact to
 * the alarm), the measured extremes of the fall, and why the event closed. Before the first
 * page arrives the panel says it is reading, so an empty log is never claimed too early.
 */
export function FallHistory({
  events,
  hasMore,
  onLoadMore,
  loading = false,
  error = null,
  onRetry,
}: FallHistoryProps) {
  const errorNotice = error ? (
    <div role="alert" className={`mb-3 flex flex-wrap items-center justify-between gap-2 ${alertClass}`}>
      <span>Riwayat tidak dapat dibaca: {error}</span>
      {onRetry ? (
        <button type="button" className={buttonClass("secondary")} onClick={onRetry}>
          Baca ulang riwayat
        </button>
      ) : null}
    </div>
  ) : null;

  if (events.length === 0) {
    if (loading && !error) {
      return (
        <p role="status" className="text-sm text-fg-muted">
          Membaca riwayat…
        </p>
      );
    }
    return (
      <div>
        {errorNotice}
        {error ? null : (
          <p role="status" className="text-sm text-fg-muted">
            Belum ada kejadian jatuh yang tercatat. Setiap alarm tercatat di sini begitu berbunyi.
          </p>
        )}
      </div>
    );
  }
  return (
    <div>
      {errorNotice}
      <ul className="divide-y divide-line border-y border-line">
        {events.map((event) => (
          <li key={event.event_id} className="py-2.5 text-sm">
            <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
              <span className="font-semibold text-fg">
                <span aria-hidden="true">{event.active ? "⚠︎ " : "✓ "}</span>
                {personLabel(event.display_id)}
                <span className="font-mono font-normal text-fg-muted"> {formatTime(event.alert_utc)}</span>
              </span>
              <span className={event.active ? "font-semibold text-danger-fg" : "text-fg-muted"}>
                {event.active ? "aktif" : describeEndReason(event.end_reason)}
              </span>
            </div>
            <p className="mt-0.5 font-mono text-xs text-fg-muted">
              respons {formatDuration(event.response_time_s)} · Vy puncak {formatNumber(event.peak_vy)} ·
              sudut min {formatNumber(event.min_torso_angle_deg, 0)}° · rasio maks{" "}
              {formatNumber(event.max_aspect_ratio)}
              {event.ended_utc !== null
                ? ` · durasi ${formatDuration(event.ended_utc - event.alert_utc)}`
                : ""}
              {event.acknowledged_utc !== null ? " · dikonfirmasi" : ""}
            </p>
          </li>
        ))}
      </ul>
      {hasMore ? (
        <button type="button" className={`mt-3 ${buttonClass("ghost")}`} onClick={onLoadMore}>
          Muat lebih lama
        </button>
      ) : null}
    </div>
  );
}
