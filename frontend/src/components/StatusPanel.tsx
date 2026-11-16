import { useId } from "react";
import type { ReactNode } from "react";

import type { StatusPayload } from "../types";
import { cameraStateWord, connectionWord, formatBytes } from "./labels";
import { alertClass } from "./ui";

interface StatusPanelProps {
  status: StatusPayload | null;
  error: string | null;
  loading: boolean;
  connection: string;
  missedMessages: number;
  recoveredEvents: number;
  lastSequence: number;
}

/**
 * System diagnostics: performance, camera, model, events, and resources.
 *
 * The dashboard is for a caregiver first, so this sits closed at the bottom of the page. The
 * one line in its summary always says whether the pose detector is ready, the measured FPS,
 * and whether the thresholds are calibrated, so a broken model is visible without opening it.
 *
 * - **Model readiness is separate from camera state**: an offline camera and an unloaded
 *   detector are different facts with different responses, so they are different rows.
 * - **Performance is measured, not claimed**: FPS is frames processed over the last few
 *   seconds of wall time, and latency is per stage (pose detection, tracking, fall logic).
 * - **The thresholds say whether they are calibrated**, so a run on literature defaults is
 *   never mistaken for one on the frozen calibration.
 */
export function StatusPanel({
  status,
  error,
  loading,
  connection,
  missedMessages,
  recoveredEvents,
  lastSequence,
}: StatusPanelProps) {
  const headingId = useId();

  const detection = status?.detection ?? {};
  const tracking = status?.tracking ?? {};
  const fall = status?.fall ?? {};
  const thresholds = fall.thresholds ?? {};
  const performance = status?.performance ?? {};
  const latency = performance.latency ?? {};
  const events = status?.events ?? {};
  const streaming = status?.streaming ?? {};
  const resources = status?.resources ?? {};
  const pipelineStats = (status?.pipeline?.stats ?? {}) as Record<string, number>;
  const liveStats = (streaming.live?.stats ?? {}) as Record<string, number>;
  const model = (detection.model ?? {}) as Record<string, unknown>;

  const fps =
    performance.fps !== null && performance.fps !== undefined ? performance.fps.toFixed(1) : "—";
  const detectorText =
    detection.enabled === false
      ? "nonaktif"
      : detection.ready === true
        ? String(model.model_name ?? "siap")
        : "belum dimuat";
  const detectorProblem = status !== null && detection.ready !== true;

  return (
    <section aria-labelledby={headingId} className="border border-line bg-surface">
      {error ? (
        <p role="alert" className={`m-4 ${alertClass}`}>
          {error}
        </p>
      ) : null}
      <details className="group">
        <summary className="flex min-h-11 flex-wrap items-center justify-between gap-x-4 gap-y-1 px-4 py-3 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-focus [&::-webkit-details-marker]:hidden">
          <h2 id={headingId} className="font-mono text-sm font-semibold text-fg">
            <span aria-hidden="true" className="mr-2 inline-block group-open:hidden">
              ▸
            </span>
            <span aria-hidden="true" className="mr-2 hidden group-open:inline-block">
              ▾
            </span>
            Diagnostik sistem
          </h2>
          {loading && status === null ? (
            <span role="status" className="text-sm text-fg-muted">
              Membaca status…
            </span>
          ) : (
            <span className="font-mono text-sm text-fg-muted">
              <span className={detectorProblem ? "font-semibold text-danger-fg" : ""}>
                {detectorProblem ? "Detektor pose belum siap" : "Detektor siap"}
              </span>
              {" · "}
              {fps} FPS · {thresholds.calibrated ? "ambang terkalibrasi" : "ambang nilai literatur"}
            </span>
          )}
        </summary>

        <div className="border-t border-line p-4">
          <p className="mb-4 font-mono text-xs text-fg-muted">
            Live feed: {connectionWord(connection).text} · pesan ke-{lastSequence}
            {missedMessages > 0 ? ` · ${missedMessages} terlewat` : ""}
            {recoveredEvents > 0 ? ` · ${recoveredEvents} dipulihkan` : ""}
          </p>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            <Group title="Kinerja">
              <Row term="FPS" value={fps} emphasis />
              <Row term="Latensi total" value={formatLatency(latency.total_ms)} />
              <Row term="Deteksi pose" value={formatLatency(latency.detection_ms)} />
              <Row term="Pelacakan" value={formatLatency(latency.tracking_ms)} />
              <Row term="Logika jatuh" value={formatLatency(latency.fall_ms)} />
            </Group>
            <Group title="Model">
              <Row term="Detektor pose" value={detectorText} />
              <Row
                term="Pelacak"
                value={tracking.enabled === false ? "nonaktif" : trackerName(tracking.tracker)}
              />
              <Row
                term="Ambang"
                value={
                  thresholds.ar_threshold !== undefined
                    ? `AR≥${thresholds.ar_threshold} · θ<${thresholds.angle_lying_deg}° · Vy>${thresholds.vy_threshold}/dtk · long-lie ${thresholds.long_lie_s} dtk`
                    : "—"
                }
              />
              <Row
                term="Kalibrasi"
                value={thresholds.calibrated ? "terkalibrasi (beku)" : "nilai literatur"}
              />
            </Group>
            <Group title="Kamera dan kejadian">
              <Row
                term="Status kamera"
                value={status?.camera.state ? cameraStateWord(status.camera.state).text : "—"}
              />
              <Row
                term="Frame dibaca"
                value={
                  pipelineStats.frames_read !== undefined
                    ? `${pipelineStats.frames_read} (dibuang ${pipelineStats.frames_dropped ?? 0})`
                    : "—"
                }
              />
              <Row term="Alarm aktif" value={String(fall.active_alarms ?? 0)} />
              <Row term="Alarm sesi ini" value={String(events.stats?.opened ?? 0)} />
              <Row term="Pengalihan identitas" value={String(fall.stats?.handovers ?? 0)} />
              <Row
                term="Klien live feed"
                value={`${liveStats.clients_connected ?? 0} terhubung · ${liveStats.dropped ?? 0} dibuang`}
              />
              <Row
                term="Memori proses"
                value={`${formatBytes(resources.rss_bytes)} · ${resources.threads ?? "—"} thread`}
              />
              <Row term="Perangkat" value={String(resources.device ?? "—")} />
            </Group>
          </div>
        </div>
      </details>
    </section>
  );
}

/** The tracker's name. The status payload sends an object with `name`, older ones a string. */
function trackerName(tracker: unknown): string {
  if (typeof tracker === "string") {
    return tracker;
  }
  if (tracker && typeof tracker === "object" && "name" in tracker) {
    return String((tracker as { name: unknown }).name);
  }
  return "—";
}

function formatLatency(stage: { mean_ms: number; p95_ms: number } | undefined): string {
  if (!stage) {
    return "—";
  }
  return `${stage.mean_ms.toFixed(1)} ms (p95 ${stage.p95_ms.toFixed(1)})`;
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <h3 className="mb-1.5 font-mono text-xs font-semibold text-fg-muted">{title}</h3>
      <dl className="divide-y divide-line border border-line bg-surface-2 text-sm">{children}</dl>
    </div>
  );
}

function Row({ term, value, emphasis = false }: { term: string; value: string; emphasis?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 px-2.5 py-1.5">
      <dt className="shrink-0 text-fg-muted">{term}</dt>
      <dd
        className={`min-w-0 text-right font-mono break-words text-fg ${emphasis ? "text-base font-semibold tabular-nums" : "text-xs"}`}
      >
        {value}
      </dd>
    </div>
  );
}
