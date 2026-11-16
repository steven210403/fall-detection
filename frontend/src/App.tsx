import { useCallback, useEffect, useMemo, useState } from "react";

import { AlarmBanner } from "./components/AlarmBanner";
import { FallHistory } from "./components/FallHistory";
import { Panel } from "./components/Panel";
import { PersonList } from "./components/PersonList";
import { PreviewPanel } from "./components/PreviewPanel";
import { StatusPanel } from "./components/StatusPanel";
import { ThemeToggle } from "./components/ThemeToggle";
import { isCameraRunning } from "./components/labels";
import { activeAlarms, mergeEvents, useLiveFeed } from "./hooks/useLiveFeed";
import { useStatus } from "./hooks/useStatus";
import { useTheme } from "./hooks/useTheme";
import { api, ApiError } from "./services/api";
import type { Camera, FallEvent } from "./types";

/** How often the open alarms are re-read, as a safety net beside the live feed. */
const ACTIVE_POLL_MS = 3000;

/**
 * The fall-monitoring dashboard.
 *
 * It composes the annotated camera preview, the alarm banner, each person's fall state with
 * its measurements, the fall history, and the system status. Three decisions worth reading:
 *
 * - **An alarm is never lost to a slow connection.** Alarms arrive on the live feed the
 *   moment they are stored, and the open alarms are also re-read every few seconds, so a
 *   message the socket dropped still raises the banner.
 * - **Nothing is drawn over the video.** The backend draws boxes, skeletons, and states
 *   into the JPEG from the same frame the decisions came from.
 * - **A missing measurement is shown as missing** (`—`), never as zero.
 *
 * Layout is caregiver first: the alarm banner, then the people and their countdowns, then the
 * video, the history, and a closed diagnostics section. On a wide screen the video sits left
 * and the people right. Each panel reports its own errors, so a failed history read never
 * appears under the camera.
 */
export default function App() {
  const theme = useTheme();
  const [camera, setCamera] = useState<Camera | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [history, setHistory] = useState<FallEvent[]>([]);
  const [historyCursor, setHistoryCursor] = useState<string | null>(null);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [acknowledgeError, setAcknowledgeError] = useState<string | null>(null);
  const [polledActive, setPolledActive] = useState<FallEvent[]>([]);
  const [muted, setMuted] = useState(false);
  const [busy, setBusy] = useState(false);

  const feed = useLiveFeed();
  const statusFeed = useStatus(2000);

  const cameraId = camera?.camera_id ?? statusFeed.status?.camera.camera_id ?? null;

  const refreshCamera = useCallback(async () => {
    try {
      const cameras = await api.cameras();
      setCamera(cameras[0] ?? null);
      setCameraError(null);
    } catch (cause) {
      setCameraError(cause instanceof ApiError ? cause.message : "tidak dapat membaca kamera");
    }
  }, []);

  const refreshHistory = useCallback(async () => {
    try {
      const page = await api.fallEvents({ limit: 50 });
      setHistory(page.events);
      setHistoryCursor(page.next_cursor);
      setHistoryTotal(page.total);
      setHistoryError(null);
    } catch (cause) {
      setHistoryError(cause instanceof ApiError ? cause.message : "server tidak menjawab");
    } finally {
      setHistoryLoaded(true);
    }
  }, []);

  useEffect(() => {
    void (async () => {
      await Promise.all([refreshCamera(), refreshHistory()]);
    })();
  }, [refreshCamera, refreshHistory]);

  useEffect(() => {
    // Safety net: re-read the open alarms periodically, so an alarm message dropped by a
    // slow socket still reaches the banner.
    let cancelled = false;
    const poll = async () => {
      try {
        const page = await api.fallEvents({ active: true, limit: 20 });
        if (!cancelled) {
          setPolledActive(page.events);
        }
      } catch {
        // The status panel reports a broken backend; the banner keeps its last state.
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), ACTIVE_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  // When the set of open alarms changes (one opened or closed), re-read the history so a
  // closure the socket missed is not shown as still open.
  const activeKey = polledActive.map((event) => event.event_id).join(",");
  useEffect(() => {
    void (async () => {
      await refreshHistory();
    })();
  }, [activeKey, refreshHistory]);

  const events = useMemo(() => mergeEvents(history, feed.events), [history, feed.events]);
  const alarms = useMemo(() => activeAlarms(events, polledActive), [events, polledActive]);

  const loadOlder = useCallback(async () => {
    if (!historyCursor) {
      return;
    }
    try {
      const page = await api.fallEvents({ limit: 50, cursor: historyCursor });
      setHistory((existing) => mergeEvents(existing, page.events));
      setHistoryCursor(page.next_cursor);
      setHistoryError(null);
    } catch (cause) {
      setHistoryError(cause instanceof ApiError ? cause.message : "server tidak menjawab");
    }
  }, [historyCursor]);

  const withBusy = async (action: () => Promise<void>, onError: (message: string) => void) => {
    setBusy(true);
    try {
      await action();
    } catch (cause) {
      onError(cause instanceof ApiError ? cause.message : "server tidak menjawab");
    } finally {
      setBusy(false);
    }
  };

  const startCamera = () =>
    withBusy(async () => {
      if (cameraId) {
        await api.startCamera(cameraId);
        await refreshCamera();
      }
    }, setCameraError);

  const stopCamera = () =>
    withBusy(async () => {
      if (cameraId) {
        await api.stopCamera(cameraId);
        await refreshCamera();
      }
    }, setCameraError);

  const acknowledge = (eventId: string) =>
    withBusy(async () => {
      const event = await api.acknowledge(eventId);
      setHistory((existing) => mergeEvents(existing, [event]));
      setPolledActive((existing) =>
        existing.map((item) => (item.event_id === event.event_id ? event : item)),
      );
      setAcknowledgeError(null);
    }, setAcknowledgeError);

  const cameraState = statusFeed.status?.camera.state ?? feed.cameraState ?? camera?.state ?? null;

  return (
    <div className="min-h-dvh">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-3 px-4 py-2 sm:px-6">
          <div className="min-w-0">
            <h1 className="font-mono text-lg font-bold text-fg">Pemantauan Jatuh Lansia</h1>
            {statusFeed.status?.camera.name ? (
              <p className="truncate text-sm text-fg-muted">Sumber: {statusFeed.status.camera.name}</p>
            ) : null}
          </div>
          <ThemeToggle preference={theme.preference} onChange={theme.setPreference} />
        </div>
      </header>

      <main className="mx-auto grid max-w-[1440px] grid-cols-1 items-start gap-5 px-4 py-5 sm:px-6 lg:grid-cols-12">
        {alarms.length > 0 ? (
          <div className="order-1 lg:col-span-12">
            <AlarmBanner
              alarms={alarms}
              muted={muted}
              onToggleMute={() => setMuted((value) => !value)}
              onAcknowledge={(eventId) => void acknowledge(eventId)}
              busy={busy}
              error={acknowledgeError}
            />
          </div>
        ) : null}

        <div className="order-3 lg:order-2 lg:col-span-7 lg:row-span-2">
          {cameraId ? (
            <PreviewPanel
              cameraId={cameraId}
              previewUrl={api.previewUrl(cameraId)}
              snapshotUrl={api.snapshotUrl(cameraId)}
              connection={feed.connection}
              cameraState={cameraState}
              frameSeq={feed.frameSeq}
              previewBytes={undefined}
              hasFrame={feed.frameSeq !== null}
              onStart={() => void startCamera()}
              onStop={() => void stopCamera()}
              busy={busy}
              error={cameraError ?? feed.error}
            />
          ) : (
            <Panel title="Kamera">
              {cameraError ? (
                <p role="alert" className="mb-3 text-sm text-danger-fg">
                  Daftar kamera tidak dapat dibaca: {cameraError}
                </p>
              ) : null}
              <p className="text-sm text-fg-muted">
                Belum ada kamera. Isi <code className="font-mono text-fg">FALL_VIDEO__SOURCE_REF</code> dan
                media root di <code className="font-mono text-fg">backend/.env</code>, lalu muat ulang halaman.
              </p>
            </Panel>
          )}
        </div>

        <div className="order-2 lg:order-3 lg:col-span-5">
          <Panel
            title="Orang terpantau"
            description="Keadaan setiap orang di depan kamera. Jika seseorang tergeletak, hitung mundur alarm muncul di kartunya."
            actions={<span className="font-mono text-sm tabular-nums text-fg-muted">{feed.tracks.length} orang</span>}
          >
            <PersonList
              tracks={feed.tracks}
              feedLive={feed.connection === "live"}
              cameraRunning={isCameraRunning(cameraState)}
            />
          </Panel>
        </div>

        <div className="order-4 lg:col-span-5">
          <Panel
            title="Riwayat jatuh"
            description="Setiap baris adalah satu kejadian jatuh, dengan waktu alarm dan waktu respons."
            actions={
              <span className="font-mono text-sm tabular-nums text-fg-muted">
                {historyLoaded ? `${historyTotal} tersimpan` : "—"}
              </span>
            }
          >
            <FallHistory
              events={events}
              hasMore={historyCursor !== null}
              onLoadMore={() => void loadOlder()}
              loading={!historyLoaded}
              error={historyError}
              onRetry={() => void refreshHistory()}
            />
          </Panel>
        </div>

        <div className="order-5 lg:col-span-12">
          <StatusPanel
            status={statusFeed.status}
            error={statusFeed.error}
            loading={statusFeed.loading}
            connection={feed.connection}
            missedMessages={feed.missedMessages}
            recoveredEvents={feed.recoveredEvents}
            lastSequence={feed.lastSequence}
          />
        </div>
      </main>
    </div>
  );
}
