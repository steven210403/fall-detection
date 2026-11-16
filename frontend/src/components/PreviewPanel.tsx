import type { ConnectionState } from "../hooks/useLiveFeed";
import { cameraStateWord, connectionWord, formatBytes, isCameraRunning } from "./labels";
import { Panel } from "./Panel";
import { alertClass, buttonClass, chipClass } from "./ui";

interface PreviewPanelProps {
  cameraId: string;
  previewUrl: string;
  snapshotUrl: string;
  connection: ConnectionState;
  cameraState: string | null;
  frameSeq: number | null;
  previewBytes?: number;
  /** Whether a frame has been encoded; the MJPEG stream 404s until one has. */
  hasFrame: boolean;
  onStart: () => void;
  onStop: () => void;
  busy?: boolean;
  error?: string | null;
}

/**
 * The annotated preview and camera controls.
 *
 * The backend draws the boxes, skeletons, and `#id` captions into the JPEG from the same frame
 * the fall decisions came from, so the client renders the image as delivered and never
 * overlays metadata on it: a newer snapshot over an older frame would pair the wrong data.
 * The per-person details live in the person list, framed with the same corner marks.
 *
 * The camera state and the live-feed state are shown in words, because a frozen preview looks
 * identical whether the camera stopped or the socket dropped, and those need different
 * responses.
 */
export function PreviewPanel({
  cameraId,
  previewUrl,
  snapshotUrl,
  connection,
  cameraState,
  frameSeq,
  previewBytes,
  hasFrame,
  onStart,
  onStop,
  busy = false,
  error = null,
}: PreviewPanelProps) {
  const running = isCameraRunning(cameraState);
  const camera = cameraStateWord(cameraState);
  const feed = connectionWord(connection);

  return (
    <Panel
      title="Kamera"
      description="Setiap orang di gambar diberi kotak dan nomor yang sama dengan kartunya di daftar Orang terpantau."
      actions={
        <>
          <button
            type="button"
            onClick={onStart}
            disabled={busy || running}
            className={buttonClass("primary")}
          >
            Mulai kamera
          </button>
          <button
            type="button"
            onClick={onStop}
            disabled={busy || !running}
            className={buttonClass("secondary")}
          >
            Hentikan kamera
          </button>
        </>
      }
    >
      <p role="status" className="flex flex-wrap items-center gap-2">
        <span className={chipClass}>
          <span aria-hidden="true">{camera.glyph}</span>
          Kamera: <span className="text-fg">{camera.text}</span>
        </span>
        <span className={chipClass}>
          <span aria-hidden="true">{feed.glyph}</span>
          Live feed: <span className="text-fg">{feed.text}</span>
        </span>
        {frameSeq !== null ? <span className={`${chipClass} tabular-nums`}>frame {frameSeq}</span> : null}
        {previewBytes ? <span className={chipClass}>JPEG {formatBytes(previewBytes)}</span> : null}
      </p>

      <div className="marks mt-3 aspect-video border border-line bg-viewport">
        {hasFrame ? (
          // ``key`` on the camera id so switching cameras remounts the stream instead of
          // reusing a multipart response that belongs to the previous one.
          <img
            key={cameraId}
            src={previewUrl}
            alt="Pratinjau kamera beranotasi dengan kerangka tubuh dan status jatuh"
            width={1280}
            height={720}
            fetchPriority="high"
            className="block h-full w-full object-contain"
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
            <svg
              aria-hidden="true"
              viewBox="0 0 24 24"
              className="h-8 w-8 text-[#8b9bb4]"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.5}
              strokeLinecap="square"
            >
              <path d="M15 10l5-3v10l-5-3M3 7h12v10H3z" />
            </svg>
            <p className="max-w-sm text-sm text-[#8b9bb4]">
              {running
                ? "Belum ada frame. Kamera sedang menyambung; gambar muncul setelah frame pertama diproses."
                : "Belum ada frame. Tekan Mulai kamera untuk menampilkan gambar."}
            </p>
          </div>
        )}
      </div>

      {hasFrame ? (
        // Only offered once a frame exists: before that the endpoint answers 404.
        <p className="mt-2">
          <a
            className="inline-flex min-h-11 items-center text-sm font-medium text-fg underline underline-offset-4 hover:decoration-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
            href={snapshotUrl}
          >
            Buka frame diam (JPEG)
          </a>
        </p>
      ) : null}

      {error ? (
        <p role="alert" className={`mt-3 ${alertClass}`}>
          {error}
        </p>
      ) : null}
    </Panel>
  );
}
