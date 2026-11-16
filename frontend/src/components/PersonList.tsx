import type { TrackSnapshot } from "../types";
import { formatDuration, formatNumber, isFallInProgress, personLabel, presentationFor } from "./labels";

interface PersonListProps {
  tracks: TrackSnapshot[];
  /** Whether the live feed is connected. Without it the list is not measured, not empty. */
  feedLive?: boolean;
  /** Whether the camera is running. A stopped camera watches nobody. */
  cameraRunning?: boolean;
}

/**
 * Each tracked person's fall state and the three measurements it was decided from.
 *
 * The torso angle, aspect ratio, and hip velocity are shown as measured, with `—` when a
 * value could not be measured on the latest frame (never a zero). While a person lies after
 * a sudden descent, the long-lie countdown is the largest thing on the card.
 *
 * An empty list is only "nobody in view" when the feed is connected and the camera runs;
 * otherwise the list says why it cannot know.
 */
export function PersonList({ tracks, feedLive = true, cameraRunning = true }: PersonListProps) {
  if (!feedLive && tracks.length === 0) {
    return (
      <p role="status" className="text-sm text-fg-muted">
        Koneksi live feed terputus, jadi keadaan orang di depan kamera tidak diketahui. Halaman
        menyambung ulang sendiri, dan daftar terisi lagi setelah tersambung.
      </p>
    );
  }
  if (tracks.length === 0) {
    return (
      <p role="status" className="text-sm text-fg-muted">
        {cameraRunning
          ? "Tidak ada orang yang terlacak di depan kamera saat ini."
          : "Tidak ada orang yang terlacak karena kamera tidak berjalan. Tekan Mulai kamera untuk memantau."}
      </p>
    );
  }
  return (
    <div>
      {!feedLive ? (
        <p role="status" className="mb-3 border border-danger-fg px-3 py-2 text-sm text-danger-fg">
          Koneksi live feed terputus. Kartu di bawah adalah data terakhir sebelum terputus, bukan
          keadaan sekarang.
        </p>
      ) : null}
      <ul className="space-y-3">
        {tracks.map((track) => {
          const fall = track.fall;
          const state = fall?.state ?? "unknown";
          const presentation = presentationFor(state);
          const features = fall?.features;
          const name = personLabel(track.display_id);
          const countdown = fall?.long_lie_remaining_s;
          return (
            <li
              key={track.track_uuid}
              className={`marks border border-line bg-surface-2 p-3 ${isFallInProgress(state) ? "marks-danger" : ""}`}
              aria-label={`${name}: ${presentation.text}`}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-mono text-base font-semibold text-fg">{name}</span>
                <span
                  className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-sm font-semibold ${presentation.className}`}
                  title={presentation.hint}
                >
                  <span aria-hidden="true">{presentation.glyph}</span>
                  {presentation.text}
                </span>
              </div>
              {track.stale ? (
                <p className="mt-1 text-sm text-fg-muted">
                  Tidak terlihat selama {formatDuration(track.time_since_last_observation_s)}
                </p>
              ) : null}
              {countdown !== null && countdown !== undefined ? (
                <p className="mt-3 border-2 border-danger-fg px-3 py-2 text-danger-fg">
                  <span className="block text-sm font-semibold">Alarm berbunyi jika ia tidak bangkit</span>
                  <span className="font-mono text-2xl font-bold tabular-nums">
                    Alarm dalam {formatDuration(countdown)}
                  </span>
                </p>
              ) : null}
              <dl className="mt-3 grid grid-cols-3 divide-x divide-line border border-line bg-surface text-sm">
                <Measure term="Sudut torso" value={`${formatNumber(features?.torso_angle_deg, 0)}°`} />
                <Measure term="Rasio W/H" value={formatNumber(features?.aspect_ratio)} />
                <Measure term="Vy (tubuh/dtk)" value={formatNumber(features?.vy_norm)} />
              </dl>
              <p className="mt-2 text-sm text-fg-muted">
                Lama di status ini: {formatDuration(fall?.state_duration_s)}. Titik kunci terlihat:{" "}
                {features?.visible_keypoints ?? "—"} dari 17.
              </p>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Measure({ term, value }: { term: string; value: string }) {
  return (
    <div className="min-w-0 px-2 py-1.5">
      <dt className="text-xs text-fg-muted">{term}</dt>
      <dd className="font-mono text-base tabular-nums text-fg">{value.startsWith("—") ? "—" : value}</dd>
    </div>
  );
}
