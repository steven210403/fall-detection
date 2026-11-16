"""Preview frame encoding (thesis plan §8, §18.1, §18.2).

§18.2 chooses MJPEG for the MVP: "High because every frame is JPEG | Adequate for a
5 FPS preview; limited browser frame control | Low [complexity]". §8 caps it at 5 FPS
and 960x540, and §9.2 says what to do under load — *"Reduce preview work first during
overload"* — which is why the encoder is rate-limited rather than encoding every
decoded frame and letting the client discard the excess.

Two rules from §18.1 shape the API of this module:

* **"The backend draws the preview from the same frame used for tracking outputs."**
  The annotated array is handed in already drawn from one specific frame, and the
  envelope that comes out names that frame's sequence number. A consumer therefore
  cannot accidentally pair a JPEG with another frame's metadata.
* **Bandwidth is measured, not assumed.** §18.2 gives the arithmetic
  (``bytes x FPS x 8 / 1e6``) as "an arithmetic example, not a benchmark", so the
  encoded size and the resulting rate are reported rather than asserted.

JPEG encoding uses OpenCV, imported inside the method so importing
``fall_detection.streaming`` stays cheap and free of a GUI build — the same rule
``detection.annotate`` and ``features.motion`` follow.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fall_detection.config import StreamingSettings
from fall_detection.core.clock import Clock, SystemClock
from fall_detection.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    import numpy as np

__all__ = ["FrameEnvelope", "PreviewStats", "PreviewStream", "fit_within"]

logger = get_logger("streaming.frames")


def fit_within(width: int, height: int, *, max_width: int, max_height: int) -> tuple[int, int]:
    """Scale a frame size down to fit a box, preserving its aspect ratio.

    §18.1 requires the interface to "account for display letterboxing", and the
    cleanest way to keep that honest is to never change the aspect ratio here: the
    preview is a uniformly scaled copy, so a normalized zone coordinate maps onto the
    displayed image by one scalar. A frame already inside the box is returned
    unchanged rather than upscaled, because upscaling would spend bandwidth on
    invented pixels.

    Args:
        width: Source width in pixels.
        height: Source height in pixels.
        max_width: Width cap (§8: 960).
        max_height: Height cap (§8: 540).

    Returns:
        ``(width, height)`` of the preview, at least 1x1.
    """
    if width <= 0 or height <= 0:
        return 1, 1
    scale = min(max_width / width, max_height / height, 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


@dataclass(frozen=True, slots=True)
class FrameEnvelope:
    """One encoded preview frame and the frame it came from.

    Attributes:
        jpeg: Encoded JPEG bytes.
        camera_id: Source camera UUID.
        stream_session_id: Session the frame belongs to.
        frame_seq: Sequence number of the *source* frame the JPEG was drawn from.
        width: Encoded width in pixels.
        height: Encoded height in pixels.
        source_width: Source frame width, so a client can map normalized
            coordinates without being told the scale separately.
        source_height: Source frame height.
        captured_monotonic: Monotonic seconds the source frame was assembled.
        captured_utc: UTC epoch seconds the source frame was assembled.
        source_timestamp: Presentation timestamp in seconds, or ``None``.
        encoded_monotonic: Monotonic seconds the JPEG was produced.
        sequence: Monotonic counter over published preview frames, so a client can
            detect a gap without decoding the JPEG.
    """

    jpeg: bytes
    camera_id: uuid.UUID
    stream_session_id: uuid.UUID
    frame_seq: int
    width: int
    height: int
    source_width: int
    source_height: int
    captured_monotonic: float
    captured_utc: float
    source_timestamp: float | None
    encoded_monotonic: float
    sequence: int

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable metadata, without the JPEG bytes."""
        return {
            "camera_id": str(self.camera_id),
            "stream_session_id": str(self.stream_session_id),
            "frame_seq": self.frame_seq,
            "sequence": self.sequence,
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_timestamp": self.source_timestamp,
            "captured_monotonic": self.captured_monotonic,
            "captured_utc": self.captured_utc,
            "encoded_monotonic": self.encoded_monotonic,
            "jpeg_bytes": len(self.jpeg),
        }


@dataclass(slots=True)
class PreviewStats:
    """Counters and measurements describing preview encoding."""

    published: int = 0
    skipped_rate: int = 0
    encode_errors: int = 0
    last_encode_ms: float = 0.0
    mean_encode_ms: float = 0.0
    last_jpeg_bytes: int = 0
    mean_jpeg_bytes: float = 0.0
    last_sequence: int = 0
    first_published_monotonic: float | None = None
    last_published_monotonic: float | None = None

    @property
    def measured_fps(self) -> float:
        """Publish rate measured over the run, or ``0.0`` before two frames.

        Reported instead of the configured cap, because the cap is what the run
        *asked* for and this is what it achieved (§8: no figure here is a target).
        """
        first = self.first_published_monotonic
        last = self.last_published_monotonic
        if first is None or last is None or self.published < 2:
            return 0.0
        span = last - first
        if span <= 0.0:
            return 0.0
        return (self.published - 1) / span

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        §18.2 supplies the bandwidth arithmetic as ``bytes x FPS x 8 / 1e6``, and it
        is applied here to the *measured* mean JPEG size and the achieved publish
        rate. Both are measurements of this run at this quality setting, not a
        benchmark and not a bandwidth guarantee.
        """
        measured_fps = self.measured_fps
        megabits = self.mean_jpeg_bytes * measured_fps * 8.0 / 1_000_000.0
        return {
            "published": self.published,
            "skipped_rate": self.skipped_rate,
            "encode_errors": self.encode_errors,
            "last_encode_ms": round(self.last_encode_ms, 3),
            "mean_encode_ms": round(self.mean_encode_ms, 3),
            "last_jpeg_bytes": self.last_jpeg_bytes,
            "mean_jpeg_bytes": round(self.mean_jpeg_bytes, 1),
            "measured_fps": round(measured_fps, 3),
            "megabits_per_s": round(megabits, 4),
            "last_sequence": self.last_sequence,
        }


class PreviewStream:
    """Hold the newest annotated preview frame, encoded at a bounded rate.

    Args:
        settings: ``settings.streaming`` supplies the rate cap, the size cap, and the
            JPEG quality.
        clock: Injectable clock, used for the rate decision and the envelope.
    """

    def __init__(self, settings: StreamingSettings, *, clock: Clock | None = None) -> None:
        self._settings = settings
        self._clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._latest: FrameEnvelope | None = None
        self._sequence = 0
        self._last_encoded_monotonic: float | None = None
        self._stats = PreviewStats()

    # -- properties ----------------------------------------------------

    @property
    def stats(self) -> PreviewStats:
        """Live counters."""
        return self._stats

    @property
    def enabled(self) -> bool:
        """Whether preview encoding is enabled."""
        return self._settings.enabled

    @property
    def min_interval_s(self) -> float:
        """Shortest interval between two encoded frames (``1 / preview_fps``)."""
        return 1.0 / self._settings.preview_fps if self._settings.preview_fps > 0 else 0.0

    def latest(self) -> FrameEnvelope | None:
        """The newest encoded frame, or ``None`` before the first one."""
        with self._lock:
            return self._latest

    def snapshot(self) -> dict[str, Any]:
        """Status payload describing the preview stream."""
        with self._lock:
            latest = self._latest
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "preview_fps": self._settings.preview_fps,
            "max_width": self._settings.preview_max_width,
            "max_height": self._settings.preview_max_height,
            "jpeg_quality": self._settings.jpeg_quality,
            "stats": self._stats.as_dict(),
        }
        if latest is not None:
            payload["last_frame"] = latest.as_dict()
        return payload

    # -- publishing ----------------------------------------------------

    def due(self) -> bool:
        """Whether a :meth:`publish` now would be encoded rather than rate-skipped.

        Lets a caller skip preparing a frame (drawing captions on a copy) that the
        rate cap would discard; :meth:`publish` still applies the cap itself.
        """
        if not self.enabled:
            return False
        now = self._clock.monotonic()
        with self._lock:
            last = self._last_encoded_monotonic
        return last is None or (now - last) >= self.min_interval_s

    def publish(
        self,
        frame: np.ndarray,
        *,
        camera_id: uuid.UUID,
        stream_session_id: uuid.UUID,
        frame_seq: int,
        captured_monotonic: float,
        captured_utc: float,
        source_timestamp: float | None,
    ) -> FrameEnvelope | None:
        """Encode and store a preview frame, if the rate cap allows it.

        Args:
            frame: The annotated BGR frame, drawn from the frame identified by the
                other arguments — §18.1's "same frame" rule is the caller's
                responsibility and the envelope records which frame it was.
            camera_id: Source camera UUID.
            stream_session_id: Session the frame belongs to.
            frame_seq: Sequence number of the source frame.
            captured_monotonic: Monotonic seconds the source frame was assembled.
            captured_utc: UTC epoch seconds the source frame was assembled.
            source_timestamp: Presentation timestamp in seconds, or ``None``.

        Returns:
            The envelope when a frame was encoded, or ``None`` when the rate cap
            skipped this one or encoding failed.

        A skipped frame is counted, not silently dropped: §8's preview cap is a
        deliberate reduction of work, and reporting how many frames it cost is what
        keeps the cap from looking like a decoder problem.
        """
        if not self.enabled:
            return None

        now = self._clock.monotonic()
        with self._lock:
            last = self._last_encoded_monotonic
            if last is not None and (now - last) < self.min_interval_s:
                self._stats.skipped_rate += 1
                return None

        try:
            import cv2
        except Exception as exc:  # noqa: BLE001 - a missing decoder must not stop the camera
            self._stats.encode_errors += 1
            logger.warning("OpenCV is unavailable for preview encoding: %s", exc)
            return None

        try:
            source_height, source_width = int(frame.shape[0]), int(frame.shape[1])
            target_width, target_height = fit_within(
                source_width,
                source_height,
                max_width=self._settings.preview_max_width,
                max_height=self._settings.preview_max_height,
            )
            image = frame
            if (target_width, target_height) != (source_width, source_height):
                image = cv2.resize(
                    frame, (target_width, target_height), interpolation=cv2.INTER_AREA
                )
            ok, buffer = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self._settings.jpeg_quality]
            )
            if not ok:
                raise RuntimeError("cv2.imencode reported failure")
            jpeg = buffer.tobytes()
        except Exception as exc:  # noqa: BLE001 - a preview failure must not stop the camera
            self._stats.encode_errors += 1
            logger.warning("could not encode a preview frame: %s", exc)
            return None

        encode_ms = (self._clock.monotonic() - now) * 1000.0
        with self._lock:
            self._sequence += 1
            envelope = FrameEnvelope(
                jpeg=jpeg,
                camera_id=camera_id,
                stream_session_id=stream_session_id,
                frame_seq=frame_seq,
                width=target_width,
                height=target_height,
                source_width=source_width,
                source_height=source_height,
                captured_monotonic=captured_monotonic,
                captured_utc=captured_utc,
                source_timestamp=source_timestamp,
                encoded_monotonic=self._clock.monotonic(),
                sequence=self._sequence,
            )
            self._latest = envelope
            self._last_encoded_monotonic = now
            self._record(envelope, encode_ms)
            return envelope

    def clear(self) -> None:
        """Drop the stored frame, as a session change requires (§9.2).

        The rate gate is reset too: a new stream session should produce its first
        preview immediately rather than inheriting the previous session's timing.
        """
        with self._lock:
            self._latest = None
            self._last_encoded_monotonic = None

    def _record(self, envelope: FrameEnvelope, encode_ms: float) -> None:
        """Update the measured counters; caller holds the lock."""
        stats = self._stats
        published = stats.published + 1
        stats.mean_encode_ms = (stats.mean_encode_ms * stats.published + encode_ms) / published
        stats.mean_jpeg_bytes = (
            stats.mean_jpeg_bytes * stats.published + len(envelope.jpeg)
        ) / published
        stats.published = published
        stats.last_encode_ms = encode_ms
        stats.last_jpeg_bytes = len(envelope.jpeg)
        stats.last_sequence = envelope.sequence
        if stats.first_published_monotonic is None:
            stats.first_published_monotonic = envelope.encoded_monotonic
        stats.last_published_monotonic = envelope.encoded_monotonic
