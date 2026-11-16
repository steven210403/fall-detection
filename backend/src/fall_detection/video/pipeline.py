"""Camera pipeline supervisor.

Owns the decode loop for one camera and implements the lifecycle rules in
thesis plan §9.2 and FR-09/FR-11:

* **One supervisor per camera**, created once and startable/stopable repeatedly.
  Models and decoders are never duplicated; :meth:`start` is idempotent.
* **States** are ``idle``, ``connecting``, ``live``, ``reconnecting``,
  ``offline``, and ``ended`` — distinct values so the operator can tell "not
  started" from "was live and dropped" (FR-09).
* **Reconnect** uses the backoff schedule from
  :class:`~fall_detection.video.reconnect.ReconnectPolicy`, and every attempt
  creates a *new* stream session, which is the signal later phases use to clear
  tracker state and activity buffers.
* **Clean EOF** ends the source normally: the session is closed with
  ``source_ended`` and the state becomes ``ended`` rather than retrying a file
  that has already finished.
* **Stop** is blocking and idempotent: it stops the decoder, joins the threads,
  closes the session row, and leaves the supervisor restartable.

Frames are consumed by an optional ``on_frame`` callback. Phase 3 attaches the
detector there; until then the callback is absent and frames are simply counted,
which is exactly what the Phase 2 replay verification needs.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NotRequired, TypedDict

from sqlalchemy.orm import Session

from fall_detection.config import Settings
from fall_detection.core.clock import Clock, SystemClock
from fall_detection.core.contracts import CameraState, FramePacket, StreamSessionInfo
from fall_detection.core.errors import (
    CameraUnavailableError,
    CctvError,
    DecoderError,
    DecoderStallError,
    InvalidSourceError,
    PipelineStateError,
)
from fall_detection.core.logging import get_logger
from fall_detection.database import (
    CameraRepository,
    DatabaseWriter,
    StreamSessionRepository,
    utc_now,
)
from fall_detection.video.decoder import DecoderMetadata, FfmpegDecoder, StreamMetadata
from fall_detection.video.queue import FrameQueue, QueueStats
from fall_detection.video.reconnect import ReconnectPolicy
from fall_detection.video.sources import VideoSource, describe_source, parse_source

__all__ = [
    "FrameConsumer",
    "PipelineStats",
    "PipelineStatus",
    "PipelineSupervisor",
]

logger = get_logger("video.pipeline")

FrameConsumer = Callable[[FramePacket], None]


class PipelineStatus(TypedDict):
    """Status payload for ``GET /api/v1/status`` (thesis plan §17.2).

    Typed so the API layer and its tests can rely on the shape without
    re-deriving it; a plain ``dict[str, object]`` would hide every nested key
    behind ``object``. ``source`` is optional because it is absent until the
    source has been resolved.
    """

    camera_id: str
    state: str
    state_age_s: float
    stream_session_id: str | None
    attempt: int | None
    stream: StreamMetadata | None
    stats: dict[str, object]
    queue: QueueStats | None
    reconnect_policy: dict[str, object]
    source: NotRequired[dict[str, object]]


@dataclass(slots=True)
class PipelineStats:
    """Counters reported through ``GET /api/v1/status`` (thesis plan §17.2)."""

    sessions_opened: int = 0
    sessions_closed: int = 0
    reconnects: int = 0
    frames_read: int = 0
    frames_dropped: int = 0
    decode_errors: int = 0
    last_frame_monotonic: float | None = None
    last_error: str | None = None
    started_monotonic: float | None = None
    ended_monotonic: float | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable snapshot."""
        return {
            "sessions_opened": self.sessions_opened,
            "sessions_closed": self.sessions_closed,
            "reconnects": self.reconnects,
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "decode_errors": self.decode_errors,
            "last_frame_monotonic": self.last_frame_monotonic,
            "last_error": self.last_error,
            "started_monotonic": self.started_monotonic,
            "ended_monotonic": self.ended_monotonic,
        }


@dataclass(slots=True)
class _LiveSession:
    """Bookkeeping for the session currently being consumed."""

    info: StreamSessionInfo
    decoder: FfmpegDecoder
    metadata: DecoderMetadata
    frames_read: int = 0
    frames_dropped: int = 0
    termination_reason: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


class PipelineSupervisor:
    """Run one camera's decode loop with reconnection and clean shutdown.

    Args:
        settings: Application settings; supplies the video, media-root, and
            RTSP configuration.
        camera_id: Camera UUID this supervisor owns.
        writer: The single database writer.
        on_frame: Optional consumer called for every decoded frame. Exceptions
            from the consumer are logged and counted, never allowed to kill the
            decode loop.
        clock: Injectable clock.
        policy: Optional reconnect policy override (tests use a fast one).
        ffmpeg_bin: FFmpeg executable.
        ffprobe_bin: FFprobe executable.
        extra_input_args: Extra FFmpeg input arguments, used by tests to inject
            decoder failures.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        camera_id: uuid.UUID,
        writer: DatabaseWriter,
        on_frame: FrameConsumer | None = None,
        clock: Clock | None = None,
        policy: ReconnectPolicy | None = None,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        extra_input_args: tuple[str, ...] = (),
    ) -> None:
        self._settings = settings
        self._camera_id = camera_id
        self._writer = writer
        self._on_frame = on_frame
        self._clock = clock or SystemClock()
        self._policy = policy or ReconnectPolicy(
            backoff_s=settings.video.reconnect_backoff_s,
            jitter_ratio=settings.video.reconnect_jitter_ratio,
            max_attempts=settings.video.max_reconnect_attempts,
        )
        self._ffmpeg_bin = ffmpeg_bin
        self._ffprobe_bin = ffprobe_bin
        self._extra_input_args = extra_input_args

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._state = CameraState.IDLE
        self._state_changed_monotonic = self._clock.monotonic()
        self._session: _LiveSession | None = None
        self._source: VideoSource | None = None
        self._session_lock = threading.Lock()
        self._stats = PipelineStats()
        self._consumer_errors = 0
        # Outcome of the most recent connection, read by the reconnect loop.
        self._last_ended_cleanly = False
        self._last_error: CctvError | None = None

    # -- state ---------------------------------------------------------

    @property
    def camera_id(self) -> uuid.UUID:
        """Camera UUID owned by this supervisor."""
        return self._camera_id

    @property
    def state(self) -> CameraState:
        """Current camera state (FR-09)."""
        with self._lock:
            return self._state

    @property
    def stats(self) -> PipelineStats:
        """Live counters."""
        return self._stats

    @property
    def consumer_errors(self) -> int:
        """Number of frames whose consumer callback raised."""
        return self._consumer_errors

    def is_running(self) -> bool:
        """Whether the supervisor thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def current_session(self) -> StreamSessionInfo | None:
        """Session info for the currently live stream, if any."""
        with self._lock:
            return self._session.info if self._session is not None else None

    def frame_queue(self) -> FrameQueue[FramePacket] | None:
        """Queue of the currently live decoder, if any."""
        with self._lock:
            return self._session.decoder.queue if self._session is not None else None

    def current_decoder(self) -> FfmpegDecoder | None:
        """Decoder of the currently live session, if any.

        Exposed for diagnostics and for fault-injection verification: §22
        requires that a decoder hang or abrupt exit is recovered from, which can
        only be exercised by disturbing a live decoder.
        """
        with self._lock:
            return self._session.decoder if self._session is not None else None

    def _set_state(self, state: CameraState) -> None:
        with self._lock:
            if self._state == state:
                return
            previous = self._state
            self._state = state
            self._state_changed_monotonic = self._clock.monotonic()
        logger.info("camera %s state %s -> %s", self._camera_id, previous, state)

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Start decoding.

        Idempotent: calling :meth:`start` while running is a no-op, so a
        duplicated API request cannot create a second decoder (FR-11).

        Raises:
            ConfigurationError: The source is not configured.
            InvalidSourceError: The source exists but cannot be used.
        """
        with self._lock:
            if self.is_running():
                logger.debug("camera %s start ignored: already running", self._camera_id)
                return
            # Validate the source before the thread starts so a configuration
            # error surfaces synchronously in the API response rather than only
            # in the log of a background thread.
            self._source = self._resolve_source()
            self._stop_event.clear()
            self._wake_event.clear()
            self._stats = PipelineStats(started_monotonic=self._clock.monotonic())
            self._set_state(CameraState.CONNECTING)
            self._thread = threading.Thread(
                target=self._run, name=f"pipeline-{self._camera_id}", daemon=True
            )
            self._thread.start()

    def stop(self, *, timeout: float = 10.0, reason: str = "stopped") -> None:
        """Stop decoding and finalise the active session.

        Blocking and idempotent. Closes the event/decoder, joins the thread,
        closes the session row with ``reason``, and leaves the object
        restartable. Safe to call from the lifespan even if never started.

        Args:
            timeout: Seconds to wait for the decode thread.
            reason: Termination reason recorded on the session row.
        """
        with self._lock:
            thread = self._thread
            self._stop_event.set()
            self._wake_event.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        # The thread closes the session in its own ``finally`` block; closing
        # again here covers the "never started" and "thread died early" cases
        # because close_session is idempotent.
        self._close_session(reason=reason)
        with self._lock:
            self._thread = None
            if self._state not in (CameraState.IDLE,):
                self._set_state(CameraState.IDLE)

    def snapshot(self) -> PipelineStatus:
        """Return a status payload for the API."""
        with self._lock:
            state = self._state
            session = self._session.info if self._session is not None else None
            metadata = self._session.metadata if self._session is not None else None
            queue = self._session.decoder.queue if self._session is not None else None
        payload = PipelineStatus(
            camera_id=str(self._camera_id),
            state=state.value,
            state_age_s=round(self._clock.monotonic() - self._state_changed_monotonic, 3),
            stream_session_id=str(session.stream_session_id) if session else None,
            attempt=session.attempt if session else None,
            stream=metadata.describe() if metadata else None,
            stats=self._stats.as_dict(),
            queue=queue.stats() if queue is not None else None,
            reconnect_policy=self._policy.describe(),
        )
        if self._source is not None:
            payload["source"] = describe_source(self._source)
        return payload

    # -- internals -----------------------------------------------------

    def _resolve_source(self) -> VideoSource:
        return parse_source(
            mode=self._settings.video.mode,
            source_ref=self._settings.video.source_ref,
            media_roots=self._settings.media_root_paths,
            rtsp_url=self._settings.rtsp_url,
            webcam_device=self._settings.video.webcam_device,
            webcam_width=self._settings.video.webcam_width,
            webcam_height=self._settings.video.webcam_height,
            webcam_fps=self._settings.video.webcam_fps,
        )

    def _run(self) -> None:
        """Decode loop: connect, consume, and reconnect until stopped."""
        attempt = 1
        while not self._stop_event.is_set():
            try:
                self._consume_once(attempt)
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                self._last_error = exc if isinstance(exc, CctvError) else None
                self._stats.decode_errors += 1
                self._stats.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("camera %s attempt %d failed: %s", self._camera_id, attempt, exc)

            if self._stop_event.is_set():
                return

            if self._last_ended_cleanly:
                # A finite source reached its end; retrying would replay the file
                # forever, so the camera moves to ``ended`` and waits for an
                # explicit start.
                self._set_state(CameraState.ENDED)
                self._stats.ended_monotonic = self._clock.monotonic()
                return

            if isinstance(self._last_error, InvalidSourceError):
                # Retrying cannot fix a rejected path or scheme.
                self._set_state(CameraState.OFFLINE)
                return

            if not self._policy.should_retry(attempt):
                logger.error("camera %s giving up after %d attempts", self._camera_id, attempt)
                self._set_state(CameraState.OFFLINE)
                return

            self._set_state(CameraState.RECONNECTING)
            delay = self._policy.delay_for(attempt)
            logger.info(
                "camera %s reconnecting in %.2fs (attempt %d)", self._camera_id, delay, attempt + 1
            )
            # A wake from stop() must not wait out the full backoff.
            if self._wake_event.wait(timeout=delay):
                return
            attempt += 1

    def _consume_once(self, attempt: int) -> None:
        """Decode from one connection until it ends, fails, or is stopped."""
        source = self._source
        if source is None:
            raise CameraUnavailableError("no source resolved")

        session_id = uuid.uuid4()
        now_monotonic = self._clock.monotonic()
        info = StreamSessionInfo(
            camera_id=self._camera_id,
            stream_session_id=session_id,
            started_utc=self._clock.utc_timestamp(),
            started_monotonic=now_monotonic,
            attempt=attempt,
        )

        settings = self._settings
        decoder = FfmpegDecoder(
            source,
            camera_id=self._camera_id,
            stream_session_id=session_id,
            queue_capacity=(
                settings.video.live_queue_capacity
                if source.is_live
                else settings.video.offline_queue_capacity
            ),
            drop_oldest=source.is_live,
            read_timeout_s=settings.video.read_timeout_s,
            fallback_fps=settings.video.fallback_fps,
            clock=self._clock,
            ffmpeg_bin=self._ffmpeg_bin,
            ffprobe_bin=self._ffprobe_bin,
            extra_input_args=self._extra_input_args,
        )

        try:
            decoder.open()
        except (CameraUnavailableError, DecoderError) as exc:
            raise CameraUnavailableError(str(exc)) from exc

        live = _LiveSession(info=info, decoder=decoder, metadata=decoder.metadata)  # type: ignore[arg-type]
        with self._lock:
            self._session = live
        self._stats.sessions_opened += 1
        if attempt > 1:
            self._stats.reconnects += 1
        self._record_session_open(info)
        self._set_state(CameraState.LIVE)
        logger.info(
            "camera %s live session=%s attempt=%d %dx%d @%.3ffps",
            self._camera_id,
            session_id,
            attempt,
            decoder.metadata.width,  # type: ignore[union-attr]
            decoder.metadata.height,  # type: ignore[union-attr]
            decoder.metadata.fps,  # type: ignore[union-attr]
        )

        self._last_ended_cleanly = False
        try:
            self._drain(live)
        except DecoderStallError, DecoderError:
            live.termination_reason = "decoder_error"
            raise
        except InvalidSourceError:
            live.termination_reason = "source_invalid"
            raise
        finally:
            frames_read = live.frames_read
            frames_dropped = live.frames_dropped
            decoder.close()
            with self._lock:
                self._session = None
            self._record_session_close(
                info,
                reason=live.termination_reason or "stopped",
                frames_read=frames_read,
                frames_dropped=frames_dropped,
            )

    def _drain(self, live: _LiveSession) -> None:
        """Read frames from one session until stopped, EOF, or failure."""
        decoder = live.decoder
        while not self._stop_event.is_set():
            packet = decoder.read()
            if packet is None:
                if self._stop_event.is_set():
                    # A stop request makes ``read`` return ``None`` early; this
                    # is a deliberate shutdown, not the end of the source.
                    live.termination_reason = "stopped"
                    return
                if decoder.frames_read == 0:
                    raise DecoderError("decoder produced no frames before exiting")
                # A finite source reached a clean end; retrying would replay it.
                live.termination_reason = "source_ended"
                self._last_ended_cleanly = True
                return

            live.frames_read += 1
            live.frames_dropped = decoder.frames_dropped
            self._stats.frames_read += 1
            self._stats.frames_dropped = live.frames_dropped
            self._stats.last_frame_monotonic = packet.receive_monotonic

            consumer = self._on_frame
            if consumer is not None:
                try:
                    consumer(packet)
                except Exception:  # noqa: BLE001 - a bad consumer must not stop decoding
                    self._consumer_errors += 1
                    logger.exception(
                        "consumer failed for camera %s frame %d",
                        self._camera_id,
                        packet.frame_seq,
                    )

            if packet.is_last:
                live.termination_reason = "source_ended"
                self._last_ended_cleanly = True
                return

        live.termination_reason = live.termination_reason or "stopped"

    def _record_session_open(self, info: StreamSessionInfo) -> None:
        """Insert the session row through the single writer."""

        def work(session: Session) -> None:
            CameraRepository(session).get_or_create(
                camera_id=self._camera_id,
                name=self._settings.camera_name,
                source_type=self._settings.video.mode,
                source_ref=self._source.safe_ref if self._source else "",
            )
            StreamSessionRepository(session).open(
                camera_id=self._camera_id,
                attempt=info.attempt,
                stream_session_id=info.stream_session_id,
                started_at=utc_now(),
            )

        self._writer.write(work)

    def _record_session_close(
        self,
        info: StreamSessionInfo,
        *,
        reason: str,
        frames_read: int,
        frames_dropped: int,
    ) -> None:
        """Close the session row exactly once."""
        session_id = info.stream_session_id

        def work(session: Session) -> None:
            StreamSessionRepository(session).close(
                stream_session_id=session_id,
                termination_reason=reason,
                frames_read=frames_read,
                frames_dropped=frames_dropped,
            )

        with contextlib.suppress(CctvError, OSError):
            self._writer.write(work)
        self._stats.sessions_closed += 1

    def _close_session(self, *, reason: str) -> None:
        """Close whatever session is still open for this camera.

        Called by :meth:`stop`; the session object may already be gone when the
        decode thread finished on its own, in which case this is a no-op.
        """
        with self._lock:
            live = self._session
            self._session = None
        if live is None:
            return
        live.decoder.close()
        self._record_session_close(
            live.info,
            reason=live.termination_reason or reason,
            frames_read=live.frames_read,
            frames_dropped=live.frames_dropped,
        )


def build_supervisor(
    *,
    settings: Settings,
    camera_id: uuid.UUID,
    writer: DatabaseWriter,
    on_frame: FrameConsumer | None = None,
) -> PipelineSupervisor:
    """Construct a supervisor from settings.

    Provided so the API layer and CLIs share one construction path and cannot
    drift on decoding defaults.
    """
    return PipelineSupervisor(
        settings=settings,
        camera_id=camera_id,
        writer=writer,
        on_frame=on_frame,
        policy=ReconnectPolicy(
            backoff_s=settings.video.reconnect_backoff_s,
            jitter_ratio=settings.video.reconnect_jitter_ratio,
            max_attempts=settings.video.max_reconnect_attempts,
        ),
    )


def wait_for_state(
    supervisor: PipelineSupervisor,
    state: CameraState,
    *,
    timeout: float = 30.0,
    poll: float = 0.02,
) -> bool:
    """Block until ``supervisor`` reaches ``state`` or ``timeout`` elapses.

    Returns:
        ``True`` if the state was reached. Used by the replay CLI and tests so
        neither needs to sleep for a fixed guess.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if supervisor.state == state:
            return True
        time.sleep(poll)
    return supervisor.state == state


def ensure_stopped(supervisor: PipelineSupervisor) -> None:
    """Stop a supervisor if it is running; convenience for CLI cleanup."""
    with contextlib.suppress(CctvError):
        supervisor.stop()


def raise_if_not_startable(supervisor: PipelineSupervisor) -> None:
    """Raise :class:`PipelineStateError` when the supervisor cannot start."""
    if supervisor.is_running():
        raise PipelineStateError("pipeline is already running")
