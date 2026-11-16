"""FFmpeg subprocess decoder.

Thesis plan §3 and §9.1 fix the decoding contract:

* FFmpeg runs as a **subprocess controlled by Python**, always invoked with an
  argument list and ``shell=False`` (§24.2).
* Raw ``bgr24`` frames are piped to stdout and reshaped with NumPy, so no extra
  Python binding is required.
* Presentation timestamps come from FFmpeg's ``showinfo`` filter on stderr and
  are attached to the matching frame. MP4 time is never derived from loop speed.
* A **reader thread** drains stdout continuously so FFmpeg's pipe buffer cannot
  build a delayed backlog; frames land in a bounded
  :class:`~fall_detection.video.queue.FrameQueue` whose drop policy follows the
  live/offline mode.
* A **stall timeout** raises :class:`~fall_detection.core.errors.DecoderStallError`
  when no frame arrives, which is the fault-injection case listed in §22.

Input options come from the source (:meth:`VideoSource.ffmpeg_input_args`): only
``rtsp`` gets ``-rtsp_transport tcp``, and only ``webcam`` selects a capture demuxer
(AVFoundation, V4L2 or DirectShow) with an explicit capture size and rate. A webcam
is never probed with FFprobe (opening the camera twice is slow and can fail while
it is busy); its requested size is used and the output is scaled to it, so every
frame on the pipe has exactly the expected byte count.
"""

from __future__ import annotations

import contextlib
import json
import re
import select
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import IO, TYPE_CHECKING, TypedDict

from fall_detection.core.clock import Clock, SystemClock
from fall_detection.core.contracts import FramePacket
from fall_detection.core.errors import DecoderError, DecoderStallError
from fall_detection.core.logging import get_logger, redact_text
from fall_detection.video.queue import FrameQueue, QueueClosedError
from fall_detection.video.sources import VideoSource, WebcamSource, describe_source

if TYPE_CHECKING:  # pragma: no cover - numpy is imported lazily inside the reader
    import numpy as np

__all__ = ["DecoderMetadata", "FfmpegDecoder", "StreamMetadata"]

logger = get_logger("video.decoder")

# ``showinfo`` prints one line per frame containing ``pts_time:<float>``. The
# pattern is deliberately strict: a malformed line is ignored rather than
# producing a wrong timestamp.
_PTS_RE = re.compile(r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:(-?\d+(?:\.\d+)?)")
_SIZE_RE = re.compile(r"\bs:(\d+)x(\d+)\b")

# What to check when a camera opens but delivers no frame, per capture demuxer.
_WEBCAM_SILENT_HINTS = {
    # AVFoundation opens the device but delivers nothing when macOS has not granted
    # camera access to the app that launched the process.
    "avfoundation": (
        "grant camera access to the terminal app that runs the backend (System "
        "Settings > Privacy & Security > Camera), then restart that app"
    ),
    "v4l2": (
        "check that no other program holds the camera and that this user may read "
        "the device (membership of the 'video' group)"
    ),
    "dshow": (
        "allow desktop apps to use the camera (Settings > Privacy & security > "
        "Camera) and close any other program that holds it"
    ),
}


class StreamMetadata(TypedDict):
    """JSON-serialisable stream properties for status payloads."""

    width: int
    height: int
    fps: float
    codec: str | None
    has_source_timestamps: bool
    container_duration_s: float | None
    ffmpeg_version: str | None


@dataclass(frozen=True, slots=True)
class DecoderMetadata:
    """Stream properties discovered before the first frame is read."""

    width: int
    height: int
    fps: float
    codec: str | None
    has_source_timestamps: bool
    container_duration_s: float | None = None
    ffmpeg_version: str | None = None

    @property
    def frame_bytes(self) -> int:
        """Bytes in one ``bgr24`` frame."""
        return self.width * self.height * 3

    def describe(self) -> StreamMetadata:
        """Return a JSON-serialisable summary."""
        return StreamMetadata(
            width=self.width,
            height=self.height,
            fps=self.fps,
            codec=self.codec,
            has_source_timestamps=self.has_source_timestamps,
            container_duration_s=self.container_duration_s,
            ffmpeg_version=self.ffmpeg_version,
        )


@dataclass(slots=True)
class _Counters:
    """Mutable counters shared with the reader thread."""

    frames_read: int = 0
    frames_dropped: int = 0
    last_frame_monotonic: float = field(default=0.0)
    eof: bool = False
    stalled: bool = False
    error: str | None = None


class FfmpegDecoder:
    """Read frames from an MP4 file or RTSP stream via an FFmpeg subprocess.

    Args:
        source: A validated source from
            :func:`fall_detection.video.sources.parse_source`.
        camera_id: Camera UUID stamped on every packet.
        stream_session_id: Session UUID stamped on every packet. A reconnect
            creates a new decoder with a new session UUID.
        queue_capacity: Bounded queue size.
        drop_oldest: ``True`` in live mode, ``False`` offline.
        read_timeout_s: Stall timeout for one frame.
        fallback_fps: Rate used only when the container reports none.
        clock: Injectable clock.
        ffmpeg_bin: FFmpeg executable name or path.
        ffprobe_bin: FFprobe executable name or path.
        width: Optional explicit frame width, bypassing FFprobe.
        height: Optional explicit frame height, bypassing FFprobe.
        extra_input_args: Additional arguments placed before ``-i``. Used by
            tests to inject a decoder failure; never populated from user input.
    """

    def __init__(
        self,
        source: VideoSource,
        *,
        camera_id: uuid.UUID,
        stream_session_id: uuid.UUID,
        queue_capacity: int = 2,
        drop_oldest: bool = True,
        read_timeout_s: float = 15.0,
        fallback_fps: float = 25.0,
        clock: Clock | None = None,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        width: int | None = None,
        height: int | None = None,
        extra_input_args: tuple[str, ...] = (),
    ) -> None:
        self._source = source
        self._camera_id = camera_id
        self._stream_session_id = stream_session_id
        self._read_timeout_s = read_timeout_s
        self._fallback_fps = fallback_fps
        self._clock = clock or SystemClock()
        self._ffmpeg_bin = ffmpeg_bin
        self._ffprobe_bin = ffprobe_bin
        self._width_override = width
        self._height_override = height
        self._extra_input_args = extra_input_args

        self._queue: FrameQueue[FramePacket] = FrameQueue(
            queue_capacity, drop_oldest=drop_oldest, name="decoder"
        )
        self._counters = _Counters()
        self._metadata: DecoderMetadata | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._pts: list[float] = []
        self._pts_lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=12)
        self._stop = threading.Event()
        self._frame_seq = 0

    # -- introspection -------------------------------------------------

    @property
    def camera_id(self) -> uuid.UUID:
        return self._camera_id

    @property
    def stream_session_id(self) -> uuid.UUID:
        return self._stream_session_id

    @property
    def metadata(self) -> DecoderMetadata | None:
        """Stream metadata, available after :meth:`open`."""
        return self._metadata

    @property
    def frames_read(self) -> int:
        """Frames successfully decoded into packets."""
        return self._counters.frames_read

    @property
    def frames_dropped(self) -> int:
        """Frames discarded because the bounded queue was full."""
        return self._queue.dropped

    @property
    def queue(self) -> FrameQueue[FramePacket]:
        """The bounded frame queue feeding the pipeline."""
        return self._queue

    @property
    def error(self) -> str | None:
        """Reader-thread error text, if the subprocess failed."""
        return self._counters.error

    def is_alive(self) -> bool:
        """Whether the FFmpeg process is still running."""
        process = self._process
        return process is not None and process.poll() is None

    def kill_process(self) -> bool:
        """Kill FFmpeg abruptly to simulate a camera dropout.

        Unlike :meth:`close`, this does *not* set the internal stop flag, so the
        reader thread observes a non-zero exit status and reports a decode
        failure — exactly the fault §22 requires the reconnect logic to survive.
        It exists for fault-injection verification and tests.

        Returns:
            ``True`` if a running process was killed.
        """
        process = self._process
        if process is None or process.poll() is not None:
            return False
        process.kill()
        return True

    def returncode(self) -> int | None:
        """FFmpeg exit status, or ``None`` while running."""
        process = self._process
        return None if process is None else process.poll()

    # -- lifecycle -----------------------------------------------------

    def open(self) -> None:
        """Probe the stream and start FFmpeg plus the reader threads.

        Raises:
            DecoderError: FFmpeg or FFprobe is missing, or probing failed.
        """
        if self._process is not None:
            raise DecoderError("decoder is already open")

        self._require_binary(self._ffmpeg_bin)
        self._metadata = self._probe()
        command = self._build_command(self._metadata)
        logger.info(
            "starting decoder source=%s session=%s %dx%d @%.3ffps",
            json.dumps(describe_source(self._source)),
            self._stream_session_id,
            self._metadata.width,
            self._metadata.height,
            self._metadata.fps,
        )
        try:
            self._process = subprocess.Popen(  # noqa: S603 - argument list, shell=False
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError as exc:
            raise DecoderError(f"failed to start ffmpeg: {exc}") from exc

        self._counters.last_frame_monotonic = self._clock.monotonic()
        self._stop.clear()
        self._stderr_thread = threading.Thread(
            target=self._consume_stderr, name="ffmpeg-stderr", daemon=True
        )
        self._reader = threading.Thread(target=self._read_frames, name="ffmpeg-reader", daemon=True)
        self._stderr_thread.start()
        self._reader.start()

    def read(self, timeout: float | None = None) -> FramePacket | None:
        """Return the next packet, or ``None`` at end of stream.

        Args:
            timeout: Seconds to wait for a frame. Defaults to the configured
                stall timeout.

        Raises:
            DecoderStallError: No frame arrived within the timeout while FFmpeg
                is still running.
            DecoderError: The reader thread recorded a decode failure.
        """
        wait = self._read_timeout_s if timeout is None else timeout
        packet = self._queue.get(wait)
        if packet is not None:
            return packet

        if self._counters.error is not None:
            raise DecoderError(self._counters.error)
        if self._counters.stalled:
            raise DecoderStallError(
                self._stall_message(),
                detail=f"timeout_s={wait} frames_read={self._counters.frames_read}",
            )
        returncode = self.returncode()
        if returncode is not None and returncode != 0:
            raise DecoderError(
                "ffmpeg exited with a non-zero status",
                detail=f"returncode={returncode} frames_read={self._counters.frames_read} "
                f"stderr={self._stderr_excerpt()}",
            )
        if self._counters.eof or not self.is_alive():
            # Exit status 0 (or a stop request) with nothing left to read is a
            # clean end of stream, including the ``ended`` state for a file.
            return None
        if self._stop.is_set():
            return None
        raise DecoderStallError(
            self._stall_message(),
            detail=f"timeout_s={wait} frames_read={self._counters.frames_read}",
        )

    def _stall_message(self) -> str:
        """Explain a stall; a webcam that never sends a frame is usually a permission issue."""
        message = "no frame decoded within the stall timeout"
        if isinstance(self._source, WebcamSource) and self._counters.frames_read == 0:
            message += (
                "; the webcam opened but sent no frame — "
                + _WEBCAM_SILENT_HINTS[self._source.backend]
            )
        return message

    def close(self, *, timeout: float = 5.0) -> None:
        """Stop FFmpeg, join the threads, and close the queue.

        Safe to call more than once and from any thread.
        """
        self._stop.set()
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # FFmpeg ignored SIGTERM (a stuck RTSP read); escalate.
                    process.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=timeout)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    with contextlib.suppress(OSError):
                        stream.close()
            self._process = None

        for thread in (self._reader, self._stderr_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout)
        self._reader = None
        self._stderr_thread = None
        self._queue.close()

    # -- internals -----------------------------------------------------

    def _require_binary(self, name: str) -> None:
        if shutil.which(name) is None:
            raise DecoderError(f"required executable not found on PATH: {name}")

    def _probe(self) -> DecoderMetadata:
        """Read stream metadata with FFprobe, or fall back to overrides."""
        return probe_source(
            self._source,
            ffprobe_bin=self._ffprobe_bin,
            ffmpeg_bin=self._ffmpeg_bin,
            width=self._width_override,
            height=self._height_override,
            fallback_fps=self._fallback_fps,
        )

    def _build_command(self, metadata: DecoderMetadata) -> list[str]:
        """Build the FFmpeg argument list."""
        command: list[str] = [
            self._ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",  # required so showinfo's per-frame lines reach stderr
        ]
        command += list(self._source.ffmpeg_input_args())
        command += list(self._extra_input_args)
        video_filter = "showinfo"
        if isinstance(self._source, WebcamSource):
            # AVFoundation may deliver a size other than the one requested; scaling
            # pins the raw frame size to the metadata the reader reshapes with.
            video_filter = f"showinfo,scale={metadata.width}:{metadata.height}"
        command += [
            "-i",
            self._source.ffmpeg_input(),
            # ``showinfo`` emits ``pts_time`` per frame; ``-an`` drops audio so
            # the pipe carries only video.
            "-an",
            "-vf",
            video_filter,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        return command

    def _consume_stderr(self) -> None:
        """Parse ``pts_time`` values from FFmpeg's stderr stream.

        Lines are also retained in a bounded tail. They are *not* interpreted as
        errors here: FFmpeg's progress and input banners contain words like
        "error" and file paths can contain anything, so only the process exit
        status decides whether decoding failed.
        """
        process = self._process
        if process is None or process.stderr is None:
            return
        stream: IO[bytes] = process.stderr
        try:
            for raw_line in stream:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                match = _PTS_RE.search(line)
                if match:
                    with self._pts_lock:
                        self._pts.append(float(match.group(2)))
                    continue
                # ``showinfo`` also emits continuation lines (colour metadata)
                # that carry no diagnostic value; keeping them would push the
                # real error out of the bounded tail.
                if line and "Parsed_showinfo" not in line:
                    with self._pts_lock:
                        self._stderr_tail.append(redact_text(line))
        except OSError, ValueError:
            # The pipe closed during shutdown; nothing to report.
            return

    def _stderr_excerpt(self) -> str:
        """Return the last few stderr lines, credential-redacted."""
        with self._pts_lock:
            lines = list(self._stderr_tail)
        return " | ".join(lines[-4:])

    def _next_pts(self, wait: float) -> float | None:
        """Return the next presentation timestamp, waiting briefly for stderr."""
        deadline = self._clock.monotonic() + wait
        while True:
            with self._pts_lock:
                if self._pts:
                    return self._pts.pop(0)
            if self._clock.monotonic() >= deadline or self._stop.is_set():
                return None
            time.sleep(0.001)

    def _read_frames(self) -> None:
        """Reader thread: drain stdout and enqueue packets."""
        process = self._process
        metadata = self._metadata
        if process is None or process.stdout is None or metadata is None:
            return

        stdout = process.stdout
        frame_bytes = metadata.frame_bytes
        partial = b""

        try:
            while not self._stop.is_set():
                if not self._wait_readable(stdout, self._read_timeout_s):
                    if self._stop.is_set():
                        break
                    if process.poll() is not None:
                        break
                    # FFmpeg is alive but silent: record a stall so ``read``
                    # reports DecoderStallError rather than a clean EOF.
                    self._counters.stalled = True
                    break

                chunk = stdout.read(frame_bytes - len(partial))
                if not chunk:
                    # Clean EOF, possibly mid-frame (a truncated final frame is
                    # discarded rather than reshaped into garbage).
                    break
                partial += chunk
                if len(partial) < frame_bytes:
                    continue

                buffer = _to_array(partial, metadata.height, metadata.width)
                partial = b""
                now_monotonic = self._clock.monotonic()
                # ``showinfo`` emits the timestamp before the frame's pixels
                # finish arriving, so a short wait is normally enough; ``None``
                # is recorded honestly when the source has no timestamps.
                pts = self._next_pts(0.05)

                packet = FramePacket(
                    camera_id=self._camera_id,
                    stream_session_id=self._stream_session_id,
                    frame_seq=self._frame_seq,
                    frame=buffer,
                    source_timestamp=pts,
                    receive_monotonic=now_monotonic,
                    receive_utc=self._clock.utc_timestamp(),
                )
                self._frame_seq += 1
                self._counters.last_frame_monotonic = now_monotonic
                try:
                    self._queue.put(packet, size=packet.nbytes())
                except QueueClosedError:
                    break
                self._counters.frames_read += 1
        except (OSError, ValueError) as exc:
            self._counters.error = f"decoder read failed: {exc}"
        finally:
            self._finalize_process()
            self._counters.eof = True
            self._queue.close()

    def _finalize_process(self) -> None:
        """Reap FFmpeg and record a non-zero exit status as a decode error.

        stdout closing usually means FFmpeg has exited, but the exit status is
        only known after ``wait``. Recording it here (rather than inferring from
        stderr text) is what distinguishes a clean end of stream from a decoder
        that failed, for example because of an invalid input option.
        """
        process = self._process
        if process is None or self._stop.is_set():
            return
        try:
            returncode = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            return
        if returncode != 0 and self._counters.error is None:
            self._counters.error = (
                f"ffmpeg exited with status {returncode}: {self._stderr_excerpt()}"
            )

    def _wait_readable(self, stream: IO[bytes], timeout: float) -> bool:
        """Wait until ``stream`` has data or has reached end of file.

        POSIX uses ``select`` on the pipe. Windows ``select`` accepts only sockets, so
        there the pipe is polled with ``PeekNamedPipe`` (:func:`_wait_pipe_windows`).
        """
        fileno = getattr(stream, "fileno", None)
        if fileno is None:  # pragma: no cover - real pipes always expose fileno
            return True
        if sys.platform == "win32":
            return _wait_pipe_windows(stream, timeout, self._stop)
        try:
            ready, _, _ = select.select([stream], [], [], timeout)
        except OSError, ValueError:
            return False
        return bool(ready)


_WINDOWS_PIPE_POLL_S = 0.002
"""Poll interval while a Windows pipe is empty: short next to a 33 ms frame period."""


def _wait_pipe_windows(stream: IO[bytes], timeout: float, stop: threading.Event) -> bool:
    """Windows stand-in for ``select`` on an anonymous pipe.

    Returns ``True`` as soon as bytes are waiting or the pipe is broken (FFmpeg exited,
    so the next ``read`` returns end of file), and ``False`` when ``timeout`` passes
    with the pipe still open and empty, which the reader records as a stall. The stop
    event ends the wait early so ``close`` is not held up by a full poll.
    """
    if sys.platform != "win32":  # pragma: no cover - narrows the platform for type checkers
        raise RuntimeError("_wait_pipe_windows is only meaningful on Windows")
    import ctypes  # noqa: PLC0415 - Windows only
    import msvcrt  # noqa: PLC0415 - Windows only
    from ctypes import wintypes  # noqa: PLC0415 - Windows only

    peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
    peek.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    peek.restype = wintypes.BOOL
    try:
        handle = msvcrt.get_osfhandle(stream.fileno())
    except OSError, ValueError:
        return True  # the pipe is already closed; ``read`` reports end of file
    available = wintypes.DWORD(0)
    deadline = time.monotonic() + timeout
    while True:
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            return True  # ERROR_BROKEN_PIPE: the writer is gone, ``read`` sees EOF
        if available.value > 0:
            return True
        if stop.is_set() or time.monotonic() >= deadline:
            return False
        time.sleep(_WINDOWS_PIPE_POLL_S)


def _parse_fps(value: str | None) -> float | None:
    """Parse an FFmpeg rational frame rate such as ``"25/1"`` or ``"30000/1001"``."""
    if not value:
        return None
    try:
        rate = Fraction(value)
    except ValueError, ZeroDivisionError:
        return None
    if rate <= 0:
        return None
    return float(rate)


def _to_array(buffer: bytes, height: int, width: int) -> np.ndarray:
    """Reshape a raw ``bgr24`` buffer into an ``(H, W, 3)`` uint8 array.

    ``np.frombuffer`` returns a read-only view over ``bytes``, and the buffer is
    a reused ``bytes`` accumulator, so the array is copied to give every packet
    its own writable frame.
    """
    import numpy as np

    array = np.frombuffer(buffer, dtype=np.uint8)
    return array.reshape(height, width, 3).copy()


def probe_source(
    source: VideoSource,
    *,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
    width: int | None = None,
    height: int | None = None,
    fallback_fps: float = 25.0,
) -> DecoderMetadata:
    """Read stream properties with FFprobe.

    Args:
        source: A validated source.
        ffprobe_bin: FFprobe executable.
        ffmpeg_bin: FFmpeg executable, used only to record its version string.
        width: Explicit width, bypassing FFprobe entirely.
        height: Explicit height, bypassing FFprobe entirely.
        fallback_fps: Rate used only when the container reports none.

    Raises:
        DecoderError: FFprobe is missing, failed, or reported no video stream.
    """
    if isinstance(source, WebcamSource) and not (width and height):
        return DecoderMetadata(
            width=source.width,
            height=source.height,
            fps=source.fps,
            codec=None,
            has_source_timestamps=True,
            ffmpeg_version=ffmpeg_version(ffmpeg_bin),
        )

    if width and height:
        return DecoderMetadata(
            width=width,
            height=height,
            fps=fallback_fps,
            codec=None,
            has_source_timestamps=True,
        )

    if shutil.which(ffprobe_bin) is None:
        raise DecoderError(f"required executable not found on PATH: {ffprobe_bin}")

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,time_base",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        source.ffmpeg_input(),
    ]
    try:
        result = subprocess.run(  # noqa: S603 - argument list, shell=False
            command, capture_output=True, text=True, timeout=30.0, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise DecoderError("ffprobe timed out reading the source") from exc
    if result.returncode != 0:
        raise DecoderError(
            "ffprobe failed to read the source",
            detail=redact_text(result.stderr.strip()[:400]),
        )
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
    except (ValueError, KeyError, IndexError) as exc:
        raise DecoderError("ffprobe returned no usable video stream") from exc

    fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate"))
    duration = None
    with contextlib.suppress(TypeError, ValueError, KeyError):
        duration = float(payload["format"]["duration"])

    return DecoderMetadata(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps or fallback_fps,
        codec=stream.get("codec_name"),
        has_source_timestamps=bool(stream.get("time_base")),
        container_duration_s=duration,
        ffmpeg_version=ffmpeg_version(ffmpeg_bin),
    )


def ffmpeg_version(ffmpeg_bin: str = "ffmpeg") -> str | None:
    """Return the first line of ``ffmpeg -version``, or ``None``."""
    with contextlib.suppress(OSError, subprocess.SubprocessError, IndexError):
        output = subprocess.run(  # noqa: S603 - argument list, shell=False
            [ffmpeg_bin, "-version"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        ).stdout
        return output.splitlines()[0].strip()
    return None
