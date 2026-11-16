"""Shared typed contracts.

Thesis plan §17.1 fixes one internal contract chain and requires that CV modules
exchange dataclasses rather than ORM models or Pydantic models:

    FramePacket → DetectionBatch → TrackObservations → PersonFeatures
                → ActivityPrediction → ActivityEvent

Phase 2 implements the first link, :class:`FramePacket`, because the video layer
must carry source and receive timestamps before any detection code exists
(thesis plan §9.1). Phase 3 adds ``DetectionBatch`` in the ``detection``
package; it is intentionally absent here.

``core`` must not import FastAPI, SQLAlchemy, Pydantic, or headless-CV types.
NumPy is referenced only under ``TYPE_CHECKING`` so importing
``fall_detection.core`` stays cheap and dependency-light — the API lifespan
imports this module on every start.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    import numpy as np

__all__ = [
    "CameraState",
    "FramePacket",
    "FrameSource",
    "StreamSessionInfo",
]


class CameraState(StrEnum):
    """Distinct camera states required by FR-09 (thesis plan §7).

    ``CONNECTING`` covers process start and first-frame waits; ``RECONNECTING``
    covers a retry after a failure. They are separate so the operator can tell
    "not started yet" from "was live and dropped", and an offline camera never
    fails the liveness probe (thesis plan §17.2).
    """

    IDLE = "idle"
    CONNECTING = "connecting"
    LIVE = "live"
    RECONNECTING = "reconnecting"
    OFFLINE = "offline"
    ENDED = "ended"


@dataclass(frozen=True, slots=True)
class StreamSessionInfo:
    """Identity of one uninterrupted decoding session.

    A reconnect starts a *new* stream session and clears tracker state
    (thesis plan §9.2). ``camera_id + stream_session_id`` is the internal key
    prefix; the same camera therefore has many sessions over time.

    Attributes:
        camera_id: Configured camera.
        stream_session_id: UUID of this session, assigned before decoding.
        started_utc: Session start as a UTC epoch timestamp.
        started_monotonic: Session start on the monotonic clock, for durations.
        attempt: 1 for the first connection, 2 for the first reconnect, and so on.
    """

    camera_id: uuid.UUID
    stream_session_id: uuid.UUID
    started_utc: float
    started_monotonic: float
    attempt: int = 1


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One decoded video frame and the timing metadata it must carry.

    Thesis plan §9.1 requires every frame to carry ``source_id``,
    ``stream_session_id``, ``frame_seq``, a source timestamp when available, and
    a receive monotonic timestamp. MP4 uses presentation timestamps and must not
    derive time from loop speed; RTSP stores receive timestamps and uses source
    timestamps when the server provides them.

    Attributes:
        camera_id: Source camera UUID.
        stream_session_id: Session UUID, reset on every reconnect.
        frame_seq: Zero-based counter within the session, used to detect drops.
        frame: Decoded BGR ``uint8`` array of shape ``(H, W, 3)``.
        source_timestamp: Presentation timestamp in seconds, or ``None`` when
            the source does not provide one. ``None`` is never replaced by a
            synthetic value — honesty about missing source time is required to
            avoid claiming capture-to-display latency (thesis plan §9.1).
        receive_monotonic: Monotonic seconds when the reader assembled the frame.
        receive_utc: UTC epoch seconds when the reader assembled the frame.
        is_last: ``True`` for the final frame of a finite source (clean EOF).
    """

    camera_id: uuid.UUID
    stream_session_id: uuid.UUID
    frame_seq: int
    frame: np.ndarray
    source_timestamp: float | None
    receive_monotonic: float
    receive_utc: float
    is_last: bool = False

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(height, width, channels)`` of the decoded frame."""
        height, width, channels = self.frame.shape
        return int(height), int(width), int(channels)

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.frame.shape[1])

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.frame.shape[0])

    def nbytes(self) -> int:
        """Size of the frame buffer, used for queue-memory accounting."""
        return int(self.frame.nbytes)


@runtime_checkable
class FrameSource(Protocol):
    """Anything that yields :class:`FramePacket` objects for one camera.

    The pipeline supervisor and the replay CLI depend only on this protocol, so
    tests can substitute a synthetic source without starting FFmpeg.
    """

    @property
    def camera_id(self) -> uuid.UUID:
        """UUID of the camera being read."""
        ...

    def open(self) -> None:
        """Start decoding and block until the first frame metadata is known."""
        ...

    def read(self, timeout: float | None = None) -> FramePacket | None:
        """Return the next packet, or ``None`` on timeout or end of stream."""
        ...

    def close(self) -> None:
        """Stop decoding and release the subprocess and its threads."""
        ...
