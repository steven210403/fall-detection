"""Tracking contracts.

Thesis plan §17.1 fixes the internal contract chain::

    FramePacket → DetectionBatch → TrackObservations → PersonFeatures
                → ActivityPrediction → ActivityEvent

Phase 4 implements the third link. The chain exists so no downstream module has
to reach back into an earlier layer for context, which is why
:class:`TrackObservations` repeats the frame identity and both timestamps that
:class:`~fall_detection.detection.contracts.DetectionBatch` already carried.

Two distinctions from §11.2 are encoded in the types rather than in prose:

* A **detection** is one bounding-box observation; a **track** is a temporary
  association identifier within one stream session. :class:`TrackObservation` is
  therefore always tied to a detection that was actually observed in this frame —
  there is no field that could hold a Kalman prediction, so a lost track cannot
  contribute evidence by construction.
* The **integer tracker ID is for display and debugging only** (§11.2). Identity
  is :attr:`TrackObservation.track_uuid`, whose internal key is
  ``camera_id + stream_session_uuid + track_uuid``.

This module stays free of NumPy, OpenCV, PyTorch, and SQLAlchemy, like the other
contract modules: importing ``fall_detection.tracking`` must not pull in the ML
stack, because the API imports the package on every start.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "TRACK_TERMINATION_REASONS",
    "TrackObservation",
    "TrackObservations",
    "TrackState",
    "TrackSummary",
    "TrackTermination",
]

TRACK_TERMINATION_REASONS: tuple[str, ...] = (
    "lost_timeout",  # the loss tolerance elapsed with no matching observation
    "session_ended",  # a reconnect or restart began a new stream session
    "stopped",  # the operator stopped the pipeline
)
"""Reasons a track can end, mirroring the session reasons in §19.

Kept as a plain tuple so the database check constraint, the API layer, and the
tests all read one list instead of repeating string literals.
"""


class TrackState(StrEnum):
    """Lifecycle state of one track (thesis plan §11.2).

    ``LOST`` is a *recoverable* state, not an ended one: §11.2 allows a new
    detection to be associated with the track again before termination. While a
    track is lost its buffer must receive missing observations instead of
    evidence, and after ``stale_after_s`` its last state must not be used as
    evidence at all.
    """

    ACTIVE = "active"
    """Observed in the frame this summary was built for."""

    LOST = "lost"
    """No matching observation in this frame; held for possible recovery."""

    TERMINATED = "terminated"
    """Ended. A person seen again receives a *new* track UUID (§11.2)."""


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """One person observation attributed to a track, for one frame.

    Every instance corresponds to a real detection in this frame — §11.2 forbids
    extending activity with a lost track's motion prediction, and this contract
    offers no field in which such a prediction could be carried.

    Attributes:
        track_uuid: Stable identity for the life of the track (§11.2 internal
            key ``camera_id + stream_session_uuid + track_uuid``).
        display_id: Integer tracker ID. Display and debugging only; never an
            identity and never a primary key.
        camera_id: Source camera UUID.
        stream_session_id: Session UUID, so a buffer can detect a reset.
        bbox: ``(x1, y1, x2, y2)`` in original-frame pixels.
        confidence: Detector score of the matched detection. Not a probability
            that the association is correct.
        class_name: Detection class, always ``"person"`` in Phase 4.
        frame_seq: Frame counter within the session.
        observed_monotonic: Monotonic seconds when the detection was produced.
        source_timestamp: Presentation timestamp in seconds, or ``None`` when the
            source provided none; never synthesised (§9.1).
        receive_monotonic: Monotonic seconds when the frame was assembled.
        receive_utc: UTC epoch seconds when the frame was assembled.
        first_seen_monotonic: Monotonic seconds when this track started.
        age_s: Time since the track started.
        observed_frames: Number of frames with an observation, including this one.
        gap_frames: Consecutive frames without an observation immediately before
            this one. ``0`` for an uninterrupted observation, positive after a
            recovery, which is how the feature buffer learns a gap occurred.
        recovered: ``True`` when this observation ended a gap, so the caller can
            see that a recovery happened without diffing state itself.
        keypoints: The matched detection's 17 COCO keypoints ``(x, y, conf)``, or
            ``None`` for a box-only model. They are this frame's measurement, read
            back through the detection index, never a prediction.
    """

    track_uuid: uuid.UUID
    display_id: int
    camera_id: uuid.UUID
    stream_session_id: uuid.UUID
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str
    frame_seq: int
    observed_monotonic: float
    source_timestamp: float | None
    receive_monotonic: float
    receive_utc: float
    first_seen_monotonic: float
    age_s: float
    observed_frames: int
    gap_frames: int = 0
    recovered: bool = False
    keypoints: tuple[tuple[float, float, float], ...] | None = None

    @property
    def width(self) -> float:
        """Box width in pixels (never negative)."""
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        """Box height in pixels (never negative)."""
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple[float, float]:
        """Box centre in pixels."""
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for status and live payloads."""
        return {
            "track_uuid": str(self.track_uuid),
            "display_id": self.display_id,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "class_name": self.class_name,
            "frame_seq": self.frame_seq,
            "age_s": round(self.age_s, 3),
            "observed_frames": self.observed_frames,
            "gap_frames": self.gap_frames,
            "recovered": self.recovered,
        }


@dataclass(frozen=True, slots=True)
class TrackSummary:
    """Current state of one track, for status payloads and the next phase.

    A summary describes a track that may currently be lost, so it deliberately
    carries the *last observed* geometry rather than a predicted box. Consumers
    that need evidence must read :class:`TrackObservation`; consumers that need
    to know a track still exists read this.

    Attributes:
        track_uuid: Stable identity.
        display_id: Integer tracker ID (display only).
        camera_id: Source camera UUID.
        stream_session_id: Session UUID.
        state: ``active``, ``lost``, or ``terminated``.
        first_seen_monotonic: Monotonic seconds when the track started.
        last_observed_monotonic: Monotonic seconds of the newest observation.
        age_s: Time since the track started, at the moment of building.
        time_since_last_observation_s: Gap length at the moment of building.
        stale: ``True`` once the gap exceeds ``stale_after_s`` (0.5 s by default,
            §11.2), which is the point where the last state stops being usable as
            evidence. Computed when the summary is built, so a summary held for
            later must be rebuilt rather than trusted.
        observed_frames: Frames with a matched observation.
        missing_frames: Frames without one while the track stayed alive.
        bbox: Last observed box, or ``None`` before the first observation.
        confidence: Last observed detection score, or ``None``.
        termination_reason: Set only for a terminated track; one of
            :data:`TRACK_TERMINATION_REASONS`.
    """

    track_uuid: uuid.UUID
    display_id: int
    camera_id: uuid.UUID
    stream_session_id: uuid.UUID
    state: TrackState
    first_seen_monotonic: float
    last_observed_monotonic: float
    age_s: float
    time_since_last_observation_s: float
    stale: bool
    observed_frames: int
    missing_frames: int
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    termination_reason: str | None = None

    @property
    def is_live(self) -> bool:
        """Whether the track is still active or held for recovery."""
        return self.state in (TrackState.ACTIVE, TrackState.LOST)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for status payloads.

        §18.1 requires unknown and stale states to be visible in the interface,
        so ``stale`` and ``state`` are reported as explicit fields rather than
        being folded into a colour.
        """
        payload: dict[str, Any] = {
            "track_uuid": str(self.track_uuid),
            "display_id": self.display_id,
            "state": self.state.value,
            "age_s": round(self.age_s, 3),
            "time_since_last_observation_s": round(self.time_since_last_observation_s, 3),
            "stale": self.stale,
            "observed_frames": self.observed_frames,
            "missing_frames": self.missing_frames,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
        }
        if self.termination_reason is not None:
            payload["termination_reason"] = self.termination_reason
        return payload


@dataclass(frozen=True, slots=True)
class TrackTermination:
    """One track ending, reported on the update that detected it.

    Termination is terminal (§11.2): the track's buffer and any workstation
    assignment are closed, and the same person appearing later gets a new
    ``track_uuid``. The integer ID is not resurrected either, because the
    lifecycle retires it for the rest of the stream session.

    Attributes:
        track_uuid: Identity that ended.
        display_id: Integer tracker ID that ended.
        camera_id: Source camera UUID.
        stream_session_id: Session UUID.
        reason: One of :data:`TRACK_TERMINATION_REASONS`.
        ended_monotonic: Monotonic seconds when the track ended.
        ended_utc: UTC epoch seconds when the track ended.
        duration_s: Time from first observation to termination.
        observed_frames: Frames that had a matched observation.
        missing_frames: Frames without one while the track stayed alive.
    """

    track_uuid: uuid.UUID
    display_id: int
    camera_id: uuid.UUID
    stream_session_id: uuid.UUID
    reason: str
    ended_monotonic: float
    ended_utc: float
    duration_s: float
    observed_frames: int
    missing_frames: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for status payloads."""
        return {
            "track_uuid": str(self.track_uuid),
            "display_id": self.display_id,
            "reason": self.reason,
            "duration_s": round(self.duration_s, 3),
            "observed_frames": self.observed_frames,
            "missing_frames": self.missing_frames,
        }


@dataclass(frozen=True, slots=True)
class TrackObservations:
    """Everything tracking produced from one :class:`DetectionBatch`.

    Attributes:
        camera_id: Source camera UUID.
        stream_session_id: Session UUID; a change means the tracker was reset.
        frame_seq: Frame counter within the session.
        observations: One entry per person observed in this frame. This is the
            only place activity evidence can come from.
        tracks: Summary of every live track after this update, including the ones
            that were lost in this frame, so the caller can insert missing
            observations into a buffer (§11.2).
        terminated: Tracks that ended during this update.
        frame_shape: ``(height, width, channels)`` of the frame the detections
            came from.
        source_timestamp: Presentation timestamp in seconds, or ``None``.
        receive_monotonic: Monotonic seconds the frame was assembled.
        receive_utc: UTC epoch seconds the frame was assembled.
        tracking_monotonic: Monotonic seconds when tracking finished.
        tracker_name: Configured tracker that produced this update.
    """

    camera_id: uuid.UUID
    stream_session_id: uuid.UUID
    frame_seq: int
    observations: tuple[TrackObservation, ...]
    tracks: tuple[TrackSummary, ...]
    terminated: tuple[TrackTermination, ...]
    frame_shape: tuple[int, int, int]
    source_timestamp: float | None
    receive_monotonic: float
    receive_utc: float
    tracking_monotonic: float
    tracker_name: str

    def observed_uuids(self) -> tuple[uuid.UUID, ...]:
        """Track UUIDs observed in this frame, in model order."""
        return tuple(observation.track_uuid for observation in self.observations)

    def lost_tracks(self) -> tuple[TrackSummary, ...]:
        """Live tracks that were not observed in this frame."""
        return tuple(track for track in self.tracks if track.state is TrackState.LOST)

    def by_uuid(self) -> dict[uuid.UUID, TrackObservation]:
        """Index this frame's observations by track UUID."""
        return {observation.track_uuid: observation for observation in self.observations}

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for status and live payloads."""
        return {
            "stream_session_id": str(self.stream_session_id),
            "frame_seq": self.frame_seq,
            "tracker": self.tracker_name,
            "observations": [item.as_dict() for item in self.observations],
            "tracks": [item.as_dict() for item in self.tracks],
            "terminated": [item.as_dict() for item in self.terminated],
        }
