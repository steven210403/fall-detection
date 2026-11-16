"""Fall-detection contracts (THESIS_PLAN §4–§5).

The fall layer turns one tracked person's keypoints into three measurements and one
state::

    TrackObservation → PoseGeometry (AR, torso angle, hip centre)
                     → FallFeatures (+ hip vertical velocity Vy)
                     → FallDecision (state of the per-person state machine)

Every measurement is either a number or ``None``. ``None`` means "not measurable on this
frame" (for example the hips were not visible), and the state machine never reads it as
"upright" or "lying": an unmeasurable frame does not advance or reset a fall.

This module is stdlib-only, so the API and the tests can import it without the ML stack.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

__all__ = [
    "FallBatch",
    "FallDecision",
    "FallFeatures",
    "FallState",
    "FallTransition",
    "PoseGeometry",
    "TimeSource",
]

TimeSource = Literal["source", "monotonic"]
"""Which clock an elapsed time was measured on: the source's presentation/acquisition
timestamps (preferred) or the process's monotonic clock (fallback)."""


class FallState(StrEnum):
    """States of the per-person fall state machine (THESIS_PLAN §5)."""

    UNKNOWN = "unknown"
    """No measurement has been possible yet for this person."""
    NORMAL = "normal"
    """Upright or moving normally; no fall in progress."""
    FALLING = "falling"
    """A sudden downward hip velocity was observed; waiting for an impact posture."""
    IMPACT = "impact"
    """The body reached a lying posture after the velocity spike; the long-lie timer runs."""
    FALLEN = "fallen"
    """The person stayed down for the long-lie time without recovering: alarm."""


@dataclass(frozen=True, slots=True)
class PoseGeometry:
    """Geometric measurements of one person in one frame.

    Attributes:
        aspect_ratio: Bounding-box ``W / H``. ``None`` for a degenerate box.
        torso_angle_deg: Angle between the shoulder→hip vector and the horizontal, in
            ``[0, 90]`` degrees: ~90 upright, ~0 lying. ``None`` when no shoulder or no
            hip keypoint is visible.
        torso_side: ``"both"`` when both shoulders and both hips were visible, ``"partial"``
            when the angle was measured from the visible side only, ``None`` when no
            angle was measured.
        hip_center: Hip midpoint ``(x, y)`` in pixels (the centre of mass the velocity is
            measured on), or ``None`` when no hip keypoint is visible.
        body_length_px: ``max(W, H)`` of the box, a body-length estimate that does not
            depend on posture; used to normalise the velocity.
        visible_keypoints: Number of keypoints above the confidence floor.
    """

    aspect_ratio: float | None
    torso_angle_deg: float | None
    torso_side: Literal["both", "partial"] | None
    hip_center: tuple[float, float] | None
    body_length_px: float | None
    visible_keypoints: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "aspect_ratio": _round(self.aspect_ratio),
            "torso_angle_deg": _round(self.torso_angle_deg, 2),
            "torso_side": self.torso_side,
            "hip_center": None
            if self.hip_center is None
            else [round(self.hip_center[0], 1), round(self.hip_center[1], 1)],
            "body_length_px": _round(self.body_length_px, 1),
            "visible_keypoints": self.visible_keypoints,
        }


@dataclass(frozen=True, slots=True)
class FallFeatures:
    """The three fall parameters for one person in one frame.

    Attributes:
        track_uuid: Tracked identity.
        display_id: Integer tracker ID (display only).
        frame_seq: Frame counter within the session.
        t: Observation time in seconds on ``time_source``'s clock. Only differences of
            ``t`` within one track are meaningful.
        utc: UTC epoch seconds of the frame, used for stored event times.
        time_source: Which clock ``t`` came from.
        geometry: Aspect ratio, torso angle and hip centre.
        vy_px_s: Downward hip velocity in pixels per second (image y grows downward,
            so a fall is positive). ``None`` when it could not be measured.
        vy_norm: ``vy_px_s`` divided by the person's reference body length, i.e. body
            lengths per second. This is the value the threshold is applied to, so the
            same threshold works at 2, 3 or 4 m from the camera.
        source_timestamp: The frame's source timestamp, when the source has one.
        height_ratio: Box height divided by the person's recent standing height (the
            tallest upright box in the last ``standing_window_s``); ``None`` before any
            upright observation.
    """

    track_uuid: uuid.UUID
    display_id: int
    frame_seq: int
    t: float
    utc: float
    time_source: TimeSource
    geometry: PoseGeometry
    vy_px_s: float | None
    vy_norm: float | None
    source_timestamp: float | None = None
    height_ratio: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            **self.geometry.as_dict(),
            "vy_px_s": _round(self.vy_px_s, 2),
            "vy_norm": _round(self.vy_norm),
            "height_ratio": _round(self.height_ratio),
            "time_source": self.time_source,
        }


@dataclass(frozen=True, slots=True)
class FallTransition:
    """One observed state change, kept for evaluation and debugging."""

    track_uuid: uuid.UUID
    from_state: FallState
    to_state: FallState
    t: float
    utc: float
    frame_seq: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "track_uuid": str(self.track_uuid),
            "from": self.from_state.value,
            "to": self.to_state.value,
            "t": round(self.t, 4),
            "utc": self.utc,
            "frame_seq": self.frame_seq,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class FallDecision:
    """The state of one person after one frame.

    Attributes:
        features: The measurements the decision was made from.
        state: The state after this frame.
        state_since_t: Observation time the current state was entered.
        transitions: State changes that happened on this frame (usually zero or one).
        long_lie_remaining_s: Seconds left before the alarm while in ``impact``;
            ``None`` in any other state.
        event_id: Identity of the open fall event while the alarm is active.
        alert_fired: ``True`` on the frame the alarm fired.
    """

    features: FallFeatures
    state: FallState
    state_since_t: float
    transitions: tuple[FallTransition, ...] = ()
    long_lie_remaining_s: float | None = None
    event_id: uuid.UUID | None = None
    alert_fired: bool = False

    @property
    def track_uuid(self) -> uuid.UUID:
        """Tracked identity this decision belongs to."""
        return self.features.track_uuid

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for the live payload."""
        return {
            "state": self.state.value,
            "state_duration_s": round(max(0.0, self.features.t - self.state_since_t), 3),
            "long_lie_remaining_s": _round(self.long_lie_remaining_s, 2),
            "event_id": None if self.event_id is None else str(self.event_id),
            "frame_seq": self.features.frame_seq,
            "features": self.features.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class FallBatch:
    """Everything the fall layer produced from one frame."""

    stream_session_id: uuid.UUID
    frame_seq: int
    decisions: tuple[FallDecision, ...]
    closed_tracks: tuple[uuid.UUID, ...] = ()
    alerts: tuple[uuid.UUID, ...] = field(default=())

    def by_track(self) -> dict[uuid.UUID, FallDecision]:
        """Index this frame's decisions by track UUID."""
        return {decision.track_uuid: decision for decision in self.decisions}


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)
