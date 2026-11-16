"""API-boundary Pydantic models (thesis plan §17.2).

§17.1 fixes where Pydantic belongs: "Use Pydantic at configuration/API boundaries; do
not import ORM models into CV modules." These models are the request and response
contracts of the minimum API, and the CV layers never see them — they exchange
dataclasses.

Two conventions run through every model here:

* **Field names match the internal contract's JSON view.** A response is the contract's
  ``as_dict()`` validated into a typed model, so a client cannot receive a field the
  internal record does not have, and adding a field is a one-line change in two places
  rather than a silent drift.
* **Nothing carries a credential.** §17.2 requires ``GET /api/v1/cameras`` to return
  "camera metadata without credentials", so every camera payload is built from
  ``VideoSource.describe()``/``safe_ref``, which redact the userinfo section.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "API_SCHEMA_VERSION",
    "CameraModel",
    "CameraStateModel",
    "ErrorBody",
    "ErrorResponse",
    "FallEventModel",
    "FallEventPageModel",
    "FallFeaturesModel",
    "FallStateModel",
    "LiveEnvelope",
    "PreviewFrameModel",
    "StatusModel",
    "TrackModel",
]

API_SCHEMA_VERSION = 1
"""Version of the HTTP and WebSocket payloads.

Reported in the live envelope (§17.2) so a client can tell whether it understands what
it is being sent before it renders anything. A breaking change to a payload shape must
increment this rather than quietly changing a field's meaning.
"""


class ErrorBody(BaseModel):
    """Machine-readable error detail.

    §17.2 requires "consistent error codes for invalid sources, camera failures,
    unavailable models, and database errors". The ``code`` is the stable identifier
    from :class:`~fall_detection.core.errors.CctvError`, so a client switches on it
    rather than on a message that may be reworded.
    """

    code: str
    message: str
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Envelope for every error the API returns."""

    error: ErrorBody


class CameraModel(BaseModel):
    """One configured camera, without credentials (§17.2)."""

    camera_id: str
    name: str
    source_type: str
    source_ref: str
    state: str
    """One of the FR-09 states: ``idle``, ``connecting``, ``live``, ``reconnecting``,
    ``offline``, ``ended``. Distinct values so the operator can tell "not started" from
    "was live and dropped"."""
    stream_session_id: str | None = None
    attempt: int | None = None
    stream: dict[str, Any] | None = None
    source: dict[str, Any] | None = None
    """Credential-free description of the resolved source."""


class CameraStateModel(BaseModel):
    """Result of a start or stop request."""

    camera_id: str
    state: str
    stream_session_id: str | None = None
    started: bool = False
    stopped: bool = False


class FallFeaturesModel(BaseModel):
    """The three fall parameters for one person on one frame.

    ``None`` means "not measurable on this frame", never zero.
    """

    aspect_ratio: float | None = None
    torso_angle_deg: float | None = None
    torso_side: Literal["both", "partial"] | None = None
    hip_center: list[float] | None = None
    body_length_px: float | None = None
    visible_keypoints: int = 0
    vy_px_s: float | None = None
    vy_norm: float | None = None
    time_source: Literal["source", "monotonic"]


class FallStateModel(BaseModel):
    """One person's fall state."""

    state: Literal["unknown", "normal", "falling", "impact", "fallen"]
    state_duration_s: float
    long_lie_remaining_s: float | None = None
    event_id: str | None = None
    frame_seq: int
    features: FallFeaturesModel


class TrackModel(BaseModel):
    """One live tracked person in a live payload."""

    track_uuid: str
    display_id: int
    state: str
    bbox: list[float] | None = None
    age_s: float = 0.0
    time_since_last_observation_s: float = 0.0
    stale: bool = False
    confidence: float | None = None
    fall: FallStateModel | None = None


class PreviewFrameModel(BaseModel):
    """Metadata of the JPEG the preview endpoint is serving (§18.1, §18.2).

    The dimensions are reported together with the *source* dimensions, so a client can
    map a normalized zone coordinate onto the displayed image without knowing the scale
    separately — the letterboxing rule §18.1 asks the ROI editor to respect.
    """

    frame_seq: int
    sequence: int
    width: int
    height: int
    source_width: int
    source_height: int
    jpeg_bytes: int
    source_timestamp: float | None = None
    captured_monotonic: float
    captured_utc: float


class LiveEnvelope(BaseModel):
    """One message on ``WS /api/v1/live`` (§17.2).

    ``sequence`` is monotonic per process, so a client that receives 5 then 8 knows it
    missed two messages and can recover the interval with the event cursor. That is the
    recovery path §17.2 requires for a slow client, which by design receives "the latest
    snapshot, not a backlog".
    """

    schema_version: int
    kind: Literal["hello", "snapshot", "event", "status"]
    sequence: int
    emitted_monotonic: float
    emitted_utc: float
    camera_id: str | None = None
    stream_session_id: str | None = None
    frame_seq: int | None = None
    source_timestamp: float | None = None
    receive_monotonic: float | None = None
    receive_utc: float | None = None
    preview: PreviewFrameModel | None = None
    tracks: list[TrackModel] = Field(default_factory=list)
    active_alarms: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    event: dict[str, Any] | None = None
    status: dict[str, Any] | None = None
    latest_cursor: str | None = None
    """Cursor to resume event history from after a reconnect (§17.2)."""


class FallEventModel(BaseModel):
    """One stored fall event.

    The ``*_utc`` times are observation times; ``created_utc`` is when the row was
    written. ``active`` is ``True`` while the alarm has not closed.
    """

    event_id: str
    track_uuid: str
    camera_id: str
    stream_session_id: str
    display_id: int | None = None
    falling_utc: float
    impact_utc: float
    alert_utc: float
    ended_utc: float | None = None
    end_reason: str | None = None
    active: bool
    response_time_s: float
    peak_vy: float | None = None
    min_torso_angle_deg: float | None = None
    max_aspect_ratio: float | None = None
    alert_source_timestamp: float | None = None
    acknowledged_utc: float | None = None
    created_utc: float | None = None
    schema_version: int = 1


class FallEventPageModel(BaseModel):
    """One page of fall-event history; ``next_cursor`` is ``None`` at the end."""

    events: list[FallEventModel] = Field(default_factory=list)
    next_cursor: str | None = None
    limit: int
    total: int


class StatusModel(BaseModel):
    """``GET /api/v1/status``: camera, model, sampling, and resources (§17.2).

    The payload is deliberately one object rather than several endpoints: an operator
    diagnosing a problem needs the camera state, the model state, the sampling settings,
    and the resource measurements together, and §17.2 lists them as one endpoint.
    """

    schema_version: int
    camera: dict[str, Any] = Field(default_factory=dict)
    pipeline: dict[str, Any] = Field(default_factory=dict)
    detection: dict[str, Any] = Field(default_factory=dict)
    tracking: dict[str, Any] = Field(default_factory=dict)
    fall: dict[str, Any] = Field(default_factory=dict)
    performance: dict[str, Any] = Field(default_factory=dict)
    events: dict[str, Any] = Field(default_factory=dict)
    streaming: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)
