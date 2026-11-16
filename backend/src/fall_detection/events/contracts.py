"""Fall-event contracts.

A :class:`FallEventRecord` is one detected fall of one tracked person, as stored and as
sent to clients. It is stdlib-only, so the database layer can build it without
importing anything heavier, and the API can serialise it without the ORM.

The four state-machine times (``falling``, ``impact``, ``alert``, ``ended``) are
**observation times**: the UTC instants of the frames on which each transition was
observed. ``created_utc`` is when the row was written. Keeping the two apart is what
lets the alarm response time be measured from evidence rather than from the database.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

__all__ = ["FallEventPage", "FallEventRecord", "OpenFallEvent"]


@dataclass(frozen=True, slots=True)
class FallEventRecord:
    """One stored fall event."""

    event_id: uuid.UUID
    track_uuid: uuid.UUID
    camera_id: uuid.UUID
    stream_session_id: uuid.UUID
    display_id: int | None
    falling_utc: float
    impact_utc: float
    alert_utc: float
    ended_utc: float | None
    end_reason: str | None
    response_time_s: float
    peak_vy: float | None
    min_torso_angle_deg: float | None
    max_aspect_ratio: float | None
    alert_source_timestamp: float | None
    acknowledged_utc: float | None
    created_utc: float
    schema_version: int = 1

    @property
    def active(self) -> bool:
        """Whether the alarm is still active (the event has not closed)."""
        return self.ended_utc is None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "event_id": str(self.event_id),
            "track_uuid": str(self.track_uuid),
            "camera_id": str(self.camera_id),
            "stream_session_id": str(self.stream_session_id),
            "display_id": self.display_id,
            "falling_utc": self.falling_utc,
            "impact_utc": self.impact_utc,
            "alert_utc": self.alert_utc,
            "ended_utc": self.ended_utc,
            "end_reason": self.end_reason,
            "active": self.active,
            "response_time_s": round(self.response_time_s, 3),
            "peak_vy": _round(self.peak_vy),
            "min_torso_angle_deg": _round(self.min_torso_angle_deg),
            "max_aspect_ratio": _round(self.max_aspect_ratio),
            "alert_source_timestamp": self.alert_source_timestamp,
            "acknowledged_utc": self.acknowledged_utc,
            "created_utc": self.created_utc,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class FallEventPage:
    """One page of fall-event history, newest alert first."""

    events: tuple[FallEventRecord, ...]
    next_cursor: str | None
    limit: int
    total: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "events": [event.as_dict() for event in self.events],
            "next_cursor": self.next_cursor,
            "limit": self.limit,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class OpenFallEvent:
    """What the fall layer hands the writer when an alarm fires.

    Attributes:
        event_id: Identity chosen by the fall layer, so the close can name it.
        track_uuid: The tracked person.
        camera_id: Source camera.
        stream_session_id: Session the track belongs to.
        display_id: Integer tracker ID, for display only.
        falling_utc: Observation time the velocity spike was seen.
        impact_utc: Observation time the lying posture was first confirmed.
        alert_utc: Observation time the long-lie confirmation elapsed.
        response_time_s: Seconds from the impact to the alarm, measured on the source's own
            clock (not on wall time, which a file replayed faster than real time compresses).
        peak_vy: Peak normalised downward hip velocity during the fall.
        min_torso_angle_deg: Smallest torso angle seen so far.
        max_aspect_ratio: Largest bounding-box aspect ratio seen so far.
        alert_source_timestamp: Source presentation timestamp of the alert frame.
    """

    event_id: uuid.UUID
    track_uuid: uuid.UUID
    camera_id: uuid.UUID
    stream_session_id: uuid.UUID
    display_id: int | None
    falling_utc: float
    impact_utc: float
    alert_utc: float
    response_time_s: float
    peak_vy: float | None
    min_torso_angle_deg: float | None
    max_aspect_ratio: float | None
    alert_source_timestamp: float | None


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
