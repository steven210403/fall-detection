"""ORM models.

``camera`` is a configured video source; ``stream_session`` is one uninterrupted
decoding attempt (a reconnect opens a new one, so session identity keeps two decoding
runs apart); ``track`` is the durable record of one tracked person within a session;
``fall_event`` is one detected fall of one tracked person.

A fall event is written the moment its alarm fires (the person has been lying for the
long-lie confirmation time) and updated when it closes, because an alarm that only
reached the database after the person stood up again would be useless to a caregiver.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, Index, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fall_detection.database.base import Base

__all__ = [
    "FALL_EVENT_END_REASONS",
    "TERMINATION_REASONS",
    "TRACK_TERMINATION_REASONS",
    "Camera",
    "FallEvent",
    "StreamSession",
    "Track",
]

# Reasons a session can end. Kept as a check constraint rather than a database
# enum so SQLite and PostgreSQL agree and Alembic autogenerate stays stable
# (thesis plan §19: string enums, portable types).
TERMINATION_REASONS = (
    "stopped",  # operator requested a stop
    "source_ended",  # clean end of a finite source (MP4 EOF)
    "decoder_error",  # FFmpeg failed or stalled repeatedly
    "source_invalid",  # the source could not be used at all
    "shutdown",  # process shutdown while the session was live
)

# Reasons a track can end (thesis plan §11.2). The vocabulary is owned by
# ``tracking.contracts`` because it is a tracking concept, but the constraint is
# declared here so the database can reject an invalid value; a test asserts the
# two tuples stay identical, which is cheaper than importing a CV module into the
# database layer.
TRACK_TERMINATION_REASONS = (
    "lost_timeout",  # the loss tolerance elapsed with no matching observation
    "session_ended",  # a reconnect or restart began a new stream session
    "stopped",  # the operator stopped the pipeline
)

# Why a fall event closed. Owned by ``config.settings.FALL_EVENT_END_REASONS``; a test
# asserts the two tuples stay identical, which is cheaper than importing configuration
# into the database layer.
FALL_EVENT_END_REASONS = (
    "recovered",  # the person stood up again after the alert
    "track_lost",  # the tracked identity ended while the alert was active
    "session_ended",  # a reconnect or restart began a new stream session
    "stopped",  # the operator stopped the pipeline or the process shut down
    "max_duration",  # the configured ceiling closed a very long alert
)


class Camera(Base):
    """A configured video source.

    ``source_ref`` stores a safe reference (never a credential) as required by
    thesis plan §19 and §24.2.
    """

    __tablename__ = "camera"
    __table_args__ = (
        CheckConstraint(
            "source_type IN ('mp4', 'rtsp', 'webcam')",
            name="source_type_allowed",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)

    sessions: Mapped[list[StreamSession]] = relationship(
        back_populates="camera",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Camera(id={self.id!r}, name={self.name!r}, source_type={self.source_type!r})"


class StreamSession(Base):
    """One uninterrupted decoding attempt for a camera.

    A reconnect creates a new row, so ``stream_session.id`` is the value the
    tracker and activity buffers key on and the value that resets their state
    (thesis plan §9.2). ``attempt`` counts connection attempts within the current
    run: 1 is the first connect, 2 the first reconnect.

    ``ended_at`` and ``termination_reason`` are written by the single database
    writer when the session closes; a session left open by a crash is closed
    with reason ``shutdown`` on the next start.
    """

    __tablename__ = "stream_session"
    __table_args__ = (
        CheckConstraint(
            "termination_reason IS NULL OR termination_reason IN ("
            + ", ".join(f"'{reason}'" for reason in TERMINATION_REASONS)
            + ")",
            name="termination_reason_allowed",
        ),
        CheckConstraint("attempt >= 1", name="attempt_positive"),
        CheckConstraint(
            "ended_at IS NULL OR ended_at >= started_at",
            name="ended_after_started",
        ),
        Index("ix_stream_session_camera_started", "camera_id", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("camera.id", ondelete="CASCADE"),
        nullable=False,
    )
    attempt: Mapped[int] = mapped_column(nullable=False, default=1)
    started_at: Mapped[datetime] = mapped_column(nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(nullable=True)
    termination_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    frames_read: Mapped[int] = mapped_column(nullable=False, default=0)
    frames_dropped: Mapped[int] = mapped_column(nullable=False, default=0)

    camera: Mapped[Camera] = relationship(back_populates="sessions")
    tracks: Mapped[list[Track]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"StreamSession(id={self.id!r}, camera_id={self.camera_id!r}, attempt={self.attempt!r})"
        )


class Track(Base):
    """One tracked person within one stream session (thesis plan §11.2, §19).

    The row exists so fall events have a durable identity to attach to. Two
    rules from §11.2 are enforced by the schema rather than by convention:

    * **The integer tracker ID is not a key.** ``tracker_id`` is a plain column
      for display and debugging; ``id`` is the UUID that identifies the track. A
      tracker may restart its integer IDs on a new session, and the same integer
      may therefore appear in two different sessions — or, after a terminated
      track, in the same session.
    * **Termination consumes the identity.** ``terminated_at`` and
      ``termination_reason`` are written when the track ends, and a person who
      returns afterwards gets a new row, because ``TrackLifecycle`` never
      resurrects a terminated UUID.

    ``first_seen_at``/``last_seen_at`` are observation times. For a track that
    ended after a timeout, ``last_seen_at`` is the *last observation*, not the
    moment the tolerance expired, so the stored interval always describes
    evidence rather than a grace period (FR-07).
    """

    __tablename__ = "track"
    __table_args__ = (
        CheckConstraint(
            "termination_reason IS NULL OR termination_reason IN ("
            + ", ".join(f"'{reason}'" for reason in TRACK_TERMINATION_REASONS)
            + ")",
            name="termination_reason_allowed",
        ),
        CheckConstraint("tracker_id >= 1", name="tracker_id_positive"),
        CheckConstraint("observed_frames >= 0", name="observed_frames_non_negative"),
        CheckConstraint(
            "last_seen_at >= first_seen_at",
            name="last_seen_after_first_seen",
        ),
        CheckConstraint(
            "terminated_at IS NULL OR terminated_at >= first_seen_at",
            name="terminated_after_first_seen",
        ),
        Index("ix_track_session_tracker", "stream_session_id", "tracker_id"),
        Index("ix_track_session_first_seen", "stream_session_id", "first_seen_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stream_session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("stream_session.id", ondelete="CASCADE"),
        nullable=False,
    )
    tracker_id: Mapped[int] = mapped_column(nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(nullable=False)
    terminated_at: Mapped[datetime | None] = mapped_column(nullable=True)
    termination_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    observed_frames: Mapped[int] = mapped_column(nullable=False, default=0)
    missing_frames: Mapped[int] = mapped_column(nullable=False, default=0)

    session: Mapped[StreamSession] = relationship(back_populates="tracks")
    fall_events: Mapped[list[FallEvent]] = relationship(
        back_populates="track",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"Track(id={self.id!r}, stream_session_id={self.stream_session_id!r}, "
            f"tracker_id={self.tracker_id!r})"
        )


class FallEvent(Base):
    """One detected fall of one tracked person.

    The four state-machine timestamps are **observation times** (the frames on which
    the transition was observed), while ``created_at`` is the moment the row was
    written, so the alarm response time can be measured from the evidence rather than
    from the database.

    The row is inserted when the alarm fires (``alert_at``) with ``ended_at`` and
    ``end_reason`` empty, and both are filled when the event closes. The two are
    constrained to be set together, so an event is either open or fully closed.
    """

    __tablename__ = "fall_event"
    __table_args__ = (
        CheckConstraint(
            "end_reason IS NULL OR end_reason IN ("
            + ", ".join(f"'{reason}'" for reason in FALL_EVENT_END_REASONS)
            + ")",
            name="end_reason_allowed",
        ),
        CheckConstraint(
            "(ended_at IS NULL AND end_reason IS NULL) "
            "OR (ended_at IS NOT NULL AND end_reason IS NOT NULL)",
            name="closed_together",
        ),
        CheckConstraint("impact_at >= falling_at", name="impact_after_falling"),
        CheckConstraint("alert_at >= impact_at", name="alert_after_impact"),
        CheckConstraint("ended_at IS NULL OR ended_at >= alert_at", name="ended_after_alert"),
        CheckConstraint(
            "acknowledged_at IS NULL OR acknowledged_at >= alert_at",
            name="acknowledged_after_alert",
        ),
        CheckConstraint("schema_version >= 1", name="schema_version_positive"),
        CheckConstraint("response_time_s >= 0", name="response_time_non_negative"),
        Index("ix_fall_event_alert", "alert_at"),
        Index("ix_fall_event_track_alert", "track_id", "alert_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("track.id", ondelete="CASCADE"),
        nullable=False,
    )
    falling_at: Mapped[datetime] = mapped_column(nullable=False)
    impact_at: Mapped[datetime] = mapped_column(nullable=False)
    alert_at: Mapped[datetime] = mapped_column(nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(nullable=True)
    end_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    response_time_s: Mapped[float] = mapped_column(nullable=False, default=0.0)
    peak_vy: Mapped[float | None] = mapped_column(nullable=True)
    min_torso_angle_deg: Mapped[float | None] = mapped_column(nullable=True)
    max_aspect_ratio: Mapped[float | None] = mapped_column(nullable=True)
    alert_source_timestamp: Mapped[float | None] = mapped_column(nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    schema_version: Mapped[int] = mapped_column(nullable=False, default=1)

    track: Mapped[Track] = relationship(back_populates="fall_events")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"FallEvent(id={self.id!r}, track_id={self.track_id!r}, alert_at={self.alert_at!r})"
