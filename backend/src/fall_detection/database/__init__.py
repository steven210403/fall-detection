"""Database engine, session factory, ORM base, and repositories."""

from fall_detection.database.base import Base
from fall_detection.database.models import (
    FALL_EVENT_END_REASONS,
    TERMINATION_REASONS,
    TRACK_TERMINATION_REASONS,
    Camera,
    FallEvent,
    StreamSession,
    Track,
)
from fall_detection.database.repositories import (
    CameraRepository,
    FallEventRepository,
    StreamSessionRepository,
    TrackRepository,
    decode_cursor,
    encode_cursor,
    to_fall_event_record,
    utc_now,
)
from fall_detection.database.session import (
    DatabaseWriter,
    create_engine,
    create_session_factory,
)

__all__ = [
    "FALL_EVENT_END_REASONS",
    "TERMINATION_REASONS",
    "TRACK_TERMINATION_REASONS",
    "Base",
    "Camera",
    "CameraRepository",
    "DatabaseWriter",
    "FallEvent",
    "FallEventRepository",
    "StreamSession",
    "StreamSessionRepository",
    "Track",
    "TrackRepository",
    "create_engine",
    "create_session_factory",
    "decode_cursor",
    "encode_cursor",
    "to_fall_event_record",
    "utc_now",
]
