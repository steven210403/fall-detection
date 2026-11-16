"""Repositories.

Repositories are the only place SQLAlchemy query syntax appears. Callers pass and
receive plain values (UUIDs, datetimes, counters, contract records) rather than ORM
instances, which keeps the computer-vision modules free of ORM imports.

The fall-event repository reads history with a keyset cursor ``(alert_at, id)``
rather than an offset, because the camera keeps writing events while a client pages.
"""

from __future__ import annotations

import base64
import binascii
import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import delete, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from fall_detection.core.errors import NotFoundError
from fall_detection.database.models import Camera, FallEvent, StreamSession, Track

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from fall_detection.events.contracts import FallEventPage, FallEventRecord

__all__ = [
    "CameraRepository",
    "FallEventRepository",
    "StreamSessionRepository",
    "TrackRepository",
    "decode_cursor",
    "encode_cursor",
    "to_fall_event_record",
    "utc_now",
]


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    Timestamps are stored in UTC everywhere. Video-relative time is carried
    separately on the frame packet, never substituted here.
    """
    return datetime.now(UTC)


class CameraRepository:
    """Read and create camera rows."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, camera_id: uuid.UUID) -> Camera | None:
        """Return the camera with ``camera_id``, or ``None``."""
        return self._session.get(Camera, camera_id)

    def get_or_create(
        self,
        *,
        camera_id: uuid.UUID,
        name: str,
        source_type: str,
        source_ref: str,
    ) -> Camera:
        """Return an existing camera, creating it when absent.

        Idempotent so that repeated pipeline starts do not fail on a duplicate
        primary key.
        """
        camera = self.get(camera_id)
        if camera is not None:
            return camera

        camera = Camera(
            id=camera_id,
            name=name,
            source_type=source_type,
            source_ref=source_ref,
            created_at=utc_now(),
        )
        self._session.add(camera)
        self._session.flush()
        return camera

    def list_all(self) -> list[Camera]:
        """Return every configured camera, oldest first."""
        return list(self._session.scalars(select(Camera).order_by(Camera.created_at, Camera.name)))


class StreamSessionRepository:
    """Open, close, and recover stream sessions."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def open(
        self,
        *,
        camera_id: uuid.UUID,
        attempt: int = 1,
        stream_session_id: uuid.UUID | None = None,
        started_at: datetime | None = None,
    ) -> uuid.UUID:
        """Insert a new stream session and return its UUID.

        The caller may supply the UUID so the in-memory
        :class:`~fall_detection.core.contracts.StreamSessionInfo` and the database
        row share one identity from the first frame.
        """
        session_id = stream_session_id or uuid.uuid4()
        self._session.add(
            StreamSession(
                id=session_id,
                camera_id=camera_id,
                attempt=attempt,
                started_at=started_at or utc_now(),
            )
        )
        self._session.flush()
        return session_id

    def close(
        self,
        *,
        stream_session_id: uuid.UUID,
        termination_reason: str,
        frames_read: int = 0,
        frames_dropped: int = 0,
        ended_at: datetime | None = None,
    ) -> None:
        """Finalise a session with its end time, reason, and frame counters."""
        self._session.execute(
            update(StreamSession)
            .where(StreamSession.id == stream_session_id)
            .values(
                ended_at=ended_at or utc_now(),
                termination_reason=termination_reason,
                frames_read=frames_read,
                frames_dropped=frames_dropped,
            )
        )

    def get(self, stream_session_id: uuid.UUID) -> StreamSession | None:
        """Return one session row, or ``None``."""
        return self._session.get(StreamSession, stream_session_id)

    def list_open(self) -> list[StreamSession]:
        """Return sessions that were never closed."""
        return list(
            self._session.scalars(
                select(StreamSession)
                .where(StreamSession.ended_at.is_(None))
                .order_by(StreamSession.started_at)
            )
        )

    def close_orphans(self, *, reason: str = "shutdown") -> int:
        """Close every open session left behind by a previous process.

        Called once at startup: a session still marked open cannot be live
        because this process just started, so leaving it open would make the
        history ambiguous. Returns the number of rows closed.
        """
        orphans = self.list_open()
        if not orphans:
            return 0
        now = utc_now()
        for orphan in orphans:
            orphan.ended_at = now
            orphan.termination_reason = reason
        self._session.flush()
        return len(orphans)


class TrackRepository:
    """Open, close, and read tracked identities.

    Thesis plan §11.2: the UUID is the identity, the integer tracker ID is for
    display and debugging. Every method therefore takes and returns UUIDs, and no
    caller has to know the integer at all.

    Track rows are written through the single :class:`DatabaseWriter`, so the
    inference thread never opens an ad-hoc session (§17).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def open(
        self,
        *,
        track_id: uuid.UUID,
        stream_session_id: uuid.UUID,
        tracker_id: int,
        first_seen_at: datetime | None = None,
    ) -> uuid.UUID:
        """Insert a new track and return its UUID.

        The caller supplies the UUID minted by
        :class:`~fall_detection.tracking.lifecycle.TrackLifecycle` so the runtime
        identity and the stored row share one value from the first observation.
        """
        seen = first_seen_at or utc_now()
        self._session.add(
            Track(
                id=track_id,
                stream_session_id=stream_session_id,
                tracker_id=tracker_id,
                first_seen_at=seen,
                last_seen_at=seen,
            )
        )
        self._session.flush()
        return track_id

    def close(
        self,
        *,
        track_id: uuid.UUID,
        termination_reason: str,
        last_seen_at: datetime | None = None,
        terminated_at: datetime | None = None,
        observed_frames: int = 0,
        missing_frames: int = 0,
    ) -> None:
        """Finalise a track with its end time, reason, and frame counters.

        ``last_seen_at`` is the last *observation* rather than the moment the
        decision was made, which is what FR-07 requires of stored times.
        """
        end = last_seen_at or utc_now()
        self._session.execute(
            update(Track)
            .where(Track.id == track_id)
            .values(
                last_seen_at=end,
                terminated_at=terminated_at or end,
                termination_reason=termination_reason,
                observed_frames=observed_frames,
                missing_frames=missing_frames,
            )
        )

    def touch(
        self,
        *,
        track_id: uuid.UUID,
        last_seen_at: datetime,
        observed_frames: int,
        missing_frames: int,
    ) -> None:
        """Update a still-live track's counters.

        Optional: the track row is correct without it, because :meth:`close`
        writes the final figures. It exists so a long session does not lose the
        counters if the process is killed before the track ends.
        """
        self._session.execute(
            update(Track)
            .where(Track.id == track_id)
            .values(
                last_seen_at=last_seen_at,
                observed_frames=observed_frames,
                missing_frames=missing_frames,
            )
        )

    def get(self, track_id: uuid.UUID) -> Track | None:
        """Return one track row, or ``None``."""
        return self._session.get(Track, track_id)

    def list_for_session(self, stream_session_id: uuid.UUID) -> list[Track]:
        """Return every track of one session, ordered by first observation."""
        return list(
            self._session.scalars(
                select(Track)
                .where(Track.stream_session_id == stream_session_id)
                .order_by(Track.first_seen_at, Track.tracker_id)
            )
        )

    def list_open(self) -> list[Track]:
        """Return tracks that were never closed."""
        return list(
            self._session.scalars(
                select(Track).where(Track.terminated_at.is_(None)).order_by(Track.first_seen_at)
            )
        )


def encode_cursor(*, alert_at: datetime, event_id: uuid.UUID) -> str:
    """Encode a keyset cursor for event history (§17.2).

    The cursor is ``(alert_at, id)`` rather than an offset. An offset would skip or repeat
    a row whenever an event is written between two page requests, which is exactly what
    happens while the camera is live — and §17.2 requires the cursor to "recover gaps"
    after a reconnect, so a cursor that shifted under a concurrent write would recover the
    wrong interval.

    The timestamp is carried as an **epoch float computed from a UTC-labelled value**.
    SQLite stores these columns without a timezone, and Python's ``datetime.timestamp()``
    interprets a *naive* datetime as **local** time — so encoding the naive column value
    directly would shift the cursor by the machine's UTC offset and the next page would
    come back empty. A round trip through a formatted string has the same problem. Reading
    the value as UTC and converting explicitly is the only form that survives the trip.

    Encoded as URL-safe base64 of a compact JSON pair so the value is opaque to a client
    and stable across processes. It is *not* signed: it carries no privilege, because the
    query it resumes is already restricted by the caller's session.
    """
    as_utc = alert_at if alert_at.tzinfo is not None else alert_at.replace(tzinfo=UTC)
    payload = json.dumps({"t": as_utc.timestamp(), "i": str(event_id)}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    """Decode a cursor produced by :func:`encode_cursor`.

    Returns a **naive UTC** datetime, which is the form SQLite stores and compares in. An
    aware value would be converted by the driver against the local offset and match nothing.

    Raises:
        NotFoundError: The cursor is not one this project produced. Reported as a missing
            entity rather than a validation error because a cursor is opaque: a client
            cannot tell a malformed value from one belonging to another deployment, and both
            mean "there is nothing to resume from".
    """
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        decoded = json.loads(raw)
        timestamp = float(decoded["t"])
        event_id = uuid.UUID(str(decoded["i"]))
    except (ValueError, TypeError, KeyError, binascii.Error, UnicodeDecodeError) as exc:
        raise NotFoundError("the event cursor is not valid") from exc
    return _utc_naive(timestamp), event_id


def _context_loaders() -> tuple[Any, ...]:
    """Eager-load the track and session an event needs for its contract."""
    from sqlalchemy.orm import selectinload

    return (selectinload(FallEvent.track).selectinload(Track.session),)


class FallEventRepository:
    """Write and read stored fall events.

    An event is inserted when its alarm fires and closed later with
    :meth:`close`; :meth:`acknowledge` records that an operator saw the alarm.
    Reads are keyset-paginated on ``(alert_at, id)``, newest first.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, row: dict[str, Any]) -> uuid.UUID:
        """Insert one event (its alarm has just fired) and return its id."""
        event = FallEvent(**row)
        self._session.add(event)
        self._session.flush()
        return event.id

    def close(
        self,
        event_id: uuid.UUID,
        *,
        ended_at: datetime,
        end_reason: str,
        peak_vy: float | None = None,
        min_torso_angle_deg: float | None = None,
        max_aspect_ratio: float | None = None,
    ) -> bool:
        """Close an open event; returns ``False`` when it was missing or already closed."""
        values: dict[str, Any] = {"ended_at": ended_at, "end_reason": end_reason}
        if peak_vy is not None:
            values["peak_vy"] = peak_vy
        if min_torso_angle_deg is not None:
            values["min_torso_angle_deg"] = min_torso_angle_deg
        if max_aspect_ratio is not None:
            values["max_aspect_ratio"] = max_aspect_ratio
        result = self._session.execute(
            update(FallEvent)
            .where(FallEvent.id == event_id, FallEvent.ended_at.is_(None))
            .values(**values)
        )
        return int(cast("CursorResult[Any]", result).rowcount or 0) == 1

    def acknowledge(self, event_id: uuid.UUID, *, at: datetime) -> FallEventRecord | None:
        """Record an operator acknowledgement (idempotent); ``None`` when missing."""
        row = self._session.get(FallEvent, event_id)
        if row is None:
            return None
        if row.acknowledged_at is None:
            alert_at = row.alert_at
            when = at.replace(tzinfo=None) if at.tzinfo is not None else at
            # Clock skew between the observation clock and the wall clock must not
            # violate the ``acknowledged_after_alert`` constraint.
            row.acknowledged_at = max(when, alert_at)
            self._session.flush()
        return self.get(event_id)

    def get(self, event_id: uuid.UUID) -> FallEventRecord | None:
        """Return one event as its contract record, or ``None``."""
        row = self._session.scalars(
            select(FallEvent).options(*_context_loaders()).where(FallEvent.id == event_id)
        ).one_or_none()
        return None if row is None else to_fall_event_record(row)

    def count(self) -> int:
        """Total number of stored events."""
        return int(self._session.scalar(select(func.count()).select_from(FallEvent)) or 0)

    def list_open(self) -> list[FallEventRecord]:
        """Events whose alarm is still active, oldest first."""
        rows = self._session.scalars(
            select(FallEvent)
            .options(*_context_loaders())
            .where(FallEvent.ended_at.is_(None))
            .order_by(FallEvent.alert_at, FallEvent.id)
        )
        return [to_fall_event_record(row) for row in rows]

    def close_orphans(self, *, ended_at: datetime, reason: str = "stopped") -> int:
        """Close events left open by a previous process."""
        result = self._session.execute(
            update(FallEvent)
            .where(FallEvent.ended_at.is_(None))
            .values(ended_at=func.max(FallEvent.alert_at, ended_at), end_reason=reason)
        )
        return int(cast("CursorResult[Any]", result).rowcount or 0)

    def page(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        track_id: uuid.UUID | None = None,
        stream_session_id: uuid.UUID | None = None,
        since_utc: float | None = None,
        until_utc: float | None = None,
        open_only: bool = False,
    ) -> FallEventPage:
        """Read one page of history, newest alert first."""
        from fall_detection.events.contracts import FallEventPage

        filters = self._filters(
            track_id=track_id,
            stream_session_id=stream_session_id,
            since_utc=since_utc,
            until_utc=until_utc,
            open_only=open_only,
        )
        statement = select(FallEvent)
        if stream_session_id is not None:
            statement = statement.join(Track, Track.id == FallEvent.track_id)
        statement = statement.where(*filters)
        if cursor is not None:
            cursor_time, cursor_id = decode_cursor(cursor)
            statement = statement.where(
                (FallEvent.alert_at < cursor_time)
                | ((FallEvent.alert_at == cursor_time) & (FallEvent.id < cursor_id))
            )
        rows = list(
            self._session.scalars(
                statement.options(*_context_loaders())
                .order_by(FallEvent.alert_at.desc(), FallEvent.id.desc())
                .limit(limit + 1)
            )
        )
        has_more = len(rows) > limit
        visible = rows[:limit]
        next_cursor = None
        if has_more and visible:
            last = visible[-1]
            next_cursor = encode_cursor(alert_at=last.alert_at, event_id=last.id)
        count = select(func.count()).select_from(FallEvent)
        if stream_session_id is not None:
            count = count.join(Track, Track.id == FallEvent.track_id)
        total = int(self._session.scalar(count.where(*filters)) or 0)
        return FallEventPage(
            events=tuple(to_fall_event_record(row) for row in visible),
            next_cursor=next_cursor,
            limit=limit,
            total=total,
        )

    @staticmethod
    def _filters(
        *,
        track_id: uuid.UUID | None,
        stream_session_id: uuid.UUID | None,
        since_utc: float | None,
        until_utc: float | None,
        open_only: bool,
    ) -> list[Any]:
        filters: list[Any] = []
        if track_id is not None:
            filters.append(FallEvent.track_id == track_id)
        if stream_session_id is not None:
            filters.append(Track.stream_session_id == stream_session_id)
        if since_utc is not None:
            filters.append(FallEvent.alert_at >= _utc_naive(since_utc))
        if until_utc is not None:
            filters.append(FallEvent.alert_at <= _utc_naive(until_utc))
        if open_only:
            filters.append(FallEvent.ended_at.is_(None))
        return filters

    def delete_before(self, cutoff: datetime) -> int:
        """Delete closed events that ended before ``cutoff`` (FR-12 retention)."""
        if cutoff.tzinfo is not None:
            cutoff = cutoff.astimezone(UTC).replace(tzinfo=None)
        result = self._session.execute(
            delete(FallEvent).where(FallEvent.ended_at.is_not(None), FallEvent.ended_at < cutoff)
        )
        return int(cast("CursorResult[Any]", result).rowcount or 0)


def to_fall_event_record(row: FallEvent) -> FallEventRecord:
    """Convert an ORM row into the immutable event contract."""
    from fall_detection.events.contracts import FallEventRecord

    track = row.track
    session = None if track is None else track.session
    return FallEventRecord(
        event_id=row.id,
        track_uuid=row.track_id,
        camera_id=uuid.UUID(int=0) if session is None else session.camera_id,
        stream_session_id=uuid.UUID(int=0) if track is None else track.stream_session_id,
        display_id=None if track is None else track.tracker_id,
        falling_utc=_epoch(row.falling_at),
        impact_utc=_epoch(row.impact_at),
        alert_utc=_epoch(row.alert_at),
        ended_utc=None if row.ended_at is None else _epoch(row.ended_at),
        end_reason=row.end_reason,
        response_time_s=row.response_time_s,
        peak_vy=row.peak_vy,
        min_torso_angle_deg=row.min_torso_angle_deg,
        max_aspect_ratio=row.max_aspect_ratio,
        alert_source_timestamp=row.alert_source_timestamp,
        acknowledged_utc=None if row.acknowledged_at is None else _epoch(row.acknowledged_at),
        created_utc=_epoch(row.created_at),
        schema_version=row.schema_version,
    )


def _utc_naive(epoch: float) -> datetime:
    """Convert a UTC epoch timestamp to the naive-UTC form the columns store.

    Every ``DateTime`` column in this schema is timezone-naive and holds UTC (§19). An
    aware datetime must therefore never be handed to a comparison: the driver would apply
    the session's offset and the predicate would match nothing. The bug is silent — an
    empty page is indistinguishable from "the history is empty" — which is why the
    conversion lives in one place with this note rather than at each call site.
    """
    return datetime.fromtimestamp(epoch, tz=UTC).replace(tzinfo=None)


def _epoch(value: datetime) -> float:
    """Convert a stored datetime back to a UTC epoch timestamp.

    SQLite returns a naive datetime for a column declared without a timezone, and §19
    requires every stored timestamp to be UTC. ``datetime.timestamp()`` interprets a naive
    value as **local** time, so the value is labelled UTC first and only then converted:
    otherwise every interval in the history would be shifted by the machine's offset, and
    the shift would silently change with the machine's timezone.
    """
    as_utc = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return as_utc.timestamp()
