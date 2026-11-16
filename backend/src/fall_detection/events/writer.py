"""Fall-event writer: the only path from an alarm to the database and the clients.

An alarm is **persisted and broadcast the moment it fires**, then updated when the event
closes. The publish happens after the transaction returns, never before it, so a client
can never show an alarm the history does not contain.

Two failure modes are handled differently on purpose:

* **Writing an event must not stop the pipeline.** This runs inside the frame loop, so a
  persistence failure is counted and logged; the alarm is still broadcast (marked
  ``persisted: false``), because a caregiver must see a fall even when the disk is full.
* **Reading history must not lie.** A history query that cannot reach the database
  raises :class:`~fall_detection.core.errors.DatabaseError`, because an empty history
  and a broken database are different answers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fall_detection.core.errors import DatabaseError
from fall_detection.core.logging import get_logger
from fall_detection.database import DatabaseWriter, FallEventRepository, utc_now
from fall_detection.events.contracts import FallEventPage, FallEventRecord, OpenFallEvent

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from fall_detection.streaming.snapshots import SnapshotBroadcaster

__all__ = ["FallEventStats", "FallEventWriter"]

logger = get_logger("events.writer")


@dataclass(slots=True)
class FallEventStats:
    """Counters describing event persistence in this process."""

    opened: int = 0
    closed: int = 0
    write_errors: int = 0
    published: int = 0
    publish_errors: int = 0
    history_reads: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable view."""
        return {
            "opened": self.opened,
            "closed": self.closed,
            "write_errors": self.write_errors,
            "published": self.published,
            "publish_errors": self.publish_errors,
            "history_reads": self.history_reads,
        }


class FallEventWriter:
    """Store fall events and publish them once they are committed.

    Args:
        writer: The process's single database writer. Optional, so an offline
            evaluation can run the fall layer without a database.
        broadcaster: Optional client broadcaster.
    """

    def __init__(
        self,
        *,
        writer: DatabaseWriter | None = None,
        broadcaster: SnapshotBroadcaster | None = None,
    ) -> None:
        self._writer = writer
        self._broadcaster = broadcaster
        self._stats = FallEventStats()

    @property
    def persistent(self) -> bool:
        """Whether a database writer is attached."""
        return self._writer is not None

    @property
    def stats(self) -> FallEventStats:
        """Live counters."""
        return self._stats

    def attach_broadcaster(self, broadcaster: SnapshotBroadcaster) -> None:
        """Attach the client broadcaster, so committed events are published."""
        self._broadcaster = broadcaster

    # -- writes ---------------------------------------------------------

    def open(self, event: OpenFallEvent) -> bool:
        """Store an alarm that has just fired, then publish it.

        Returns:
            ``True`` when the row was committed.
        """
        persisted = False
        writer = self._writer
        if writer is not None:
            row = {
                "id": event.event_id,
                "track_id": event.track_uuid,
                "falling_at": _naive(event.falling_utc),
                "impact_at": _naive(event.impact_utc),
                "alert_at": _naive(event.alert_utc),
                "response_time_s": event.response_time_s,
                "peak_vy": event.peak_vy,
                "min_torso_angle_deg": event.min_torso_angle_deg,
                "max_aspect_ratio": event.max_aspect_ratio,
                "alert_source_timestamp": event.alert_source_timestamp,
                "created_at": utc_now().replace(tzinfo=None),
                "schema_version": 1,
            }
            try:
                writer.write(lambda session: FallEventRepository(session).create(row))
                persisted = True
            except Exception as exc:  # noqa: BLE001 - the frame loop must keep running
                self._stats.write_errors += 1
                logger.warning("could not store fall event %s: %s", event.event_id, exc)
        self._stats.opened += 1
        self._publish(
            {
                "kind": "opened",
                "persisted": persisted,
                "event": _open_payload(event),
            }
        )
        logger.warning(
            "FALL ALARM track=%s display_id=%s event=%s",
            event.track_uuid,
            event.display_id,
            event.event_id,
        )
        return persisted

    def close(
        self,
        event_id: uuid.UUID,
        *,
        ended_utc: float,
        end_reason: str,
        peak_vy: float | None = None,
        min_torso_angle_deg: float | None = None,
        max_aspect_ratio: float | None = None,
    ) -> bool:
        """Close an open event, then publish the closure."""
        closed = False
        writer = self._writer
        if writer is not None:
            try:
                closed = writer.write(
                    lambda session: FallEventRepository(session).close(
                        event_id,
                        ended_at=_naive(ended_utc),
                        end_reason=end_reason,
                        peak_vy=peak_vy,
                        min_torso_angle_deg=min_torso_angle_deg,
                        max_aspect_ratio=max_aspect_ratio,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - the frame loop must keep running
                self._stats.write_errors += 1
                logger.warning("could not close fall event %s: %s", event_id, exc)
        self._stats.closed += 1
        self._publish(
            {
                "kind": "closed",
                "persisted": closed,
                "event": {
                    "event_id": str(event_id),
                    "ended_utc": ended_utc,
                    "end_reason": end_reason,
                    "active": False,
                },
            }
        )
        return closed

    def acknowledge(self, event_id: uuid.UUID) -> FallEventRecord | None:
        """Record an operator acknowledgement and publish it; ``None`` when missing."""
        writer = self._require_writer()
        try:
            record = writer.write(
                lambda session: FallEventRepository(session).acknowledge(event_id, at=utc_now())
            )
        except Exception as exc:  # noqa: BLE001 - any failure is a database failure here
            raise DatabaseError(f"could not acknowledge event {event_id}: {exc}") from exc
        if record is not None:
            self._publish({"kind": "acknowledged", "persisted": True, "event": record.as_dict()})
        return record

    def _publish(self, payload: dict[str, Any]) -> None:
        broadcaster = self._broadcaster
        if broadcaster is None:
            return
        try:
            broadcaster.publish_event(payload)
            self._stats.published += 1
        except Exception as exc:  # noqa: BLE001 - a slow client must not stop the camera
            self._stats.publish_errors += 1
            logger.warning("could not publish a fall event: %s", exc)

    # -- reads ----------------------------------------------------------

    def history(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        track_uuid: uuid.UUID | None = None,
        session_id: uuid.UUID | None = None,
        since_utc: float | None = None,
        until_utc: float | None = None,
        open_only: bool = False,
    ) -> FallEventPage:
        """Read a page of fall-event history, newest alert first.

        Raises:
            DatabaseError: The history could not be read.
            NotFoundError: The cursor is malformed.
        """
        writer = self._require_writer()
        from fall_detection.core.errors import NotFoundError

        try:
            page = writer.read(
                lambda session: FallEventRepository(session).page(
                    limit=limit,
                    cursor=cursor,
                    track_id=track_uuid,
                    stream_session_id=session_id,
                    since_utc=since_utc,
                    until_utc=until_utc,
                    open_only=open_only,
                )
            )
        except NotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure is a database failure here
            raise DatabaseError(f"could not read fall-event history: {exc}") from exc
        self._stats.history_reads += 1
        return page

    def get(self, event_id: uuid.UUID) -> FallEventRecord | None:
        """Read one stored event, or ``None``."""
        writer = self._require_writer()
        try:
            return writer.read(lambda session: FallEventRepository(session).get(event_id))
        except Exception as exc:  # noqa: BLE001 - any failure is a database failure here
            raise DatabaseError(f"could not read event {event_id}: {exc}") from exc

    def count(self) -> int:
        """Total number of stored events."""
        writer = self._writer
        if writer is None:
            return 0
        return writer.read(lambda session: FallEventRepository(session).count())

    def close_orphans(self, *, reason: str = "stopped") -> int:
        """Close events a previous process left open; returns the number closed."""
        writer = self._writer
        if writer is None:
            return 0
        now = utc_now().replace(tzinfo=None)
        return writer.write(
            lambda session: FallEventRepository(session).close_orphans(ended_at=now, reason=reason)
        )

    def snapshot(self) -> dict[str, Any]:
        """Status payload describing the event layer."""
        return {"persistent": self.persistent, "stats": self._stats.as_dict()}

    def _require_writer(self) -> DatabaseWriter:
        writer = self._writer
        if writer is None:
            raise DatabaseError("fall-event history is unavailable: no database writer is attached")
        return writer


def _naive(epoch: float) -> datetime:
    """UTC epoch → the naive-UTC datetime the columns store."""
    return datetime.fromtimestamp(epoch, tz=UTC).replace(tzinfo=None)


def _open_payload(event: OpenFallEvent) -> dict[str, Any]:
    return {
        "event_id": str(event.event_id),
        "track_uuid": str(event.track_uuid),
        "camera_id": str(event.camera_id),
        "stream_session_id": str(event.stream_session_id),
        "display_id": event.display_id,
        "falling_utc": event.falling_utc,
        "impact_utc": event.impact_utc,
        "alert_utc": event.alert_utc,
        "ended_utc": None,
        "end_reason": None,
        "active": True,
        "response_time_s": round(event.response_time_s, 3),
        "peak_vy": event.peak_vy,
        "min_torso_angle_deg": event.min_torso_angle_deg,
        "max_aspect_ratio": event.max_aspect_ratio,
        "alert_source_timestamp": event.alert_source_timestamp,
        "acknowledged_utc": None,
    }
