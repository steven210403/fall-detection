"""Client snapshot delivery and backpressure (thesis plan §17.2, §8).

§17.2 fixes the delivery contract in four lines, and each becomes a mechanism here:

* *"Send track snapshots at 2 Hz; send events/status immediately."* — snapshots pass
  through a throttle; events and status changes bypass it.
* *"Slow clients receive the latest snapshot, not a backlog."* — each client owns a
  **bounded** queue that discards its oldest message instead of growing. The queue is
  the generic :class:`~fall_detection.video.queue.FrameQueue` the project already uses
  for frames, constructed with ``drop_oldest=True``.
* *"Send events after database commit."* — enforced by
  :class:`~fall_detection.events.writer.EventWriter`, which publishes only once its
  transaction has returned; this module simply carries what it is handed.
* *"On reconnect, fetch history using a cursor to recover gaps."* — every message
  carries a monotonic ``sequence``, so a client can see that it missed something and
  ask the history endpoint for the interval it lost. The drop count is reported, so
  the gap is never silent.

The broadcaster is transport-agnostic: it hands out queues and knows nothing about
WebSockets, which keeps the backpressure rules testable without a socket, a browser,
or an event loop.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any

from fall_detection.config import StreamingSettings
from fall_detection.core.clock import Clock, SystemClock
from fall_detection.core.logging import get_logger
from fall_detection.video.queue import FrameQueue, QueueClosedError

__all__ = [
    "SNAPSHOT_SCHEMA_VERSION",
    "BroadcasterStats",
    "ClientHandle",
    "MessageKind",
    "SnapshotBroadcaster",
]

logger = get_logger("streaming.snapshots")

SNAPSHOT_SCHEMA_VERSION = 1
"""Version of the live envelope a client receives (§17.2).

§17.2 requires the live payload to "include schema version", so a client can tell
whether it understands what it is being sent before it renders anything.
"""


class MessageKind:
    """Message kinds published to clients.

    Plain string constants rather than an enum because they travel in JSON and are
    read by a TypeScript client whose type is a string union; keeping one vocabulary
    in both places is what stops a typo from silently dropping a message.
    """

    SNAPSHOT = "snapshot"
    EVENT = "event"
    STATUS = "status"
    HELLO = "hello"


@dataclass(slots=True)
class BroadcasterStats:
    """Counters describing client delivery.

    ``dropped`` is the number of messages a slow client never received. §17.2 accepts
    that loss ("the latest snapshot, not a backlog") but requires the client to be able
    to recover it, so the count and the per-client sequence gap are both reported.
    """

    published: int = 0
    events: int = 0
    status_changes: int = 0
    throttled: int = 0
    dropped: int = 0
    clients_connected: int = 0
    clients_disconnected: int = 0
    rejected: int = 0
    delivery_errors: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable view."""
        return {
            "published": self.published,
            "events": self.events,
            "status_changes": self.status_changes,
            "throttled": self.throttled,
            "dropped": self.dropped,
            "clients_connected": self.clients_connected,
            "clients_disconnected": self.clients_disconnected,
            "rejected": self.rejected,
            "delivery_errors": self.delivery_errors,
        }


@dataclass(slots=True)
class ClientHandle:
    """One connected client and the queue it reads from.

    Attributes:
        client_id: Random identifier, used in logs and status payloads.
        queue: Bounded message queue. Capacity 1 means "latest snapshot only"; the
            default of 2 lets one message be in flight while the next waits.
        delivered: Messages this client actually received.
        last_sequence: Sequence number of the last message it received, so a gap is
            detectable by the client and by :meth:`SnapshotBroadcaster.snapshot`.
        user: Operator account that opened the connection.
    """

    client_id: str
    queue: FrameQueue[dict[str, Any]]
    user: str = "operator"
    delivered: int = 0
    last_sequence: int = 0

    def close(self) -> None:
        """Close the client's queue, waking any reader."""
        self.queue.close()

    def take(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Read the next message, counting it on delivery."""
        message = self.queue.get(timeout)
        if message is not None:
            self.delivered += 1
            sequence = message.get("sequence")
            if isinstance(sequence, int):
                self.last_sequence = sequence
        return message


class SnapshotBroadcaster:
    """Fan out live snapshots, events, and status to connected clients.

    Args:
        settings: ``settings.streaming`` supplies the snapshot interval and the client
            queue capacity and count.
        clock: Injectable clock, used for the snapshot throttle and message times.
    """

    def __init__(self, settings: StreamingSettings, *, clock: Clock | None = None) -> None:
        self._settings = settings
        self._clock = clock or SystemClock()
        self._lock = threading.RLock()
        self._clients: dict[str, ClientHandle] = {}
        self._sequence = 0
        self._last_snapshot_monotonic: float | None = None
        self._stats = BroadcasterStats()
        self._latest_snapshot: dict[str, Any] | None = None
        self._latest_status: dict[str, Any] | None = None

    # -- properties ----------------------------------------------------

    @property
    def stats(self) -> BroadcasterStats:
        """Live counters."""
        return self._stats

    @property
    def schema_version(self) -> int:
        """Live envelope schema version (§17.2)."""
        return SNAPSHOT_SCHEMA_VERSION

    def client_count(self) -> int:
        """Number of connected clients."""
        with self._lock:
            return len(self._clients)

    def clients(self) -> tuple[ClientHandle, ...]:
        """Connected clients, ordered by identifier."""
        with self._lock:
            return tuple(self._clients[key] for key in sorted(self._clients))

    def latest_snapshot(self) -> dict[str, Any] | None:
        """Most recent snapshot, so a newly connected client starts with state."""
        with self._lock:
            return self._latest_snapshot

    def latest_status(self) -> dict[str, Any] | None:
        """Most recent status publication, or ``None``."""
        with self._lock:
            return self._latest_status

    # -- connections ---------------------------------------------------

    def connect(self, *, user: str = "operator") -> ClientHandle:
        """Register a client and return its handle.

        Raises:
            PermissionError: The configured client ceiling is reached. Refusing a
                client is better than letting a runaway browser grow the process
                without limit, and §17.2's "not a backlog" rule applies to the number
                of readers as much as to the depth of one queue.
        """
        with self._lock:
            if len(self._clients) >= self._settings.max_clients:
                self._stats.rejected += 1
                raise PermissionError(
                    f"the live feed already has {len(self._clients)} clients; "
                    f"streaming.max_clients is {self._settings.max_clients}"
                )
            client_id = uuid.uuid4().hex[:12]
            handle = ClientHandle(
                client_id=client_id,
                queue=FrameQueue[dict[str, Any]](
                    self._settings.client_queue_capacity,
                    drop_oldest=True,
                    name=f"client-{client_id}",
                ),
                user=user,
            )
            self._clients[client_id] = handle
            self._stats.clients_connected += 1
            latest = self._latest_snapshot
            status = self._latest_status
        # A new client starts with the newest snapshot and status, so the interface
        # is populated immediately instead of waiting for the next throttle window.
        if latest is not None:
            handle.queue.put(latest)
        if status is not None:
            handle.queue.put(status)
        logger.info("live client connected id=%s clients=%d", client_id, self.client_count())
        return handle

    def disconnect(self, handle: ClientHandle) -> None:
        """Remove a client and close its queue; idempotent."""
        with self._lock:
            removed = self._clients.pop(handle.client_id, None)
        if removed is not None:
            self._stats.clients_disconnected += 1
        handle.close()
        logger.info(
            "live client disconnected id=%s clients=%d", handle.client_id, self.client_count()
        )

    def close_all(self) -> None:
        """Disconnect every client, as shutdown requires."""
        with self._lock:
            handles = list(self._clients.values())
            self._clients.clear()
        for handle in handles:
            handle.close()
        if handles:
            self._stats.clients_disconnected += len(handles)

    # -- publishing ----------------------------------------------------

    def publish_snapshot(self, payload: dict[str, Any], *, force: bool = False) -> bool:
        """Publish a track snapshot, throttled to the configured interval.

        Args:
            payload: Envelope body, without the sequence or kind.
            force: Bypass the throttle. Used for the first snapshot of a session,
                where a client that just connected should not wait.

        Returns:
            ``True`` when the message was published.
        """
        now = self._clock.monotonic()
        with self._lock:
            last = self._last_snapshot_monotonic
            if not force and last is not None and (now - last) < self._settings.snapshot_interval_s:
                self._stats.throttled += 1
                return False
            self._last_snapshot_monotonic = now
        self._publish(MessageKind.SNAPSHOT, payload)
        return True

    def publish_event(self, payload: dict[str, Any]) -> None:
        """Publish one committed activity event immediately (§17.2)."""
        self._publish(MessageKind.EVENT, payload)

    def publish_status(self, payload: dict[str, Any]) -> None:
        """Publish a status change immediately (§17.2)."""
        self._publish(MessageKind.STATUS, payload)

    def snapshot(self) -> dict[str, Any]:
        """Status payload describing the live feed."""
        with self._lock:
            clients = [
                {
                    "client_id": handle.client_id,
                    "user": handle.user,
                    "delivered": handle.delivered,
                    "last_sequence": handle.last_sequence,
                    "pending": handle.queue.qsize(),
                    "dropped": handle.queue.dropped,
                }
                for handle in self._clients.values()
            ]
            sequence = self._sequence
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_interval_s": self._settings.snapshot_interval_s,
            "client_queue_capacity": self._settings.client_queue_capacity,
            "max_clients": self._settings.max_clients,
            "sequence": sequence,
            "clients": clients,
            "stats": self._stats.as_dict(),
        }

    # -- internals -----------------------------------------------------

    def _publish(self, kind: str, payload: dict[str, Any]) -> None:
        """Build the envelope and enqueue it for every client."""
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            envelope: dict[str, Any] = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "kind": kind,
                "sequence": sequence,
                "emitted_monotonic": self._clock.monotonic(),
                "emitted_utc": self._clock.utc_timestamp(),
                **payload,
            }
            handles = list(self._clients.values())
            if kind is MessageKind.SNAPSHOT:
                self._latest_snapshot = envelope
            elif kind is MessageKind.STATUS:
                self._latest_status = envelope
            self._stats.published += 1
            if kind is MessageKind.EVENT:
                self._stats.events += 1
            elif kind is MessageKind.STATUS:
                self._stats.status_changes += 1

        for handle in handles:
            try:
                # ``drop_oldest`` is what implements §17.2's "latest snapshot, not a
                # backlog": a full queue discards its oldest message and the count is
                # reported, so the client can recover the gap through the event cursor.
                if handle.queue.put(envelope):
                    self._stats.dropped += 1
            except QueueClosedError:
                self._stats.delivery_errors += 1
