"""Bounded frame queue.

Thesis plan §9.1 requires two different overload behaviours from one queue:

* **Live** — capacity two, drop the oldest frame, and *count the drop rate*. The
  reader must keep draining the decoder so FFmpeg's pipe buffer never introduces
  delayed video, so a full queue must never block the producer.
* **Offline** — deterministic sampling with *no dropping because the machine is
  slow*. A full queue blocks the producer, which applies backpressure through
  FFmpeg's stdout pipe instead of silently losing frames.

:class:`FrameQueue` implements both with a single flag. It also bounds client
snapshot queues in later phases, so it is deliberately generic and free of any
CV imports.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TypedDict, TypeVar

__all__ = ["FrameQueue", "QueueClosedError", "QueueStats"]

T = TypeVar("T")


class QueueStats(TypedDict):
    """Diagnostics snapshot for one queue.

    Typed so callers (the status endpoint, the replay CLI) can read individual
    fields without casting away ``int | bool | str``.
    """

    name: str
    capacity: int
    depth: int
    max_depth: int
    dropped: int
    drop_oldest: bool
    bytes_buffered: int
    closed: bool


class QueueClosedError(RuntimeError):
    """Raised when producing into a queue that has been closed."""


class FrameQueue[T]:
    """A bounded queue with an explicit drop policy.

    Args:
        capacity: Maximum number of buffered items. Must be at least one.
        drop_oldest: ``True`` for live mode (never block, discard the oldest
            item and count it); ``False`` for offline mode (block the producer
            until space is available).
        name: Label used in log messages and diagnostics.

    Attributes:
        dropped: Number of items discarded because the queue was full.
        max_depth: Highest occupancy observed, used to report buffer quality.
    """

    def __init__(self, capacity: int, *, drop_oldest: bool, name: str = "frames") -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._drop_oldest = drop_oldest
        self._name = name
        self._items: deque[T] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._bytes_buffered = 0
        self._dropped = 0
        self._max_depth = 0

    @property
    def name(self) -> str:
        """Human-readable queue label."""
        return self._name

    @property
    def capacity(self) -> int:
        """Configured maximum occupancy."""
        return self._capacity

    @property
    def drop_oldest(self) -> bool:
        """Whether the queue discards the oldest item instead of blocking."""
        return self._drop_oldest

    @property
    def dropped(self) -> int:
        """Total items discarded because of a full queue."""
        with self._condition:
            return self._dropped

    @property
    def max_depth(self) -> int:
        """Highest occupancy observed since construction."""
        with self._condition:
            return self._max_depth

    @property
    def bytes_buffered(self) -> int:
        """Sum of ``nbytes`` reported by buffered items, for memory accounting."""
        with self._condition:
            return self._bytes_buffered

    def qsize(self) -> int:
        """Current number of buffered items."""
        with self._condition:
            return len(self._items)

    def empty(self) -> bool:
        """Whether the queue currently holds no items."""
        with self._condition:
            return not self._items

    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        with self._condition:
            return self._closed

    def put(self, item: T, *, size: int = 0) -> bool:
        """Add ``item`` to the queue.

        Args:
            item: The value to enqueue.
            size: Byte size of the item for memory accounting (0 to ignore).

        Returns:
            ``True`` when an existing item had to be dropped to make room.

        Raises:
            QueueClosedError: If the queue was closed.
        """
        dropped_item = False
        with self._condition:
            if self._closed:
                raise QueueClosedError(f"queue {self._name!r} is closed")

            if len(self._items) >= self._capacity:
                if self._drop_oldest:
                    self._discard_oldest_locked()
                    dropped_item = True
                else:
                    # Offline mode: wait for the consumer. The wait is bounded
                    # only by close(); a stopped consumer closes the queue.
                    while len(self._items) >= self._capacity and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        raise QueueClosedError(f"queue {self._name!r} is closed")

            self._items.append(item)
            self._bytes_buffered += size if size > 0 else _item_size(item)
            self._max_depth = max(self._max_depth, len(self._items))
            self._condition.notify_all()
        return dropped_item

    def get(self, timeout: float | None = None) -> T | None:
        """Remove and return the oldest item.

        Args:
            timeout: Seconds to wait. ``None`` blocks until an item arrives or
                the queue closes; ``0`` polls.

        Returns:
            The oldest item, or ``None`` when the timeout elapsed or the queue
            was closed while empty.
        """
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        with self._condition:
            while not self._items and not self._closed:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if not self._items:
                return None
            item = self._items.popleft()
            self._bytes_buffered = max(self._bytes_buffered - _item_size(item), 0)
            self._condition.notify_all()
            return item

    def drain(self) -> list[T]:
        """Remove and return every buffered item."""
        with self._condition:
            items = list(self._items)
            self._items.clear()
            self._bytes_buffered = 0
            self._condition.notify_all()
            return items

    def close(self) -> None:
        """Mark the queue closed and wake every waiter.

        Buffered items remain readable so a consumer can finish the backlog; a
        producer attempting to enqueue afterwards gets
        :class:`QueueClosedError`.
        """
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _discard_oldest_locked(self) -> None:
        """Drop the oldest item; caller must hold the condition."""
        dropped = self._items.popleft()
        self._bytes_buffered = max(self._bytes_buffered - _item_size(dropped), 0)
        self._dropped += 1

    def stats(self) -> QueueStats:
        """Return a snapshot of queue diagnostics for status reporting."""
        with self._condition:
            return QueueStats(
                name=self._name,
                capacity=self._capacity,
                depth=len(self._items),
                max_depth=self._max_depth,
                dropped=self._dropped,
                drop_oldest=self._drop_oldest,
                bytes_buffered=self._bytes_buffered,
                closed=self._closed,
            )


def _item_size(item: object) -> int:
    """Best-effort byte size of a buffered item."""
    nbytes: object = getattr(item, "nbytes", None)
    if callable(nbytes):
        try:
            return int(nbytes())  # type: ignore[operator]
        except Exception:  # noqa: BLE001 - accounting must never break the pipeline
            return 0
    return 0
