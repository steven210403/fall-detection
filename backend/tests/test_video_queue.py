"""Tests for the bounded frame queue's live and offline policies."""

from __future__ import annotations

import threading
import time

import pytest

from fall_detection.video.queue import FrameQueue, QueueClosedError


class _Sized:
    """Item exposing ``nbytes`` so byte accounting can be asserted."""

    def __init__(self, value: int, nbytes: int) -> None:
        self.value = value
        self._nbytes = nbytes

    def nbytes(self) -> int:
        return self._nbytes


class TestConstruction:
    @pytest.mark.parametrize("capacity", [0, -1])
    def test_rejects_non_positive_capacity(self, capacity: int) -> None:
        with pytest.raises(ValueError, match="capacity"):
            FrameQueue(capacity, drop_oldest=True)


class TestLivePolicy:
    def test_drops_oldest_and_counts_the_drop(self) -> None:
        queue: FrameQueue[int] = FrameQueue(2, drop_oldest=True)
        assert queue.put(1) is False
        assert queue.put(2) is False
        assert queue.put(3) is True
        assert queue.dropped == 1
        assert queue.get(0) == 2  # oldest (1) was discarded
        assert queue.get(0) == 3

    def test_never_blocks_the_producer(self) -> None:
        # Live mode must keep draining the decoder: a full queue can never make
        # ``put`` wait, whatever the consumer is doing (thesis plan §9.1).
        queue: FrameQueue[int] = FrameQueue(2, drop_oldest=True)
        finished = threading.Event()

        def producer() -> None:
            for value in range(500):
                queue.put(value)
            finished.set()

        thread = threading.Thread(target=producer)
        thread.start()
        assert finished.wait(timeout=5.0), "live put blocked the producer"
        thread.join(timeout=1.0)
        assert queue.qsize() == 2
        assert queue.dropped == 498
        assert queue.max_depth == 2


class TestOfflinePolicy:
    def test_blocks_producer_instead_of_dropping(self) -> None:
        # Offline mode applies backpressure so no sample is lost because the
        # machine is slow (thesis plan §9.1).
        queue: FrameQueue[int] = FrameQueue(2, drop_oldest=False)
        assert queue.put(1) is False
        assert queue.put(2) is False

        blocked = threading.Event()

        def producer() -> None:
            queue.put(3)
            blocked.set()

        thread = threading.Thread(target=producer)
        thread.start()
        time.sleep(0.1)
        assert not blocked.is_set(), "offline put must wait for space"
        assert queue.dropped == 0

        assert queue.get(0) == 1
        assert blocked.wait(timeout=5.0)
        thread.join(timeout=1.0)
        assert queue.drain() == [2, 3]
        assert queue.dropped == 0

    def test_close_releases_a_blocked_producer(self) -> None:
        queue: FrameQueue[int] = FrameQueue(1, drop_oldest=False)
        queue.put(1)
        errors: list[Exception] = []

        def producer() -> None:
            try:
                queue.put(2)
            except QueueClosedError as exc:  # expected on close
                errors.append(exc)

        thread = threading.Thread(target=producer)
        thread.start()
        time.sleep(0.1)
        queue.close()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert len(errors) == 1


class TestGetAndClose:
    def test_get_times_out_when_empty(self) -> None:
        queue: FrameQueue[int] = FrameQueue(2, drop_oldest=True)
        start = time.monotonic()
        assert queue.get(timeout=0.05) is None
        assert time.monotonic() - start >= 0.04

    def test_get_wakes_on_put(self) -> None:
        queue: FrameQueue[int] = FrameQueue(2, drop_oldest=True)
        result: list[int | None] = []

        def consumer() -> None:
            result.append(queue.get(timeout=5.0))

        thread = threading.Thread(target=consumer)
        thread.start()
        time.sleep(0.05)
        queue.put(42)
        thread.join(timeout=5.0)
        assert result == [42]

    def test_close_leaves_buffered_items_readable(self) -> None:
        queue: FrameQueue[int] = FrameQueue(3, drop_oldest=True)
        queue.put(1)
        queue.put(2)
        queue.close()
        assert queue.closed() is True
        assert queue.get(0) == 1
        assert queue.get(0) == 2
        assert queue.get(0) is None

    def test_put_after_close_raises(self) -> None:
        queue: FrameQueue[int] = FrameQueue(2, drop_oldest=True)
        queue.close()
        with pytest.raises(QueueClosedError):
            queue.put(1)


class TestAccounting:
    def test_byte_accounting_uses_item_nbytes(self) -> None:
        queue: FrameQueue[_Sized] = FrameQueue(4, drop_oldest=True)
        queue.put(_Sized(1, 100))
        queue.put(_Sized(2, 250))
        assert queue.bytes_buffered == 350
        queue.get(0)
        assert queue.bytes_buffered == 250

    def test_byte_accounting_recovers_after_drop(self) -> None:
        queue: FrameQueue[_Sized] = FrameQueue(1, drop_oldest=True)
        queue.put(_Sized(1, 100))
        queue.put(_Sized(2, 300))
        assert queue.bytes_buffered == 300
        assert queue.dropped == 1

    def test_stats_snapshot(self) -> None:
        queue: FrameQueue[int] = FrameQueue(2, drop_oldest=True, name="preview")
        queue.put(1)
        stats = queue.stats()
        assert stats["name"] == "preview"
        assert stats["capacity"] == 2
        assert stats["depth"] == 1
        assert stats["closed"] is False

    def test_drain_clears_and_returns_everything(self) -> None:
        queue: FrameQueue[int] = FrameQueue(4, drop_oldest=True)
        for value in range(4):
            queue.put(value)
        assert queue.drain() == [0, 1, 2, 3]
        assert queue.empty()
        assert queue.bytes_buffered == 0
