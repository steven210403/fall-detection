"""Preview and client backpressure tests (thesis plan §8, §17.2, §18.1, §18.2).

§17.2 states the delivery rules and each is asserted here:

* *"Send track snapshots at 2 Hz; send events/status immediately."* — snapshots are
  throttled; events and status are not.
* *"Slow clients receive the latest snapshot, not a backlog."* — a full client queue
  discards its oldest message rather than growing, and the drop is counted.
* *"On reconnect, fetch history using a cursor to recover gaps."* — every message carries a
  monotonic sequence, so a client can see that it missed one.

§18.1 supplies the frame rule — "The backend draws the preview from the same frame used for
tracking outputs" — and §8 supplies the caps. The preview tests assert the aspect ratio is
preserved and the rate is bounded, because those are what keep a normalized zone coordinate
mappable onto the displayed image.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from fall_detection.core.clock import ManualClock
from fall_detection.streaming import (
    SNAPSHOT_SCHEMA_VERSION,
    MessageKind,
    PreviewStream,
    SnapshotBroadcaster,
    fit_within,
)

CAMERA = uuid.UUID("22222222-2222-2222-2222-222222222222")
SESSION = uuid.UUID("11111111-1111-1111-1111-111111111111")


def frame(width: int = 1920, height: int = 1080) -> np.ndarray:
    """A small synthetic BGR frame."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image[10:40, 10:40] = 255
    return image


class TestFitWithin:
    def test_a_wider_frame_is_scaled_by_width(self) -> None:
        assert fit_within(1920, 1080, max_width=960, max_height=540) == (960, 540)

    def test_a_taller_frame_is_scaled_by_height(self) -> None:
        assert fit_within(480, 960, max_width=960, max_height=540) == (270, 540)

    def test_the_aspect_ratio_is_preserved(self) -> None:
        # §18.1 asks the ROI editor to "account for display letterboxing", and the simplest
        # way to keep that honest is to never change the aspect ratio: one scalar maps a
        # normalized coordinate onto the displayed image.
        for width, height in ((1920, 1080), (1280, 720), (640, 480), (800, 800)):
            scaled_width, scaled_height = fit_within(width, height, max_width=960, max_height=540)
            assert scaled_width / scaled_height == pytest.approx(width / height, rel=0.01)

    def test_a_frame_inside_the_cap_is_not_upscaled(self) -> None:
        # Upscaling would spend bandwidth on invented pixels.
        assert fit_within(320, 240, max_width=960, max_height=540) == (320, 240)

    def test_a_degenerate_frame_still_yields_one_pixel(self) -> None:
        assert fit_within(0, 0, max_width=960, max_height=540) == (1, 1)


class TestPreviewStream:
    def test_the_preview_is_capped_and_the_source_size_is_reported(self, make_settings) -> None:
        # §8: "Preview: 5 FPS, initially capped at 960x540." The envelope carries both sizes
        # so a client can map a normalized coordinate without knowing the scale.
        stream = PreviewStream(make_settings().streaming, clock=ManualClock())
        envelope = stream.publish(
            frame(1920, 1080), **{**_ARGS, "frame_seq": 7, "source_timestamp": 0.5}
        )
        assert envelope is not None
        assert (envelope.width, envelope.height) == (960, 540)
        assert (envelope.source_width, envelope.source_height) == (1920, 1080)
        assert envelope.frame_seq == 7
        assert envelope.source_timestamp == 0.5
        assert envelope.jpeg[:2] == b"\xff\xd8"  # the JPEG start-of-image marker

    def test_the_rate_cap_skips_frames_and_counts_them(self, make_settings) -> None:
        # §9.2: "Reduce preview work first during overload." The cap is a deliberate
        # reduction of work, so the frames it costs are counted rather than silently lost.
        clock = ManualClock()
        stream = PreviewStream(make_settings().streaming, clock=clock)
        assert stream.publish(frame(), **_ARGS) is not None
        for _ in range(3):
            clock.advance(0.05)
            assert stream.publish(frame(), **_ARGS) is None
        assert stream.stats.skipped_rate == 3
        clock.advance(0.25)
        assert stream.publish(frame(), **_ARGS) is not None
        assert stream.stats.published == 2

    def test_due_predicts_whether_publish_would_encode(self, make_settings) -> None:
        # The consumer only prepares a captioned copy when the cap would encode it.
        clock = ManualClock()
        stream = PreviewStream(make_settings().streaming, clock=clock)
        assert stream.due()
        assert stream.publish(frame(), **_ARGS) is not None
        clock.advance(0.05)
        assert not stream.due()
        assert stream.publish(frame(), **_ARGS) is None
        clock.advance(0.2)
        assert stream.due()
        assert not PreviewStream(make_settings(streaming={"enabled": False}).streaming).due()

    def test_the_measured_jpeg_size_and_bandwidth_are_reported(self, make_settings) -> None:
        # §18.2 gives the bandwidth arithmetic as "an arithmetic example, not a benchmark",
        # so the payload reports the *measured* size and rate it derived the figure from.
        # Two frames are needed for a rate: with one frame there is no interval to divide
        # by, and ``measured_fps`` reports 0.0 rather than inventing a number.
        clock = ManualClock()
        stream = PreviewStream(make_settings().streaming, clock=clock)
        stream.publish(frame(), **_ARGS)
        clock.advance(0.5)
        stream.publish(frame(), **_ARGS)
        described = stream.snapshot()
        assert described["stats"]["published"] == 2
        assert described["stats"]["mean_jpeg_bytes"] > 0
        assert described["stats"]["measured_fps"] == pytest.approx(2.0, rel=0.2)
        # §18.2's arithmetic applied to the measured size and the measured rate.
        assert described["stats"]["megabits_per_s"] > 0
        assert described["stats"]["last_encode_ms"] >= 0.0

    def test_a_cleared_stream_has_no_frame_and_ignores_the_rate_gate(self, make_settings) -> None:
        # §9.2: a reconnect must not keep serving the previous session's frame.
        clock = ManualClock()
        stream = PreviewStream(make_settings().streaming, clock=clock)
        stream.publish(frame(), **_ARGS)
        stream.clear()
        assert stream.latest() is None
        # The rate gate is reset too, so the new session's first frame is immediate.
        assert stream.publish(frame(), **_ARGS) is not None

    def test_a_disabled_stream_publishes_nothing(self, make_settings) -> None:
        stream = PreviewStream(make_settings(streaming={"enabled": False}).streaming)
        assert stream.publish(frame(), **_ARGS) is None
        assert stream.latest() is None

    def test_the_sequence_advances_with_every_frame(self, make_settings) -> None:
        clock = ManualClock()
        stream = PreviewStream(make_settings().streaming, clock=clock)
        first = stream.publish(frame(), **_ARGS)
        clock.advance(1.0)
        second = stream.publish(frame(), **_ARGS)
        assert first is not None and second is not None
        assert second.sequence == first.sequence + 1

    def test_the_envelope_metadata_excludes_the_bytes(self, make_settings) -> None:
        stream = PreviewStream(make_settings().streaming, clock=ManualClock())
        envelope = stream.publish(frame(), **_ARGS)
        assert envelope is not None
        payload = envelope.as_dict()
        assert "jpeg" not in payload
        assert payload["jpeg_bytes"] == len(envelope.jpeg)


_ARGS: dict = {
    "camera_id": CAMERA,
    "stream_session_id": SESSION,
    "frame_seq": 1,
    "captured_monotonic": 1.0,
    "captured_utc": 1_700_000_001.0,
    "source_timestamp": None,
}
"""The keyword arguments :meth:`PreviewStream.publish` needs beyond the frame."""


class TestSnapshotBroadcaster:
    def test_snapshots_are_throttled_to_the_configured_interval(self, make_settings) -> None:
        # §17.2: "Send track snapshots at 2 Hz." At a 0.5 s interval, four rapid publishes
        # produce one message.
        clock = ManualClock()
        broadcaster = SnapshotBroadcaster(make_settings().streaming, clock=clock)
        client = broadcaster.connect()
        assert broadcaster.publish_snapshot({"frame_seq": 1}, force=True) is True
        for step in range(3):
            clock.advance(0.1)
            assert broadcaster.publish_snapshot({"frame_seq": step + 2}) is False
        assert broadcaster.stats.throttled == 3
        clock.advance(0.5)
        assert broadcaster.publish_snapshot({"frame_seq": 9}) is True
        assert client.queue.qsize() == 2

    def test_events_and_status_bypass_the_throttle(self, make_settings) -> None:
        # §17.2 sends events and status "immediately", so the snapshot throttle must not
        # hold them back: they are published the moment they happen, whatever the snapshot
        # window is doing.
        clock = ManualClock()
        broadcaster = SnapshotBroadcaster(make_settings().streaming, clock=clock)
        client = broadcaster.connect()
        broadcaster.publish_snapshot({"frame_seq": 1}, force=True)
        # A second snapshot inside the throttle window is refused, so the only messages the
        # client can be holding are the event and the status change.
        clock.advance(0.1)
        assert broadcaster.publish_snapshot({"frame_seq": 2}) is False
        broadcaster.publish_event({"label": "walking"})
        broadcaster.publish_status({"state": "live"})
        messages = [client.take(timeout=0.5) for _ in range(2)]
        assert all(message is not None for message in messages)
        kinds = [message["kind"] for message in messages if message is not None]
        assert kinds == [MessageKind.EVENT, MessageKind.STATUS]
        assert broadcaster.stats.events == 1
        assert broadcaster.stats.status_changes == 1

    def test_a_capacity_of_one_keeps_only_the_newest_message(self, make_settings) -> None:
        # The extreme form of §17.2's rule: with room for a single message, a slow client
        # receives the newest one and the older ones are counted as dropped, which is what
        # tells it to recover the gap through the event cursor.
        broadcaster = SnapshotBroadcaster(
            make_settings(streaming={"client_queue_capacity": 1}).streaming
        )
        client = broadcaster.connect()
        broadcaster.publish_snapshot({"frame_seq": 1}, force=True)
        broadcaster.publish_event({"label": "walking"})
        assert client.queue.qsize() == 1
        assert client.queue.dropped == 1
        message = client.take(timeout=0.5)
        assert message is not None
        assert message["kind"] == MessageKind.EVENT

    def test_every_message_carries_a_monotonic_sequence_and_the_schema_version(
        self, make_settings
    ) -> None:
        # §17.2 requires the live payload to include a schema version, and the sequence is
        # what lets a client notice a gap and recover it through the event cursor.
        broadcaster = SnapshotBroadcaster(make_settings().streaming)
        client = broadcaster.connect()
        broadcaster.publish_snapshot({"frame_seq": 1}, force=True)
        broadcaster.publish_event({"label": "walking"})
        first = client.take(timeout=0.1)
        second = client.take(timeout=0.1)
        assert first is not None and second is not None
        assert first["schema_version"] == SNAPSHOT_SCHEMA_VERSION == 1
        assert second["sequence"] == first["sequence"] + 1

    def test_a_slow_client_loses_the_oldest_message_rather_than_growing(
        self, make_settings
    ) -> None:
        # §17.2: "Slow clients receive the latest snapshot, not a backlog." With a capacity
        # of two, a third message evicts the first and the eviction is counted.
        broadcaster = SnapshotBroadcaster(
            make_settings(streaming={"client_queue_capacity": 2}).streaming
        )
        client = broadcaster.connect()
        for step in range(4):
            broadcaster.publish_event({"n": step})
        assert client.queue.dropped == 2
        assert client.queue.qsize() == 2
        assert broadcaster.stats.dropped == 2
        # What the client still holds is the newest pair, so the gap is recoverable.
        remaining = [client.take(timeout=0.1) for _ in range(2)]
        assert [message["n"] for message in remaining if message is not None] == [2, 3]

    def test_a_new_client_starts_with_the_latest_snapshot_and_status(self, make_settings) -> None:
        # Otherwise the interface would be empty until the next throttle window, which reads
        # as a camera problem rather than as a fresh connection.
        broadcaster = SnapshotBroadcaster(make_settings().streaming)
        broadcaster.publish_snapshot({"frame_seq": 42}, force=True)
        broadcaster.publish_status({"state": "live"})
        client = broadcaster.connect()
        payloads = [client.take(timeout=0.1) for _ in range(3)]
        kinds = {payload["kind"] for payload in payloads if payload is not None}
        assert MessageKind.SNAPSHOT in kinds
        assert MessageKind.STATUS in kinds

    def test_the_client_ceiling_refuses_rather_than_growing(self, make_settings) -> None:
        broadcaster = SnapshotBroadcaster(make_settings(streaming={"max_clients": 2}).streaming)
        broadcaster.connect()
        broadcaster.connect()
        with pytest.raises(PermissionError, match="max_clients"):
            broadcaster.connect()
        assert broadcaster.stats.rejected == 1
        assert broadcaster.client_count() == 2

    def test_disconnect_removes_the_client_and_is_idempotent(self, make_settings) -> None:
        broadcaster = SnapshotBroadcaster(make_settings().streaming)
        client = broadcaster.connect()
        broadcaster.disconnect(client)
        broadcaster.disconnect(client)
        assert broadcaster.client_count() == 0
        assert broadcaster.stats.clients_connected == 1
        assert broadcaster.stats.clients_disconnected == 1

    def test_close_all_disconnects_everyone(self, make_settings) -> None:
        broadcaster = SnapshotBroadcaster(make_settings().streaming)
        broadcaster.connect()
        broadcaster.connect()
        broadcaster.close_all()
        assert broadcaster.client_count() == 0

    def test_publishing_to_a_closed_client_is_counted_not_raised(self, make_settings) -> None:
        # A closed queue is a client that went away between the snapshot and its delivery;
        # the broadcaster must survive it.
        broadcaster = SnapshotBroadcaster(make_settings().streaming)
        client = broadcaster.connect()
        client.close()
        broadcaster.publish_event({"label": "walking"})
        assert broadcaster.stats.delivery_errors == 1

    def test_no_clients_means_publishing_is_cheap_and_counted(self, make_settings) -> None:
        broadcaster = SnapshotBroadcaster(make_settings().streaming)
        broadcaster.publish_event({"label": "walking"})
        assert broadcaster.stats.published == 1
        assert broadcaster.stats.events == 1

    def test_the_snapshot_reports_client_delivery_state(self, make_settings) -> None:
        # A client's own sequence gap is what tells it to recover through the cursor, so the
        # registry reports each client's last sequence and its drop count (§17.2).
        broadcaster = SnapshotBroadcaster(
            make_settings(streaming={"client_queue_capacity": 1}).streaming
        )
        client = broadcaster.connect()
        broadcaster.publish_event({"n": 0})
        client.take(timeout=0.1)
        broadcaster.publish_event({"n": 1})
        broadcaster.publish_event({"n": 2})
        described = broadcaster.snapshot()
        assert described["stats"]["dropped"] == 1
        assert described["clients"][0]["client_id"] == client.client_id
        assert described["clients"][0]["delivered"] == 1
        assert described["clients"][0]["last_sequence"] == 1

    def test_the_snapshot_carries_the_schema_version(self, make_settings) -> None:
        broadcaster = SnapshotBroadcaster(make_settings().streaming)
        assert broadcaster.snapshot()["schema_version"] == SNAPSHOT_SCHEMA_VERSION
