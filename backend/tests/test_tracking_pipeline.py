"""End-to-end wiring: the real detector and tracker attached to the real pipeline.

Thesis plan §9 places detection then tracking directly after decoding, and the
Phase 4 acceptance criterion is "multi-person tracking; session reset; no buffer
mixing" reached through that pipeline. This module is the Phase 4 acceptance test
for the wiring: it decodes a synthetic clip with FFmpeg, runs the real YOLO26n
model, and feeds the resulting batches into the real tracker through
:class:`~fall_detection.tracking.consumer.DetectionTrackingConsumer`.

Everything asserted here is **behavioural** — that a tracking update happens per
decoded frame, that the session is the live one, that a reconnect resets tracking
state, and that the tracker's output is consistent with the frame it came from.
Nothing asserts detection or tracking accuracy: the synthetic clip contains no
people, and the pilot that would justify an accuracy claim has not run
(``docs/FALL_PILOT_REPORT.md``).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from fall_detection.config import Settings
from fall_detection.core.clock import SystemClock
from fall_detection.core.contracts import CameraState, FramePacket
from fall_detection.database import DatabaseWriter
from fall_detection.detection import DetectionService
from fall_detection.tracking import DetectionTrackingConsumer, TrackingService
from fall_detection.video.pipeline import PipelineSupervisor, wait_for_state
from fall_detection.video.reconnect import ReconnectPolicy


def build_consumer(settings: Settings, writer: DatabaseWriter) -> DetectionTrackingConsumer:
    """Build the Phase 4 frame consumer the way ``api.app`` does."""
    detection = DetectionService(settings)
    detection.load()
    tracking = TrackingService(settings, writer=writer)
    tracking.validate()
    return DetectionTrackingConsumer(detection, tracking)


def tracking_supervisor(
    *,
    settings: Settings,
    writer: DatabaseWriter,
    consumer: DetectionTrackingConsumer,
    extra_input_args: tuple[str, ...] = (),
) -> PipelineSupervisor:
    return PipelineSupervisor(
        settings=settings,
        camera_id=consumer.tracking.session_id or uuid.uuid4(),
        writer=writer,
        on_frame=consumer,
        policy=ReconnectPolicy(backoff_s=(0.01, 0.02), jitter_ratio=0.0, max_attempts=3),
        extra_input_args=extra_input_args,
    )


class TestDetectionTrackingPipeline:
    def test_pipeline_detects_and_tracks_every_frame(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
        detector_weights: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
            detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320},
        )
        consumer = build_consumer(settings, writer)
        supervisor = tracking_supervisor(settings=settings, writer=writer, consumer=consumer)
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.ENDED, timeout=60.0)
        finally:
            supervisor.stop()

        assert supervisor.stats.frames_read > 0
        assert supervisor.consumer_errors == 0
        # One detection pass and one tracking update per decoded frame.
        assert consumer.detection.stats.frames_processed == supervisor.stats.frames_read
        assert consumer.tracking.stats.updates == supervisor.stats.frames_read
        assert consumer.updates == supervisor.stats.frames_read

    def test_tracking_uses_the_live_stream_session(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
        detector_weights: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
            detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320},
        )
        consumer = build_consumer(settings, writer)
        supervisor = tracking_supervisor(settings=settings, writer=writer, consumer=consumer)
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.ENDED, timeout=60.0)
            batch = consumer.detection.latest_batch()
            session = consumer.tracking.session_id
        finally:
            supervisor.stop()

        assert batch is not None
        assert session == batch.stream_session_id
        # The tracker must have been built for this session exactly once.
        assert consumer.tracking.stats.sessions_started == 1
        assert consumer.tracking.info.reset_count == 1

    def test_reconnect_starts_a_new_tracking_session(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
        detector_weights: Path,
    ) -> None:
        # §9.2: a reconnect creates a new stream session and clears old tracks and
        # buffers. Driving it through the real decoder proves the reset is wired to
        # session identity and not to a manual call.
        settings = make_settings(
            video={
                "mode": "mp4",
                "source_ref": str(synthetic_video),
                "read_timeout_s": 30.0,
            },
            media_roots={"roots": [str(synthetic_video.parent)]},
            detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320},
        )
        consumer = build_consumer(settings, writer)
        supervisor = tracking_supervisor(settings=settings, writer=writer, consumer=consumer)
        supervisor.start()
        try:
            # Wait until the decode loop is live, then kill FFmpeg to force a
            # reconnect.
            assert wait_for_state(supervisor, CameraState.LIVE, timeout=30.0)
            deadline = SystemClock().monotonic() + 10.0
            while SystemClock().monotonic() < deadline:
                decoder = supervisor.current_decoder()
                if decoder is not None and decoder.frames_read >= 1:
                    decoder.kill_process()
                    break
            assert wait_for_state(supervisor, CameraState.ENDED, timeout=60.0)
        finally:
            supervisor.stop()

        assert supervisor.stats.sessions_opened >= 2, supervisor.snapshot()["stats"]
        assert consumer.tracking.stats.sessions_started >= 2
        # No buffer mixing: the tracker was reset once per session, so no
        # association state survived the reconnect.
        assert consumer.tracking.info.reset_count == consumer.tracking.stats.sessions_started

    def test_stopping_the_pipeline_ends_the_tracking_session(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
        detector_weights: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
            detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320},
        )
        consumer = build_consumer(settings, writer)
        supervisor = tracking_supervisor(settings=settings, writer=writer, consumer=consumer)
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.LIVE, timeout=30.0)
        finally:
            supervisor.stop()

        # AppState.shutdown() calls end_session; this asserts the call is safe and
        # leaves no live track, which is what the lifespan relies on.
        ended = consumer.tracking.end_session(reason="stopped")
        assert ended == ()
        assert consumer.tracking.session_id is None
        assert consumer.tracking.snapshot()["ready"] is False

    def test_consumer_snapshot_combines_both_services(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
        detector_weights: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
            detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320},
        )
        consumer = build_consumer(settings, writer)
        supervisor = tracking_supervisor(settings=settings, writer=writer, consumer=consumer)
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.LIVE, timeout=30.0)
            snapshot = consumer.snapshot()
        finally:
            supervisor.stop()

        assert set(snapshot) == {"detection", "tracking", "consumer_updates"}
        assert snapshot["tracking"]["tracker"]["name"] == settings.tracking.tracker

    def test_annotated_frames_are_cached_alongside_tracking(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
        detector_weights: Path,
    ) -> None:
        # §18.1: the preview must come from the same frame the tracking outputs
        # came from, and the consumer must expose both from that one pass.
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
            detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320},
        )
        consumer = build_consumer(settings, writer)
        supervisor = tracking_supervisor(settings=settings, writer=writer, consumer=consumer)
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.ENDED, timeout=60.0)
        finally:
            supervisor.stop()

        annotated = consumer.latest_annotated()
        tracks = consumer.latest_tracks()
        batch = consumer.detection.latest_batch()
        assert annotated is not None
        assert tracks is not None
        assert batch is not None
        assert annotated.shape == batch.frame_shape
        assert tracks.frame_seq == batch.frame_seq


class TestConsumerWithoutDetection:
    def test_disabled_detection_skips_tracking_rather_than_faking_it(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
    ) -> None:
        # Tracking consumes detections, not pixels, so with detection off there is
        # nothing to track and no empty batch may be invented.
        settings = make_settings(detection={"enabled": False})
        detection_service = DetectionService(settings)
        tracking = TrackingService(settings, writer=writer)
        consumer = DetectionTrackingConsumer(detection_service, tracking)

        packet = FramePacket(
            camera_id=uuid.uuid4(),
            stream_session_id=uuid.uuid4(),
            frame_seq=0,
            frame=np.zeros((32, 32, 3), dtype=np.uint8),
            source_timestamp=0.0,
            receive_monotonic=0.0,
            receive_utc=1_700_000_000.0,
        )
        consumer(packet)
        assert consumer.updates == 0
        assert tracking.stats.updates == 0
        assert tracking.session_id is None

    def test_disabled_tracking_runs_detection_only(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        detector_weights: Path,
    ) -> None:
        settings = make_settings(
            detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320},
            tracking={"enabled": False},
        )
        detection_service = DetectionService(settings)
        detection_service.load()
        tracking = TrackingService(settings, writer=writer)
        consumer = DetectionTrackingConsumer(detection_service, tracking)

        packet = FramePacket(
            camera_id=uuid.uuid4(),
            stream_session_id=uuid.uuid4(),
            frame_seq=0,
            frame=np.zeros((320, 320, 3), dtype=np.uint8),
            source_timestamp=0.0,
            receive_monotonic=0.0,
            receive_utc=1_700_000_000.0,
        )
        consumer(packet)
        assert detection_service.stats.frames_processed == 1
        assert consumer.updates == 0
        assert consumer.latest_annotated() is not None


class TestConsumerErrors:
    def test_tracking_failure_propagates_to_the_supervisor(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        detector_weights: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The supervisor counts and logs a failing consumer and keeps decoding, so
        # the consumer must let a real tracker failure through rather than hiding
        # it. One bad frame must therefore not be fatal, and must be visible.
        settings = make_settings(
            detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320}
        )
        detection_service = DetectionService(settings)
        detection_service.load()
        tracking = TrackingService(settings, writer=writer)

        def boom(_batch: object) -> None:
            raise RuntimeError("tracker exploded")

        monkeypatch.setattr(tracking, "update", boom)
        consumer = DetectionTrackingConsumer(detection_service, tracking)

        packet = FramePacket(
            camera_id=uuid.uuid4(),
            stream_session_id=uuid.uuid4(),
            frame_seq=0,
            frame=np.zeros((320, 320, 3), dtype=np.uint8),
            source_timestamp=0.0,
            receive_monotonic=0.0,
            receive_utc=1_700_000_000.0,
        )
        with pytest.raises(RuntimeError, match="tracker exploded"):
            consumer(packet)
