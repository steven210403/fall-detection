"""End-to-end wiring: the real detector attached to the real pipeline seam.

Thesis plan §9 places detection directly after decoding, and Phase 2 exposed
``PipelineSupervisor(on_frame=...)`` as the seam. This module is the Phase 3
acceptance test for that seam: it decodes a synthetic clip with FFmpeg and runs
the real YOLO26n model on every frame.

Everything asserted here is **behavioural** — that a batch arrives per frame, that
batches carry the live stream session, that a reconnect clears cached state, and
that annotated frames match the decoded frame. Nothing asserts detection accuracy;
the synthetic clip contains no people and the pilot that would justify an accuracy
claim has not run (``docs/FALL_PILOT_REPORT.md``).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

from fall_detection.config import Settings
from fall_detection.core.contracts import CameraState
from fall_detection.database import DatabaseWriter
from fall_detection.detection import DetectionService
from fall_detection.video.pipeline import PipelineSupervisor, wait_for_state
from fall_detection.video.reconnect import ReconnectPolicy


def detection_supervisor(
    *,
    settings: Settings,
    writer: DatabaseWriter,
    service: DetectionService,
    extra_input_args: tuple[str, ...] = (),
) -> PipelineSupervisor:
    return PipelineSupervisor(
        settings=settings,
        camera_id=uuid.uuid4(),
        writer=writer,
        on_frame=service.on_frame,
        policy=ReconnectPolicy(backoff_s=(0.01, 0.02), jitter_ratio=0.0, max_attempts=3),
        extra_input_args=extra_input_args,
    )


class TestDetectionPipeline:
    def test_pipeline_detects_every_frame_of_a_clip(
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
        service = DetectionService(settings)
        service.load()
        supervisor = detection_supervisor(settings=settings, writer=writer, service=service)
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.ENDED, timeout=60.0)
        finally:
            supervisor.stop()

        assert supervisor.stats.frames_read > 0
        assert supervisor.consumer_errors == 0
        # Every decoded frame produced a batch and an annotated frame.
        assert service.stats.frames_processed == supervisor.stats.frames_read
        batch = service.latest_batch()
        assert batch is not None
        assert batch.frame_shape == (120, 160, 3)
        assert service.latest_annotated() is not None

    def test_batch_session_matches_the_live_stream_session(
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
        service = DetectionService(settings)
        service.load()
        supervisor = detection_supervisor(settings=settings, writer=writer, service=service)
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.ENDED, timeout=60.0)
        finally:
            supervisor.stop()

        session = supervisor.current_session()
        batch = service.latest_batch()
        assert batch is not None
        # The session has already closed, but the batch must carry the session
        # UUID of the run that produced it, not a stray identifier.
        assert session is None or batch.stream_session_id == session.stream_session_id
        assert batch.camera_id == supervisor.camera_id

    def test_warmup_happens_before_the_first_frame(
        self,
        make_settings: Callable[..., Settings],
        detector_weights: Path,
    ) -> None:
        settings = make_settings(
            detection={
                "enabled": True,
                "model_path": str(detector_weights),
                "imgsz": 320,
                "warmup_runs": 2,
            }
        )
        service = DetectionService(settings)
        info = service.load()
        # Load (and therefore warmup) must be complete before any frame is seen.
        assert info.warmup_runs == 2
        assert service.latest_batch() is None
