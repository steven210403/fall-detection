"""Pipeline supervisor tests: states, sessions, reconnection, and shutdown.

The supervisor is exercised against the real FFmpeg decoder using locally
generated synthetic clips, so the tests need no authorized footage and make no
claim about detection or activity accuracy. Failure paths inject a bad FFmpeg
option, which fails the decoder exactly as a camera dropout would and therefore
drives the reconnect/offline logic without a network dependency.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import select

from fall_detection.config import Settings
from fall_detection.core.contracts import CameraState, FramePacket
from fall_detection.core.errors import ConfigurationError, InvalidSourceError
from fall_detection.database import DatabaseWriter, StreamSession
from fall_detection.video.pipeline import PipelineSupervisor, build_supervisor, wait_for_state
from fall_detection.video.reconnect import ReconnectPolicy


def fast_policy(**overrides: object) -> ReconnectPolicy:
    """A policy with negligible delays so reconnect tests stay fast."""
    params: dict[str, object] = {
        "backoff_s": (0.01, 0.02, 0.04),
        "jitter_ratio": 0.0,
        "max_attempts": 3,
    }
    params.update(overrides)
    return ReconnectPolicy(**params)  # type: ignore[arg-type]


def block_consumer(release: threading.Event) -> Callable[[FramePacket], None]:
    """Return a frame consumer that blocks until ``release`` is set.

    Used to hold the pipeline in ``live`` while the test inspects it, and to
    exercise the live drop policy without depending on machine speed.
    """

    def consumer(_packet: FramePacket) -> None:
        release.wait(timeout=5.0)

    return consumer


def make_supervisor(
    *,
    settings: Settings,
    writer: DatabaseWriter,
    on_frame: Callable[[FramePacket], None] | None = None,
    policy: ReconnectPolicy | None = None,
    extra_input_args: tuple[str, ...] = (),
) -> PipelineSupervisor:
    return PipelineSupervisor(
        settings=settings,
        camera_id=uuid.uuid4(),
        writer=writer,
        on_frame=on_frame,
        policy=policy or fast_policy(),
        extra_input_args=extra_input_args,
    )


class TestLifecycle:
    def test_reaches_ended_after_clean_eof(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(settings=settings, writer=writer)
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.ENDED, timeout=30.0)
            assert supervisor.stats.sessions_opened == 1
            assert supervisor.stats.frames_read > 0
            assert supervisor.stats.decode_errors == 0
        finally:
            supervisor.stop()

    def test_offline_camera_does_not_fail_liveness(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
    ) -> None:
        # FR-09 / §17.2: an offline camera is a state, not a process failure.
        settings = make_settings(video={"mode": "mp4", "source_ref": None})
        supervisor = make_supervisor(settings=settings, writer=writer)
        assert supervisor.state is CameraState.IDLE
        with pytest.raises(ConfigurationError):
            supervisor.start()
        assert supervisor.state is CameraState.IDLE

    def test_start_is_idempotent(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        # FR-11: a repeated start must not create a second decoder or session.
        release = threading.Event()

        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(
            settings=settings, writer=writer, on_frame=block_consumer(release)
        )
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.LIVE, timeout=20.0)
            supervisor.start()  # must be a no-op
            supervisor.start()
            assert supervisor.stats.sessions_opened == 1
            assert wait_for_state(supervisor, CameraState.LIVE, timeout=5.0)
        finally:
            release.set()
            supervisor.stop()

    def test_stop_returns_to_idle_and_is_idempotent(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        release = threading.Event()

        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(
            settings=settings, writer=writer, on_frame=block_consumer(release)
        )
        supervisor.start()
        assert wait_for_state(supervisor, CameraState.LIVE, timeout=20.0)
        release.set()
        supervisor.stop()
        assert supervisor.state is CameraState.IDLE
        assert supervisor.is_running() is False
        supervisor.stop()  # second stop must be harmless
        assert supervisor.state is CameraState.IDLE

    def test_never_started_stop_is_harmless(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
    ) -> None:
        settings = make_settings(video={"mode": "mp4", "source_ref": None})
        supervisor = make_supervisor(settings=settings, writer=writer)
        supervisor.stop()
        assert supervisor.state is CameraState.IDLE

    def test_reaches_live_while_consuming(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        seen: list[CameraState] = []
        release = threading.Event()

        def consumer(_packet: FramePacket) -> None:
            seen.append(supervisor.state)
            release.wait(timeout=5.0)

        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(settings=settings, writer=writer, on_frame=consumer)
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.LIVE, timeout=20.0)
            release.set()
        finally:
            supervisor.stop()
        assert CameraState.LIVE in seen

    def test_can_restart_after_end(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(settings=settings, writer=writer)
        supervisor.start()
        assert wait_for_state(supervisor, CameraState.ENDED, timeout=30.0)
        supervisor.stop()

        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.ENDED, timeout=30.0)
            assert supervisor.stats.sessions_opened == 1  # counters reset per run
        finally:
            supervisor.stop()


class TestSessionPersistence:
    def test_clean_eof_closes_the_session_as_source_ended(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(settings=settings, writer=writer)
        supervisor.start()
        assert wait_for_state(supervisor, CameraState.ENDED, timeout=30.0)
        supervisor.stop()

        rows = writer.read(lambda session: list(session.scalars(select(StreamSession))))
        assert len(rows) == 1
        row = rows[0]
        assert row.attempt == 1
        assert row.termination_reason == "source_ended"
        assert row.ended_at is not None
        assert row.frames_read > 0
        assert row.frames_dropped == 0

    def test_stop_closes_the_session_as_stopped(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        release = threading.Event()

        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(
            settings=settings,
            writer=writer,
            on_frame=block_consumer(release),
        )
        supervisor.start()
        assert wait_for_state(supervisor, CameraState.LIVE, timeout=20.0)
        release.set()
        supervisor.stop()

        rows = writer.read(lambda session: list(session.scalars(select(StreamSession))))
        assert len(rows) == 1
        assert rows[0].termination_reason in {"stopped", "source_ended"}
        assert rows[0].ended_at is not None

    def test_no_session_is_left_open(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(settings=settings, writer=writer)
        supervisor.start()
        assert wait_for_state(supervisor, CameraState.ENDED, timeout=30.0)
        supervisor.stop()

        open_rows = writer.read(
            lambda session: list(
                session.scalars(select(StreamSession).where(StreamSession.ended_at.is_(None)))
            )
        )
        assert open_rows == []

    def test_camera_row_is_reused_without_credentials(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        from fall_detection.database import Camera

        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(settings=settings, writer=writer)
        supervisor.start()
        assert wait_for_state(supervisor, CameraState.ENDED, timeout=30.0)
        supervisor.stop()

        cameras = writer.read(lambda session: list(session.scalars(select(Camera))))
        assert len(cameras) == 1
        assert cameras[0].source_type == "mp4"
        assert "://" not in cameras[0].source_ref or "@" not in cameras[0].source_ref


class TestReconnection:
    def test_decoder_failure_reconnects_then_goes_offline(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(
            settings=settings,
            writer=writer,
            policy=fast_policy(max_attempts=3),
            extra_input_args=("-definitely-not-an-option",),
        )
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.OFFLINE, timeout=40.0)
            stats = supervisor.stats
            assert stats.sessions_opened == 3
            assert stats.reconnects == 2
            assert stats.decode_errors == 3
        finally:
            supervisor.stop()

    def test_each_attempt_opens_a_distinct_session(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(
            settings=settings,
            writer=writer,
            policy=fast_policy(max_attempts=2),
            extra_input_args=("-definitely-not-an-option",),
        )
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.OFFLINE, timeout=40.0)
        finally:
            supervisor.stop()

        rows = writer.read(
            lambda session: list(
                session.scalars(select(StreamSession).order_by(StreamSession.attempt))
            )
        )
        assert [row.attempt for row in rows] == [1, 2]
        assert len({row.id for row in rows}) == 2  # a reconnect never reuses an ID

    def test_reconnection_clears_state_between_sessions(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(
            settings=settings,
            writer=writer,
            policy=fast_policy(max_attempts=2),
            extra_input_args=("-definitely-not-an-option",),
        )
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.OFFLINE, timeout=40.0)
        finally:
            supervisor.stop()
        # The live session must be gone after a failure, not retained.
        assert supervisor.current_session() is None
        assert supervisor.frame_queue() is None

    def test_invalid_source_is_not_retried(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        tmp_path: Path,
    ) -> None:
        # A rejected path cannot be fixed by retrying, so ``start`` raises.
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(tmp_path / "outside.mp4")},
            media_roots={"roots": [str(tmp_path / "authorized")]},
        )
        (tmp_path / "authorized").mkdir()
        (tmp_path / "outside.mp4").write_bytes(b"x")
        supervisor = make_supervisor(settings=settings, writer=writer)
        with pytest.raises(InvalidSourceError):
            supervisor.start()
        assert supervisor.state is CameraState.IDLE


class TestConsumerIsolation:
    def test_consumer_errors_do_not_stop_decoding(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        calls = 0

        def broken_consumer(_packet: FramePacket) -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("consumer exploded")

        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(settings=settings, writer=writer, on_frame=broken_consumer)
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.ENDED, timeout=30.0)
        finally:
            supervisor.stop()
        assert calls > 0
        assert supervisor.consumer_errors == calls


class TestSnapshot:
    def test_snapshot_reports_state_and_stream(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(settings=settings, writer=writer)
        payload = supervisor.snapshot()
        assert payload["state"] == "idle"
        assert payload["camera_id"] == str(supervisor.camera_id)
        assert payload["stream"] is None
        assert payload["reconnect_policy"]["max_attempts"] == 3

    def test_snapshot_includes_stream_metadata_when_live(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        release = threading.Event()
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = make_supervisor(
            settings=settings,
            writer=writer,
            on_frame=block_consumer(release),
        )
        supervisor.start()
        try:
            assert wait_for_state(supervisor, CameraState.LIVE, timeout=20.0)
            payload = supervisor.snapshot()
            assert payload["state"] == "live"
            assert payload["stream_session_id"] is not None
            stream = payload["stream"]
            queue = payload["queue"]
            assert stream is not None and queue is not None
            assert stream["width"] == 160
            assert queue["capacity"] >= 2
        finally:
            release.set()
            supervisor.stop()


class TestBuildSupervisor:
    def test_build_helper_uses_settings(
        self,
        make_settings: Callable[..., Settings],
        writer: DatabaseWriter,
        synthetic_video: Path,
    ) -> None:
        settings = make_settings(
            video={"mode": "mp4", "source_ref": str(synthetic_video)},
            media_roots={"roots": [str(synthetic_video.parent)]},
        )
        supervisor = build_supervisor(settings=settings, camera_id=uuid.uuid4(), writer=writer)
        assert supervisor.state is CameraState.IDLE
        described = supervisor.snapshot()["reconnect_policy"]
        assert described["backoff_s"] == list(settings.video.reconnect_backoff_s)
