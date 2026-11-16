"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from fall_detection.config import Settings
from fall_detection.database import (
    Base,
    CameraRepository,
    DatabaseWriter,
    StreamSessionRepository,
    TrackRepository,
    create_engine,
    create_session_factory,
    utc_now,
)
from fall_detection.detection.contracts import Detection, DetectionBatch

# Tests never read the developer's ``backend/.env``: a local setting (an enabled detector,
# an MPS device, a media root) must not change what a default ``Settings()`` means.
Settings.model_config["env_file"] = None


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    """File-backed SQLite URL.

    A file (not ``:memory:``) is required because WAL and foreign-key behaviour
    are per-connection and Alembic opens its own connection.
    """
    return f"sqlite+pysqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def engine(sqlite_url: str) -> Iterator[Engine]:
    engine = create_engine(sqlite_url)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    factory: sessionmaker[Session] = create_session_factory(engine)
    with factory() as session:
        yield session


@pytest.fixture
def writer(engine: Engine) -> DatabaseWriter:
    """The single-writer wrapper over the test engine."""
    return DatabaseWriter(create_session_factory(engine))


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    """An authorized media directory for source-validation tests."""
    root = tmp_path / "media"
    root.mkdir()
    return root


@pytest.fixture
def synthetic_video(media_root: Path) -> Path:
    """A short synthetic MP4 inside an authorized root.

    Generated with FFmpeg's ``testsrc2`` source. It is a decoder fixture only:
    it proves nothing about detection or activity accuracy (thesis plan §13.1).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")
    path = media_root / "synthetic.mp4"
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg could not build the fixture: {result.stderr.strip()[:200]}")
    return path


@pytest.fixture
def slow_video(media_root: Path) -> Path:
    """A tiny 1 fps clip, used to force a decoder stall deterministically.

    16x16 makes decoding instantaneous, but frames arrive one second apart, so a
    sub-second read timeout must report a stall rather than block forever.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")
    path = media_root / "slow.mp4"
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=16x16:rate=1:duration=5",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg could not build the fixture: {result.stderr.strip()[:200]}")
    return path


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    """Build isolated settings without touching the process environment."""

    def factory(**overrides: object) -> Settings:
        base: dict[str, object] = {
            "environment": "test",
            "database_url": f"sqlite+pysqlite:///{tmp_path / 'settings.db'}",
            "log_level": "WARNING",
            "_env_file": None,
        }
        base.update(overrides)
        return Settings(**base)  # type: ignore[arg-type]

    return factory


@pytest.fixture
def app_settings(make_settings: Callable[..., Settings]) -> Callable[..., Settings]:
    """Settings whose database already has the migrated schema.

    The lifespan never creates tables, so tests that exercise the application
    must prepare the schema the way ``alembic upgrade head`` would.
    """

    def factory(**overrides: object) -> Settings:
        base: dict[str, object] = {
            # A pinned camera UUID, so a test can address the camera by name. Without it the
            # UUID is derived from the source reference (see ``Settings.camera_uuid``), which
            # is the right production behaviour and an awkward one to assert against.
            "camera_id": str(TEST_CAMERA_ID),
        }
        base.update(overrides)
        settings = make_settings(**base)
        engine = create_engine(settings.database_url)
        Base.metadata.create_all(engine)
        engine.dispose()
        return settings

    return factory


@pytest.fixture
def detector_weights() -> Path:
    """Path to the YOLOv8n-pose weights, or skip when they are absent.

    Weights are gitignored (``*.pt``) and downloaded by ``scripts/phase0_ml_smoke.py``,
    so a fresh checkout legitimately has none until that script has run. Tests that need
    a model skip rather than fail, matching the existing pattern for an absent FFmpeg.
    """
    path = Path(__file__).resolve().parents[2] / "models" / "yolov8n-pose.pt"
    if not path.is_file():
        pytest.skip(f"detector weights not present at {path}")
    return path


TRACKING_CAMERA_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
"""Fixed camera for tracking fixtures, so a batch carries a predictable identity."""

TEST_CAMERA_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
"""Camera the ``app_settings`` fixture pins, so an API test can address it by name."""

TRACKING_SESSION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
"""Fixed first stream session for tracking fixtures."""


@pytest.fixture
def person_detection() -> Callable[..., Detection]:
    """Build a person detection with a box at a given position.

    Tracking tests drive the real tracker with synthetic boxes, so they need
    detections whose only meaningful property is geometry. Nothing here is a
    detection-quality fixture: the boxes are idealised, and no test may derive an
    accuracy figure from them (thesis plan §15, §13.1).
    """

    def factory(
        x: float,
        y: float = 100.0,
        *,
        width: float = 60.0,
        height: float = 200.0,
        confidence: float = 0.9,
        class_name: str = "person",
        class_id: int = 0,
        keypoints: tuple[tuple[float, float, float], ...] | None = None,
    ) -> Detection:
        return Detection(
            bbox=(x, y, x + width, y + height),
            confidence=confidence,
            class_name=class_name,
            class_id=class_id,
            keypoints=keypoints,
        )

    return factory


@pytest.fixture
def detection_batch() -> Callable[..., DetectionBatch]:
    """Build a :class:`DetectionBatch` for one frame.

    The batch repeats the frame identity and timestamps so tracking never needs a
    decoded frame, which is exactly how the Phase 3 contract is designed. Frame
    size defaults to 640x480, large enough for the synthetic boxes.
    """

    def factory(
        frame_seq: int,
        detections: Sequence[Detection] = (),
        *,
        stream_session_id: uuid.UUID = TRACKING_SESSION_ID,
        camera_id: uuid.UUID = TRACKING_CAMERA_ID,
        frame_shape: tuple[int, int, int] = (480, 640, 3),
        source_timestamp: float | None = None,
        receive_monotonic: float = 0.0,
        receive_utc: float = 1_700_000_000.0,
    ) -> DetectionBatch:
        return DetectionBatch(
            camera_id=camera_id,
            stream_session_id=stream_session_id,
            frame_seq=frame_seq,
            detections=tuple(detections),
            frame_shape=frame_shape,
            source_timestamp=source_timestamp,
            receive_monotonic=receive_monotonic,
            receive_utc=receive_utc,
            inference_monotonic=receive_monotonic,
            model_name="yolo26n.pt",
        )

    return factory


@pytest.fixture
def stream_session_row(writer: DatabaseWriter) -> Callable[..., uuid.UUID]:
    """Create the ``stream_session`` row a track row must reference.

    Track persistence has a foreign key to ``stream_session``, and the video
    pipeline is the only writer of those rows. Tracking tests that exercise
    persistence therefore create the parent row the way the pipeline does,
    instead of weakening the constraint.
    """

    def factory(
        *,
        stream_session_id: uuid.UUID = TRACKING_SESSION_ID,
        camera_id: uuid.UUID = TRACKING_CAMERA_ID,
        attempt: int = 1,
    ) -> uuid.UUID:
        def work(session: Session) -> None:
            CameraRepository(session).get_or_create(
                camera_id=camera_id,
                name="tracking-test-camera",
                source_type="mp4",
                source_ref="synthetic.mp4",
            )
            StreamSessionRepository(session).open(
                camera_id=camera_id,
                attempt=attempt,
                stream_session_id=stream_session_id,
                started_at=utc_now(),
            )

        writer.write(work)
        return stream_session_id

    return factory


@pytest.fixture
def track_row(writer: DatabaseWriter) -> Callable[..., uuid.UUID]:
    """Persist a ``track`` row so a fall event can reference it.

    ``fall_event`` has a strict foreign key to ``track``, so event tests create the
    parent row the way tracking does rather than weakening the constraint.
    """

    def factory(
        *,
        track_id: uuid.UUID = TRACKING_SESSION_ID,
        stream_session_id: uuid.UUID = TRACKING_SESSION_ID,
        tracker_id: int = 1,
    ) -> uuid.UUID:
        def work(session: Session) -> None:
            TrackRepository(session).open(
                track_id=track_id,
                stream_session_id=stream_session_id,
                tracker_id=tracker_id,
            )

        writer.write(work)
        return track_id

    return factory


@pytest.fixture
def api_client() -> Iterator[Callable[..., TestClient]]:
    """Build a ``TestClient`` over the real application.

    The API has no login, so the client needs no session or header. The returned callable
    takes the same keyword overrides as ``app_settings``.
    """
    clients: list[TestClient] = []

    def factory(settings_factory: Callable[..., Settings], **overrides: object) -> TestClient:
        from fall_detection.api.app import create_app

        client = TestClient(create_app(settings_factory(**overrides)))
        client.__enter__()
        clients.append(client)
        return client

    try:
        yield factory
    finally:
        for client in clients:
            client.__exit__(None, None, None)
