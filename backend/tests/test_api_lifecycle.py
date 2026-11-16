"""API lifecycle tests for the Phase 1 foundation.

The lifespan owns the database writer and the orphan-session recovery, so these
tests assert the wiring rather than business endpoints: there are still only
health endpoints (FR-11 and the camera API arrive in later phases).

The lifespan deliberately does *not* create tables (migrations are an explicit
operator step), so the ``app_settings`` fixture prepares the schema the way
``alembic upgrade head`` would.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from fall_detection.api.app import create_app
from fall_detection.api.state import AppState
from fall_detection.config import Settings
from fall_detection.core.contracts import CameraState, StreamSessionInfo
from fall_detection.database import Camera, StreamSession
from fall_detection.fall.consumer import FallConsumer
from fall_detection.tracking.consumer import DetectionTrackingConsumer


def app_state(client: TestClient) -> AppState:
    """Return the lifespan-built state for a test client.

    ``TestClient.app`` is typed as an ASGI callable, so the attribute is reached
    through a helper rather than by casting at every call site.
    """
    app = cast(FastAPI, client.app)
    return cast(AppState, app.state.cctv)


def build_client(factory: Callable[..., Settings]) -> TestClient:
    return TestClient(create_app(factory()))


def test_lifespan_attaches_state(app_settings: Callable[..., Settings]) -> None:
    with build_client(app_settings) as client:
        state = app_state(client)
        assert state.writer is not None
        assert state.engine is not None
        # Phase 6 builds the supervisor in the lifespan, because the camera endpoints
        # are what start it (FR-11: start/stop never duplicates a decoder).
        assert state.pipeline is not None
        assert state.pipeline.state is CameraState.IDLE


def test_health_endpoints_still_work(app_settings: Callable[..., Settings]) -> None:
    with build_client(app_settings) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ok"


def test_missing_schema_degrades_readiness_without_failing_liveness(
    make_settings: Callable[..., Settings],
) -> None:
    # No schema prepared: the operator forgot ``alembic upgrade head``. Liveness
    # must still answer, and readiness must name the problem (thesis plan §17.2).
    settings = make_settings()
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").json()["status"] == "ok"
        body = client.get("/health/ready").json()
        database = next(c for c in body["checks"] if c["name"] == "database")
        assert database["status"] == "degraded"
        assert "alembic upgrade head" in database["detail"]
        assert body["status"] == "degraded"


def test_lifespan_recovers_orphaned_sessions(
    app_settings: Callable[..., Settings],
) -> None:
    # A session left open by a previous process must be closed at startup, or
    # the history would show a live session that does not exist (thesis plan §19).
    settings = app_settings()
    with TestClient(create_app(settings)) as client:
        state = app_state(client)
        camera_id = uuid.uuid4()
        session_id = uuid.uuid4()

        def write(session: object) -> None:
            session.add(  # type: ignore[attr-defined]
                Camera(
                    id=camera_id,
                    name="c",
                    source_type="mp4",
                    source_ref="a.mp4",
                    created_at=datetime.now(UTC),
                )
            )
            session.add(  # type: ignore[attr-defined]
                StreamSession(
                    id=session_id,
                    camera_id=camera_id,
                    attempt=1,
                    started_at=datetime.now(UTC),
                )
            )

        state.writer.write(write)

    # Reopen: the new process must find and close the orphan.
    with TestClient(create_app(settings)) as client:
        state = app_state(client)
        assert state.orphaned_sessions_closed == 1
        rows = state.writer.read(lambda session: list(session.scalars(select(StreamSession))))
        assert rows[0].termination_reason == "shutdown"
        assert rows[0].ended_at is not None


def test_lifespan_is_reentrant(app_settings: Callable[..., Settings]) -> None:
    # Two starts must not fail; the second run finds no orphans.
    settings = app_settings()
    with TestClient(create_app(settings)) as client:
        assert app_state(client).orphaned_sessions_closed == 0
    with TestClient(create_app(settings)) as client:
        assert app_state(client).orphaned_sessions_closed == 0


def test_openapi_exposes_the_api(app_settings: Callable[..., Settings]) -> None:
    # The two health probes remain, and every camera, fall-event and status path exists.
    with build_client(app_settings) as client:
        schema = client.get("/openapi.json").json()
    paths = set(schema["paths"])
    assert {"/health/live", "/health/ready"} <= paths
    assert {
        "/api/v1/cameras",
        "/api/v1/cameras/{camera_id}/start",
        "/api/v1/cameras/{camera_id}/stop",
        "/api/v1/cameras/{camera_id}/preview.mjpeg",
        "/api/v1/fall-events",
        "/api/v1/fall-events/{event_id}",
        "/api/v1/fall-events/{event_id}/acknowledge",
        "/api/v1/status",
    } <= paths
    assert not any("zone" in path for path in paths)


def test_lifespan_builds_the_tracker_once(app_settings: Callable[..., Settings]) -> None:
    # §3 gives one inference thread the model *and* tracking state, so the tracker
    # is built once in the lifespan next to the detector rather than per camera.
    with build_client(app_settings) as client:
        state = app_state(client)
        assert state.tracking is not None
        assert state.tracking.info.name == state.settings.tracking.tracker
        assert state.tracker_error is None
        # Detection is disabled by default, so there is no batch to consume and no
        # consumer is built; tracking must not be fed pixels. The supervisor still
        # exists, because the camera endpoints must be able to report why a start fails.
        assert state.consumer is None


def test_tracking_is_disabled_by_configuration(
    app_settings: Callable[..., Settings],
) -> None:
    with build_client(lambda: app_settings(tracking={"enabled": False})) as client:
        assert app_state(client).tracking is None


def test_shutdown_ends_the_tracking_session(app_settings: Callable[..., Settings]) -> None:
    # A track left live at shutdown would keep an open row and a stale buffer, so
    # the lifespan must close the session before releasing the tracker.
    settings = app_settings()
    with build_client(lambda: settings) as client:
        tracking = app_state(client).tracking
        assert tracking is not None
        tracking.begin_session(
            StreamSessionInfo(
                camera_id=uuid.uuid4(),
                stream_session_id=uuid.uuid4(),
                started_utc=1_700_000_000.0,
                started_monotonic=0.0,
            )
        )
    # After the client closed, the tracker was unloaded and the session ended.
    assert tracking.session_id is None


def test_consumer_is_built_when_detection_and_tracking_are_enabled(
    app_settings: Callable[..., Settings],
    detector_weights: Path,
) -> None:
    # Phase 6 attaches this consumer to the pipeline, so it must exist whenever
    # both services do.
    settings = app_settings(
        detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320}
    )
    with build_client(lambda: settings) as client:
        state = app_state(client)
        assert state.detection is not None
        assert state.tracking is not None
        assert state.consumer is not None
        assert state.consumer.tracking is state.tracking
        assert state.consumer.detection is state.detection


def test_lifespan_builds_the_fall_service_once(
    app_settings: Callable[..., Settings],
) -> None:
    # The fall layer loads no weights, so it is built even without a detector, and its
    # event writer is attached so an alarm can be stored and published.
    with build_client(app_settings) as client:
        state = app_state(client)
        assert state.fall is not None
        assert state.fall.enabled is True
        assert state.fall_error is None
        assert state.events is not None
        assert state.events.persistent is True


def test_fall_detection_is_disabled_by_configuration(
    app_settings: Callable[..., Settings],
) -> None:
    with build_client(lambda: app_settings(fall={"enabled": False})) as client:
        state = app_state(client)
        assert state.fall is None
        assert state.events is None


def test_an_unreadable_calibration_is_recorded_not_raised(
    app_settings: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    # A missing calibration file must not stop the API; readiness reports it instead.
    settings = app_settings(fall={"calibration_path": str(tmp_path / "missing.json")})
    with build_client(lambda: settings) as client:
        state = app_state(client)
        assert state.fall is None
        assert state.fall_error is not None
        ready = client.get("/health/ready").json()
        fall = next(check for check in ready["checks"] if check["name"] == "fall")
        assert fall["status"] == "degraded"


def test_the_consumer_runs_every_stage_when_everything_is_enabled(
    app_settings: Callable[..., Settings],
    detector_weights: Path,
) -> None:
    settings = app_settings(
        detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320}
    )
    with build_client(lambda: settings) as client:
        state = app_state(client)
        consumer = state.consumer
        assert isinstance(consumer, FallConsumer)
        assert consumer.fall is state.fall
        assert consumer.tracking is state.tracking
        assert consumer.detection is state.detection


def test_the_consumer_falls_back_to_detection_and_tracking_when_fall_is_disabled(
    app_settings: Callable[..., Settings],
    detector_weights: Path,
) -> None:
    settings = app_settings(
        detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320},
        fall={"enabled": False},
    )
    with build_client(lambda: settings) as client:
        state = app_state(client)
        assert isinstance(state.consumer, DetectionTrackingConsumer)


def test_no_consumer_without_detection(
    app_settings: Callable[..., Settings],
) -> None:
    # Fall detection consumes tracking output, which consumes detections, so without a
    # detector there is nothing to measure.
    with build_client(app_settings) as client:
        state = app_state(client)
        assert state.detection is None
        assert state.fall is not None
        assert state.consumer is None


def test_start_is_refused_when_the_detector_failed_to_load(
    app_settings: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    # Missing weights are recorded, not raised, at startup; starting the camera must then
    # be refused with that reason instead of decoding frames that every one fails.
    settings = app_settings(
        detection={"enabled": True, "model_path": str(tmp_path / "missing-pose.pt")}
    )
    with build_client(lambda: settings) as client:
        state = app_state(client)
        assert state.model_error is not None
        response = client.post(f"/api/v1/cameras/{state.camera_id}/start")
        assert response.status_code == 409
        assert "missing-pose.pt" in response.text
        assert state.camera_state is CameraState.IDLE
