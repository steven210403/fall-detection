"""FastAPI application factory and lifespan.

The lifespan builds everything exactly once — the database writer, the pose detector,
the tracker, the fall service and its event writer, the preview stream, the live
broadcaster, and the pipeline supervisor with the frame consumer attached — because the
architecture is one Uvicorn process, one inference thread owning model and tracking
state, and one database writer path. A subsystem that fails to load is *recorded* rather
than raised, so ``/health/live`` and ``/health/ready`` can report the problem.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fall_detection import __version__
from fall_detection.api.health import router as health_router
from fall_detection.api.state import AppState
from fall_detection.config import Settings, get_settings
from fall_detection.core.errors import CctvError
from fall_detection.core.logging import configure_logging, get_logger

__all__ = ["create_app"]

logger = get_logger("api")


def _build_state(settings: Settings) -> AppState:
    """Create the runtime state container for one application instance.

    A missing schema is *recorded* rather than raised: the process stays alive so
    ``/health/live`` and ``/health/ready`` can report the problem, which matches
    the rule that liveness must not depend on a subsystem being ready
    (thesis plan §17.2). Tables are never created implicitly — migrations are an
    explicit operator step.
    """
    from sqlalchemy.exc import OperationalError

    from fall_detection.database import (
        DatabaseWriter,
        StreamSessionRepository,
        create_engine,
        create_session_factory,
    )

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    writer = DatabaseWriter(session_factory)

    # A session left open belongs to a previous process, so close it now rather
    # than showing a phantom live session in the history (thesis plan §19).
    orphaned = 0
    database_error: str | None = None
    try:
        orphaned = writer.write(lambda session: StreamSessionRepository(session).close_orphans())
    except OperationalError as exc:
        database_error = (
            "database schema is not present; run 'alembic upgrade head' "
            f"(the application never creates tables implicitly): {exc.orig}"
        )
        logger.error("database not ready: %s", database_error)

    return AppState(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        writer=writer,
        orphaned_sessions_closed=orphaned,
        database_error=database_error,
    )


def _build_detection_state(state: AppState) -> None:
    """Attach the detector to ``state`` and load it once.

    Detection is off by default (``FALL_DETECTION__ENABLED``), which keeps the
    Phase 2 ingestion tests and a database-only deployment free of model I/O. When
    it is on, a load failure is recorded on the state instead of raised, so the
    process still starts and readiness reports the model as unavailable
    (thesis plan §17.2: "Present model readiness and camera status separately").
    """
    if not state.settings.detection.enabled:
        logger.info("detection disabled; no model loaded")
        return

    from fall_detection.detection import DetectionService

    service = DetectionService(state.settings)
    state.detection = service
    try:
        info = service.load()
    except CctvError as exc:
        # A missing weight file must not stop the API: the operator needs the
        # readiness endpoint to tell them what is wrong.
        state.model_error = f"{type(exc).__name__}: {exc.message}"
        logger.error("detector not ready: %s", state.model_error)
        return
    logger.info(
        "detector ready model=%s device=%s imgsz=%d",
        info.model_name,
        info.device,
        info.imgsz,
    )


def _build_tracking_state(state: AppState) -> None:
    """Attach the tracker to ``state`` and validate its configuration.

    Thesis plan §3 fixes one inference thread owning model *and* tracking state, so
    the tracker is built here, once, next to the detector rather than per camera.
    It needs no weights and no warmup, which is why it is enabled by default; the
    switch exists for a detection-only ablation run.

    A construction failure is recorded rather than raised, matching the detector:
    a tracker problem degrades readiness and must not stop the process.
    """
    if not state.settings.tracking.enabled:
        logger.info("tracking disabled; no tracker constructed")
        return

    from fall_detection.tracking import TrackingService

    service = TrackingService(state.settings, writer=state.writer)
    state.tracking = service
    try:
        # Validating here gives the operator a startup-time error instead of a
        # mid-stream one; building the tracker instance itself is deferred to the
        # first stream session, because a tracker holds per-session state.
        service.validate()
    except CctvError as exc:
        state.tracker_error = f"{type(exc).__name__}: {exc.message}"
        logger.error("tracker not ready: %s", state.tracker_error)
        return
    info = service.info
    logger.info(
        "tracker ready name=%s buffer_frames=%d loss_tolerance=%.2fs stale_after=%.2fs",
        info.name,
        info.configured_buffer_frames,
        state.settings.tracking.loss_tolerance_s,
        state.settings.tracking.stale_after_s,
    )


def _build_fall_state(state: AppState) -> None:
    """Attach the broadcaster, the preview, the fall-event writer, and the fall service.

    Built in dependency order: the broadcaster exists before the writer (so a committed
    alarm can be published), and the writer before the fall service (so an alarm reaches
    it). Alarms a previous process left open are closed first, with reason ``stopped``.
    """
    settings = state.settings
    from fall_detection.streaming import PreviewStream, SnapshotBroadcaster

    if settings.streaming.enabled:
        state.broadcaster = SnapshotBroadcaster(settings.streaming)
        state.preview = PreviewStream(settings.streaming)
    else:
        logger.info("streaming disabled; no preview or live feed")

    if not settings.fall.enabled:
        logger.info("fall detection disabled; no fall service constructed")
        return

    from fall_detection.events import FallEventWriter
    from fall_detection.fall import FallService

    state.events = FallEventWriter(writer=state.writer, broadcaster=state.broadcaster)
    if state.database_error is None:
        try:
            state.orphaned_events_closed = state.events.close_orphans(reason="stopped")
        except Exception as exc:  # noqa: BLE001 - a cleanup must not stop the process
            logger.warning("could not close orphaned fall events: %s", exc)
    try:
        service = FallService(settings, events=state.events)
    except Exception as exc:  # noqa: BLE001 - an unreadable calibration must not stop the API
        state.fall_error = f"{type(exc).__name__}: {exc}"
        logger.error("fall detection not ready: %s", state.fall_error)
        return
    state.fall = service
    thresholds = service.thresholds
    logger.info(
        "fall detection ready AR>=%.2f lying<%.0fdeg upright>%.0fdeg Vy>%.2f/s "
        "impact_window=%.1fs long_lie=%.1fs calibrated=%s",
        thresholds.ar_threshold,
        thresholds.angle_lying_deg,
        thresholds.angle_upright_deg,
        thresholds.vy_threshold,
        thresholds.impact_window_s,
        thresholds.long_lie_s,
        settings.fall.calibration_path is not None,
    )


def _build_consumer(state: AppState) -> None:
    """Attach the frame consumer the pipeline will call, and build the supervisor.

    Detection and tracking are the minimum (tracking consumes detections, not pixels);
    the fall and preview stages are added when the fall service exists. A missing stage
    stops the composition at the last one that exists, so the pipeline never receives a
    consumer that silently produces nothing.
    """
    from fall_detection.fall.consumer import FallConsumer
    from fall_detection.tracking.consumer import DetectionTrackingConsumer
    from fall_detection.video.pipeline import build_supervisor

    detection = state.detection
    tracking = state.tracking
    fall = state.fall
    consumer: DetectionTrackingConsumer | FallConsumer | None = None
    if detection is not None and tracking is not None:
        consumer = DetectionTrackingConsumer(detection, tracking)
        if fall is not None:
            consumer = FallConsumer(
                detection,
                tracking,
                fall,
                preview=state.preview,
                broadcaster=state.broadcaster,
            )
    state.consumer = consumer
    state.pipeline = build_supervisor(
        settings=state.settings,
        camera_id=state.camera_id,
        writer=state.writer,
        on_frame=consumer,
    )
    logger.info(
        "pipeline supervisor built camera=%s consumer=%s",
        state.camera_id,
        "none" if consumer is None else type(consumer).__name__,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application.

    Args:
        settings: Optional settings override, used by tests. When omitted the
            cached process-wide settings are used.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("api")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = _build_state(settings)
        # Load the detector once, before serving requests, so readiness reflects
        # the model this process will actually use (thesis plan §9.2, §17.2).
        _build_detection_state(state)
        # Build the tracker next to the detector, because §3 gives one inference
        # thread both (FR-11).
        _build_tracking_state(state)
        # Then the fall service, its event writer, and the preview stream, so the
        # whole pipeline has one owner and one construction order.
        _build_fall_state(state)
        # Finally the consumer and the supervisor that drives it.
        _build_consumer(state)
        # FR-12: apply the retention policy once at startup. Off by default, so this is a
        # no-op unless an operator enabled it; a failure is recorded rather than raised.
        state.apply_retention()
        app.state.cctv = state
        logger.info(
            "application started environment=%s device=%s database=%s orphans_closed=%d",
            settings.environment,
            settings.device,
            "sqlite" if settings.database_url.startswith("sqlite") else "other",
            state.orphaned_sessions_closed,
        )
        try:
            yield
        finally:
            # Close the pipeline before the engine so no writer thread races the
            # dispose call. Shutting down also closes every open fall alarm and ends the
            # tracking session, so no row is left describing a live thing.
            state.shutdown()
            state.engine.dispose()
            logger.info("application stopped")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        docs_url="/docs" if settings.environment != "production" else None,
        lifespan=lifespan,
    )
    _install_error_handlers(app)
    app.include_router(health_router)
    _include_api_routers(app)
    return app


def _include_api_routers(app: FastAPI) -> None:
    """Register the §17.2 minimum API.

    Imported inside the function so a deployment that only needs the health probes — a
    database migration check, for instance — never imports the fall, event, and
    streaming layers.
    """
    from fall_detection.api.cameras import router as camera_router
    from fall_detection.api.events import router as event_router
    from fall_detection.api.live import router as live_router
    from fall_detection.api.status import router as status_router

    app.include_router(camera_router)
    app.include_router(event_router)
    app.include_router(status_router)
    # Registered last because it owns a WebSocket route rather than a path prefix.
    app.include_router(live_router)


def _install_error_handlers(app: FastAPI) -> None:
    """Map every typed error to a consistent JSON body (§17.2).

    §17.2 requires "consistent error codes for invalid sources, camera failures,
    unavailable models, and database errors", and §22 requires these paths to be tested.
    One handler for :class:`~fall_detection.core.errors.CctvError` covers all of them
    because the class carries its own stable ``code``, and a second handler wraps
    FastAPI's validation errors in the same envelope so a client parses one shape.
    """
    from fall_detection.api.deps import error_payload, status_for_error

    @app.exception_handler(CctvError)
    async def _cctv_error(_request: Request, exc: CctvError) -> JSONResponse:
        return JSONResponse(status_code=status_for_error(exc), content=error_payload(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "the request body or parameters are not valid",
                    "detail": str(exc.errors()[:3]),
                }
            },
        )


app = create_app()
