"""Application runtime state.

Holds the objects that live for the duration of one ASGI application: settings, the
database engine and single writer, the detector, the tracker, the fall service, the fall
event writer, the preview stream, the live broadcaster, and the camera pipeline
supervisor. The lifespan owns construction and teardown in one place, and route handlers
reach them through ``request.app.state.cctv`` without a module-level global.

The supervisor is created once with the consumer attached, so a duplicated start request
cannot create a second decoder, and :meth:`start_camera`/:meth:`stop_camera` are the only
paths that drive it.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from fall_detection.config import Settings
from fall_detection.core.contracts import CameraState, StreamSessionInfo
from fall_detection.core.errors import PipelineStateError
from fall_detection.core.logging import get_logger
from fall_detection.database.session import DatabaseWriter
from fall_detection.streaming.snapshots import SnapshotBroadcaster

if TYPE_CHECKING:  # pragma: no cover - avoids a circular import at runtime
    from fall_detection.detection.service import DetectionService
    from fall_detection.events.writer import FallEventWriter
    from fall_detection.fall.consumer import FallConsumer
    from fall_detection.fall.service import FallService
    from fall_detection.streaming.frames import PreviewStream
    from fall_detection.tracking.consumer import DetectionTrackingConsumer
    from fall_detection.tracking.service import TrackingService
    from fall_detection.video.pipeline import PipelineSupervisor

logger = get_logger("api.state")

FrameConsumer = "DetectionTrackingConsumer | FallConsumer"
"""The consumer attached to ``PipelineSupervisor(on_frame=...)``."""

__all__ = ["FrameConsumer", "AppState"]


@dataclass
class AppState:
    """Per-application runtime state."""

    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    writer: DatabaseWriter
    orphaned_sessions_closed: int = 0
    database_error: str | None = None
    """Set when the schema is missing or unusable, so the readiness probe can
    report a degraded database without the process failing liveness."""

    retention: dict[str, Any] | None = None
    """Outcome of the last FR-12 retention pass, or ``None`` when none has run.

    Reported in the status payload so an operator can see whether the policy is on and
    what the last pass deleted; a policy that silently removed history would be
    indistinguishable from one that did nothing.
    """

    pipeline: PipelineSupervisor | None = None
    """The one supervisor for this process's camera (FR-11).

    Built by the lifespan with the consumer attached, so a start request cannot create a
    second decoder and a stop cannot leave the first one running.
    """

    detection: DetectionService | None = None
    """The one detector for this process (thesis plan §9.2: models are loaded
    once). ``None`` when detection is disabled in configuration."""

    tracking: TrackingService | None = None
    """The one tracker for this process (thesis plan §3: one inference thread owns
    model and tracking state). ``None`` when tracking is disabled."""

    fall: FallService | None = None
    """The one fall service for this process: per-person geometry, velocity and state
    machine. ``None`` when fall detection is disabled."""
    events: FallEventWriter | None = None
    """The fall-event writer: the only path from an alarm to the database and the
    connected clients, persisting before it publishes."""
    preview: PreviewStream | None = None
    """The preview stream, or ``None`` when streaming is disabled (§18.2)."""

    broadcaster: SnapshotBroadcaster | None = None
    """The live broadcaster. Always present when streaming is enabled, because the
    WebSocket endpoint needs it even before a camera starts."""

    consumer: FallConsumer | DetectionTrackingConsumer | None = None
    """The object the pipeline calls for every decoded frame.

    Built when detection and tracking are both enabled; the fall and preview stages are
    added when fall detection is enabled."""

    model_error: str | None = None
    """Set when the detector could not be loaded, so readiness can report model
    unavailability separately from camera state (thesis plan §17.2)."""

    tracker_error: str | None = None
    """Set when the tracker could not be constructed, reported separately for the
    same reason as ``model_error``."""

    fall_error: str | None = None
    """Set when the fall layer could not be constructed (for example an unreadable
    calibration file), reported separately from model and tracker readiness."""
    orphaned_events_closed: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    _camera_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _camera_id: uuid.UUID | None = field(default=None, repr=False)

    # -- camera lifecycle ----------------------------------------------

    @property
    def camera_id(self) -> uuid.UUID:
        """UUID of the configured camera.

        Derived from the source reference unless ``FALL_CAMERA_ID`` fixes one (see
        :meth:`Settings.camera_uuid`), so restarting the process continues the same
        camera's history rather than creating a new one per run.
        """
        with self._camera_lock:
            if self._camera_id is None:
                self._camera_id = self.settings.camera_uuid()
            return self._camera_id

    @property
    def camera_state(self) -> CameraState:
        """Current camera state (FR-09).

        ``idle`` when no supervisor exists, so the endpoint can report a camera that was
        never started without inventing a state.
        """
        pipeline = self.pipeline
        return CameraState.IDLE if pipeline is None else pipeline.state

    def start_camera(self) -> CameraState:
        """Start the configured source.

        Idempotent: :meth:`PipelineSupervisor.start` is a no-op while running, so a
        duplicated request cannot create a second decoder (FR-11).

        Raises:
            PipelineStateError: No pipeline is available, which means detection and
                tracking were both disabled and there is nothing to run, or the
                detector is enabled but its weights failed to load.
            ConfigurationError: The source is not configured.
            InvalidSourceError: The source exists but cannot be used.
        """
        pipeline = self.pipeline
        if pipeline is None:
            raise PipelineStateError(
                "no pipeline is available; enable detection and tracking, or configure "
                "a source, before starting the camera"
            )
        detection = self.detection
        if detection is not None and not detection.loaded:
            # Starting anyway would decode frames only to fail every one of them in
            # the detector; refusing names the load error the operator has to fix.
            reason = self.model_error or "detector not loaded"
            raise PipelineStateError(f"cannot start the camera: {reason}")
        pipeline.start()
        self.publish_status("camera_start_requested")
        return pipeline.state

    def stop_camera(self, *, reason: str = "stopped") -> CameraState:
        """Stop the pipeline and close its session.

        Blocking and idempotent, like :meth:`PipelineSupervisor.stop`. The fall session is
        ended too, so every open alarm closes with a recorded reason instead of being
        left to the next run.
        """
        pipeline = self.pipeline
        if pipeline is None:
            return CameraState.IDLE
        pipeline.stop(reason=reason)
        fall = self.fall
        if fall is not None:
            fall.end_session(reason=reason)
        preview = self.preview
        if preview is not None:
            preview.clear()
        self.publish_status("camera_stopped")
        return pipeline.state

    def current_session(self) -> StreamSessionInfo | None:
        """The live session of the running pipeline, or ``None``.

        One accessor rather than a repeated ``pipeline.current_session()`` chain, so the two
        callers below cannot disagree about whether a session exists — and so the optional
        pipeline is unwrapped once.
        """
        pipeline = self.pipeline
        return None if pipeline is None else pipeline.current_session()

    def publish_status(self, reason: str) -> None:
        """Publish a camera status change immediately (§17.2).

        Events and status are published the moment they happen while track snapshots are
        throttled to 2 Hz, so a client learns about ``connecting`` → ``live`` →
        ``reconnecting`` without waiting for the next snapshot window.
        """
        broadcaster = self.broadcaster
        if broadcaster is None:
            return
        session = self.current_session()
        broadcaster.publish_status(
            {
                "reason": reason,
                "camera_id": str(self.camera_id),
                "state": self.camera_state.value,
                "stream_session_id": None if session is None else str(session.stream_session_id),
            }
        )

    def snapshot(self) -> dict[str, Any]:
        """Status payload for ``GET /api/v1/status`` (§17.2).

        Composed from the services' own snapshots so a figure has exactly one owner, and
        so a service that is disabled reports itself as disabled rather than as absent.
        """
        from fall_detection.schemas.api import API_SCHEMA_VERSION

        pipeline = self.pipeline
        session = self.current_session()
        return {
            "schema_version": API_SCHEMA_VERSION,
            "camera": {
                "camera_id": str(self.camera_id),
                "state": self.camera_state.value,
                "name": self.settings.camera_name,
                "source_type": self.settings.video.mode,
                "stream_session_id": None if session is None else str(session.stream_session_id),
                "attempt": None if session is None else session.attempt,
            },
            "pipeline": {} if pipeline is None else pipeline.snapshot(),
            "detection": {} if self.detection is None else self.detection.snapshot(),
            "tracking": {} if self.tracking is None else self.tracking.snapshot(),
            "fall": {} if self.fall is None else self.fall.snapshot(),
            "performance": self.performance_snapshot(),
            "events": {} if self.events is None else self.events.snapshot(),
            "streaming": {
                "preview": {} if self.preview is None else self.preview.snapshot(),
                "live": {} if self.broadcaster is None else self.broadcaster.snapshot(),
            },
            "resources": self.resource_snapshot(),
            "retention": self.retention_payload(),
        }

    def performance_snapshot(self) -> dict[str, Any]:
        """FPS and per-stage latency measured by the consumer, when one runs."""
        consumer = self.consumer
        performance = getattr(consumer, "performance", None)
        return {} if performance is None else performance.snapshot()

    def retention_payload(self) -> dict[str, Any]:
        """FR-12's policy and the last pass's outcome (§17.2, §24.1).

        A disabled policy says so; an enabled one reports what it deleted and how many
        events remain. Reporting ``None`` for the last pass is deliberate: no pass has
        run yet, which is different from a pass that deleted nothing.
        """
        from fall_detection.database.maintenance import describe_retention, policy_from_settings

        return {
            "policy": describe_retention(policy_from_settings(self.settings)),
            "last_pass": self.retention,
        }

    def apply_retention(self) -> dict[str, Any]:
        """Run one FR-12 retention pass and record it for the status payload.

        Called once at startup, which is enough for a single-process deployment: the
        policy's unit is days, so a per-request check would add a query to every call and
        delete the same rows nothing more often.
        """
        from fall_detection.database.maintenance import apply_retention, policy_from_settings

        policy = policy_from_settings(self.settings)
        try:
            result = apply_retention(self.writer, policy)
        except Exception as exc:  # noqa: BLE001 - a failed cleanup must not stop the process
            self.retention = {
                "applied": False,
                "error": f"{type(exc).__name__}: {exc}",
                "note": "the retention pass failed; the history was left untouched",
            }
            logger.error("retention pass failed: %s", self.retention["error"])
            return self.retention
        self.retention = result.as_dict()
        return self.retention

    def resource_snapshot(self) -> dict[str, Any]:
        """Process resource measurements for the status payload (§17.2).

        §8 asks for a status payload that reports "throughput, latency, drop rate, and
        memory usage" and §23.1 warns that Apple Silicon's unified memory must not be
        presented as discrete VRAM. The values here are therefore the *process's*: its
        resident set, its CPU time, its thread count, and the machine's load average.
        None of them is an accelerator-memory figure, and none is a performance claim.
        """
        import threading

        from fall_detection.core.process import load_average, process_usage

        usage = process_usage()
        load = load_average()
        return {
            "rss_bytes": usage.peak_rss_bytes,
            "cpu_user_s": round(usage.cpu_user_s, 3),
            "cpu_system_s": round(usage.cpu_system_s, 3),
            "threads": threading.active_count(),
            # ``None`` where the OS has no load average (Windows), never a fake zero.
            "load_average": None if load is None else [round(value, 3) for value in load],
            "device": self.settings.device,
            "note": (
                "process measurements; on Apple Silicon the accelerator shares this "
                "memory, so this is not a discrete VRAM figure (§23.1)"
            ),
        }

    def shutdown(self) -> None:
        """Stop the pipeline and release every service; idempotent.

        Order matters: the pipeline is stopped first so no frame consumer is still
        running, then the fall layer ends its session (which closes every open alarm with
        a recorded reason), then tracking ends its session (which terminates live tracks
        and closes their rows), then the clients are disconnected, and finally the model
        is released.
        """
        pipeline = self.pipeline
        if pipeline is not None:
            pipeline.stop()
            self.pipeline = None
        fall = self.fall
        if fall is not None:
            fall.end_session(reason="stopped")
        tracking = self.tracking
        if tracking is not None:
            tracking.end_session(reason="shutdown")
        broadcaster = self.broadcaster
        if broadcaster is not None:
            broadcaster.close_all()
        detection = self.detection
        if detection is not None:
            detection.unload()
            self.detection = None
        if tracking is not None:
            tracking.unload()
            self.tracking = None
        self.fall = None
        self.consumer = None
