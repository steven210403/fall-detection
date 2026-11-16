"""Health routers.

The fall and event readiness checks sit alongside the
existing dependency, database, detection, tracking, and feature checks. §17.2 requires
"model readiness and camera status separately", and the same separation applies to each
subsystem: a disabled component is not a degradation, an enabled one that failed to load
is, and the reason is always in the payload.
"""

from __future__ import annotations

import importlib.metadata as metadata

from fastapi import APIRouter, Request
from sqlalchemy import text

from fall_detection.core.clock import SystemClock
from fall_detection.schemas.health import HealthLive, HealthReady, ReadinessCheck

router = APIRouter(tags=["health"])

_clock = SystemClock()


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@router.get("/health/live", response_model=HealthLive, summary="Process liveness")
def health_live() -> HealthLive:
    """Liveness only: an offline camera must not fail this probe."""
    return HealthLive(
        service="fall-detection",
        version=_distribution_version("fall-detection") or "0.0.0",
    )


@router.get("/health/ready", response_model=HealthReady, summary="Dependency readiness")
def health_ready(request: Request) -> HealthReady:
    """Report local dependency readiness separately from camera state.

    The database probe uses the engine the lifespan already built, so readiness
    always describes the database this process actually uses rather than a
    second connection built from ambient settings.
    """
    checks: list[ReadinessCheck] = []

    # Import name -> distribution name (they differ for OpenCV).
    distributions = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "torch": "torch",
        "torchvision": "torchvision",
        "ultralytics": "ultralytics",
    }
    for import_name, distribution in distributions.items():
        version = _distribution_version(distribution)
        checks.append(
            ReadinessCheck(
                name=import_name,
                status="ok" if version else "unavailable",
                detail=version,
            )
        )

    checks.append(
        ReadinessCheck(
            name="clock",
            status="ok" if _clock.monotonic() > 0 else "degraded",
            detail="monotonic",
        )
    )

    checks.append(_database_check(request))
    detection = _detection_check(request)
    if detection is not None:
        # Only reported when detection is enabled: a deliberately disabled
        # detector is not a degradation, and readiness must stay meaningful.
        checks.append(detection)
    tracking = _tracking_check(request)
    if tracking is not None:
        # §17.2 separates tracker readiness from model readiness: a camera or
        # tracker problem must not fail liveness, and the operator needs to know
        # which of the two is unavailable.
        checks.append(tracking)
    fall = _fall_check(request)
    if fall is not None:
        # Reported separately: the fall logic loads no weights, so a problem here is a
        # configuration problem rather than a model one.
        checks.append(fall)
    events = _events_check(request)
    if events is not None:
        # Event storage is reported separately because failing to store an alarm is
        # different from failing to detect a fall.
        checks.append(events)

    status = "ok" if all(c.status == "ok" for c in checks) else "degraded"
    return HealthReady(status=status, checks=checks)


def _database_check(request: Request) -> ReadinessCheck:
    """Report whether the migrated schema is present.

    A missing schema degrades readiness rather than failing the process, so the
    operator can still reach this endpoint to find out what is wrong
    (thesis plan §17.2).
    """
    state = getattr(request.app.state, "cctv", None)
    engine = getattr(state, "engine", None)
    if engine is None:
        return ReadinessCheck(
            name="database", status="degraded", detail="application state unavailable"
        )
    try:
        with engine.connect() as connection:
            # A zero-row query against the newest table: it validates both
            # connectivity and that migrations have been applied.
            connection.execute(text("SELECT ended_at FROM stream_session WHERE 1 = 0")).first()
    except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
        return ReadinessCheck(
            name="database",
            status="degraded",
            detail=f"schema unavailable; run 'alembic upgrade head' ({type(exc).__name__})",
        )
    return ReadinessCheck(name="database", status="ok", detail="sqlite schema present")


def _detection_check(request: Request) -> ReadinessCheck | None:
    """Report whether the detector is loaded and ready.

    Returns ``None`` when detection is disabled, because an intentionally
    disabled detector is not a degradation. When enabled, the model is reported
    separately from the camera (thesis plan §17.2), and an offline camera never
    affects this check.
    """
    state = getattr(request.app.state, "cctv", None)
    if state is None:
        return None
    service = getattr(state, "detection", None)
    if service is None:
        if getattr(state.settings.detection, "enabled", False):
            detail = state.model_error or "detector not loaded"
            return ReadinessCheck(name="detection", status="degraded", detail=detail)
        return None
    info = service.info
    if info is None:
        return ReadinessCheck(name="detection", status="degraded", detail="model not loaded")
    return ReadinessCheck(
        name="detection",
        status="ok",
        detail=f"{info.model_name} on {info.device} @ {info.imgsz}px",
    )


def _tracking_check(request: Request) -> ReadinessCheck | None:
    """Report whether the configured tracker is available.

    Returns ``None`` when tracking is disabled, because an intentionally disabled
    tracker is not a degradation. When enabled, the check is reported separately
    from the detector so §17.2's "present model readiness and camera status
    separately" holds for tracking too. The check deliberately does not report a
    live session: a camera that was never started is a *camera* state, not a
    tracker problem.
    """
    state = getattr(request.app.state, "cctv", None)
    if state is None:
        return None
    service = getattr(state, "tracking", None)
    if service is None:
        if getattr(state.settings.tracking, "enabled", False):
            detail = state.tracker_error or "tracker not constructed"
            return ReadinessCheck(name="tracking", status="degraded", detail=detail)
        return None
    info = service.info
    return ReadinessCheck(
        name="tracking",
        status="ok",
        detail=(
            f"{info.name} buffer={info.configured_buffer_frames} frames "
            f"@ {state.settings.tracking.assumed_update_rate_hz:.0f}Hz assumed"
        ),
    )


def _fall_check(request: Request) -> ReadinessCheck | None:
    """Report whether fall detection is configured, and with which thresholds."""
    state = getattr(request.app.state, "cctv", None)
    if state is None:
        return None
    service = getattr(state, "fall", None)
    if service is None:
        if getattr(state.settings.fall, "enabled", False):
            detail = state.fall_error or "fall service not constructed"
            return ReadinessCheck(name="fall", status="degraded", detail=detail)
        return None
    thresholds = service.thresholds
    calibrated = state.settings.fall.calibration_path is not None
    return ReadinessCheck(
        name="fall",
        status="ok",
        detail=(
            f"AR>={thresholds.ar_threshold:.2f}, lying<{thresholds.angle_lying_deg:.0f}deg, "
            f"Vy>{thresholds.vy_threshold:.2f}/s, long-lie {thresholds.long_lie_s:.1f}s "
            f"({'calibrated' if calibrated else 'literature defaults'})"
        ),
    )


def _events_check(request: Request) -> ReadinessCheck | None:
    """Report whether event storage is available.

    Returns ``None`` when events are disabled. When enabled, the check reports whether a
    database writer is attached, because an event layer without one runs the state machine
    and discards the intervals — a legitimate offline configuration that an operator must
    still be able to see.
    """
    state = getattr(request.app.state, "cctv", None)
    if state is None:
        return None
    if not getattr(state.settings.fall, "enabled", False):
        return None
    writer = getattr(state, "events", None)
    if writer is None:
        return ReadinessCheck(
            name="events", status="degraded", detail="event writer not constructed"
        )
    if not writer.persistent:
        return ReadinessCheck(
            name="events",
            status="degraded",
            detail="no database writer is attached; alarms would not be stored",
        )
    return ReadinessCheck(
        name="events",
        status="ok",
        detail=f"history storage ready; {writer.stats.opened} alarm(s) this run",
    )


__all__ = ["router"]
