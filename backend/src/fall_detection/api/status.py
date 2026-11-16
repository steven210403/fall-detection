"""Status endpoint (thesis plan §17.2).

§17.2 lists ``GET /api/v1/status`` as "Camera, model, sampling, and resources", and §8
asks the interface to show "throughput, latency, drop rate, and memory usage". This
endpoint is the one place those are read, and it composes them from the services' own
snapshots rather than recomputing anything: a figure has exactly one owner, and a service
that is disabled reports itself as disabled rather than as absent.

Two honesty rules from the plan are visible here:

* **Model readiness is separate from camera state** (§17.2: "Present model readiness and
  camera status separately"). An offline camera never makes the model look broken, and a
  missing weight file never makes the camera look offline.
* **No figure is an accuracy claim.** Every number is a counter, a configuration value,
  or a measured time from this process. The latency figures are the *pipeline's* stage
  times; the fall decision delay (the long-lie confirmation) is a configuration value
  reported under ``fall``, never folded into a latency figure.
"""

from __future__ import annotations

from fastapi import APIRouter

from fall_detection.api.deps import AppStateDependency
from fall_detection.schemas.api import StatusModel

__all__ = ["router"]

router = APIRouter(prefix="/api/v1", tags=["status"])


@router.get("/status", response_model=StatusModel, summary="Camera, model, and resource status")
def read_status(state: AppStateDependency) -> StatusModel:
    """Return the composed status payload.

    Includes the sampling configuration §17.2 asks for — the detection and tracking
    rates, the feature rate, and the preview rate — because a run's figures are only
    interpretable alongside the rates that produced them (§8).
    """
    return StatusModel.model_validate(state.snapshot())


@router.get("/config", summary="Resolved configuration that affects a run")
def read_config(state: AppStateDependency) -> dict[str, object]:
    """Return the configuration a run is using, without credentials or paths.

    Reported separately from :func:`read_status` because it changes only on restart while
    status changes per frame. The payload names the resolved *decisions* — tracker,
    thresholds, window lengths, preview caps — and never a credential or an absolute
    media path (§24.2).
    """
    settings = state.settings
    return {
        "environment": settings.environment,
        "device": settings.device,
        "video": {
            "mode": settings.video.mode,
            "read_timeout_s": settings.video.read_timeout_s,
            "live_queue_capacity": settings.video.live_queue_capacity,
            "offline_queue_capacity": settings.video.offline_queue_capacity,
        },
        "detection": {
            "enabled": settings.detection.enabled,
            "imgsz": settings.detection.imgsz,
            "conf": settings.detection.conf,
            "classes": list(settings.detection.classes),
        },
        "tracking": {
            "enabled": settings.tracking.enabled,
            "tracker": settings.tracking.tracker,
            "loss_tolerance_s": settings.tracking.loss_tolerance_s,
            "stale_after_s": settings.tracking.stale_after_s,
            "with_reid": settings.tracking.with_reid,
        },
        "fall": {
            "enabled": settings.fall.enabled,
            "keypoint_min_conf": settings.fall.keypoint_min_conf,
            "ar_threshold": settings.fall.ar_threshold,
            "angle_lying_deg": settings.fall.angle_lying_deg,
            "angle_upright_deg": settings.fall.angle_upright_deg,
            "vy_threshold": settings.fall.vy_threshold,
            "impact_window_s": settings.fall.impact_window_s,
            "long_lie_s": settings.fall.long_lie_s,
            "require_both": settings.fall.require_both,
            "recovery_hold_s": settings.fall.recovery_hold_s,
            "calibrated": settings.fall.calibration_path is not None,
        },
        "streaming": {
            "enabled": settings.streaming.enabled,
            "preview_fps": settings.streaming.preview_fps,
            "preview_max_width": settings.streaming.preview_max_width,
            "preview_max_height": settings.streaming.preview_max_height,
            "client_queue_capacity": settings.streaming.client_queue_capacity,
            "snapshot_interval_s": settings.streaming.snapshot_interval_s,
        },
    }
