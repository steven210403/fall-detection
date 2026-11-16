"""API-boundary Pydantic models: health probes and the HTTP/WebSocket contracts."""

from fall_detection.schemas.api import (
    API_SCHEMA_VERSION,
    CameraModel,
    CameraStateModel,
    ErrorBody,
    ErrorResponse,
    FallEventModel,
    FallEventPageModel,
    FallFeaturesModel,
    FallStateModel,
    LiveEnvelope,
    PreviewFrameModel,
    StatusModel,
    TrackModel,
)
from fall_detection.schemas.health import HealthLive, HealthReady, ReadinessCheck

__all__ = [
    "API_SCHEMA_VERSION",
    "CameraModel",
    "CameraStateModel",
    "ErrorBody",
    "ErrorResponse",
    "FallEventModel",
    "FallEventPageModel",
    "FallFeaturesModel",
    "FallStateModel",
    "HealthLive",
    "HealthReady",
    "LiveEnvelope",
    "PreviewFrameModel",
    "ReadinessCheck",
    "StatusModel",
    "TrackModel",
]
