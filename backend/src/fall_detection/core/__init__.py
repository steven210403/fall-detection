"""Shared typed contracts, clocks, errors, and logging.

This package is the bottom of the dependency graph (thesis plan §17.1): it must
not import FastAPI, SQLAlchemy, Pydantic, or any CV library. Subpackages import
from here; nothing here imports from them.
"""

from fall_detection.core.clock import (
    AcquisitionClockBase,
    Clock,
    MonotonicClock,
    SystemClock,
)
from fall_detection.core.contracts import CameraState, FramePacket, FrameSource, StreamSessionInfo
from fall_detection.core.errors import (
    CameraUnavailableError,
    CctvError,
    ConfigurationError,
    DatabaseError,
    DecoderError,
    DecoderStallError,
    InvalidSourceError,
    InvalidZoneError,
    ModelUnavailableError,
    NotFoundError,
    PipelineStateError,
)
from fall_detection.core.logging import configure_logging, get_logger, redact_text, redact_url

__all__ = [
    "AcquisitionClockBase",
    "CameraState",
    "CameraUnavailableError",
    "CctvError",
    "Clock",
    "ConfigurationError",
    "DatabaseError",
    "DecoderError",
    "DecoderStallError",
    "FramePacket",
    "FrameSource",
    "InvalidSourceError",
    "InvalidZoneError",
    "ModelUnavailableError",
    "MonotonicClock",
    "NotFoundError",
    "PipelineStateError",
    "StreamSessionInfo",
    "SystemClock",
    "configure_logging",
    "get_logger",
    "redact_text",
    "redact_url",
]
