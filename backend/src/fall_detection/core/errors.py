"""Typed exception hierarchy.

Thesis plan §17.2 requires consistent error codes for invalid sources, camera
failures, unavailable models, and database errors, and §22 lists these classes
as the unit under test. Every failure the application can attribute to a
subsystem therefore raises a subclass of :class:`CctvError` instead of a bare
``RuntimeError``.

The classes carry a stable ``code`` so the API layer can map them to HTTP
responses without matching on message text, and callers can distinguish
"misconfigured by the operator" (``InvalidSourceError``) from "the camera went
away" (``CameraUnavailableError``). Decoder failures are retryable by design;
validation failures are not.
"""

from __future__ import annotations

__all__ = [
    "CameraUnavailableError",
    "CctvError",
    "ConfigurationError",
    "DatabaseError",
    "DecoderError",
    "DecoderStallError",
    "InvalidSourceError",
    "InvalidZoneError",
    "ModelUnavailableError",
    "NotFoundError",
    "PipelineStateError",
]


class CctvError(Exception):
    """Base class for every application error.

    Attributes:
        code: Stable machine-readable identifier used by the API layer.
    """

    code = "cctv_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable representation for API error bodies."""
        payload = {"code": self.code, "message": self.message}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class ConfigurationError(CctvError):
    """The process is misconfigured and cannot start a subsystem."""

    code = "configuration_error"


class InvalidSourceError(CctvError):
    """The requested video source is not usable and retrying will not help.

    Raised for an unsupported scheme, a path outside the authorized media
    roots, or a missing file. It is deliberately *not* retryable: the pipeline
    reports ``offline`` rather than looping through reconnect backoff.
    """

    code = "invalid_source"


class CameraUnavailableError(CctvError):
    """A configured camera could not be reached (network, RTSP, permissions).

    Retryable: the reconnect policy (thesis plan §9.2) applies exponential
    backoff with jitter.
    """

    code = "camera_unavailable"


class DecoderError(CctvError):
    """FFmpeg decoding failed after the process was started."""

    code = "decoder_error"


class DecoderStallError(DecoderError):
    """No frame arrived within the configured stall timeout.

    Thesis plan §22 lists decoder hangs as a specific fault to inject. A stall
    is treated as a decoder failure so the same reconnect path applies.
    """

    code = "decoder_stall"


class PipelineStateError(CctvError):
    """A lifecycle operation is invalid for the current pipeline state."""

    code = "pipeline_state"


class DatabaseError(CctvError):
    """Persistence failed."""

    code = "database_error"


class ModelUnavailableError(CctvError):
    """A model weight file or inference device is unavailable.

    Defined here so the readiness endpoint and later detection phases share one
    vocabulary (thesis plan §17.2 separates model readiness from camera state).
    """

    code = "model_unavailable"


class InvalidZoneError(CctvError):
    """A configured ROI polygon or zone assignment is invalid.

    FR-05 requires polygons to be validated and stored with versions, and FR-06
    requires a reference-workstation assignment to apply only to an active track.
    Both failures are *validation* failures rather than camera failures, so this
    error is deliberately not retryable and carries the offending input in its
    message.
    """

    code = "invalid_zone"


class NotFoundError(CctvError):
    """The requested entity does not exist.

    Covers an unknown camera, zone, track, or event cursor. A missing entity is a
    ``404`` rather than a validation failure, which is why it is its own class:
    the request was well formed and the answer is "there is nothing there".
    """

    code = "not_found"
