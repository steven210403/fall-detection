"""FastAPI dependencies and error handling (thesis plan §17.2).

**Error shape.** §17.2 requires "consistent error codes for invalid sources, camera
failures, unavailable models, and database errors". One handler maps every
:class:`~fall_detection.core.errors.CctvError` subclass to an HTTP status using its
stable ``code``, so a client switches on the code rather than on a message that may be
reworded.

The dashboard has no login: the API is unauthenticated and is meant to be reached only on
the local machine or a trusted network. The application state is reached through
``request.app.state.cctv``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, Request

from fall_detection.core.errors import CctvError, ConfigurationError

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from fall_detection.api.state import AppState

__all__ = [
    "AppStateDependency",
    "get_app_state",
    "status_for_error",
]

# HTTP status per error code. §17.2 asks for consistent codes rather than a status per
# message, so this table is the contract: a client can rely on ``invalid_source`` always
# being a 400 and ``model_unavailable`` always a 503, whatever the wording.
_STATUS_BY_CODE: dict[str, int] = {
    # The request names a source that cannot be used; retrying will not help.
    "invalid_source": 400,
    # A polygon or assignment failed validation (FR-05, FR-06).
    "invalid_zone": 400,
    # The process is misconfigured; the operator must fix the environment.
    "configuration_error": 500,
    # The entity or cursor does not exist.
    "not_found": 404,
    # The lifecycle operation is invalid for the current camera state (FR-11).
    "pipeline_state": 409,
    # The camera could not be reached; retryable through §9.2's backoff.
    "camera_unavailable": 503,
    "decoder_error": 503,
    "decoder_stall": 503,
    "model_unavailable": 503,
    # A database problem is a service condition, not a client mistake.
    "database_error": 503,
}


def status_for_error(error: CctvError) -> int:
    """Return the HTTP status for a typed application error.

    An unknown code falls back to ``500``: a new error class that nobody mapped is a
    server-side problem, and returning ``200`` or ``400`` would misreport it.
    """
    return _STATUS_BY_CODE.get(error.code, 500)


def get_app_state(request: Request) -> AppState:
    """Return the lifespan-built application state.

    Raises:
        CctvError: The state is missing, which means the lifespan did not run. FastAPI
            always runs it for a real request, so this only fires in a misconfigured
            test, and a typed error is more useful than an ``AttributeError``.
    """
    state = getattr(request.app.state, "cctv", None)
    if state is None:  # pragma: no cover - defensive; the lifespan always sets it
        raise ConfigurationError("application state is unavailable; the lifespan did not run")
    return state


AppStateDependency = Annotated["AppState", Depends(get_app_state)]
"""Application state, injected into a route that reads it."""


def error_payload(error: CctvError) -> dict[str, Any]:
    """Build the JSON body for a typed error (§17.2)."""
    return {"error": error.as_dict()}
