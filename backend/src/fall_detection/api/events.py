"""Fall-event endpoints.

* ``GET /api/v1/fall-events`` — history, newest alarm first, with a keyset cursor so a
  client that reconnects can page through what it missed while the camera keeps writing.
  ``active=true`` returns only alarms that are still open, which the dashboard polls as a
  safety net in case a live alarm message was dropped by a slow connection.
* ``GET /api/v1/fall-events/{id}`` — one event.
* ``POST /api/v1/fall-events/{id}/acknowledge`` — record that an operator saw the alarm.

A read that cannot reach the database raises ``503`` rather than returning an empty page:
an empty history and a broken database are different answers.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Path, Query

from fall_detection.api.deps import AppStateDependency
from fall_detection.core.errors import DatabaseError, ModelUnavailableError, NotFoundError
from fall_detection.events.writer import FallEventWriter
from fall_detection.schemas.api import FallEventModel, FallEventPageModel

__all__ = ["router"]

router = APIRouter(prefix="/api/v1", tags=["fall-events"])


@router.get(
    "/fall-events",
    response_model=FallEventPageModel,
    summary="Fall-event history with cursors",
)
def list_fall_events(
    state: AppStateDependency,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
    track_uuid: Annotated[uuid.UUID | None, Query()] = None,
    stream_session_id: Annotated[uuid.UUID | None, Query()] = None,
    since: Annotated[float | None, Query(ge=0)] = None,
    until: Annotated[float | None, Query(ge=0)] = None,
    active: Annotated[bool, Query()] = False,
) -> FallEventPageModel:
    """Return one page of stored fall events, newest alarm first.

    Args:
        limit: Page size.
        cursor: Opaque cursor from a previous page's ``next_cursor``.
        track_uuid: Restrict to one tracked person.
        stream_session_id: Restrict to one stream session.
        since: Only alarms at or after this UTC epoch time.
        until: Only alarms at or before this UTC epoch time.
        active: Only alarms that are still open.

    Raises:
        404: The cursor is not one this project produced.
        503: Event storage is disabled, or the history could not be read.
    """
    writer = _writer(state)
    page = writer.history(
        limit=limit,
        cursor=cursor,
        track_uuid=track_uuid,
        session_id=stream_session_id,
        since_utc=since,
        until_utc=until,
        open_only=active,
    )
    return FallEventPageModel.model_validate(page.as_dict())


@router.get(
    "/fall-events/{event_id}",
    response_model=FallEventModel,
    summary="One stored fall event",
)
def get_fall_event(
    state: AppStateDependency,
    event_id: Annotated[uuid.UUID, Path()],
) -> FallEventModel:
    """Return one stored event, or ``404``."""
    event = _writer(state).get(event_id)
    if event is None:
        raise NotFoundError(f"fall event {event_id} does not exist")
    return FallEventModel.model_validate(event.as_dict())


@router.post(
    "/fall-events/{event_id}/acknowledge",
    response_model=FallEventModel,
    summary="Acknowledge a fall alarm",
)
def acknowledge_fall_event(
    state: AppStateDependency,
    event_id: Annotated[uuid.UUID, Path()],
) -> FallEventModel:
    """Record that an operator saw the alarm (idempotent).

    Acknowledging does not close the event: it stays active until the person recovers or
    the track ends, so the history keeps the real duration of the fall.
    """
    event = _writer(state).acknowledge(event_id)
    if event is None:
        raise NotFoundError(f"fall event {event_id} does not exist")
    return FallEventModel.model_validate(event.as_dict())


def _writer(state: AppStateDependency) -> FallEventWriter:
    writer = state.events
    if writer is None:
        raise ModelUnavailableError("fall-event storage is disabled in configuration")
    if not writer.persistent:
        raise DatabaseError("fall-event history is unavailable: no database writer is attached")
    return writer
