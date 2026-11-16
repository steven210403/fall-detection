"""Live WebSocket endpoint (thesis plan §17.2).

§17.2 describes the live feed in five lines, and each becomes a mechanism:

* *"WS /api/v1/live | Track, fall-state, fall-event, and status snapshots"* — one socket carries
  all three kinds, and every envelope names its kind so a client routes without guessing.
* *"Send track snapshots at 2 Hz; send events/status immediately."* — the throttle lives
  in :class:`~fall_detection.streaming.snapshots.SnapshotBroadcaster`, and events and
  status bypass it.
* *"Slow clients receive the latest snapshot, not a backlog."* — each connection reads
  from its own **bounded** queue, which discards its oldest message when full. The
  envelope's ``sequence`` is monotonic, so a client that receives 5 then 8 knows it missed
  two messages.
* *"On reconnect, fetch history using a cursor to recover gaps."* — the first message of
  every connection is a ``hello`` carrying the current event cursor, so a client can ask
  ``GET /api/v1/fall-events`` for exactly the interval it lost.

There is no authentication or origin check: the endpoint is open, and the handshake is
refused only when the live feed is disabled or the client ceiling is reached.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from fall_detection.api.state import AppState
from fall_detection.core.logging import get_logger
from fall_detection.schemas.api import API_SCHEMA_VERSION
from fall_detection.streaming.snapshots import MessageKind

__all__ = ["router"]

logger = get_logger("api.live")

router = APIRouter(tags=["live"])

# WebSocket close codes. 1008 is "policy violation", which is the right code for a
# refused handshake: the client is not malformed (1003/1007) and the server is not
# failing (1011) — the request is not allowed.
CLOSE_POLICY_VIOLATION = 1008
CLOSE_GOING_AWAY = 1001

# How long the reader waits for the next message before checking the socket's state.
# The queue is what actually paces the stream; this timeout exists so a disconnect is
# noticed even when no message is due.
_READ_TIMEOUT_S = 1.0


@router.websocket("/api/v1/live")
async def live_feed(websocket: WebSocket) -> None:
    """Stream track, fall-state, fall-event, and status envelopes.

    The handshake is refused, before the socket is accepted, when the live feed is disabled
    or the client ceiling is reached. Refusing before ``accept()`` means a rejected client
    never receives a frame.
    """
    app_state: AppState = websocket.app.state.cctv  # type: ignore[attr-defined]
    broadcaster = app_state.broadcaster
    if broadcaster is None:
        await websocket.close(code=CLOSE_POLICY_VIOLATION, reason="the live feed is disabled")
        return

    try:
        handle = broadcaster.connect()
    except PermissionError as exc:
        await websocket.close(code=CLOSE_POLICY_VIOLATION, reason=str(exc))
        return

    await websocket.accept()
    try:
        await websocket.send_json(
            {
                "schema_version": API_SCHEMA_VERSION,
                "kind": MessageKind.HELLO,
                "sequence": 0,
                "emitted_monotonic": 0.0,
                "emitted_utc": 0.0,
                "latest_cursor": _latest_cursor(app_state),
                "status": app_state.snapshot(),
            }
        )
        await _pump(websocket, handle)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - a broken socket must not stop the process
        logger.warning("live connection ended: %s", exc)
    finally:
        broadcaster.disconnect(handle)


async def _pump(websocket: WebSocket, handle: Any) -> None:
    """Forward messages from the client's queue to the socket.

    The queue read is blocking, so it runs in a worker thread rather than on the event
    loop: a client that stops reading must not stall the loop that serves every other
    request (§8: "API remains responsive during camera disconnection or busy inference").
    """
    while True:
        message = await asyncio.to_thread(handle.take, _READ_TIMEOUT_S)
        if message is None:
            if websocket.client_state.name != "CONNECTED":
                return
            continue
        await websocket.send_json(message)


def _latest_cursor(app_state: AppState) -> str | None:
    """The cursor a client should resume history from (§17.2).

    Returns ``None`` when event storage is unavailable, which tells the client that it
    cannot recover a gap and should reload rather than silently reconnecting with a hole
    in its timeline.
    """
    writer = app_state.events
    if writer is None or not writer.persistent:
        return None
    try:
        page = writer.history(limit=1)
    except Exception as exc:  # noqa: BLE001 - a cursor is a convenience, not a gate
        logger.warning("could not read the latest event cursor: %s", exc)
        return None
    if page.events:
        from fall_detection.database.repositories import encode_cursor

        newest = page.events[0]
        return encode_cursor(alert_at=_as_datetime(newest.alert_utc), event_id=newest.event_id)
    return None


def _as_datetime(epoch: float) -> Any:
    """Convert a UTC epoch timestamp to an aware datetime for cursor encoding."""
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch, tz=UTC)
