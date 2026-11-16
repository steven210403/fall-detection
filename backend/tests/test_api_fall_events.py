"""Fall-event endpoints: history with cursors, the active filter, acknowledgement."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fall_detection.api.state import AppState
from fall_detection.config import Settings
from fall_detection.database import CameraRepository, StreamSessionRepository, TrackRepository
from fall_detection.events import OpenFallEvent

CAMERA = uuid.UUID("22222222-2222-2222-2222-222222222222")
SESSION = uuid.uuid4()
BASE = 1_700_000_000.0


def state_of(client: TestClient) -> AppState:
    return cast(AppState, cast(FastAPI, client.app).state.cctv)


def seed(state: AppState, count: int) -> list[uuid.UUID]:
    """Open ``count`` alarms on ``count`` tracks through the real writer."""
    track_ids = [uuid.uuid4() for _ in range(count)]

    def work(session: object) -> None:
        CameraRepository(session).get_or_create(  # type: ignore[arg-type]
            camera_id=CAMERA, name="cam", source_type="mp4", source_ref="a.mp4"
        )
        StreamSessionRepository(session).open(  # type: ignore[arg-type]
            stream_session_id=SESSION, camera_id=CAMERA, attempt=1
        )
        for index, track_id in enumerate(track_ids):
            TrackRepository(session).open(  # type: ignore[arg-type]
                track_id=track_id, stream_session_id=SESSION, tracker_id=index + 1
            )

    state.writer.write(work)
    writer = state.events
    assert writer is not None
    event_ids = []
    for index, track_id in enumerate(track_ids):
        event_id = uuid.uuid4()
        t = BASE + 10.0 * index
        writer.open(
            OpenFallEvent(
                event_id=event_id,
                track_uuid=track_id,
                camera_id=CAMERA,
                stream_session_id=SESSION,
                display_id=index + 1,
                falling_utc=t,
                impact_utc=t + 0.5,
                alert_utc=t + 3.5,
                response_time_s=3.0,
                peak_vy=2.0,
                min_torso_angle_deg=10.0,
                max_aspect_ratio=2.5,
                alert_source_timestamp=None,
            )
        )
        event_ids.append(event_id)
    return event_ids


def test_history_is_newest_first_with_a_cursor(
    api_client: Callable[..., TestClient], app_settings: Callable[..., Settings]
) -> None:
    client = api_client(app_settings)
    ids = seed(state_of(client), 3)
    first = client.get("/api/v1/fall-events", params={"limit": 2}).json()
    assert first["total"] == 3
    assert [event["event_id"] for event in first["events"]] == [str(ids[2]), str(ids[1])]
    assert first["next_cursor"] is not None
    second = client.get(
        "/api/v1/fall-events", params={"limit": 2, "cursor": first["next_cursor"]}
    ).json()
    assert [event["event_id"] for event in second["events"]] == [str(ids[0])]
    assert second["next_cursor"] is None


def test_an_open_alarm_is_active_until_closed(
    api_client: Callable[..., TestClient], app_settings: Callable[..., Settings]
) -> None:
    client = api_client(app_settings)
    state = state_of(client)
    ids = seed(state, 2)
    assert state.events is not None
    state.events.close(ids[0], ended_utc=BASE + 20.0, end_reason="recovered")

    active = client.get("/api/v1/fall-events", params={"active": True}).json()
    assert [event["event_id"] for event in active["events"]] == [str(ids[1])]
    closed = client.get(f"/api/v1/fall-events/{ids[0]}").json()
    assert closed["active"] is False
    assert closed["end_reason"] == "recovered"
    assert closed["response_time_s"] == 3.0


def test_acknowledging_does_not_close_the_alarm(
    api_client: Callable[..., TestClient], app_settings: Callable[..., Settings]
) -> None:
    client = api_client(app_settings)
    (event_id,) = seed(state_of(client), 1)
    response = client.post(f"/api/v1/fall-events/{event_id}/acknowledge")
    assert response.status_code == 200
    body = response.json()
    assert body["acknowledged_utc"] is not None
    assert body["active"] is True
    # Idempotent: a second acknowledgement keeps the first time.
    again = client.post(f"/api/v1/fall-events/{event_id}/acknowledge").json()
    assert again["acknowledged_utc"] == body["acknowledged_utc"]


def test_unknown_event_and_bad_cursor_are_404(
    api_client: Callable[..., TestClient], app_settings: Callable[..., Settings]
) -> None:
    client = api_client(app_settings)
    missing = uuid.uuid4()
    assert client.get(f"/api/v1/fall-events/{missing}").status_code == 404
    assert client.post(f"/api/v1/fall-events/{missing}/acknowledge").status_code == 404
    bad = client.get("/api/v1/fall-events", params={"cursor": "not-a-cursor"})
    assert bad.status_code == 404


def test_disabled_fall_detection_disables_the_history(
    api_client: Callable[..., TestClient], app_settings: Callable[..., Settings]
) -> None:
    client = api_client(app_settings, fall={"enabled": False})
    assert client.get("/api/v1/fall-events").status_code == 503


def test_alarms_left_open_by_a_previous_process_are_closed_on_start(
    api_client: Callable[..., TestClient], app_settings: Callable[..., Settings]
) -> None:
    settings = app_settings()
    client = api_client(lambda: settings)
    seed(state_of(client), 1)
    client.__exit__(None, None, None)
    restarted = api_client(lambda: settings)
    assert state_of(restarted).orphaned_events_closed == 1
    page = restarted.get("/api/v1/fall-events").json()
    assert page["events"][0]["end_reason"] == "stopped"
    assert page["events"][0]["active"] is False


def test_the_live_hello_carries_the_newest_alarm_cursor(
    api_client: Callable[..., TestClient], app_settings: Callable[..., Settings]
) -> None:
    client = api_client(app_settings)
    seed(state_of(client), 1)
    with client.websocket_connect("/api/v1/live") as socket:
        hello = socket.receive_json()
    assert hello["kind"] == "hello"
    assert hello["latest_cursor"] is not None
