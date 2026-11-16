"""The fall service over tracking output, with a real event writer and database."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from tests.helpers_fall import (
    CAMERA,
    SESSION,
    TRACK,
    fall_sequence,
    observations,
    standing,
    termination,
)

from fall_detection.config import Settings
from fall_detection.database import DatabaseWriter, FallEventRepository, TrackRepository
from fall_detection.events import FallEventWriter
from fall_detection.fall.contracts import FallState
from fall_detection.fall.service import FallService


class RecordingBroadcaster:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def publish_event(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


@pytest.fixture
def seeded(
    writer: DatabaseWriter,
    stream_session_row: Callable[..., uuid.UUID],
) -> DatabaseWriter:
    stream_session_row(camera_id=CAMERA, stream_session_id=SESSION)
    writer.write(
        lambda session: TrackRepository(session).open(
            track_id=TRACK, stream_session_id=SESSION, tracker_id=1
        )
    )
    return writer


def drive(service: FallService, frames: list[tuple[float, Any]], *, start_seq: int = 0) -> None:
    for offset, (t, pose) in enumerate(frames):
        service.update(observations(start_seq + offset, t, [(TRACK, 1, pose)]))


def test_a_fall_opens_one_persisted_and_published_alarm(
    make_settings: Callable[..., Settings], seeded: DatabaseWriter
) -> None:
    broadcaster = RecordingBroadcaster()
    events = FallEventWriter(writer=seeded, broadcaster=broadcaster)  # type: ignore[arg-type]
    service = FallService(make_settings(), events=events, record_transitions=True)
    drive(service, fall_sequence(lie_seconds=4.0))

    assert service.current_states()[TRACK].state is FallState.FALLEN
    opened = [message for message in broadcaster.events if message["kind"] == "opened"]
    assert len(opened) == 1
    assert opened[0]["persisted"] is True
    stored = seeded.read(lambda session: FallEventRepository(session).list_open())
    assert len(stored) == 1
    assert stored[0].event_id == service.current_states()[TRACK].event_id
    assert stored[0].falling_utc <= stored[0].impact_utc <= stored[0].alert_utc
    # The long-lie confirmation is the response time.
    assert stored[0].response_time_s == pytest.approx(3.0, abs=0.1)
    states = [transition.to_state for transition in service.transition_log]
    assert states[:4] == [
        FallState.NORMAL,
        FallState.FALLING,
        FallState.IMPACT,
        FallState.FALLEN,
    ]


def test_recovery_closes_the_alarm(
    make_settings: Callable[..., Settings], seeded: DatabaseWriter
) -> None:
    events = FallEventWriter(writer=seeded)
    service = FallService(make_settings(), events=events)
    frames = fall_sequence(lie_seconds=4.0)
    drive(service, frames)
    t = frames[-1][0]
    recovery = [(t + 0.1 * (index + 1), standing()) for index in range(10)]
    drive(service, recovery, start_seq=len(frames))

    assert service.current_states()[TRACK].state is FallState.NORMAL
    page = seeded.read(lambda session: FallEventRepository(session).page())
    assert page.total == 1
    assert page.events[0].end_reason == "recovered"
    assert page.events[0].ended_utc is not None


def test_a_terminated_track_closes_its_alarm(
    make_settings: Callable[..., Settings], seeded: DatabaseWriter
) -> None:
    events = FallEventWriter(writer=seeded)
    service = FallService(make_settings(), events=events)
    frames = fall_sequence(lie_seconds=4.0)
    drive(service, frames)
    t = frames[-1][0] + 3.5
    batch = service.update(observations(len(frames), t, [], terminated=[termination(TRACK, t)]))
    assert batch.closed_tracks == (TRACK,)
    page = seeded.read(lambda session: FallEventRepository(session).page())
    assert page.events[0].end_reason == "track_lost"


def test_a_new_session_closes_open_alarms(
    make_settings: Callable[..., Settings], seeded: DatabaseWriter
) -> None:
    events = FallEventWriter(writer=seeded)
    service = FallService(make_settings(), events=events)
    frames = fall_sequence(lie_seconds=4.0)
    drive(service, frames)
    service.update(observations(0, frames[-1][0] + 1.0, [], session=uuid.uuid4()))
    page = seeded.read(lambda session: FallEventRepository(session).page())
    assert page.events[0].end_reason == "session_ended"
    assert service.current_states() == {}


def test_stopping_closes_open_alarms(
    make_settings: Callable[..., Settings], seeded: DatabaseWriter
) -> None:
    events = FallEventWriter(writer=seeded)
    service = FallService(make_settings(), events=events)
    drive(service, fall_sequence(lie_seconds=4.0))
    service.end_session(reason="stopped")
    page = seeded.read(lambda session: FallEventRepository(session).page())
    assert page.events[0].end_reason == "stopped"


def test_a_standing_person_raises_nothing(make_settings: Callable[..., Settings]) -> None:
    service = FallService(make_settings(), record_transitions=True)
    drive(service, [(0.1 * index, standing()) for index in range(100)])
    assert service.current_states()[TRACK].state is FallState.NORMAL
    assert service.alert_log == ()


def test_the_alarm_works_without_a_database(make_settings: Callable[..., Settings]) -> None:
    # The offline evaluation runs without storage; alarms are still decided and logged.
    service = FallService(make_settings(), record_transitions=True)
    drive(service, fall_sequence(lie_seconds=4.0))
    assert len(service.alert_log) == 1


def test_calibrated_thresholds_override_the_defaults(
    make_settings: Callable[..., Settings], tmp_path: Any
) -> None:
    path = tmp_path / "calibration.json"
    path.write_text('{"thresholds": {"long_lie_s": 1.5, "vy_threshold": 0.8}}', encoding="utf-8")
    service = FallService(make_settings(fall={"calibration_path": str(path)}))
    assert service.thresholds.long_lie_s == 1.5
    assert service.thresholds.vy_threshold == 0.8
    assert service.snapshot()["thresholds"]["calibrated"] is True


def test_the_snapshot_counts_states(make_settings: Callable[..., Settings]) -> None:
    service = FallService(make_settings())
    drive(service, fall_sequence(lie_seconds=4.0))
    snapshot = service.snapshot()
    assert snapshot["active_alarms"] == 1
    assert snapshot["stats"]["alerts"] == 1
    assert snapshot["thresholds"]["calibrated"] is False


def test_a_person_lost_by_the_detector_after_impact_still_alarms(
    make_settings: Callable[..., Settings], seeded: DatabaseWriter
) -> None:
    # The pose model often stops seeing someone lying on the floor. The person reached
    # impact and was never seen recovering, so the alarm fires on the frame clock.
    broadcaster = RecordingBroadcaster()
    events = FallEventWriter(writer=seeded, broadcaster=broadcaster)  # type: ignore[arg-type]
    service = FallService(make_settings(), events=events, record_transitions=True)
    frames = fall_sequence(lie_seconds=0.3)
    drive(service, frames)
    assert service.current_states()[TRACK].state is FallState.IMPACT
    t = frames[-1][0]
    for index in range(1, 120):
        service.update(observations(len(frames) + index, t + index / 30.0, []))
    assert service.current_states()[TRACK].state is FallState.FALLEN
    reasons = [transition.reason for transition in service.transition_log]
    assert "long_lie_unobserved" in reasons
    assert [message["kind"] for message in broadcaster.events] == ["opened"]
