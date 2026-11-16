"""Track table and repository tests.

Thesis plan §19 lists ``track`` as the durable record of a temporary tracking
identity, and §11.2 states the rules the schema has to support: the integer
tracker ID is not a key, termination consumes the identity, and a track belongs to
one stream session.

Constraints are asserted against the real SQLite schema rather than the ORM alone,
because the ORM can be changed without a migration and the database is what the
application actually reads.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fall_detection.database import (
    TRACK_TERMINATION_REASONS,
    Camera,
    CameraRepository,
    StreamSessionRepository,
    Track,
    TrackRepository,
    utc_now,
)


@pytest.fixture
def camera_and_session(session: Session) -> tuple[uuid.UUID, uuid.UUID]:
    """Create the camera and session rows a track must reference."""
    camera_id = uuid.uuid4()
    CameraRepository(session).get_or_create(
        camera_id=camera_id,
        name="office",
        source_type="mp4",
        source_ref="office.mp4",
    )
    session_id = StreamSessionRepository(session).open(camera_id=camera_id, attempt=1)
    session.flush()
    return camera_id, session_id


class TestTrackRepository:
    def test_open_stores_the_caller_uuid(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        # The runtime identity and the stored row must share one UUID, or an
        # fall event could not be traced back to a track.
        _, session_id = camera_and_session
        track_id = uuid.uuid4()
        TrackRepository(session).open(track_id=track_id, stream_session_id=session_id, tracker_id=1)
        session.flush()
        stored = TrackRepository(session).get(track_id)
        assert stored is not None
        assert stored.id == track_id
        assert stored.tracker_id == 1
        assert stored.terminated_at is None
        assert stored.termination_reason is None
        assert stored.observed_frames == 0

    def test_open_uses_the_supplied_first_seen_time(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        _, session_id = camera_and_session
        seen = datetime(2026, 10, 14, 9, 30, tzinfo=UTC)
        track_id = uuid.uuid4()
        TrackRepository(session).open(
            track_id=track_id,
            stream_session_id=session_id,
            tracker_id=2,
            first_seen_at=seen,
        )
        session.flush()
        stored = TrackRepository(session).get(track_id)
        assert stored is not None
        # SQLite stores datetimes without a timezone, so a round trip returns a
        # naive value equal to the UTC instant that was written.
        assert stored.first_seen_at == seen.replace(tzinfo=None)
        # last_seen starts equal to first_seen so the ordering constraint holds
        # from the first write.
        assert stored.last_seen_at == seen.replace(tzinfo=None)

    def test_close_records_reason_and_counters(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        _, session_id = camera_and_session
        track_id = uuid.uuid4()
        repository = TrackRepository(session)
        repository.open(track_id=track_id, stream_session_id=session_id, tracker_id=1)
        ended = utc_now() + timedelta(seconds=4)
        repository.close(
            track_id=track_id,
            termination_reason="lost_timeout",
            last_seen_at=ended,
            terminated_at=ended,
            observed_frames=7,
            missing_frames=20,
        )
        session.flush()
        stored = repository.get(track_id)
        assert stored is not None
        assert stored.termination_reason == "lost_timeout"
        # SQLite drops the timezone offset, so compare the UTC instant.
        assert stored.terminated_at == ended.replace(tzinfo=None)
        assert stored.observed_frames == 7
        assert stored.missing_frames == 20

    def test_touch_updates_counters_without_terminating(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        _, session_id = camera_and_session
        track_id = uuid.uuid4()
        repository = TrackRepository(session)
        repository.open(track_id=track_id, stream_session_id=session_id, tracker_id=1)
        seen = utc_now() + timedelta(seconds=2)
        repository.touch(track_id=track_id, last_seen_at=seen, observed_frames=5, missing_frames=1)
        session.flush()
        stored = repository.get(track_id)
        assert stored is not None
        assert stored.last_seen_at == seen.replace(tzinfo=None)
        assert stored.observed_frames == 5
        assert stored.missing_frames == 1
        assert stored.terminated_at is None

    def test_list_for_session_is_ordered_by_first_seen(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        _, session_id = camera_and_session
        repository = TrackRepository(session)
        base = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)
        for index, tracker_id in enumerate((3, 1, 2)):
            repository.open(
                track_id=uuid.uuid4(),
                stream_session_id=session_id,
                tracker_id=tracker_id,
                first_seen_at=base + timedelta(seconds=index),
            )
        session.flush()
        rows = repository.list_for_session(session_id)
        assert [row.tracker_id for row in rows] == [3, 1, 2]

    def test_list_for_session_excludes_other_sessions(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        # §19: a track belongs to exactly one session, which is what keeps two
        # decoding runs from being mixed.
        camera_id, session_id = camera_and_session
        other_session = StreamSessionRepository(session).open(camera_id=camera_id, attempt=2)
        repository = TrackRepository(session)
        repository.open(track_id=uuid.uuid4(), stream_session_id=session_id, tracker_id=1)
        repository.open(track_id=uuid.uuid4(), stream_session_id=other_session, tracker_id=1)
        session.flush()
        assert len(repository.list_for_session(session_id)) == 1
        assert len(repository.list_for_session(other_session)) == 1

    def test_list_open_returns_unterminated_tracks(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        _, session_id = camera_and_session
        repository = TrackRepository(session)
        open_id, closed_id = uuid.uuid4(), uuid.uuid4()
        repository.open(track_id=open_id, stream_session_id=session_id, tracker_id=1)
        repository.open(track_id=closed_id, stream_session_id=session_id, tracker_id=2)
        repository.close(track_id=closed_id, termination_reason="stopped")
        session.flush()
        assert [row.id for row in repository.list_open()] == [open_id]

    def test_tracker_id_may_repeat_within_a_session(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        # §11.2 gives the integer no identity meaning, so a tracker that reissues
        # an ID must not hit a uniqueness constraint. The UUID is the identity.
        _, session_id = camera_and_session
        repository = TrackRepository(session)
        first, second = uuid.uuid4(), uuid.uuid4()
        repository.open(track_id=first, stream_session_id=session_id, tracker_id=1)
        repository.close(track_id=first, termination_reason="lost_timeout")
        repository.open(track_id=second, stream_session_id=session_id, tracker_id=1)
        session.flush()
        rows = repository.list_for_session(session_id)
        assert len(rows) == 2
        assert {row.id for row in rows} == {first, second}


class TestTrackConstraints:
    def test_tracker_id_must_be_positive(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        _, session_id = camera_and_session
        session.add(
            Track(
                id=uuid.uuid4(),
                stream_session_id=session_id,
                tracker_id=0,
                first_seen_at=utc_now(),
                last_seen_at=utc_now(),
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()

    def test_termination_reason_is_constrained(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        _, session_id = camera_and_session
        session.add(
            Track(
                id=uuid.uuid4(),
                stream_session_id=session_id,
                tracker_id=1,
                first_seen_at=utc_now(),
                last_seen_at=utc_now(),
                terminated_at=utc_now(),
                termination_reason="not_a_reason",
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()

    def test_last_seen_must_not_precede_first_seen(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        _, session_id = camera_and_session
        now = utc_now()
        session.add(
            Track(
                id=uuid.uuid4(),
                stream_session_id=session_id,
                tracker_id=1,
                first_seen_at=now,
                last_seen_at=now - timedelta(seconds=1),
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()

    def test_termination_must_not_precede_first_seen(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        _, session_id = camera_and_session
        now = utc_now()
        session.add(
            Track(
                id=uuid.uuid4(),
                stream_session_id=session_id,
                tracker_id=1,
                first_seen_at=now,
                last_seen_at=now,
                terminated_at=now - timedelta(seconds=1),
                termination_reason="stopped",
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()

    def test_session_foreign_key_is_required(self, session: Session) -> None:
        session.add(
            Track(
                id=uuid.uuid4(),
                stream_session_id=uuid.uuid4(),
                tracker_id=1,
                first_seen_at=utc_now(),
                last_seen_at=utc_now(),
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()

    def test_deleting_a_session_cascades_to_its_tracks(
        self, session: Session, camera_and_session: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        # §19: a track has no meaning without its session, so the delete cascades.
        camera_id, session_id = camera_and_session
        TrackRepository(session).open(
            track_id=uuid.uuid4(), stream_session_id=session_id, tracker_id=1
        )
        session.flush()
        camera = session.get(Camera, camera_id)
        assert camera is not None
        session.delete(camera)
        session.flush()
        assert session.scalars(select(Track)).all() == []

    def test_reason_vocabulary_matches_the_contract(self) -> None:
        # The vocabulary lives in ``tracking.contracts`` but the constraint is
        # declared in the database layer, so the two must be kept identical.
        from fall_detection.tracking import TRACK_TERMINATION_REASONS as contract_reasons

        assert contract_reasons == TRACK_TERMINATION_REASONS
