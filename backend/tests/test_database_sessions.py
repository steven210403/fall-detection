"""Stream-session persistence tests (thesis plan §19, §22)."""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from fall_detection.database import (
    Camera,
    CameraRepository,
    DatabaseWriter,
    StreamSession,
    StreamSessionRepository,
    create_session_factory,
    utc_now,
)


def make_camera(session: Session, *, name: str = "camera") -> Camera:
    camera = Camera(
        id=uuid.uuid4(),
        name=name,
        source_type="mp4",
        source_ref="fixtures/pilot.mp4",
        created_at=utc_now(),
    )
    session.add(camera)
    session.commit()
    return camera


class TestCameraRepository:
    def test_get_or_create_is_idempotent(self, session: Session) -> None:
        repo = CameraRepository(session)
        camera_id = uuid.uuid4()
        first = repo.get_or_create(
            camera_id=camera_id, name="front", source_type="mp4", source_ref="a.mp4"
        )
        session.commit()
        second = repo.get_or_create(
            camera_id=camera_id, name="other", source_type="rtsp", source_ref="rtsp://h/s"
        )
        assert first.id == second.id
        assert second.name == "front"  # an existing row is never overwritten
        assert len(repo.list_all()) == 1

    def test_get_returns_none_for_unknown_id(self, session: Session) -> None:
        assert CameraRepository(session).get(uuid.uuid4()) is None


class TestStreamSessionRepository:
    def test_open_uses_the_supplied_uuid(self, session: Session) -> None:
        camera = make_camera(session)
        session_id = uuid.uuid4()
        returned = StreamSessionRepository(session).open(
            camera_id=camera.id, attempt=1, stream_session_id=session_id
        )
        session.commit()
        assert returned == session_id
        stored = session.get(StreamSession, session_id)
        assert stored is not None
        assert stored.ended_at is None

    def test_close_records_reason_and_counters(self, session: Session) -> None:
        camera = make_camera(session)
        repo = StreamSessionRepository(session)
        session_id = repo.open(camera_id=camera.id, attempt=2)
        session.commit()

        repo.close(
            stream_session_id=session_id,
            termination_reason="decoder_error",
            frames_read=120,
            frames_dropped=7,
        )
        session.commit()

        stored = session.get(StreamSession, session_id)
        assert stored is not None
        assert stored.termination_reason == "decoder_error"
        assert stored.frames_read == 120
        assert stored.frames_dropped == 7
        assert stored.ended_at is not None
        assert stored.attempt == 2

    def test_list_open_finds_unclosed_sessions(self, session: Session) -> None:
        camera = make_camera(session)
        repo = StreamSessionRepository(session)
        first = repo.open(camera_id=camera.id, attempt=1)
        second = repo.open(camera_id=camera.id, attempt=2)
        repo.close(stream_session_id=first, termination_reason="stopped")
        session.commit()

        open_ids = {row.id for row in repo.list_open()}
        assert open_ids == {second}

    def test_close_orphans_recovers_after_a_crash(self, session: Session) -> None:
        camera = make_camera(session)
        repo = StreamSessionRepository(session)
        repo.open(camera_id=camera.id, attempt=1)
        repo.open(camera_id=camera.id, attempt=2)
        session.commit()

        closed = repo.close_orphans(reason="shutdown")
        session.commit()

        assert closed == 2
        assert repo.list_open() == []
        rows = session.scalars(select(StreamSession)).all()
        assert {row.termination_reason for row in rows} == {"shutdown"}

    def test_close_orphans_is_a_no_op_without_orphans(self, session: Session) -> None:
        make_camera(session)
        assert StreamSessionRepository(session).close_orphans() == 0


class TestConstraints:
    def test_foreign_key_cascades_from_camera(self, session: Session) -> None:
        camera = make_camera(session)
        StreamSessionRepository(session).open(camera_id=camera.id, attempt=1)
        session.commit()

        session.delete(camera)
        session.commit()
        assert session.scalars(select(StreamSession)).all() == []

    def test_session_requires_an_existing_camera(self, session: Session) -> None:
        session.add(
            StreamSession(
                id=uuid.uuid4(),
                camera_id=uuid.uuid4(),
                attempt=1,
                started_at=utc_now(),
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

    def test_attempt_must_be_positive(self, session: Session) -> None:
        camera = make_camera(session)
        session.add(
            StreamSession(
                id=uuid.uuid4(),
                camera_id=camera.id,
                attempt=0,
                started_at=utc_now(),
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

    def test_unknown_termination_reason_is_rejected(self, session: Session) -> None:
        camera = make_camera(session)
        session.add(
            StreamSession(
                id=uuid.uuid4(),
                camera_id=camera.id,
                attempt=1,
                started_at=utc_now(),
                ended_at=utc_now(),
                termination_reason="melted",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

    def test_end_before_start_is_rejected(self, session: Session) -> None:
        camera = make_camera(session)
        now = datetime.now(UTC)
        session.add(
            StreamSession(
                id=uuid.uuid4(),
                camera_id=camera.id,
                attempt=1,
                started_at=now,
                ended_at=now - timedelta(seconds=1),
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


class TestDatabaseWriter:
    def test_write_commits_and_returns(self, engine: Engine) -> None:
        writer = DatabaseWriter(create_session_factory(engine))
        camera_id = writer.write(
            lambda session: (
                CameraRepository(session)
                .get_or_create(
                    camera_id=uuid.uuid4(), name="c", source_type="mp4", source_ref="x.mp4"
                )
                .id
            )
        )
        assert writer.read(lambda session: CameraRepository(session).get(camera_id)) is not None

    def test_write_rolls_back_on_error(self, engine: Engine) -> None:
        writer = DatabaseWriter(create_session_factory(engine))

        def failing_work(session: Session) -> None:
            CameraRepository(session).get_or_create(
                camera_id=uuid.uuid4(), name="c", source_type="mp4", source_ref="x.mp4"
            )
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            writer.write(failing_work)
        assert writer.read(lambda session: CameraRepository(session).list_all()) == []

    def test_writer_is_reusable_after_a_failure(self, engine: Engine) -> None:
        writer = DatabaseWriter(create_session_factory(engine))

        def failing_work(session: Session) -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            writer.write(failing_work)
        assert writer.write(lambda session: 7) == 7

    def test_writes_are_serialised(self, engine: Engine) -> None:
        # «One database writer path» (§17): concurrent writers must never
        # interleave their transactions.
        writer = DatabaseWriter(create_session_factory(engine))
        active = 0
        max_active = 0
        guard = threading.Lock()

        def work(session: Session) -> None:
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            try:
                session.execute(select(1))
            finally:
                with guard:
                    active -= 1

        threads = [threading.Thread(target=writer.write, args=(work,)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        assert max_active == 1

    def test_read_is_not_blocked_by_the_write_lock(self, engine: Engine) -> None:
        # WAL allows concurrent readers, so a status query must succeed while a
        # write transaction holds the writer lock (§19).
        writer = DatabaseWriter(create_session_factory(engine))
        holder = threading.Event()
        release = threading.Event()

        def slow_write(session: Session) -> None:
            holder.set()
            release.wait(timeout=5.0)

        thread = threading.Thread(target=writer.write, args=(slow_write,))
        thread.start()
        assert holder.wait(timeout=5.0)
        try:
            # The write lock is held, yet this read returns immediately.
            assert writer.read(lambda session: CameraRepository(session).list_all()) == []
        finally:
            release.set()
            thread.join(timeout=5.0)


class TestModuleBoundaries:
    def test_queue_module_stays_free_of_orm_and_cv_dependencies(self) -> None:
        # §17.1: CV modules exchange dataclasses and never touch ORM models, and
        # the bounded queue also backs client snapshot queues, so it must stay
        # generic. Asserting on the source is the only way to check an *absence*
        # of imports, which no runtime assertion can detect.
        import fall_detection.video.queue as queue_module

        assert queue_module.__file__ is not None
        source = Path(queue_module.__file__).read_text(encoding="utf-8")
        assert "sqlalchemy" not in source
        assert "fall_detection.database" not in source
        assert "import numpy" not in source

    def test_session_factory_keeps_instances_alive(self, sqlite_url: str) -> None:
        factory: sessionmaker[Session] = create_session_factory(engine_for(sqlite_url))
        assert factory.kw["expire_on_commit"] is False


def engine_for(url: str) -> Engine:
    from fall_detection.database import create_engine

    return create_engine(url)
