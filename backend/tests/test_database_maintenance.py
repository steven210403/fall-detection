"""FR-12 retention and §19 backup (Phase 12).

The assertions are about the *policy's arithmetic and its safety*, not about how many
events happen to be in a fixture:

* a disabled policy deletes nothing and says so;
* an enabled one deletes only events whose **observed** end precedes the cutoff, so a row
  written late for an earlier interval is judged by when it happened (FR-07);
* an enabled policy with no window is refused at construction, because "enabled" plus "no
  window" has two incompatible readings;
* a backup is taken with the SQLite backup API, verified by opening the copy, running
  ``integrity_check``, and counting every table — a file that exists but does not open is
  not a backup.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fall_detection.database import Base, DatabaseWriter, create_engine, create_session_factory
from fall_detection.database.maintenance import (
    RetentionPolicy,
    apply_retention,
    backup_database,
    describe_retention,
    policy_from_settings,
    sqlite_path_from_url,
    verify_backup,
)
from fall_detection.database.models import Camera, FallEvent, StreamSession, Track


@pytest.fixture
def writer(tmp_path: Path) -> DatabaseWriter:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'app.db'}")
    Base.metadata.create_all(engine)
    return DatabaseWriter(create_session_factory(engine))


def _seed_event(
    writer: DatabaseWriter,
    *,
    ended_days_ago: float,
    end_reason: str = "recovered",
) -> uuid.UUID:
    """Insert one closed event whose observation end is ``ended_days_ago`` days ago."""
    camera_id = uuid.uuid4()
    session_id = uuid.uuid4()
    track_id = uuid.uuid4()
    ended = datetime.now(UTC) - timedelta(days=ended_days_ago)

    def work(session):  # type: ignore[no-untyped-def]
        session.add(
            Camera(
                id=camera_id,
                name=f"cam-{camera_id.hex[:6]}",
                source_type="mp4",
                source_ref="datasets/synthetic/pilot.mp4",
                created_at=ended - timedelta(minutes=10),
            )
        )
        session.flush()
        session.add(
            StreamSession(
                id=session_id,
                camera_id=camera_id,
                attempt=1,
                started_at=ended - timedelta(minutes=5),
                ended_at=ended,
            )
        )
        session.flush()
        session.add(
            Track(
                id=track_id,
                stream_session_id=session_id,
                tracker_id=1,
                first_seen_at=ended - timedelta(minutes=5),
                last_seen_at=ended,
            )
        )
        session.flush()
        session.add(
            FallEvent(
                id=uuid.uuid4(),
                track_id=track_id,
                falling_at=ended - timedelta(seconds=30),
                impact_at=ended - timedelta(seconds=29),
                alert_at=ended - timedelta(seconds=26),
                ended_at=ended,
                end_reason=end_reason,
                created_at=ended - timedelta(seconds=26),
                schema_version=1,
            )
        )

    writer.write(work)
    return track_id


def _url_of(writer: DatabaseWriter) -> str:
    """The engine URL a fixture's writer is bound to, for the backup tests."""
    return str(writer._session_factory.kw["bind"].url)  # type: ignore[attr-defined]


def _event_count(writer: DatabaseWriter) -> int:
    from sqlalchemy import func, select

    return int(
        writer.read(
            lambda session: session.scalar(select(func.count()).select_from(FallEvent)) or 0
        )
    )


class TestRetentionPolicy:
    def test_the_default_is_off(self) -> None:
        assert RetentionPolicy().enabled is False

    def test_an_enabled_policy_without_a_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs a window"):
            RetentionPolicy(enabled=True, event_days=None)

    def test_a_negative_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            RetentionPolicy(event_days=-1)

    def test_the_cutoff_is_the_window_before_now(self) -> None:
        policy = RetentionPolicy(enabled=True, event_days=30)
        now = datetime(2026, 10, 27, tzinfo=UTC)
        assert policy.cutoff(now) == now - timedelta(days=30)

    def test_the_payload_names_what_it_does_not_touch(self) -> None:
        payload = describe_retention(RetentionPolicy())
        assert "research" in payload["research_note"].lower()
        assert "tracks or sessions" in payload["note"]

    def test_the_policy_reads_from_settings(self) -> None:
        from fall_detection.config import Settings

        settings = Settings.model_validate({"retention": {"enabled": True, "event_days": 7}})
        policy = policy_from_settings(settings)
        assert policy.enabled is True
        assert policy.event_days == 7

    def test_settings_refuse_an_enabled_policy_without_a_window(self) -> None:
        from fall_detection.config import Settings

        with pytest.raises(ValueError, match="event_days"):
            Settings.model_validate({"retention": {"enabled": True, "event_days": None}})


class TestApplyRetention:
    def test_a_disabled_policy_deletes_nothing(self, writer: DatabaseWriter) -> None:
        _seed_event(writer, ended_days_ago=400)
        result = apply_retention(writer, RetentionPolicy(enabled=False))
        assert result.applied is False
        assert result.deleted_events == 0
        # A disabled policy does not query either, so it has no count to report.
        assert result.remaining_events is None
        assert "disabled" in (result.reason or "")
        assert _event_count(writer) == 1

    def test_a_disabled_policy_survives_a_missing_schema(self, tmp_path: Path) -> None:
        # This is the startup case: the application records a missing schema as degraded
        # readiness rather than raising, so a disabled retention pass must not fail either.
        from fall_detection.database import create_engine, create_session_factory

        engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'empty.db'}")
        writer = DatabaseWriter(create_session_factory(engine))
        result = apply_retention(writer, RetentionPolicy(enabled=False))
        assert result.applied is False
        assert result.remaining_events is None
        engine.dispose()

    def test_an_expired_event_is_deleted(self, writer: DatabaseWriter) -> None:
        _seed_event(writer, ended_days_ago=40)
        result = apply_retention(writer, RetentionPolicy(enabled=True, event_days=30))
        assert result.applied is True
        assert result.deleted_events == 1
        assert result.remaining_events == 0

    def test_a_recent_event_is_kept(self, writer: DatabaseWriter) -> None:
        _seed_event(writer, ended_days_ago=1)
        result = apply_retention(writer, RetentionPolicy(enabled=True, event_days=30))
        assert result.deleted_events == 0
        assert result.remaining_events == 1

    def test_only_the_expired_event_is_removed(self, writer: DatabaseWriter) -> None:
        _seed_event(writer, ended_days_ago=400)
        _seed_event(writer, ended_days_ago=2)
        result = apply_retention(writer, RetentionPolicy(enabled=True, event_days=30))
        assert result.deleted_events == 1
        assert result.remaining_events == 1

    def test_the_cutoff_is_reported(self, writer: DatabaseWriter) -> None:
        now = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
        result = apply_retention(writer, RetentionPolicy(enabled=True, event_days=30), now=now)
        assert result.cutoff_utc == now - timedelta(days=30)

    def test_a_zero_day_window_deletes_every_closed_event(self, writer: DatabaseWriter) -> None:
        _seed_event(writer, ended_days_ago=0.001)
        result = apply_retention(writer, RetentionPolicy(enabled=True, event_days=0))
        assert result.deleted_events == 1

    def test_retention_keeps_tracks_and_sessions(self, writer: DatabaseWriter) -> None:
        from sqlalchemy import func, select

        _seed_event(writer, ended_days_ago=400)
        apply_retention(writer, RetentionPolicy(enabled=True, event_days=30))
        tracks = writer.read(
            lambda session: session.scalar(select(func.count()).select_from(Track)) or 0
        )
        assert int(tracks) == 1
        sessions = writer.read(
            lambda session: session.scalar(select(func.count()).select_from(StreamSession)) or 0
        )
        assert int(sessions) == 1


class TestSqlitePath:
    def test_a_relative_url_maps_to_a_path(self) -> None:
        assert sqlite_path_from_url("sqlite+pysqlite:///./data/app.db") == Path("./data/app.db")

    def test_an_absolute_url_maps_to_a_path(self) -> None:
        assert sqlite_path_from_url("sqlite:////tmp/app.db") == Path("/tmp/app.db")

    def test_memory_is_not_a_file(self) -> None:
        assert sqlite_path_from_url("sqlite:///:memory:") is None

    def test_a_non_sqlite_url_is_not_a_file(self) -> None:
        assert sqlite_path_from_url("postgresql://host/db") is None


class TestBackup:
    def test_a_backup_is_verified_by_opening_it(
        self, writer: DatabaseWriter, tmp_path: Path
    ) -> None:
        _seed_event(writer, ended_days_ago=1)
        source = sqlite_path_from_url(_url_of(writer))
        assert source is not None
        result = backup_database(source, tmp_path / "backup" / "app.db")
        assert result.verified is True
        assert result.integrity_ok is True
        assert result.tables["fall_event"] == 1
        assert result.size_bytes and result.size_bytes > 0

    def test_a_missing_source_is_refused_not_created(self, tmp_path: Path) -> None:
        result = backup_database(tmp_path / "absent.db", tmp_path / "copy.db")
        assert result.verified is False
        assert result.destination is None
        assert "does not exist" in (result.reason or "")
        assert not (tmp_path / "copy.db").exists()

    def test_a_backup_includes_wal_writes(self, writer: DatabaseWriter, tmp_path: Path) -> None:
        # WAL is on, so the newest transaction lives in the -wal file until a checkpoint.
        # A byte copy without it would miss the row; the backup API must not.
        _seed_event(writer, ended_days_ago=1)
        source = sqlite_path_from_url(_url_of(writer))
        assert source is not None
        destination = tmp_path / "copy.db"
        result = backup_database(source, destination)
        with sqlite3.connect(destination) as connection:
            count = connection.execute("SELECT COUNT(*) FROM fall_event").fetchone()[0]
        assert count == 1
        assert result.tables["fall_event"] == 1

    def test_an_existing_destination_is_replaced(
        self, writer: DatabaseWriter, tmp_path: Path
    ) -> None:
        _seed_event(writer, ended_days_ago=1)
        source = sqlite_path_from_url(_url_of(writer))
        assert source is not None
        destination = tmp_path / "copy.db"
        destination.write_text("not a database")
        result = backup_database(source, destination)
        assert result.verified is True

    def test_verification_reports_a_missing_file(self, tmp_path: Path) -> None:
        payload = verify_backup(tmp_path / "nothing.db")
        assert payload["verified"] is False
        assert "no file" in payload["reason"]

    def test_verification_reports_a_non_database(self, tmp_path: Path) -> None:
        path = tmp_path / "garbage.db"
        path.write_bytes(b"this is not a sqlite file" * 10)
        payload = verify_backup(path)
        assert payload["verified"] is False
        assert payload["reason"] is not None

    def test_verification_reports_missing_tables(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.db"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE camera (id TEXT)")
        payload = verify_backup(path)
        assert payload["integrity_ok"] is True
        assert payload["verified"] is False
        assert "missing tables" in payload["reason"]
