"""Alembic migration tests.

Migrations are executed in a subprocess because Alembic's ``env.py`` owns the
connection lifecycle. The tests assert that upgrade, downgrade, and
autogenerate-detection all work against the real SQLite configuration, and that
the migrated schema matches the ORM metadata (drift between the two would make
``create_all`` and ``alembic upgrade`` produce different databases).
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


def run_alembic(*args: str, database_url: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "FALL_DATABASE_URL": database_url,
        "PYTHONPATH": str(BACKEND / "src"),
    }
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def alembic_url(tmp_path: Path) -> str:
    return f"sqlite+pysqlite:///{tmp_path / 'alembic.db'}"


@pytest.fixture
def alembic_db(tmp_path: Path) -> Path:
    return tmp_path / "alembic.db"


def test_upgrade_head(alembic_url: str) -> None:
    result = run_alembic("upgrade", "head", database_url=alembic_url)
    assert result.returncode == 0, result.stderr


def test_downgrade_base(alembic_url: str) -> None:
    upgrade = run_alembic("upgrade", "head", database_url=alembic_url)
    assert upgrade.returncode == 0, upgrade.stderr

    downgrade = run_alembic("downgrade", "base", database_url=alembic_url)
    assert downgrade.returncode == 0, downgrade.stderr


def _tables(alembic_db: Path) -> set[str]:
    with sqlite3.connect(alembic_db) as connection:
        return {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }


def test_downgrade_to_track_removes_only_fall_event(alembic_url: str, alembic_db: Path) -> None:
    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    assert run_alembic("downgrade", "0003_track", database_url=alembic_url).returncode == 0
    tables = _tables(alembic_db)
    assert "fall_event" not in tables
    assert {"camera", "stream_session", "track"} <= tables


def test_downgrade_to_stream_session_removes_track(alembic_url: str, alembic_db: Path) -> None:
    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    assert run_alembic("downgrade", "0002_stream_session", database_url=alembic_url).returncode == 0
    tables = _tables(alembic_db)
    assert "track" not in tables
    assert {"camera", "stream_session"} <= tables


def test_no_pending_autogenerate_diff(alembic_url: str) -> None:
    upgrade = run_alembic("upgrade", "head", database_url=alembic_url)
    assert upgrade.returncode == 0, upgrade.stderr

    check = run_alembic("check", database_url=alembic_url)
    assert check.returncode == 0, check.stdout + check.stderr


def test_migrated_schema_contains_all_tables(alembic_url: str, alembic_db: Path) -> None:
    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    tables = _tables(alembic_db)
    assert {"camera", "stream_session", "track", "fall_event", "alembic_version"} <= tables
    # The office-activity tables are gone.
    assert not {"zone", "track_zone_assignment", "activity_event"} & tables


def test_fall_event_foreign_key_cascades_from_track(alembic_url: str, alembic_db: Path) -> None:
    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    with sqlite3.connect(alembic_db) as connection:
        foreign_keys = connection.execute("PRAGMA foreign_key_list('fall_event')").fetchall()
    referenced = {row[2]: (row[3], row[6]) for row in foreign_keys}
    assert referenced["track"] == ("track_id", "CASCADE")


def _seed_track(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        "INSERT INTO camera (id, name, source_type, source_ref, created_at) "
        "VALUES ('c1', 'c', 'mp4', 'a.mp4', '2026-01-01T00:00:00')"
    )
    connection.execute(
        "INSERT INTO stream_session (id, camera_id, attempt, started_at, frames_read, "
        "frames_dropped) VALUES ('s1', 'c1', 1, '2026-01-01T00:00:00', 0, 0)"
    )
    connection.execute(
        "INSERT INTO track (id, stream_session_id, tracker_id, first_seen_at, "
        "last_seen_at, observed_frames, missing_frames) "
        "VALUES ('t1', 's1', 1, '2026-01-01T00:00:00', '2026-01-01T00:00:00', 1, 0)"
    )


_INSERT_EVENT = (
    "INSERT INTO fall_event (id, track_id, falling_at, impact_at, alert_at, ended_at, "
    "end_reason, created_at, response_time_s, schema_version) "
    "VALUES (?, 't1', ?, ?, ?, ?, ?, ?, 3.0, 1)"
)


def test_fall_event_end_reason_is_constrained(alembic_url: str, alembic_db: Path) -> None:
    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    with sqlite3.connect(alembic_db) as connection:
        _seed_track(connection)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                _INSERT_EVENT,
                (
                    "e1",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:01",
                    "2026-01-01 00:00:04",
                    "2026-01-01 00:00:09",
                    "because",
                    "2026-01-01 00:00:04",
                ),
            )


def test_an_open_fall_event_cannot_carry_half_a_closure(alembic_url: str, alembic_db: Path) -> None:
    # An event is either open (no end, no reason) or fully closed (both).
    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    with sqlite3.connect(alembic_db) as connection:
        _seed_track(connection)
        connection.execute(
            _INSERT_EVENT,
            (
                "open",
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:01",
                "2026-01-01 00:00:04",
                None,
                None,
                "2026-01-01 00:00:04",
            ),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                _INSERT_EVENT,
                (
                    "half",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:01",
                    "2026-01-01 00:00:04",
                    "2026-01-01 00:00:09",
                    None,
                    "2026-01-01 00:00:04",
                ),
            )


def test_fall_event_times_are_ordered(alembic_url: str, alembic_db: Path) -> None:
    # falling <= impact <= alert: an alarm cannot precede the fall it reports.
    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    with sqlite3.connect(alembic_db) as connection:
        _seed_track(connection)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                _INSERT_EVENT,
                (
                    "e2",
                    "2026-01-01 00:00:05",
                    "2026-01-01 00:00:01",
                    "2026-01-01 00:00:04",
                    None,
                    None,
                    "2026-01-01 00:00:04",
                ),
            )


def test_migrated_foreign_key_is_declared(alembic_url: str, alembic_db: Path) -> None:
    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    with sqlite3.connect(alembic_db) as connection:
        foreign_keys = connection.execute("PRAGMA foreign_key_list('stream_session')").fetchall()
    assert len(foreign_keys) == 1
    # columns: id, seq, table, from, to, on_update, on_delete, match
    assert foreign_keys[0][2] == "camera"
    assert foreign_keys[0][3] == "camera_id"
    assert foreign_keys[0][6] == "CASCADE"


def test_track_foreign_key_is_declared(alembic_url: str, alembic_db: Path) -> None:
    # §19: a track belongs to one stream session, and deleting the session must
    # take its tracks with it.
    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    with sqlite3.connect(alembic_db) as connection:
        foreign_keys = connection.execute("PRAGMA foreign_key_list('track')").fetchall()
    assert len(foreign_keys) == 1
    assert foreign_keys[0][2] == "stream_session"
    assert foreign_keys[0][3] == "stream_session_id"
    assert foreign_keys[0][6] == "CASCADE"


def test_orm_metadata_matches_migration_head(alembic_url: str, tmp_path: Path) -> None:
    """The migrated schema and ``Base.metadata`` must not drift apart."""
    orm_db = tmp_path / "orm.db"
    script = (
        "from fall_detection.database import Base, create_engine;"
        f"engine = create_engine('sqlite+pysqlite:///{orm_db}');"
        "Base.metadata.create_all(engine)"
    )
    env = {**os.environ, "PYTHONPATH": str(BACKEND / "src")}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    migrated_db = Path(alembic_url.split("///", 1)[1])

    def schema(path: Path) -> set[str]:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                "SELECT type, name, tbl_name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
            )
            return {f"{kind}:{name}:{table}" for kind, name, table in rows}

    assert schema(orm_db) == schema(migrated_db)


def test_webcam_source_type_is_allowed(alembic_url: str, alembic_db: Path) -> None:
    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    with sqlite3.connect(alembic_db) as connection:
        connection.execute(
            "INSERT INTO camera (id, name, source_type, source_ref, created_at) "
            "VALUES ('w1', 'Webcam 0', 'webcam', 'avfoundation:0', '2026-01-01T00:00:00')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO camera (id, name, source_type, source_ref, created_at) "
                "VALUES ('u1', 'u', 'usb', 'x', '2026-01-01T00:00:00')"
            )


def test_webcam_migration_keeps_existing_history(alembic_url: str, alembic_db: Path) -> None:
    # Rebuilding ``camera`` must not cascade-delete the sessions, tracks and events below it.
    assert run_alembic("upgrade", "0004_fall_event", database_url=alembic_url).returncode == 0
    with sqlite3.connect(alembic_db) as connection:
        _seed_track(connection)
    upgrade = run_alembic("upgrade", "head", database_url=alembic_url)
    assert upgrade.returncode == 0, upgrade.stderr
    with sqlite3.connect(alembic_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM stream_session").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM track").fetchone() == (1,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_webcam_downgrade_refuses_to_drop_webcam_rows(alembic_url: str, alembic_db: Path) -> None:
    assert run_alembic("upgrade", "head", database_url=alembic_url).returncode == 0
    with sqlite3.connect(alembic_db) as connection:
        connection.execute(
            "INSERT INTO camera (id, name, source_type, source_ref, created_at) "
            "VALUES ('w1', 'Webcam 0', 'webcam', 'avfoundation:0', '2026-01-01T00:00:00')"
        )
    downgrade = run_alembic("downgrade", "0004_fall_event", database_url=alembic_url)
    assert downgrade.returncode != 0
    assert "webcam" in downgrade.stderr
