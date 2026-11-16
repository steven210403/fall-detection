"""SQLAlchemy and SQLite smoke tests."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fall_detection.database import Camera


def test_sqlite_runtime_version() -> None:
    assert sqlite3.sqlite_version_info >= (3, 35), sqlite3.sqlite_version


def test_pragmas_are_applied(engine: Engine) -> None:
    with engine.connect() as connection:
        foreign_keys = connection.execute(text("PRAGMA foreign_keys")).scalar_one()
        journal_mode = connection.execute(text("PRAGMA journal_mode")).scalar_one()
        busy_timeout = connection.execute(text("PRAGMA busy_timeout")).scalar_one()

    assert foreign_keys == 1
    assert str(journal_mode).lower() == "wal"
    assert busy_timeout == 5_000


def test_camera_round_trip(session: Session) -> None:
    camera = Camera(
        id=uuid.uuid4(),
        name="Pilot camera",
        source_type="mp4",
        source_ref="fixtures/pilot.mp4",
        created_at=datetime.now(UTC),
    )
    session.add(camera)
    session.commit()

    stored = session.execute(select(Camera)).scalar_one()
    assert stored.id == camera.id
    assert stored.name == "Pilot camera"
    assert isinstance(stored.id, uuid.UUID)


def test_source_type_check_constraint(session: Session) -> None:
    session.add(
        Camera(
            id=uuid.uuid4(),
            name="Bad camera",
            source_type="usb",
            source_ref="0",
            created_at=datetime.now(UTC),
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
