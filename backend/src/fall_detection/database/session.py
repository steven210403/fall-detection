"""Engine and session construction.

SQLite is configured with WAL, foreign keys, and a busy timeout as required by
thesis plan §19. Sessions must not be shared across threads.

:class:`DatabaseWriter` implements the "one database writer path" rule from
thesis plan §17: the inference thread never touches the ORM directly, it hands a
callable to the writer, which serialises short transactions behind a lock.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from sqlalchemy import Engine, event
from sqlalchemy import create_engine as _sa_create_engine
from sqlalchemy.orm import Session, sessionmaker

T = TypeVar("T")

__all__ = [
    "DatabaseWriter",
    "create_engine",
    "create_session_factory",
]


def _enable_sqlite_pragmas(dbapi_connection: object, _record: object) -> None:
    """Apply per-connection SQLite pragmas."""
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()


def create_engine(url: str, *, echo: bool = False) -> Engine:
    """Create an engine with the project's SQLite pragmas when applicable."""
    if url.startswith("sqlite"):
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    engine = _sa_create_engine(url, echo=echo, future=True)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_pragmas)
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


class DatabaseWriter:
    """Serialise all writes through one lock and one short transaction.

    The lock guarantees a single writer, which removes SQLite ``database is
    locked`` contention without raising the busy timeout (thesis plan §19). Each
    call opens a fresh session, so a failed unit of work cannot poison the next
    one, and the session is never shared across threads.

    Args:
        session_factory: Factory created by :func:`create_session_factory`.
        lock: Optional lock, injectable for tests that assert serialisation.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        lock: threading.Lock | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._lock = lock or threading.Lock()

    def write(self, work: Callable[[Session], T]) -> T:
        """Run ``work`` in a transaction and commit, returning its result.

        The callable receives a session and must not commit or roll back itself;
        the writer owns the transaction boundary.
        """
        with self._lock, self._session_factory() as session:
            try:
                result = work(session)
                session.commit()
            except Exception:
                session.rollback()
                raise
            return result

    def read(self, work: Callable[[Session], T]) -> T:
        """Run ``work`` read-only without committing.

        Reads do not take the write lock, so a status query is not blocked by an
        in-flight event write. SQLite WAL permits concurrent readers.
        """
        with self._session_factory() as session:
            return work(session)
