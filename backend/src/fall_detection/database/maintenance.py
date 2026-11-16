"""Data retention and backup (FR-12, §19, §24.1; Phase 12).

FR-12 is "Apply data retention: expired events are deleted; aggregate results follow
research policy". §24.1 proposes *"30 days for application events; research raw data until
three months after the thesis defense, unless the institutional protocol sets another
period"*, and §19 adds the operational rule that a live database must be backed up with
SQLite's backup API *"rather than only copying the .db file"*.

Three properties shape this module:

* **Retention is off by default and deletes nothing until it is configured.** The
  repository's ``delete_before`` has existed since Phase 6 and was deliberately never
  called; enabling a policy that erased research history before the dataset phases were
  finished would have been a data loss, not a cleanup. Phase 12 is where the policy is
  wired, and the switch is still explicit.
* **The cutoff is computed from the observation columns, not the write time.** An event
  ended ``ended_at``; a row written late for an interval observed earlier is retained or
  deleted according to when it *happened* (FR-07's distinction, applied to deletion).
* **A backup is taken with ``sqlite3.Connection.backup`` and then verified.** A copy of a
  WAL-mode database without its ``-wal`` file can be missing the newest transactions, and
  a backup nobody opened is not known to be readable. The verifier runs
  ``PRAGMA integrity_check`` and counts rows in every table on the copy.

Nothing here deletes a *track* or a *session*: those are the context an event
needs to be interpretable, and FR-12 names events.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fall_detection.core.logging import get_logger

__all__ = [
    "BackupResult",
    "RetentionPolicy",
    "RetentionResult",
    "apply_retention",
    "backup_database",
    "sqlite_path_from_url",
    "verify_backup",
]

logger = get_logger("database.maintenance")

_TABLES: tuple[str, ...] = (
    "camera",
    "stream_session",
    "track",
    "fall_event",
)
"""Tables a backup verification counts, so a copy that lost one is visible."""


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """How long application events are kept.

    Attributes:
        enabled: Master switch. ``False`` deletes nothing, whatever the window says.
        event_days: Days an event is kept after it ended. ``None`` means "forever".
        research_note: What the policy deliberately does **not** touch, recorded in every
            result so a report cannot imply that research data was cleaned up.
    """

    enabled: bool = False
    event_days: int | None = 30
    research_note: str = (
        "Research raw data and derived annotations follow the institutional protocol "
        "(§24.1 proposes three months after the defence) and are never touched by this "
        "policy: it deletes closed fall events only."
    )

    def __post_init__(self) -> None:
        if self.event_days is not None and self.event_days < 0:
            raise ValueError("event_days must be >= 0")
        if self.enabled and self.event_days is None:
            raise ValueError(
                "an enabled retention policy needs a window; 'enabled' with no window would "
                "either delete everything or nothing depending on how it is read"
            )

    def cutoff(self, now: datetime | None = None) -> datetime:
        """UTC cutoff: events that ended before it are expired."""
        if self.event_days is None:  # pragma: no cover - refused in __post_init__
            raise ValueError("the policy has no window")
        reference = now or datetime.now(UTC)
        return reference - timedelta(days=self.event_days)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "enabled": self.enabled,
            "event_days": self.event_days,
            "research_note": self.research_note,
        }


@dataclass(frozen=True, slots=True)
class RetentionResult:
    """What one retention pass did.

    Attributes:
        applied: Whether anything was deleted.
        cutoff_utc: The cutoff used, or ``None`` when the policy is disabled.
        deleted_events: Rows removed.
        remaining_events: Rows left, so a reader sees the history was not emptied by
            accident. ``None`` when the policy is disabled: a disabled pass does not touch
            the database at all, so it has no count to report and must not invent one.
        reason: Why nothing happened, when nothing happened.
    """

    applied: bool
    cutoff_utc: datetime | None
    deleted_events: int
    remaining_events: int | None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "applied": self.applied,
            "cutoff_utc": None if self.cutoff_utc is None else self.cutoff_utc.isoformat(),
            "deleted_events": self.deleted_events,
            "remaining_events": self.remaining_events,
            "reason": self.reason,
        }


def sqlite_path_from_url(url: str) -> Path | None:
    """Filesystem path of a SQLite URL, or ``None`` for an in-memory or non-SQLite URL.

    ``sqlite+pysqlite:///./data/app.db`` and ``sqlite:////abs/app.db`` both map to a path;
    ``sqlite:///:memory:`` and any non-SQLite URL do not, and a caller that tried to back
    one up would be copying nothing.
    """
    if not url.startswith("sqlite"):
        return None
    _, _, tail = url.partition("///")
    if not tail or tail == ":memory:":
        return None
    return Path(tail)


def apply_retention(
    writer: Any,
    policy: RetentionPolicy,
    *,
    now: datetime | None = None,
) -> RetentionResult:
    """Delete expired closed events through the single writer (FR-12).

    Args:
        writer: The process's :class:`~fall_detection.database.session.DatabaseWriter`.
        policy: The configured policy. A disabled policy deletes nothing and says so.
        now: Injectable clock, so a test needs no real time.

    Returns:
        What the pass did, including how many events remain.

    Raises:
        ValueError: The policy is enabled without a window (refused at construction).
    """
    from fall_detection.database.repositories import FallEventRepository

    if not policy.enabled:
        # A disabled policy does not open a session. Querying a count would make every
        # startup depend on the schema being present, and a retention pass that is off
        # should be unable to fail.
        return RetentionResult(
            applied=False,
            cutoff_utc=None,
            deleted_events=0,
            remaining_events=None,
            reason="retention is disabled; nothing was deleted and nothing was queried",
        )
    cutoff = policy.cutoff(now)
    # ``delete_before`` compares against ``ended_at``, which stores the *observation* end
    # (FR-07), and the repository converts an aware value to the naive UTC the column
    # holds. Passing the aware cutoff straight through is therefore correct and is why
    # this call does not do its own conversion.
    deleted = writer.write(lambda session: FallEventRepository(session).delete_before(cutoff))
    remaining = writer.read(_count_events)
    logger.info(
        "retention deleted %d events older than %s; %d remain",
        deleted,
        cutoff.isoformat(),
        remaining,
    )
    return RetentionResult(
        applied=True,
        cutoff_utc=cutoff,
        deleted_events=int(deleted),
        remaining_events=int(remaining),
    )


def _count_events(session: Any) -> int:
    """Number of stored fall events."""
    from sqlalchemy import func, select

    from fall_detection.database.models import FallEvent

    return int(session.scalar(select(func.count()).select_from(FallEvent)) or 0)


@dataclass(frozen=True, slots=True)
class BackupResult:
    """Outcome of one backup, including the verification.

    Attributes:
        source: Path the backup was taken from.
        destination: Path written.
        size_bytes: Size of the written file.
        tables: Row counts per table, read from the *copy*.
        integrity_ok: Whether ``PRAGMA integrity_check`` returned ``ok`` on the copy.
        verified: Whether the copy opened, passed integrity, and contained every table.
        reason: Why the backup was not taken or not verified, when it was not.
    """

    source: Path
    destination: Path | None
    size_bytes: int | None
    tables: dict[str, int]
    integrity_ok: bool
    verified: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "source": str(self.source),
            "destination": None if self.destination is None else str(self.destination),
            "size_bytes": self.size_bytes,
            "tables": dict(self.tables),
            "integrity_ok": self.integrity_ok,
            "verified": self.verified,
            "reason": self.reason,
        }


def backup_database(source: Path, destination: Path) -> BackupResult:
    """Back up a live SQLite database with the backup API, then verify the copy.

    §19 requires the backup API rather than a file copy: a WAL-mode database whose ``-wal``
    file is not copied can be missing its newest transactions, and a byte copy of a file
    being written is not a consistent snapshot.

    Args:
        source: The database file. Missing or empty sources are refused, not created.
        destination: Where to write the copy. Parent directories are created.

    Returns:
        The result, with ``verified`` false and a reason when the copy is unusable.
    """
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        return BackupResult(
            source=source,
            destination=None,
            size_bytes=None,
            tables={},
            integrity_ok=False,
            verified=False,
            reason=f"source database does not exist: {source}",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    # Both connections are plain sqlite3: the backup API is a SQLite feature, and going
    # through SQLAlchemy would add a pool and a dialect for no benefit.
    with (
        sqlite3.connect(f"file:{source}?mode=ro", uri=True) as origin,
        sqlite3.connect(destination) as copy,
    ):
        origin.backup(copy)
    verification = verify_backup(destination)
    return BackupResult(
        source=source,
        destination=destination,
        size_bytes=destination.stat().st_size,
        tables=verification["tables"],
        integrity_ok=verification["integrity_ok"],
        verified=verification["verified"],
        reason=verification["reason"],
    )


def verify_backup(path: Path) -> dict[str, Any]:
    """Open a backup, run ``integrity_check``, and count every table's rows.

    A backup is only useful if it opens. This is the check that turns "the file exists"
    into "the file is a database with the expected tables".
    """
    path = Path(path)
    tables: dict[str, int] = {}
    if not path.is_file():
        return {
            "tables": tables,
            "integrity_ok": False,
            "verified": False,
            "reason": f"no file at {path}",
        }
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            integrity_ok = bool(integrity) and str(integrity[0]).lower() == "ok"
            present = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            for table in _TABLES:
                if table in present:
                    tables[table] = int(
                        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608 - table names are this module's own constant
                    )
    except sqlite3.DatabaseError as exc:
        return {
            "tables": tables,
            "integrity_ok": False,
            "verified": False,
            "reason": f"the copy is not a readable database: {exc}",
        }
    missing = sorted(set(_TABLES) - set(tables))
    if not integrity_ok:
        return {
            "tables": tables,
            "integrity_ok": False,
            "verified": False,
            "reason": "PRAGMA integrity_check did not return 'ok'",
        }
    if missing:
        return {
            "tables": tables,
            "integrity_ok": True,
            "verified": False,
            "reason": f"the copy is missing tables: {missing}",
        }
    return {"tables": tables, "integrity_ok": True, "verified": True, "reason": None}


def policy_from_settings(settings: Any) -> RetentionPolicy:
    """Build the policy from configuration, so the switch has one definition."""
    retention = settings.retention
    return RetentionPolicy(
        enabled=bool(retention.enabled),
        event_days=retention.event_days,
    )


def describe_retention(policy: RetentionPolicy) -> Mapping[str, Any]:
    """Status-payload view of the policy, for ``GET /api/v1/status``."""
    return {
        **policy.as_dict(),
        "note": (
            "FR-12: expired, closed fall events are deleted; open alarms are never deleted, "
            "and tracks or sessions are kept because an event needs them to be interpretable"
        ),
    }
