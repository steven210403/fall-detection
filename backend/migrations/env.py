"""Alembic environment.

The database URL is read from :class:`fall_detection.config.Settings` so the
application and migrations never disagree about the target database.

The engine is built by :func:`fall_detection.database.create_engine` rather than
Alembic's ``engine_from_config`` so a migration gets the same SQLite behaviour
as the application: the parent directory is created when missing, and the WAL,
foreign-key, and busy-timeout pragmas are applied. Building a second, differently
configured engine here would let a migration succeed against a connection that
does not behave like the one the application actually uses.
"""

from __future__ import annotations

import contextlib
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection

from fall_detection.config import get_settings
from fall_detection.database import Base, create_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

database_url = get_settings().database_url
config.set_main_option("sqlalchemy.url", database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database."""
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _require_foreign_key_integrity(connection: Connection) -> None:
    """Fail the migration if it left a dangling reference while keys were off."""
    violations = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(f"migration left foreign-key violations: {violations[:5]}")


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            sqlite = connection.dialect.name == "sqlite"
            if sqlite:
                # A batch-mode migration rebuilds a table by dropping and recreating
                # it. With foreign keys on, dropping a parent table (``camera``)
                # would cascade-delete every child row (sessions, tracks, events).
                # The pragma cannot change inside a transaction, so it is set first
                # and integrity is checked before the migration commits.
                connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
                connection.commit()
            try:
                context.configure(
                    connection=connection,
                    target_metadata=target_metadata,
                    render_as_batch=True,
                )
                with context.begin_transaction():
                    context.run_migrations()
                    if sqlite:
                        _require_foreign_key_integrity(connection)
            finally:
                if sqlite:
                    connection.exec_driver_sql("PRAGMA foreign_keys=ON")
                    connection.commit()
    finally:
        with contextlib.suppress(Exception):
            engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
