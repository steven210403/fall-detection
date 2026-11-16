"""Add stream_session table.

Revision ID: 0002_stream_session
Revises: 0001_initial
Create Date: 2026-10-14

Phase 1 adds the session entity that every later track and activity event
attaches to (thesis plan §19). A reconnect opens a new row, which is how the
tracker and activity buffers know to reset.

The table is created with ``batch_alter_table`` disabled because it is a pure
create; SQLite needs no table rebuild for a new table with a foreign key.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_stream_session"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TERMINATION_REASONS = ("stopped", "source_ended", "decoder_error", "source_invalid", "shutdown")


def upgrade() -> None:
    op.create_table(
        "stream_session",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("camera_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("termination_reason", sa.String(length=32), nullable=True),
        sa.Column("frames_read", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("frames_dropped", sa.Integer(), nullable=False, server_default="0"),
        sa.CheckConstraint(
            "termination_reason IS NULL OR termination_reason IN ("
            + ", ".join(f"'{reason}'" for reason in TERMINATION_REASONS)
            + ")",
            name=op.f("ck_stream_session_termination_reason_allowed"),
        ),
        sa.CheckConstraint("attempt >= 1", name=op.f("ck_stream_session_attempt_positive")),
        sa.CheckConstraint(
            "ended_at IS NULL OR ended_at >= started_at",
            name=op.f("ck_stream_session_ended_after_started"),
        ),
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["camera.id"],
            name=op.f("fk_stream_session_camera_id_camera"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_stream_session")),
    )
    op.create_index(
        "ix_stream_session_camera_started",
        "stream_session",
        ["camera_id", "started_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_stream_session_camera_started", table_name="stream_session")
    op.drop_table("stream_session")
