"""Add track table.

Revision ID: 0003_track
Revises: 0002_stream_session
Create Date: 2026-10-14

Phase 4 adds the durable record of a tracked person (thesis plan §11.2, §19):
activity events attach to ``track.id``, so the UUID that tracking mints has to be
persisted even though the integer tracker ID is only a display value.

``tracker_id`` is deliberately **not** unique within a session. §11.2 gives the
integer no identity meaning, and a tracker may restart its integer IDs; enforcing
uniqueness would turn a legitimate reissue into a constraint violation. The
UUID primary key is the identity, and an index on
``(stream_session_id, tracker_id)`` keeps lookups cheap.

``last_seen_at`` stores the last *observation*, not the moment a track was
terminated after its loss tolerance expired, so the stored interval always
describes evidence rather than a grace period (FR-07).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_track"
down_revision: str | None = "0002_stream_session"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TRACK_TERMINATION_REASONS = ("lost_timeout", "session_ended", "stopped")


def upgrade() -> None:
    op.create_table(
        "track",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("stream_session_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("tracker_id", sa.Integer(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("terminated_at", sa.DateTime(), nullable=True),
        sa.Column("termination_reason", sa.String(length=32), nullable=True),
        sa.Column("observed_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("missing_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.CheckConstraint(
            "termination_reason IS NULL OR termination_reason IN ("
            + ", ".join(f"'{reason}'" for reason in TRACK_TERMINATION_REASONS)
            + ")",
            name=op.f("ck_track_termination_reason_allowed"),
        ),
        sa.CheckConstraint("tracker_id >= 1", name=op.f("ck_track_tracker_id_positive")),
        sa.CheckConstraint(
            "observed_frames >= 0", name=op.f("ck_track_observed_frames_non_negative")
        ),
        sa.CheckConstraint(
            "last_seen_at >= first_seen_at", name=op.f("ck_track_last_seen_after_first_seen")
        ),
        sa.CheckConstraint(
            "terminated_at IS NULL OR terminated_at >= first_seen_at",
            name=op.f("ck_track_terminated_after_first_seen"),
        ),
        sa.ForeignKeyConstraint(
            ["stream_session_id"],
            ["stream_session.id"],
            name=op.f("fk_track_stream_session_id_stream_session"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_track")),
    )
    op.create_index(
        "ix_track_session_tracker",
        "track",
        ["stream_session_id", "tracker_id"],
    )
    op.create_index(
        "ix_track_session_first_seen",
        "track",
        ["stream_session_id", "first_seen_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_track_session_first_seen", table_name="track")
    op.drop_index("ix_track_session_tracker", table_name="track")
    op.drop_table("track")
