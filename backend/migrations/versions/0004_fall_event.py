"""Fall events.

Revision ID: 0004_fall_event
Revises: 0003_track
Create Date: 2026-11-05

One row per detected fall of one tracked person. The row is inserted when the alarm
fires and closed later, so ``ended_at``/``end_reason`` are nullable and constrained to be
set together.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_fall_event"
down_revision: str | None = "0003_track"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FALL_EVENT_END_REASONS = (
    "recovered",
    "track_lost",
    "session_ended",
    "stopped",
    "max_duration",
)


def upgrade() -> None:
    op.create_table(
        "fall_event",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("track_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("falling_at", sa.DateTime(), nullable=False),
        sa.Column("impact_at", sa.DateTime(), nullable=False),
        sa.Column("alert_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("end_reason", sa.String(length=32), nullable=True),
        sa.Column("response_time_s", sa.Double(), nullable=False),
        sa.Column("peak_vy", sa.Double(), nullable=True),
        sa.Column("min_torso_angle_deg", sa.Double(), nullable=True),
        sa.Column("max_aspect_ratio", sa.Double(), nullable=True),
        sa.Column("alert_source_timestamp", sa.Double(), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "end_reason IS NULL OR end_reason IN ("
            + ", ".join(f"'{reason}'" for reason in FALL_EVENT_END_REASONS)
            + ")",
            name=op.f("ck_fall_event_end_reason_allowed"),
        ),
        sa.CheckConstraint(
            "(ended_at IS NULL AND end_reason IS NULL) "
            "OR (ended_at IS NOT NULL AND end_reason IS NOT NULL)",
            name=op.f("ck_fall_event_closed_together"),
        ),
        sa.CheckConstraint(
            "impact_at >= falling_at", name=op.f("ck_fall_event_impact_after_falling")
        ),
        sa.CheckConstraint("alert_at >= impact_at", name=op.f("ck_fall_event_alert_after_impact")),
        sa.CheckConstraint(
            "ended_at IS NULL OR ended_at >= alert_at", name=op.f("ck_fall_event_ended_after_alert")
        ),
        sa.CheckConstraint(
            "acknowledged_at IS NULL OR acknowledged_at >= alert_at",
            name=op.f("ck_fall_event_acknowledged_after_alert"),
        ),
        sa.CheckConstraint(
            "schema_version >= 1", name=op.f("ck_fall_event_schema_version_positive")
        ),
        sa.CheckConstraint(
            "response_time_s >= 0", name=op.f("ck_fall_event_response_time_non_negative")
        ),
        sa.ForeignKeyConstraint(
            ["track_id"],
            ["track.id"],
            name=op.f("fk_fall_event_track_id_track"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fall_event")),
    )
    op.create_index("ix_fall_event_alert", "fall_event", ["alert_at"])
    op.create_index("ix_fall_event_track_alert", "fall_event", ["track_id", "alert_at"])


def downgrade() -> None:
    op.drop_index("ix_fall_event_track_alert", table_name="fall_event")
    op.drop_index("ix_fall_event_alert", table_name="fall_event")
    op.drop_table("fall_event")
