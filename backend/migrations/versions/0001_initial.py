"""Initial schema: camera table.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-14

Phase 0 creates only the ``camera`` table. Later phases add the remaining
entities from thesis plan §19 in separate revisions.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "camera",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("source_ref", sa.String(length=500), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "source_type IN ('mp4', 'rtsp')",
            name=op.f("ck_camera_source_type_allowed"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_camera")),
    )


def downgrade() -> None:
    op.drop_table("camera")
