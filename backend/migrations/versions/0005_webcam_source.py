"""Allow the built-in webcam as a camera source type.

Revision ID: 0005_webcam_source
Revises: 0004_fall_event
Create Date: 2026-11-06

The report's primary live source is the laptop's built-in webcam, read by FFmpeg
through AVFoundation. ``camera.source_type`` gains ``'webcam'`` next to ``'mp4'`` and
``'rtsp'``. SQLite cannot alter a check constraint in place, so the table is rebuilt
in batch mode.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005_webcam_source"
down_revision: str | None = "0004_fall_event"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _replace_source_type_check(condition: str) -> None:
    with op.batch_alter_table("camera") as batch:
        batch.drop_constraint(op.f("ck_camera_source_type_allowed"), type_="check")
        batch.create_check_constraint("source_type_allowed", condition)


def upgrade() -> None:
    _replace_source_type_check("source_type IN ('mp4', 'rtsp', 'webcam')")


def downgrade() -> None:
    # A webcam camera row cannot satisfy the old constraint; refuse rather than
    # delete history silently.
    connection = op.get_bind()
    webcam_rows = connection.exec_driver_sql(
        "SELECT COUNT(*) FROM camera WHERE source_type = 'webcam'"
    ).scalar_one()
    if webcam_rows:
        raise RuntimeError(
            f"{webcam_rows} camera row(s) use source_type 'webcam'; "
            "remove them before downgrading below 0005_webcam_source"
        )
    _replace_source_type_check("source_type IN ('mp4', 'rtsp')")
