"""Annotated-frame rendering.

The preview is drawn from the **same frame** the detections came from, never by
overlaying newer metadata on an older frame. This module takes the frame and its
:class:`DetectionBatch` together and returns a new array; it keeps no state.

Each person is drawn with their box and, for a pose model, their 17-keypoint skeleton
(the draft's "visualisasi bounding box dan garis skeleton tubuh"). Labels always carry
text, so a viewer who cannot perceive colour still reads them.

OpenCV is imported lazily so importing ``fall_detection.detection`` does not pull in a
CV build.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fall_detection.detection.contracts import DetectionBatch

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from collections.abc import Sequence

    import numpy as np

__all__ = [
    "CLASS_COLORS",
    "DRAWN_CLASSES",
    "annotate_alert",
    "annotate_captions",
    "annotate_frame",
]

# Detection classes drawn on the preview. Every other class is detected but not drawn.
DRAWN_CLASSES: frozenset[str] = frozenset({"person"})

# BGR colours, one per detection class. Chosen to stay distinguishable in a
# greyscale print of the thesis appendix as well as on screen.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (80, 200, 80),
}

_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
_SKELETON_COLOR = (255, 200, 0)
_JOINT_COLOR = (0, 255, 255)
_KEYPOINT_MIN_CONF = 0.3
_DEFAULT_COLOR = (0, 165, 255)


def annotate_frame(frame: np.ndarray, batch: DetectionBatch) -> np.ndarray:
    """Return a copy of ``frame`` with each person detection drawn on it.

    Detections whose class is not in :data:`DRAWN_CLASSES` are skipped; the batch
    itself is untouched, so downstream stages still see every object.

    Args:
        frame: The decoded BGR ``uint8`` frame matching ``batch``.
        batch: Detections produced from **this** frame.

    Returns:
        A new contiguous BGR frame. The input is never modified, so the caller
        can reuse the original for a second consumer (for example the JPEG
        preview and the feature extractor).

    Raises:
        ValueError: The frame's shape does not match ``batch.frame_shape``, which
            would mean the batch belongs to a different frame — exactly the
            mismatch §18.1 forbids drawing.
    """
    import cv2

    if tuple(int(part) for part in frame.shape) != tuple(batch.frame_shape):
        raise ValueError(
            f"frame shape {tuple(frame.shape)} does not match batch frame_shape "
            f"{batch.frame_shape}; refusing to overlay mismatched detections"
        )

    canvas = frame.copy()
    for detection in batch.detections:
        if detection.class_name not in DRAWN_CLASSES:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        color = CLASS_COLORS.get(detection.class_name, _DEFAULT_COLOR)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if detection.keypoints:
            _draw_skeleton(canvas, detection.keypoints)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Keep the caption inside the frame even for a box against the top edge.
        text_y = max(y1, text_height + baseline)
        cv2.rectangle(
            canvas,
            (x1, text_y - text_height - baseline),
            (x1 + text_width, text_y),
            color,
            thickness=-1,
        )
        cv2.putText(
            canvas,
            label,
            (x1, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


_CAPTION_BACKGROUND = (32, 32, 32)
_CAPTION_TEXT = (255, 255, 255)


def annotate_captions(
    frame: np.ndarray,
    captions: Sequence[tuple[tuple[float, float, float, float], tuple[str, ...]]],
    *,
    backgrounds: Sequence[tuple[int, int, int] | None] | None = None,
) -> np.ndarray:
    """Draw text lines under the top edge of each box, in place, and return ``frame``.

    The caller passes boxes from the frame being drawn — tracked boxes from the same
    packet — so a caption can never sit on a box from another frame (§18.1). The lines
    are plain text on a dark plate, so they read without colour.

    Args:
        frame: A BGR ``uint8`` frame the caller owns (normally the copy
            :func:`annotate_frame` returned).
        captions: ``(bbox, lines)`` pairs; a pair with no lines is skipped.
    """
    import cv2

    height = int(frame.shape[0])
    for index, (bbox, lines) in enumerate(captions):
        if not lines:
            continue
        background = _CAPTION_BACKGROUND
        if backgrounds is not None and index < len(backgrounds):
            background = backgrounds[index] or _CAPTION_BACKGROUND
        x1, y1 = int(round(bbox[0])), int(round(bbox[1]))
        y = max(0, y1) + 4
        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            bottom = min(height - 1, y + text_height + baseline + 4)
            cv2.rectangle(
                frame,
                (x1 + 2, bottom - text_height - baseline - 4),
                (x1 + 2 + text_width + 6, bottom),
                background,
                thickness=-1,
            )
            cv2.putText(
                frame,
                line,
                (x1 + 5, bottom - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                _CAPTION_TEXT,
                1,
                cv2.LINE_AA,
            )
            y = bottom
    return frame


def _draw_skeleton(
    canvas: np.ndarray,
    keypoints: Sequence[tuple[float, float, float]],
) -> None:
    """Draw the COCO skeleton for one person, skipping invisible joints."""
    import cv2

    visible = [
        (int(round(x)), int(round(y))) if conf >= _KEYPOINT_MIN_CONF else None
        for x, y, conf in keypoints
    ]
    for start, end in _SKELETON_EDGES:
        if start >= len(visible) or end >= len(visible):
            continue
        a, b = visible[start], visible[end]
        if a is not None and b is not None:
            cv2.line(canvas, a, b, _SKELETON_COLOR, 2, cv2.LINE_AA)
    for point in visible:
        if point is not None:
            cv2.circle(canvas, point, 3, _JOINT_COLOR, -1, cv2.LINE_AA)


_ALERT_COLOR = (0, 0, 255)


def annotate_alert(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a red border and a banner on ``frame`` in place, and return it.

    Used while any person is in the ``fallen`` state, so the alarm is visible in the
    video itself as well as in the dashboard.
    """
    import cv2

    height, width = int(frame.shape[0]), int(frame.shape[1])
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), _ALERT_COLOR, 8)
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(
        frame, (0, 0), (min(width, text_width + 24), text_height + baseline + 16), _ALERT_COLOR, -1
    )
    cv2.putText(
        frame,
        text,
        (12, text_height + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame
