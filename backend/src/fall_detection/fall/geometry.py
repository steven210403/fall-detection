"""Body geometry from a bounding box and 17 COCO keypoints (THESIS_PLAN §4.1–§4.2).

Two of the draft's three fall parameters are computed here:

* **Aspect ratio** ``AR = W / H`` of the person's box. Upright people are taller than
  wide (AR < 1); a person lying on the floor is wider than tall (AR >= 1).
* **Torso angle** θ between the vector from the shoulder midpoint to the hip midpoint
  and the horizontal: θ = atan2(|Δy|, |Δx|), in [0°, 90°]. Upright ≈ 70–90°, lying
  < 30°. The absolute values make the angle independent of which way the person faces
  or falls.

The hip midpoint (keypoints 11 and 12) is the body's centre of mass; the third parameter,
its vertical velocity, is computed from it in :mod:`fall_detection.fall.kinematics`.

A keypoint below ``min_conf`` is treated as not visible. When only one side's shoulder
or hip is visible, the visible one stands in for the midpoint and the angle is reported
as ``"partial"``; when no shoulder or no hip is visible, the angle is ``None`` rather
than a guess.

Pure Python: no NumPy, no OpenCV.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from fall_detection.fall.contracts import PoseGeometry

__all__ = [
    "LEFT_HIP",
    "LEFT_SHOULDER",
    "RIGHT_HIP",
    "RIGHT_SHOULDER",
    "SKELETON_EDGES",
    "aspect_ratio",
    "compute_geometry",
    "midpoint",
    "torso_angle_deg",
]

Box = tuple[float, float, float, float]
Keypoint = tuple[float, float, float]
Point = tuple[float, float]

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

SKELETON_EDGES: tuple[tuple[int, int], ...] = (
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
"""The COCO skeleton as keypoint-index pairs, for drawing."""


def aspect_ratio(box: Box) -> float | None:
    """Bounding-box ``W / H``; ``None`` for a box with no height."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    if height <= 0.0 or width < 0.0:
        return None
    return width / height


def midpoint(
    keypoints: Sequence[Keypoint],
    left: int,
    right: int,
    *,
    min_conf: float,
) -> tuple[Point | None, bool]:
    """Midpoint of a left/right keypoint pair.

    Returns:
        ``(point, both_visible)``. When only one side is visible its position is
        returned with ``both_visible=False``; when neither is, ``(None, False)``.
    """
    left_point = _visible(keypoints, left, min_conf)
    right_point = _visible(keypoints, right, min_conf)
    if left_point is not None and right_point is not None:
        return (
            (left_point[0] + right_point[0]) / 2.0,
            (left_point[1] + right_point[1]) / 2.0,
        ), True
    if left_point is not None:
        return left_point, False
    if right_point is not None:
        return right_point, False
    return None, False


def torso_angle_deg(shoulder: Point, hip: Point) -> float | None:
    """Angle of the shoulder→hip vector to the horizontal, in ``[0, 90]`` degrees.

    ``None`` when the two points coincide, because a zero-length torso has no direction.
    """
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    if dx == 0.0 and dy == 0.0:
        return None
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def compute_geometry(
    box: Box,
    keypoints: Sequence[Keypoint] | None,
    *,
    min_conf: float,
) -> PoseGeometry:
    """Measure one person's aspect ratio, torso angle and hip centre."""
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    body_length = max(width, height)
    ratio = aspect_ratio(box)
    if not keypoints:
        return PoseGeometry(
            aspect_ratio=ratio,
            torso_angle_deg=None,
            torso_side=None,
            hip_center=None,
            body_length_px=body_length if body_length > 0.0 else None,
            visible_keypoints=0,
        )
    shoulder, shoulders_both = midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, min_conf=min_conf)
    hip, hips_both = midpoint(keypoints, LEFT_HIP, RIGHT_HIP, min_conf=min_conf)
    angle = None if shoulder is None or hip is None else torso_angle_deg(shoulder, hip)
    side = None
    if angle is not None:
        side = "both" if shoulders_both and hips_both else "partial"
    visible = sum(1 for point in keypoints if len(point) > 2 and point[2] >= min_conf)
    return PoseGeometry(
        aspect_ratio=ratio,
        torso_angle_deg=angle,
        torso_side=side,
        hip_center=hip,
        body_length_px=body_length if body_length > 0.0 else None,
        visible_keypoints=visible,
    )


def _visible(keypoints: Sequence[Keypoint], index: int, min_conf: float) -> Point | None:
    if index >= len(keypoints):
        return None
    point = keypoints[index]
    confidence = point[2] if len(point) > 2 else 1.0
    if confidence < min_conf:
        return None
    return float(point[0]), float(point[1])
