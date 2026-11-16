"""Fall detection: body geometry, hip velocity, and the per-person state machine.

Only the stdlib-only pieces are re-exported here; the consumer (which pulls in the
tracking layer) is imported from :mod:`fall_detection.fall.consumer` directly.
"""

from __future__ import annotations

from fall_detection.fall.contracts import (
    FallBatch,
    FallDecision,
    FallFeatures,
    FallState,
    FallTransition,
    PoseGeometry,
)
from fall_detection.fall.fsm import FallStateMachine, read_posture
from fall_detection.fall.geometry import aspect_ratio, compute_geometry, torso_angle_deg
from fall_detection.fall.kinematics import HipVelocityEstimator
from fall_detection.fall.service import FallService

__all__ = [
    "FallBatch",
    "FallDecision",
    "FallFeatures",
    "FallService",
    "FallState",
    "FallStateMachine",
    "FallTransition",
    "HipVelocityEstimator",
    "PoseGeometry",
    "aspect_ratio",
    "compute_geometry",
    "read_posture",
    "torso_angle_deg",
]
