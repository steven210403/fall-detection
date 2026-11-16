"""Synthetic poses and observations for the fall tests.

The poses are idealised skeletons: they test what the geometry, velocity, and state
machine rules *say*, never how well a model sees a real fall. No test may derive an
accuracy figure from them.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from fall_detection.config import FallSettings
from fall_detection.fall.contracts import FallFeatures
from fall_detection.fall.geometry import compute_geometry
from fall_detection.tracking.contracts import (
    TrackObservation,
    TrackObservations,
    TrackState,
    TrackSummary,
    TrackTermination,
)

Keypoint = tuple[float, float, float]
Box = tuple[float, float, float, float]

CAMERA = uuid.UUID("22222222-2222-2222-2222-222222222222")
SESSION = uuid.UUID("11111111-1111-1111-1111-111111111111")
TRACK = uuid.UUID("33333333-3333-3333-3333-333333333333")

BODY = 170.0
"""Body length in pixels of the synthetic person."""


def standing(x: float = 300.0, top: float = 100.0) -> tuple[Box, tuple[Keypoint, ...]]:
    """An upright person: box taller than wide, vertical torso."""
    box = (x - 25.0, top, x + 25.0, top + BODY)
    return box, _skeleton(shoulder=(x, top + 30.0), hip=(x, top + 90.0), spread=(15.0, 0.0))


def lying(y: float = 400.0, left: float = 200.0) -> tuple[Box, tuple[Keypoint, ...]]:
    """A person lying on the floor: box wider than tall, horizontal torso."""
    box = (left, y - 25.0, left + BODY, y + 25.0)
    return box, _skeleton(shoulder=(left + 30.0, y), hip=(left + 90.0, y), spread=(0.0, 15.0))


def sitting(x: float = 300.0, top: float = 180.0) -> tuple[Box, tuple[Keypoint, ...]]:
    """A seated person: box roughly square, torso still upright."""
    box = (x - 45.0, top, x + 45.0, top + 100.0)
    return box, _skeleton(shoulder=(x, top + 20.0), hip=(x + 5.0, top + 75.0), spread=(15.0, 0.0))


def bending(x: float = 300.0, top: float = 200.0) -> tuple[Box, tuple[Keypoint, ...]]:
    """Bent forward to pick something up: torso tilted ~40 deg, box wider than tall."""
    box = (x - 60.0, top, x + 60.0, top + 100.0)
    return box, _skeleton(shoulder=(x - 40.0, top + 20.0), hip=(x, top + 55.0), spread=(0.0, 10.0))


def invisible(box: Box) -> tuple[Box, tuple[Keypoint, ...]]:
    """A box whose keypoints the model did not see."""
    return box, tuple((0.0, 0.0, 0.0) for _ in range(17))


def _skeleton(
    *, shoulder: tuple[float, float], hip: tuple[float, float], spread: tuple[float, float]
) -> tuple[Keypoint, ...]:
    points: list[Keypoint] = [(shoulder[0], shoulder[1] - 25.0, 0.9) for _ in range(5)]
    dx, dy = spread
    points.append((shoulder[0] - dx, shoulder[1] - dy, 0.9))  # left shoulder
    points.append((shoulder[0] + dx, shoulder[1] + dy, 0.9))  # right shoulder
    points.extend([(shoulder[0], shoulder[1] + 20.0, 0.9)] * 4)  # elbows, wrists
    points.append((hip[0] - dx, hip[1] - dy, 0.9))  # left hip
    points.append((hip[0] + dx, hip[1] + dy, 0.9))  # right hip
    points.extend([(hip[0], hip[1] + 40.0, 0.9)] * 4)  # knees, ankles
    return tuple(points)


def features(
    t: float,
    pose: tuple[Box, tuple[Keypoint, ...]],
    *,
    vy_norm: float | None = 0.0,
    settings: FallSettings | None = None,
    frame_seq: int = 0,
) -> FallFeatures:
    """Build one :class:`FallFeatures` sample with an explicit velocity."""
    box, keypoints = pose
    min_conf = (settings or FallSettings()).keypoint_min_conf
    return FallFeatures(
        track_uuid=TRACK,
        display_id=1,
        frame_seq=frame_seq,
        t=t,
        utc=1_700_000_000.0 + t,
        time_source="source",
        geometry=compute_geometry(box, keypoints, min_conf=min_conf),
        vy_px_s=None if vy_norm is None else vy_norm * BODY,
        vy_norm=vy_norm,
        source_timestamp=t,
    )


def observations(
    frame_seq: int,
    t: float,
    people: Sequence[tuple[uuid.UUID, int, tuple[Box, tuple[Keypoint, ...]]]],
    *,
    terminated: Sequence[TrackTermination] = (),
    session: uuid.UUID = SESSION,
) -> TrackObservations:
    """Build one frame of tracking output for the fall service."""
    observed = tuple(
        TrackObservation(
            track_uuid=track_uuid,
            display_id=display_id,
            camera_id=CAMERA,
            stream_session_id=session,
            bbox=pose[0],
            confidence=0.9,
            class_name="person",
            frame_seq=frame_seq,
            observed_monotonic=t,
            source_timestamp=t,
            receive_monotonic=t,
            receive_utc=1_700_000_000.0 + t,
            first_seen_monotonic=0.0,
            age_s=t,
            observed_frames=frame_seq + 1,
            keypoints=pose[1],
        )
        for track_uuid, display_id, pose in people
    )
    summaries = tuple(
        TrackSummary(
            track_uuid=item.track_uuid,
            display_id=item.display_id,
            camera_id=CAMERA,
            stream_session_id=session,
            state=TrackState.ACTIVE,
            bbox=item.bbox,
            confidence=item.confidence,
            first_seen_monotonic=0.0,
            last_observed_monotonic=t,
            age_s=t,
            time_since_last_observation_s=0.0,
            stale=False,
            observed_frames=item.observed_frames,
            missing_frames=0,
        )
        for item in observed
    )
    return TrackObservations(
        camera_id=CAMERA,
        stream_session_id=session,
        frame_seq=frame_seq,
        observations=observed,
        tracks=summaries,
        terminated=tuple(terminated),
        frame_shape=(480, 640, 3),
        source_timestamp=t,
        receive_monotonic=t,
        receive_utc=1_700_000_000.0 + t,
        tracking_monotonic=t,
        tracker_name="bytetrack",
    )


def fall_sequence(
    *, fps: float = 30.0, lie_seconds: float = 4.0, start: float = 0.0
) -> list[tuple[float, tuple[Box, tuple[Keypoint, ...]]]]:
    """Stand for 1 s, fall over 0.4 s, then lie still for ``lie_seconds``."""
    frames: list[tuple[float, tuple[Box, tuple[Keypoint, ...]]]] = []
    step = 1.0 / fps
    t = start
    for _ in range(int(fps)):
        frames.append((t, standing()))
        t += step
    fall_frames = int(0.4 * fps)
    for index in range(1, fall_frames + 1):
        drop = 300.0 * index / fall_frames
        frames.append((t, standing(top=100.0 + drop)))
        t += step
    for _ in range(int(lie_seconds * fps)):
        frames.append((t, lying()))
        t += step
    return frames


def termination(
    track_uuid: uuid.UUID, t: float, *, reason: str = "lost_timeout"
) -> TrackTermination:
    """A track that ended at ``t``."""
    return TrackTermination(
        track_uuid=track_uuid,
        display_id=1,
        camera_id=CAMERA,
        stream_session_id=SESSION,
        reason=reason,
        ended_monotonic=t,
        ended_utc=1_700_000_000.0 + t,
        duration_s=t,
        observed_frames=1,
        missing_frames=0,
    )
