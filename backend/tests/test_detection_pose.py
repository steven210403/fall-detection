"""The pose detector's keypoints reach the tracked observation (real YOLOv8n-pose).

Runs the real model on Ultralytics' bundled ``bus.jpg``. It asserts wiring only — that
every person detection carries 17 ``(x, y, conf)`` keypoints and that tracking hands
them on unchanged — never detection or pose accuracy.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from fall_detection.config import Settings
from fall_detection.core.contracts import FramePacket
from fall_detection.detection import DetectionService
from fall_detection.fall.geometry import compute_geometry
from fall_detection.tracking import TrackingService

CAMERA = uuid.UUID("33333333-3333-3333-3333-333333333333")
SESSION = uuid.UUID("44444444-4444-4444-4444-444444444444")


@pytest.fixture
def bus_frame() -> object:
    cv2 = pytest.importorskip("cv2")
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(Path(ASSETS) / "bus.jpg"))
    if image is None:
        pytest.skip("the bundled sample image is not available")
    return image


def test_keypoints_flow_from_detector_to_tracking(
    make_settings: Callable[..., Settings], detector_weights: Path, bus_frame: object
) -> None:
    settings = make_settings(
        detection={"enabled": True, "model_path": str(detector_weights), "warmup_runs": 0}
    )
    detection = DetectionService(settings)
    detection.load()
    tracking = TrackingService(settings)
    packet = FramePacket(
        camera_id=CAMERA,
        stream_session_id=SESSION,
        frame_seq=0,
        frame=bus_frame,  # type: ignore[arg-type]
        source_timestamp=0.0,
        receive_monotonic=1.0,
        receive_utc=1_700_000_000.0,
    )
    detection.on_frame(packet)
    batch = detection.latest_batch()
    assert batch is not None
    persons = batch.persons()
    assert persons, "the sample image contains people"
    for person in persons:
        assert person.keypoints is not None
        assert len(person.keypoints) == 17
        assert all(len(point) == 3 for point in person.keypoints)

    tracks = tracking.update(batch)
    assert tracks.observations
    for observation in tracks.observations:
        assert observation.keypoints is not None and len(observation.keypoints) == 17
        geometry = compute_geometry(observation.bbox, observation.keypoints, min_conf=0.3)
        # Standing pedestrians: the geometry is measurable on a real frame.
        assert geometry.aspect_ratio is not None
    detection.unload()
