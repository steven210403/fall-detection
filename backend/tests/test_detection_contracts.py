"""Detection unit tests: contracts, class mapping, and annotated frames.

These tests make **no accuracy claim**. They exercise the Phase 3 wiring with
hand-built :class:`DetectionBatch` objects and a synthetic frame, so the
contract, the class map, and the annotator are verified without a detector. A
separate module covers real model inference and skips when weights are absent.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fall_detection.core.contracts import FramePacket
from fall_detection.core.errors import ModelUnavailableError
from fall_detection.detection import (
    Detection,
    DetectionBatch,
    DetectionTiming,
    annotate_captions,
    annotate_frame,
    class_id_map,
    resolve_classes,
)

SESSION = uuid.UUID("11111111-1111-1111-1111-111111111111")
CAMERA = uuid.UUID("22222222-2222-2222-2222-222222222222")


def make_detection(
    *,
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 60.0, 120.0),
    confidence: float = 0.8,
    class_name: str = "person",
    class_id: int = 0,
) -> Detection:
    return Detection(bbox=bbox, confidence=confidence, class_name=class_name, class_id=class_id)


def make_batch(
    detections: tuple[Detection, ...] = (),
    *,
    frame_shape: tuple[int, int, int] = (120, 160, 3),
) -> DetectionBatch:
    return DetectionBatch(
        camera_id=CAMERA,
        stream_session_id=SESSION,
        frame_seq=7,
        detections=detections,
        frame_shape=frame_shape,
        source_timestamp=1.5,
        receive_monotonic=100.0,
        receive_utc=1_700_000_000.0,
        inference_monotonic=100.05,
        timing=DetectionTiming(1.0, 2.0, 0.5, 3.5),
        model_name="yolov8n-pose.pt",
    )


def make_packet(frame: np.ndarray, *, frame_seq: int = 0) -> FramePacket:
    return FramePacket(
        camera_id=CAMERA,
        stream_session_id=SESSION,
        frame_seq=frame_seq,
        frame=frame,
        source_timestamp=0.0,
        receive_monotonic=50.0,
        receive_utc=1_700_000_000.0,
    )


class TestDetection:
    def test_geometry_helpers(self) -> None:
        detection = make_detection(bbox=(10.0, 20.0, 60.0, 120.0))
        assert detection.width == 50.0
        assert detection.height == 100.0
        assert detection.area == 5000.0
        assert detection.center == (35.0, 70.0)
        assert detection.is_person()

    def test_inverted_box_reports_zero_size(self) -> None:
        detection = make_detection(bbox=(60.0, 20.0, 10.0, 120.0))
        assert detection.width == 0.0
        assert detection.area == 0.0


class TestDetectionBatch:
    def test_counts_and_grouping(self) -> None:
        batch = make_batch(
            (
                make_detection(class_name="person"),
                make_detection(class_name="person", class_id=0),
                make_detection(class_name="laptop", class_id=63),
            )
        )
        assert batch.counts() == {"person": 2, "laptop": 1}
        assert len(batch.persons()) == 2
        assert batch.frame_size == (160, 120)

    def test_batch_preserves_frame_identity(self) -> None:
        # The batch must repeat identity so a tracker never needs the frame array.
        batch = make_batch()
        assert batch.camera_id == CAMERA
        assert batch.stream_session_id == SESSION
        assert batch.frame_seq == 7
        assert batch.source_timestamp == 1.5
        assert batch.model_name == "yolov8n-pose.pt"

    def test_missing_source_timestamp_is_not_fabricated(self) -> None:
        # Thesis plan §9.1: a missing source timestamp stays None.
        batch = DetectionBatch(
            camera_id=CAMERA,
            stream_session_id=SESSION,
            frame_seq=0,
            detections=(),
            frame_shape=(10, 10, 3),
            source_timestamp=None,
            receive_monotonic=1.0,
            receive_utc=2.0,
            inference_monotonic=1.1,
        )
        assert batch.source_timestamp is None
        assert batch.timing is None


class TestClassMapping:
    def test_pose_vocabulary_maps_person(self) -> None:
        # A YOLO pose model's vocabulary is ``{0: "person"}``.
        assert resolve_classes({0: "person"}) == {0: "person"}

    def test_coco_vocabulary_keeps_only_person(self) -> None:
        names = {0: "person", 56: "chair", 62: "tv", 63: "laptop", 1: "bicycle"}
        assert resolve_classes(names) == {0: "person"}

    def test_accepts_sequence_vocabulary(self) -> None:
        assert resolve_classes(["bicycle", "person"]) == {1: "person"}

    def test_vocabulary_without_person_raises(self) -> None:
        with pytest.raises(ModelUnavailableError, match="no 'person' class"):
            resolve_classes({0: "chair", 1: "laptop"})

    def test_class_id_map_restricts_to_requested(self) -> None:
        assert class_id_map({0: "person", 63: "laptop"}, ("person",)) == {0: "person"}

    def test_class_id_map_rejects_unknown_class(self) -> None:
        with pytest.raises(ModelUnavailableError, match="unknown detection class"):
            class_id_map({0: "person"}, ("helicopter",))


class TestAnnotate:
    def test_returns_new_array_and_does_not_mutate_input(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        original = frame.copy()
        batch = make_batch((make_detection(),))
        annotated = annotate_frame(frame, batch)
        assert annotated.shape == frame.shape
        assert annotated is not frame
        assert np.array_equal(frame, original)
        # A drawn rectangle must change at least one pixel.
        assert not np.array_equal(annotated, original)

    def test_empty_batch_returns_copy(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        annotated = annotate_frame(frame, make_batch(()))
        assert annotated is not frame
        assert np.array_equal(annotated, frame)

    def test_the_skeleton_is_drawn_from_visible_keypoints(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        hidden = tuple((80.0, 60.0, 0.0) for _ in range(17))
        visible = tuple((40.0 + index * 4.0, 20.0 + index * 4.0, 0.9) for index in range(17))
        plain = annotate_frame(frame, make_batch((make_detection(),)))
        no_joints = annotate_frame(
            frame, make_batch((replace(make_detection(), keypoints=hidden),))
        )
        skeleton = annotate_frame(
            frame, make_batch((replace(make_detection(), keypoints=visible),))
        )
        # Invisible joints draw nothing beyond the box; visible ones add the skeleton.
        assert np.array_equal(plain, no_joints)
        assert not np.array_equal(plain, skeleton)

    def test_shape_mismatch_is_rejected(self) -> None:
        # §18.1: refusing to overlay a batch from a different frame.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="does not match batch frame_shape"):
            annotate_frame(frame, make_batch(frame_shape=(120, 160, 3)))

    def test_box_near_top_edge_stays_in_frame(self) -> None:
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        batch = make_batch((make_detection(bbox=(1.0, 0.0, 40.0, 30.0)),), frame_shape=(60, 60, 3))
        # Must not raise or write out of bounds.
        annotate_frame(frame, batch)


class TestCaptions:
    def test_a_caption_is_drawn_inside_its_box_and_nowhere_else(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        annotate_captions(frame, [((100.0, 50.0, 220.0, 230.0), ("#3", "stationary"))])
        drawn = np.argwhere(frame.any(axis=2))
        assert drawn.size > 0
        # The plate hangs from the box's top-left corner, below its top edge.
        assert drawn[:, 0].min() >= 50
        assert drawn[:, 1].min() >= 100

    def test_a_box_with_no_lines_draws_nothing(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        annotate_captions(frame, [((10.0, 10.0, 60.0, 100.0), ())])
        assert not frame.any()

    def test_a_caption_at_the_bottom_edge_stays_in_frame(self) -> None:
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        # Must not raise or write out of bounds.
        annotate_captions(frame, [((0.0, 55.0, 30.0, 60.0), ("#1", "phone usage"))])


class TestModuleBoundaries:
    def test_contracts_stay_free_of_cv_and_orm_dependencies(self) -> None:
        # §17.1: CV modules exchange dataclasses and never touch ORM models. The
        # contracts carry only geometry and scalars, so importing the package must
        # not pull in the ML stack. Asserting on the source is the only way to
        # check an *absence* of imports.
        import fall_detection.detection.contracts as contracts_module

        assert contracts_module.__file__ is not None
        source = Path(contracts_module.__file__).read_text(encoding="utf-8")
        assert "import numpy" not in source
        assert "import cv2" not in source
        assert "import torch" not in source
        assert "sqlalchemy" not in source
        assert "fall_detection.database" not in source

    def test_detector_imports_ml_stack_lazily(self) -> None:
        # The ML stack must be imported inside methods, never at module import
        # time, so importing fall_detection.detection (or the app with detection
        # disabled) stays cheap. A module-level import would defeat that, so this
        # asserts that only *indented* (in-method) imports name torch/ultralytics.
        import fall_detection.detection.detector as detector_module

        assert detector_module.__file__ is not None
        source = Path(detector_module.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            if "ultralytics" in stripped or "torch" in stripped:
                # A lazy import is indented; a module-level one is not.
                assert line[:1].isspace(), f"module-level ML import: {line!r}"

    def test_detection_package_does_not_import_video_or_api(self) -> None:
        # Detection sits between video and tracking in the §9 pipeline, but it
        # consumes FramePacket and never the video package or the API layer.
        import fall_detection.detection.detector as detector_module
        import fall_detection.detection.service as service_module

        for module in (detector_module, service_module):
            assert module.__file__ is not None
            source = Path(module.__file__).read_text(encoding="utf-8")
            assert "from fall_detection.video" not in source
            assert "from fall_detection.api" not in source
