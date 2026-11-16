"""Detector integration tests against the real YOLO26n weights.

These tests load the actual Phase 0 checkpoint and run inference on a synthetic
frame. They verify **wiring and contract behaviour only** — that the model loads
once, that the batch carries the frame identity, that detections are clipped, and
that annotated frames match the source frame. They make **no accuracy claim**: a
blank synthetic frame has no people, and no mAP/FPS figure is asserted anywhere.

The whole module skips when the gitignored weights are absent, exactly as the
video tests skip when FFmpeg is absent.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from fall_detection.config import Settings
from fall_detection.core.contracts import FramePacket
from fall_detection.core.errors import ModelUnavailableError
from fall_detection.detection import DetectionService, Detector, build_detector
from fall_detection.detection.detector import resolve_model_path

CAMERA = uuid.UUID("33333333-3333-3333-3333-333333333333")
SESSION = uuid.UUID("44444444-4444-4444-4444-444444444444")


def make_packet(frame: np.ndarray, *, session_id: uuid.UUID = SESSION, seq: int = 0) -> FramePacket:
    return FramePacket(
        camera_id=CAMERA,
        stream_session_id=session_id,
        frame_seq=seq,
        frame=frame,
        source_timestamp=seq / 10.0,
        receive_monotonic=1000.0 + seq,
        receive_utc=1_700_000_000.0 + seq,
    )


def detection_settings(
    make_settings: Callable[..., Settings],
    weights: Path,
    **overrides: object,
) -> Settings:
    detection: dict[str, object] = {
        "enabled": True,
        "model_path": str(weights),
        "imgsz": 640,
        "warmup_runs": 1,
    }
    detection.update(overrides)
    return make_settings(detection=detection)


class TestModelResolution:
    def test_missing_weights_raise_model_unavailable(self, tmp_path: Path) -> None:
        with pytest.raises(ModelUnavailableError, match="weights not found"):
            resolve_model_path(tmp_path / "does_not_exist.pt")

    def test_relative_default_resolves_against_repo_root(self, detector_weights: Path) -> None:
        # The committed default is a repo-relative path; it must resolve without
        # the operator rewriting configuration for every working directory.
        resolved = resolve_model_path(Path("models") / detector_weights.name)
        assert resolved.is_file()


class TestDetector:
    def test_load_is_idempotent_and_loads_once(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        # FR-11 / §9.2: models are loaded once and reused.
        detector = build_detector(detection_settings(make_settings, detector_weights))
        first = detector.load()
        second = detector.load()
        assert first is second
        assert detector.loaded

    def test_detector_info_reports_weights_hash_and_classes(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        detector = build_detector(detection_settings(make_settings, detector_weights))
        info = detector.load()
        assert info.model_name == "yolov8n-pose.pt"
        assert len(info.weights_sha256) == 64
        assert "person" in info.classes
        assert info.parameters > 0
        assert info.imgsz == 640
        # The status view must be JSON-serialisable and omit the path.
        payload = info.as_dict()
        assert "model_path" not in payload
        assert payload["classes"] == list(info.classes)

    def test_detect_before_load_raises(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        detector = build_detector(detection_settings(make_settings, detector_weights))
        with pytest.raises(ModelUnavailableError, match="not loaded"):
            detector.detect(make_packet(np.zeros((64, 64, 3), dtype=np.uint8)))

    def test_detect_returns_batch_with_frame_identity(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        detector = build_detector(detection_settings(make_settings, detector_weights))
        detector.load()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = make_packet(frame, seq=12)
        batch = detector.detect(packet)
        assert batch.camera_id == CAMERA
        assert batch.stream_session_id == SESSION
        assert batch.frame_seq == 12
        assert batch.frame_shape == (480, 640, 3)
        assert batch.frame_size == (640, 480)
        assert batch.source_timestamp == pytest.approx(1.2)
        assert batch.model_name == "yolov8n-pose.pt"
        assert batch.timing is not None
        assert batch.timing.total_ms >= 0.0

    def test_detections_are_clipped_to_frame_bounds(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        detector = build_detector(detection_settings(make_settings, detector_weights))
        detector.load()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        batch = detector.detect(make_packet(frame))
        height, width = 240, 320
        for detection in batch.detections:
            x1, y1, x2, y2 = detection.bbox
            assert 0.0 <= x1 <= width
            assert 0.0 <= y1 <= height
            assert 0.0 <= x2 <= width
            assert 0.0 <= y2 <= height

    def test_class_filter_restricts_reported_classes(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        settings = detection_settings(make_settings, detector_weights, classes=("person",))
        detector = build_detector(settings)
        info = detector.load()
        assert info.classes == ("person",)

    def test_mps_request_falls_back_to_cpu_when_unavailable(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        # §8 portability: CPU is always available. On a host without MPS the
        # detector must report the fallback device rather than crash.
        import torch

        settings = detection_settings(make_settings, detector_weights)
        settings = settings.model_copy(update={"device": "mps"})
        detector = build_detector(settings)
        info = detector.load()
        if torch.backends.mps.is_available():
            assert info.device == "mps"
        else:
            assert info.device == "cpu"

    def test_intel_cpu_runs_a_reused_openvino_export(
        self,
        make_settings: Callable[..., Settings],
        detector_weights: Path,
        tmp_path: Path,
    ) -> None:
        # The export is written next to the weights, so work on a copy in tmp_path.
        weights = tmp_path / detector_weights.name
        weights.write_bytes(detector_weights.read_bytes())
        settings = detection_settings(make_settings, weights, imgsz=320)
        settings = settings.model_copy(update={"device": "intel:cpu"})

        info = build_detector(settings).load()
        assert (info.device, info.runtime) == ("intel:cpu", "openvino")
        export = tmp_path / f"{weights.stem}_320_openvino_model"
        assert any(export.glob("*.xml"))
        mtime = export.stat().st_mtime_ns

        detector = build_detector(settings)
        detector.load()
        assert export.stat().st_mtime_ns == mtime
        batch = detector.detect(make_packet(np.zeros((240, 320, 3), dtype=np.uint8)))
        assert batch.detections == ()

    def test_intel_gpu_request_reports_the_device_actually_used(
        self, make_settings: Callable[..., Settings], detector_weights: Path, tmp_path: Path
    ) -> None:
        import openvino as ov

        weights = tmp_path / detector_weights.name
        weights.write_bytes(detector_weights.read_bytes())
        settings = detection_settings(make_settings, weights, imgsz=320)
        settings = settings.model_copy(update={"device": "intel:gpu"})
        info = build_detector(settings).load()
        if any(d.startswith("GPU") for d in ov.Core().available_devices):
            assert (info.device, info.runtime) == ("intel:gpu", "openvino")
        else:
            assert (info.device, info.runtime) == ("cpu", "pytorch")


class TestDetectionService:
    def test_disabled_service_ignores_frames(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        settings = detection_settings(make_settings, detector_weights, enabled=False)
        service = DetectionService(settings)
        assert not service.enabled
        # A disabled service must not touch the frame or load a model.
        service.on_frame(make_packet(np.zeros((64, 64, 3), dtype=np.uint8)))
        assert service.latest_batch() is None
        assert not service.loaded

    def test_service_caches_latest_batch_and_annotated_frame(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        settings = detection_settings(make_settings, detector_weights)
        service = DetectionService(settings)
        service.load()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        service.on_frame(make_packet(frame))
        assert service.latest_batch() is not None
        annotated = service.latest_annotated()
        assert annotated is not None
        assert annotated.shape == frame.shape
        assert service.stats.frames_processed == 1
        assert service.stats.last_frame_seq == 0

    def test_reconnect_clears_cached_state(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        # §9.2: a new stream session clears the previous attempt's state, so a
        # preview can never mix two decoding sessions.
        settings = detection_settings(make_settings, detector_weights)
        service = DetectionService(settings)
        service.load()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        service.on_frame(make_packet(frame, session_id=SESSION, seq=0))
        assert service.latest_batch() is not None

        other = uuid.uuid4()
        service.on_frame(make_packet(frame, session_id=other, seq=0))
        batch = service.latest_batch()
        assert batch is not None
        assert batch.stream_session_id == other

    def test_snapshot_reports_model_readiness_separately(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        settings = detection_settings(make_settings, detector_weights)
        service = DetectionService(settings)
        assert service.snapshot()["ready"] is False
        service.load()
        snapshot = service.snapshot()
        assert snapshot["ready"] is True
        assert snapshot["enabled"] is True
        assert snapshot["model"]["model_name"] == "yolov8n-pose.pt"

    def test_unload_clears_state_and_releases_model(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        settings = detection_settings(make_settings, detector_weights)
        service = DetectionService(settings)
        service.load()
        service.on_frame(make_packet(np.zeros((240, 320, 3), dtype=np.uint8)))
        service.unload()
        assert service.latest_batch() is None
        assert service.latest_annotated() is None
        assert not service.loaded

    def test_detector_injection_avoids_a_second_model(
        self, make_settings: Callable[..., Settings], detector_weights: Path
    ) -> None:
        settings = detection_settings(make_settings, detector_weights)
        detector = build_detector(settings)
        service = DetectionService(settings, detector=detector)
        service.load()
        assert service.info is detector.info
        assert isinstance(detector, Detector)
