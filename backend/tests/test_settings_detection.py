"""Settings tests for the Phase 3 detector configuration.

The detector is a new settings group (thesis plan §10, §20), so these tests assert
the defaults an operator inherits and the environment surface, including the
``__`` nested delimiter used for ``FALL_DETECTION__*``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fall_detection.config import DEFAULT_DETECTION_CLASSES, DetectionSettings, Settings
from fall_detection.config.settings import DetectionClass


class TestDetectionDefaults:
    def test_detection_is_disabled_by_default(self) -> None:
        # Importing or starting the application must never trigger a model load
        # unless the operator opts in.
        assert Settings().detection.enabled is False

    def test_default_model_is_the_lightweight_baseline(self) -> None:
        # §10.1: YOLO26n is the baseline; YOLO26s is the primary target once
        # measured. The default stays the light option.
        settings = DetectionSettings()
        assert settings.model_path.name == "yolov8n-pose.pt"

    def test_default_classes_include_person(self) -> None:
        settings = DetectionSettings()
        assert settings.classes == DEFAULT_DETECTION_CLASSES
        assert "person" in settings.classes

    def test_default_inference_size_is_640(self) -> None:
        # §10.2: 640 is the baseline input size.
        assert DetectionSettings().imgsz == 640

    def test_half_precision_is_off_by_default(self) -> None:
        # §27: eager FP32 initially; MPS half support is not guaranteed.
        assert DetectionSettings().half is False


class TestDetectionValidation:
    def test_imgsz_bounds_are_enforced(self) -> None:
        with pytest.raises(ValidationError):
            DetectionSettings(imgsz=64)
        with pytest.raises(ValidationError):
            DetectionSettings(imgsz=4096)

    def test_confidence_bounds_are_enforced(self) -> None:
        with pytest.raises(ValidationError):
            DetectionSettings(conf=1.5)

    def test_classes_must_not_be_empty(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            DetectionSettings(classes=())

    def test_classes_must_not_repeat(self) -> None:
        with pytest.raises(ValidationError, match="must not contain duplicates"):
            DetectionSettings(classes=("person", "person"))

    def test_unknown_class_is_rejected_at_parse_time(self) -> None:
        # A Literal type rejects a class the project does not model, so a typo
        # surfaces as configuration validation rather than silently dropping boxes.
        with pytest.raises(ValidationError):
            DetectionSettings(classes=("helicopter",))  # type: ignore[arg-type]

    def test_warmup_runs_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DetectionSettings(warmup_runs=99)


class TestDetectionEnvironment:
    def test_nested_detection_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FALL_DETECTION__ENABLED", "true")
        monkeypatch.setenv("FALL_DETECTION__MODEL_PATH", "models/yolo11n-pose.pt")
        monkeypatch.setenv("FALL_DETECTION__IMGSZ", "960")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.detection.enabled is True
        assert settings.detection.model_path == Path("models/yolo11n-pose.pt")
        assert settings.detection.imgsz == 960

    def test_class_list_parses_from_colon_separated_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The classes field is NoDecode, so a colon-separated env value reaches
        # the coercion validator instead of failing JSON parsing.
        monkeypatch.setenv("FALL_DETECTION__CLASSES", "person")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.detection.classes == ("person",)

    def test_a_non_person_class_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A pose model only knows ``person``; any other class is a configuration error.
        monkeypatch.setenv("FALL_DETECTION__CLASSES", "person:laptop")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_detection_group_is_part_of_settings(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        detected: tuple[DetectionClass, ...] = settings.detection.classes
        assert len(detected) == len(DEFAULT_DETECTION_CLASSES)
