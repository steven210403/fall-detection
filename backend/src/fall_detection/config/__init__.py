"""Typed configuration: one ``Settings`` class read from ``FALL_*`` environment variables."""

from __future__ import annotations

from fall_detection.config.settings import (
    COCO_KEYPOINTS,
    DEFAULT_DETECTION_CLASSES,
    FALL_EVENT_END_REASONS,
    FALL_STATES,
    DetectionClass,
    DetectionSettings,
    FallSettings,
    MediaRootsSettings,
    RetentionSettings,
    Settings,
    StreamingSettings,
    TrackerName,
    TrackingSettings,
    VideoSettings,
    WebcamBackend,
    get_settings,
    webcam_backend_for,
)

__all__ = [
    "COCO_KEYPOINTS",
    "DEFAULT_DETECTION_CLASSES",
    "FALL_EVENT_END_REASONS",
    "FALL_STATES",
    "DetectionClass",
    "DetectionSettings",
    "FallSettings",
    "MediaRootsSettings",
    "RetentionSettings",
    "Settings",
    "StreamingSettings",
    "TrackerName",
    "TrackingSettings",
    "VideoSettings",
    "WebcamBackend",
    "get_settings",
    "webcam_backend_for",
]
