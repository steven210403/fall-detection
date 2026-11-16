"""Object detection: YOLO wrapper, detection contracts, and annotated frames.

Phase 3 implements the ``FramePacket → DetectionBatch`` link of the contract
chain fixed in thesis plan §17.1, using the YOLO26 models selected in §10. It
covers the "YOLO n/s inference and annotated frames" row of the §25 roadmap.

Tracking (Phase 4), features (Phase 5), activity (Phases 5–9), and events
(Phase 6) are deliberately absent. ``DetectionService.on_frame`` is the seam the
supervisor attaches to, and :class:`DetectionBatch` is what Phase 4 consumes.
"""

from fall_detection.detection.annotate import (
    CLASS_COLORS,
    DRAWN_CLASSES,
    annotate_captions,
    annotate_frame,
)
from fall_detection.detection.classes import class_id_map, resolve_classes
from fall_detection.detection.contracts import Detection, DetectionBatch, DetectionTiming
from fall_detection.detection.detector import Detector, DetectorInfo, build_detector
from fall_detection.detection.service import DetectionService, DetectionStats, ensure_loaded

__all__ = [
    "CLASS_COLORS",
    "DRAWN_CLASSES",
    "Detection",
    "DetectionBatch",
    "DetectionService",
    "DetectionStats",
    "DetectionTiming",
    "Detector",
    "DetectorInfo",
    "annotate_captions",
    "annotate_frame",
    "build_detector",
    "class_id_map",
    "ensure_loaded",
    "resolve_classes",
]
