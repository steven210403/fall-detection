"""Detection contracts.

The internal contract chain is::

    FramePacket → DetectionBatch → TrackObservations → FallFeatures → FallDecision
                → FallEvent

:class:`DetectionBatch` carries the detections for exactly one frame together with the
timing and identity fields the later links need. Each :class:`Detection` from a pose
model also carries its 17 COCO keypoints. Contracts store only plain tuples, so importing
this package never pulls in NumPy, OpenCV, or PyTorch.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

__all__ = ["Detection", "DetectionBatch", "DetectionTiming", "Keypoint"]

Keypoint = tuple[float, float, float]
"""One body keypoint ``(x, y, confidence)`` in original-frame pixels."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One bounding-box observation in one frame.

    A detection is an *observation*, never a track: it has no identity and no
    history (thesis plan §11.2 distinguishes the two). Coordinates are pixels in
    the original frame, not the letterboxed inference tensor, so geometry in
    later phases is expressed in the same space as the zones.

    Attributes:
        bbox: ``(x1, y1, x2, y2)`` in original-frame pixels, clipped to bounds.
        confidence: Detector score for this box.
        class_name: Detection class, e.g. ``"person"``.
        class_id: Raw class index in the model vocabulary, kept for debugging
            and for reproducing a run against different weights.
        keypoints: The 17 COCO keypoints ``(x, y, conf)`` in original-frame pixels,
            in the model's order, or ``None`` when the model is not a pose model.
            A keypoint the model did not see keeps its low confidence; it is never
            dropped, so indices always mean the same joint.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str
    class_id: int
    keypoints: tuple[Keypoint, ...] | None = None

    @property
    def width(self) -> float:
        """Box width in pixels (never negative)."""
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        """Box height in pixels (never negative)."""
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Box centre in pixels, used for association and zone tests."""
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0

    def is_person(self) -> bool:
        """Whether this observation is a person box."""
        return self.class_name == "person"


@dataclass(frozen=True, slots=True)
class DetectionTiming:
    """Measured inference stages for one batch, in milliseconds.

    Values come from the Ultralytics result's own ``speed`` report plus the
    wrapper's own preprocessing/total stopwatch. They are *measurements of this
    machine*, and the thesis plan forbids presenting them as FPS or accuracy
    claims (§8, §15).

    Attributes:
        preprocess_ms: Resize/letterbox/NCHW conversion inside Ultralytics.
        inference_ms: Forward pass.
        postprocess_ms: NMS and box decoding.
        total_ms: Wall time for the whole :meth:`Detector.detect` call.
    """

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable view for status payloads."""
        return {
            "preprocess_ms": round(self.preprocess_ms, 3),
            "inference_ms": round(self.inference_ms, 3),
            "postprocess_ms": round(self.postprocess_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """All detections produced from one :class:`FramePacket`.

    The batch is the immutable hand-off between detection and tracking. It
    repeats the frame identity (``camera_id``, ``stream_session_id``,
    ``frame_seq``) and both timestamps so a tracker or feature extractor never
    has to hold a reference to the decoded frame array just to learn *which*
    frame a box belongs to.

    ``source_timestamp`` stays ``None`` when the source provided none; it is
    never replaced with a synthetic value (thesis plan §9.1).

    Attributes:
        camera_id: Source camera UUID.
        stream_session_id: Session UUID; changes on every reconnect.
        frame_seq: Zero-based frame counter within the session.
        detections: Detections for this frame, in model order.
        frame_shape: ``(height, width, channels)`` of the original frame, so
            geometry can be normalised without the array.
        source_timestamp: Presentation timestamp in seconds, or ``None``.
        receive_monotonic: Monotonic seconds the frame was assembled.
        receive_utc: UTC epoch seconds the frame was assembled.
        inference_monotonic: Monotonic seconds when inference completed.
        timing: Stage timings, or ``None`` when not measured.
        model_name: Identifier of the weights that produced the batch.
    """

    camera_id: uuid.UUID
    stream_session_id: uuid.UUID
    frame_seq: int
    detections: tuple[Detection, ...]
    frame_shape: tuple[int, int, int]
    source_timestamp: float | None
    receive_monotonic: float
    receive_utc: float
    inference_monotonic: float
    timing: DetectionTiming | None = None
    model_name: str | None = None

    @property
    def frame_size(self) -> tuple[int, int]:
        """``(width, height)`` of the source frame in pixels."""
        height, width = self.frame_shape[0], self.frame_shape[1]
        return int(width), int(height)

    def persons(self) -> tuple[Detection, ...]:
        """Person detections only, in the order the model returned them."""
        return tuple(detection for detection in self.detections if detection.is_person())

    def by_class(self) -> dict[str, tuple[Detection, ...]]:
        """Group detections by class name, preserving order within a class."""
        grouped: dict[str, list[Detection]] = {}
        for detection in self.detections:
            grouped.setdefault(detection.class_name, []).append(detection)
        return {name: tuple(items) for name, items in grouped.items()}

    def counts(self) -> dict[str, int]:
        """Detection count per class name, for status payloads."""
        return {name: len(items) for name, items in self.by_class().items()}
