"""Detection service: the seam between the video pipeline and the detector.

Thesis plan §9 shows detection sitting directly after decoding, and §3 fixes the
architecture as **one dedicated inference thread that owns model and tracking
state**. :class:`PipelineSupervisor` calls an ``on_frame`` callback for every
decoded frame; this service is what Phase 3 attaches there.

Responsibilities, and deliberately nothing more:

* Load and warm up the model **once** and reuse it for every frame (§9.2).
* Run detection on one frame at a time and keep the most recent
  :class:`DetectionBatch` and annotated frame for the API and the next phase.
* Count frames, detections, and measured inference time for status payloads.
* Clear per-session state when a reconnect opens a **new** stream session
  (§9.2: "A reconnect creates a new stream session and clears old tracks and
  buffers").

It does **not** track, assign identity, extract features, classify activity, or
write to the database. Tracking is Phase 4; features are Phase 5; events are
Phase 6. Storing the latest batch is a read-only convenience for the status
endpoint — no ORM model is touched here (thesis plan §17.1).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fall_detection.config import Settings
from fall_detection.core.clock import Clock, SystemClock
from fall_detection.core.contracts import FramePacket
from fall_detection.core.errors import ModelUnavailableError
from fall_detection.core.logging import get_logger
from fall_detection.detection.annotate import annotate_frame
from fall_detection.detection.contracts import DetectionBatch
from fall_detection.detection.detector import Detector, DetectorInfo, build_detector

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    import numpy as np

__all__ = ["DetectionService", "DetectionStats"]

logger = get_logger("detection.service")


@dataclass(slots=True)
class DetectionStats:
    """Counters and last-measured timings for status payloads.

    Every figure is a measurement of this process, not a target and not an
    accuracy claim (thesis plan §8, §15).
    """

    frames_processed: int = 0
    detections_total: int = 0
    persons_total: int = 0
    last_detections: int = 0
    last_persons: int = 0
    last_inference_ms: float = 0.0
    last_frame_seq: int | None = None
    errors: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        mean = (
            round(self.detections_total / self.frames_processed, 3)
            if self.frames_processed
            else 0.0
        )
        return {
            "frames_processed": self.frames_processed,
            "detections_total": self.detections_total,
            "persons_total": self.persons_total,
            "last_detections": self.last_detections,
            "last_persons": self.last_persons,
            "last_inference_ms": round(self.last_inference_ms, 3),
            "mean_detections_per_frame": mean,
            "last_frame_seq": self.last_frame_seq,
            "errors": self.errors,
        }


class DetectionService:
    """Own one detector and turn frames into :class:`DetectionBatch` objects.

    Args:
        settings: Application settings; ``settings.detection`` configures the
            model and ``settings.detection.enabled`` decides whether the
            pipeline calls the detector at all.
        clock: Injectable clock.
        detector: Optional detector override, used by tests.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Clock | None = None,
        detector: Detector | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or SystemClock()
        self._detector = detector or build_detector(settings, clock=self._clock)
        self._lock = threading.RLock()
        self._stats = DetectionStats()
        self._latest_batch: DetectionBatch | None = None
        self._latest_annotated: np.ndarray | None = None
        self._session_id: uuid.UUID | None = None

    # -- lifecycle -----------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether configuration enables detection."""
        return self._settings.detection.enabled

    @property
    def loaded(self) -> bool:
        """Whether the model has been loaded and warmed."""
        return self._detector.loaded

    @property
    def info(self) -> DetectorInfo | None:
        """Description of the loaded detector, or ``None`` before loading."""
        return self._detector.info

    @property
    def stats(self) -> DetectionStats:
        """Live counters."""
        return self._stats

    def load(self) -> DetectorInfo:
        """Load and warm up the model once.

        Raises:
            ModelUnavailableError: Weights or device are unavailable.
        """
        return self._detector.load()

    def unload(self) -> None:
        """Drop the model reference so shutdown releases it deterministically.

        The next :meth:`load` builds a fresh detector; nothing else is cached, so
        this cannot leak a partially initialised model.
        """
        with self._lock:
            self._latest_batch = None
            self._latest_annotated = None
            self._session_id = None
            self._detector = build_detector(self._settings, clock=self._clock)

    # -- pipeline seam -------------------------------------------------

    def on_frame(self, packet: FramePacket) -> None:
        """Detect on one frame; suitable as ``PipelineSupervisor(on_frame=...)``.

        A new ``stream_session_id`` clears the cached batch and annotated frame,
        so a preview never mixes two decoding attempts (thesis plan §9.2).

        Exceptions propagate: the supervisor counts and logs a failing consumer
        and keeps decoding, which is the correct behaviour for a single bad
        frame (a corrupted frame should not silently produce a stale box).
        """
        if not self.enabled:
            return

        if self._session_id != packet.stream_session_id:
            with self._lock:
                self._session_id = packet.stream_session_id
                self._latest_batch = None
                self._latest_annotated = None

        try:
            batch = self.process(packet)
            annotated = annotate_frame(packet.frame, batch)
        except Exception:
            # Counted here as well as by the supervisor's consumer-error counter,
            # so the detection stats describe detection failures specifically.
            with self._lock:
                self._stats.errors += 1
            raise
        with self._lock:
            self._latest_batch = batch
            self._latest_annotated = annotated

    def process(self, packet: FramePacket) -> DetectionBatch:
        """Run detection on one frame and update counters.

        Returns:
            The detection batch for this frame.
        """
        batch = self._detector.detect(packet)
        persons = batch.persons()
        with self._lock:
            self._stats.frames_processed += 1
            self._stats.detections_total += len(batch.detections)
            self._stats.persons_total += len(persons)
            self._stats.last_detections = len(batch.detections)
            self._stats.last_persons = len(persons)
            self._stats.last_frame_seq = batch.frame_seq
            if batch.timing is not None:
                self._stats.last_inference_ms = batch.timing.total_ms
        return batch

    # -- read-only accessors -------------------------------------------

    def latest_batch(self) -> DetectionBatch | None:
        """Most recent batch, or ``None`` before the first frame."""
        with self._lock:
            return self._latest_batch

    def latest_annotated(self) -> np.ndarray | None:
        """Most recent annotated frame, or ``None`` before the first frame."""
        with self._lock:
            return self._latest_annotated

    def snapshot(self) -> dict[str, Any]:
        """Status payload for ``GET /api/v1/status`` (thesis plan §17.2).

        Reports model readiness separately from camera state, exactly as §17.2
        requires ("Present model readiness and camera status separately").
        """
        info = self._detector.info
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "ready": self.loaded,
            "stats": self._stats.as_dict(),
        }
        if info is not None:
            payload["model"] = info.as_dict()
        return payload


def ensure_loaded(service: DetectionService) -> DetectorInfo:
    """Load the service's model or raise a typed error.

    Convenience for CLIs and the lifespan, mirroring ``video.ensure_stopped``.
    """
    if not service.enabled:
        raise ModelUnavailableError("detection is disabled in configuration")
    return service.load()
