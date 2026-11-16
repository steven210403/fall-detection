"""Frame consumer composing detection and tracking.

Thesis plan §9 shows one pipeline: decode → detect → track. The video layer
exposes exactly one seam, ``PipelineSupervisor(on_frame=...)``, and Phase 3
attached :meth:`DetectionService.on_frame` there. Phase 4 needs tracking to run on
the batches that detection produces, which means one consumer has to drive both
without either service reaching into the other:

    FramePacket → DetectionBatch → TrackObservations

That composition lives here, not inside ``detection`` (which must not know about
tracking) and not inside ``tracking``'s service (which must not know about
frames). Keeping it in a small module means the Phase 6 application, the Phase 4
verification CLI, and the tests all attach the same object to the same seam.

The consumer deliberately performs no work of its own beyond ordering: it does not
buffer frames, drop samples, or cache anything the two services already cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fall_detection.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    import numpy as np

    from fall_detection.core.contracts import FramePacket
    from fall_detection.detection.service import DetectionService
    from fall_detection.tracking.contracts import TrackObservations
    from fall_detection.tracking.service import TrackingService

__all__ = ["DetectionTrackingConsumer"]

logger = get_logger("tracking.consumer")


class DetectionTrackingConsumer:
    """Run detection and then tracking for every decoded frame.

    Attach an instance to ``PipelineSupervisor(on_frame=consumer)``. A frame that
    produces no detections still updates the tracker: an empty frame is how the
    tracker learns that a person is missing, and it is what drives the loss
    tolerance in :mod:`fall_detection.tracking.lifecycle`.

    Args:
        detection: The process's single detection service.
        tracking: The process's single tracking service.
    """

    def __init__(self, detection: DetectionService, tracking: TrackingService) -> None:
        self._detection = detection
        self._tracking = tracking
        self._updates = 0

    # -- introspection -------------------------------------------------

    @property
    def detection(self) -> DetectionService:
        """The detection service this consumer drives."""
        return self._detection

    @property
    def tracking(self) -> TrackingService:
        """The tracking service this consumer drives."""
        return self._tracking

    @property
    def updates(self) -> int:
        """Number of frames that reached tracking."""
        return self._updates

    def latest_annotated(self) -> np.ndarray | None:
        """Most recent annotated frame, from the detection service."""
        return self._detection.latest_annotated()

    def latest_tracks(self) -> TrackObservations | None:
        """Most recent tracked observations, from the tracking service."""
        return self._tracking.latest()

    # -- pipeline seam -------------------------------------------------

    def __call__(self, packet: FramePacket) -> None:
        """Detect on one frame, then track the resulting detections.

        Exceptions propagate to the supervisor, which counts and logs a failing
        consumer and keeps decoding — the same contract Phase 3 relies on, so one
        bad frame cannot stop the camera.

        When detection is disabled there is nothing to track: tracking consumes
        detections, not pixels, so the frame is skipped rather than turned into a
        fake empty batch.
        """
        if not self._detection.enabled:
            return
        self._detection.on_frame(packet)
        if not self._tracking.enabled:
            return
        batch = self._detection.latest_batch()
        if batch is None:
            # Detection ran but produced no batch, which should be impossible
            # after a successful ``on_frame``. Skipping keeps the loop alive; the
            # supervisor's consumer-error counter stays at zero for a non-error.
            logger.debug("no detection batch for frame %d; skipping tracking", packet.frame_seq)
            return
        self._tracking.update(batch)
        self._updates += 1

    def snapshot(self) -> dict[str, Any]:
        """Combined status payload for the camera/status endpoints."""
        return {
            "detection": self._detection.snapshot(),
            "tracking": self._tracking.snapshot(),
            "consumer_updates": self._updates,
        }
