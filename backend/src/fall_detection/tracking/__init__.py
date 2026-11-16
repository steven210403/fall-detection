"""Multi-object tracking: ByteTrack/BoT-SORT adapters and track lifecycle.

Phase 4 implements the ``DetectionBatch → TrackObservations`` link of the contract
chain fixed in thesis plan §17.1, using the trackers selected in §11.1 (ByteTrack
as the initial primary tracker, BoT-SORT as the mandatory comparison) and the
identity and loss rules in §11.2.

What this package owns:

* ``tracker``: one adapter that drives an explicitly configured Ultralytics
  tracker and converts its output rows into plain records.
* ``lifecycle``: UUID identity, the two-second loss tolerance, the 0.5-second
  staleness threshold, termination, and per-session reset.
* ``service``: the ``update(batch)`` seam, persistence of track rows, and status
  payloads.
* ``consumer``: the object attached to ``PipelineSupervisor(on_frame=...)`` that
  runs detection and then tracking.

Features (Phase 5), activity recognition (Phases 5–9), and events (Phase 6) are
deliberately absent, as are the trackers this project does not implement: DeepSORT
stays a literature-review item (§11.1), and pose-based tracking is deferred (§12.6).
"""

from fall_detection.tracking.consumer import DetectionTrackingConsumer
from fall_detection.tracking.contracts import (
    TRACK_TERMINATION_REASONS,
    TrackObservation,
    TrackObservations,
    TrackState,
    TrackSummary,
    TrackTermination,
)
from fall_detection.tracking.lifecycle import LiveTrack, ObserveResult, TrackLifecycle, TrackMatch
from fall_detection.tracking.service import TrackingService, TrackingStats, ensure_enabled
from fall_detection.tracking.tracker import (
    RawTrack,
    TrackerAdapter,
    TrackerInfo,
    buffer_frames_for,
)

__all__ = [
    "TRACK_TERMINATION_REASONS",
    "DetectionTrackingConsumer",
    "LiveTrack",
    "ObserveResult",
    "RawTrack",
    "TrackLifecycle",
    "TrackMatch",
    "TrackObservation",
    "TrackObservations",
    "TrackState",
    "TrackSummary",
    "TrackTermination",
    "TrackerAdapter",
    "TrackerInfo",
    "TrackingService",
    "TrackingStats",
    "buffer_frames_for",
    "ensure_enabled",
]
