"""Frame consumer: detection + tracking + fall detection + preview (THESIS_PLAN §3).

The video layer exposes one seam, ``PipelineSupervisor(on_frame=...)``. This consumer
runs every stage for every decoded frame::

    FramePacket → DetectionBatch (boxes + 17 keypoints) → TrackObservations
                → FallDecision per person → preview JPEG + live snapshot

Pose runs on **every** decoded frame: the hip velocity needs a dense sampling rate, so
there is no separate, slower feature grid.

The preview is drawn from the same frame the decisions came from: the annotated array
the detection service built for *this* packet, captioned with each person's display ID,
state, torso angle and aspect ratio on *this* frame's tracked box, and framed in red while
any person is in the ``fallen`` state.

Per-stage latency (detection, tracking, fall logic, end-to-end) and the processing rate
(frames per second over a sliding window) are measured here, because this is the only
place that sees all stages of one frame.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

from fall_detection.core.logging import get_logger
from fall_detection.fall.contracts import FallDecision, FallState
from fall_detection.tracking.consumer import DetectionTrackingConsumer

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    import numpy as np

    from fall_detection.core.contracts import FramePacket
    from fall_detection.detection.service import DetectionService
    from fall_detection.fall.contracts import FallBatch
    from fall_detection.fall.service import FallService
    from fall_detection.streaming.frames import FrameEnvelope, PreviewStream
    from fall_detection.streaming.snapshots import SnapshotBroadcaster
    from fall_detection.tracking.contracts import TrackObservations
    from fall_detection.tracking.service import TrackingService

__all__ = ["FallConsumer", "PerformanceMeter"]

logger = get_logger("fall.consumer")

STATE_TEXT: dict[FallState, str] = {
    FallState.UNKNOWN: "TIDAK TERUKUR",
    FallState.NORMAL: "NORMAL",
    FallState.FALLING: "JATUH?",
    FallState.IMPACT: "TERGELETAK",
    FallState.FALLEN: "JATUH - ALARM",
}
"""Preview caption per state (the dashboard is in Bahasa Indonesia)."""

_STATE_BACKGROUND: dict[FallState, tuple[int, int, int]] = {
    FallState.FALLING: (0, 140, 255),
    FallState.IMPACT: (0, 90, 230),
    FallState.FALLEN: (0, 0, 220),
}


class PerformanceMeter:
    """Processing rate and per-stage latency over a sliding window of frames.

    FPS is ``N_frames / t_total`` (the draft's definition) over the frames finished in
    the last ``window_s`` seconds of wall time, so it reports what the pipeline is doing
    now rather than an average since start-up.
    """

    def __init__(self, *, window_s: float = 5.0, max_samples: int = 2048) -> None:
        self._window_s = window_s
        self._finished: deque[float] = deque(maxlen=max_samples)
        self._stages: dict[str, deque[float]] = {}
        self._max_samples = max_samples
        self._lock = threading.Lock()
        self._frames = 0

    def record(self, finished_monotonic: float, stages_ms: dict[str, float]) -> None:
        """Record one processed frame."""
        with self._lock:
            self._frames += 1
            self._finished.append(finished_monotonic)
            for name, value in stages_ms.items():
                bucket = self._stages.setdefault(name, deque(maxlen=self._max_samples))
                bucket.append(value)

    def snapshot(self, now_monotonic: float | None = None) -> dict[str, Any]:
        """FPS and mean/p95 latency per stage over the window."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        with self._lock:
            recent = [value for value in self._finished if now - value <= self._window_s]
            fps = None
            if len(recent) >= 2:
                span = recent[-1] - recent[0]
                fps = (len(recent) - 1) / span if span > 0 else None
            stages: dict[str, dict[str, float]] = {}
            count = len(recent)
            for name, bucket in self._stages.items():
                values = list(bucket)[-max(count, 1) :]
                if not values:
                    continue
                ordered = sorted(values)
                p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
                stages[name] = {
                    "mean_ms": round(sum(values) / len(values), 3),
                    "p95_ms": round(p95, 3),
                }
            return {
                "fps": None if fps is None else round(fps, 2),
                "window_s": self._window_s,
                "frames": self._frames,
                "latency": stages,
            }


class FallConsumer:
    """Run detection, tracking, and fall detection for every decoded frame.

    Args:
        detection: The process's single detection service.
        tracking: The process's single tracking service.
        fall: The process's single fall service.
        preview: Optional preview stream.
        broadcaster: Optional live broadcaster.
    """

    def __init__(
        self,
        detection: DetectionService,
        tracking: TrackingService,
        fall: FallService,
        *,
        preview: PreviewStream | None = None,
        broadcaster: SnapshotBroadcaster | None = None,
    ) -> None:
        self._pipeline = DetectionTrackingConsumer(detection, tracking)
        self._fall = fall
        self._preview = preview
        self._broadcaster = broadcaster
        self._lock = threading.RLock()
        self._previews = 0
        self._preview_errors = 0
        self._fall_errors = 0
        self._preview_session: uuid.UUID | None = None
        self._performance = PerformanceMeter()

    # -- introspection ---------------------------------------------------

    @property
    def detection(self) -> DetectionService:
        """The detection service this consumer drives."""
        return self._pipeline.detection

    @property
    def tracking(self) -> TrackingService:
        """The tracking service this consumer drives."""
        return self._pipeline.tracking

    @property
    def fall(self) -> FallService:
        """The fall service this consumer drives."""
        return self._fall

    @property
    def updates(self) -> int:
        """Number of frames that reached tracking."""
        return self._pipeline.updates

    @property
    def performance(self) -> PerformanceMeter:
        """FPS and latency measurements."""
        return self._performance

    @property
    def fall_errors(self) -> int:
        """Frames whose fall evaluation raised."""
        with self._lock:
            return self._fall_errors

    def latest_annotated(self) -> np.ndarray | None:
        """Most recent annotated frame, from the detection service."""
        return self._pipeline.latest_annotated()

    def latest_tracks(self) -> TrackObservations | None:
        """Most recent tracked observations."""
        return self._pipeline.latest_tracks()

    def latest_preview(self) -> FrameEnvelope | None:
        """Most recent encoded preview frame, or ``None``."""
        return None if self._preview is None else self._preview.latest()

    # -- pipeline seam -----------------------------------------------------

    def __call__(self, packet: FramePacket) -> None:
        """Detect, track, and evaluate one frame, then publish it.

        Exceptions from detection and tracking propagate to the supervisor, which counts
        them and keeps decoding. The fall and preview stages swallow their own failures,
        because a missed preview must not stop the camera.
        """
        started = time.monotonic()
        self._pipeline(packet)
        tracked = time.monotonic()

        batch = self._pipeline.detection.latest_batch()
        if batch is None or batch.frame_seq != packet.frame_seq:
            return
        tracks = self._pipeline.tracking.latest()
        if tracks is None or tracks.frame_seq != packet.frame_seq:
            return

        fall_batch: FallBatch | None = None
        if self._fall.enabled:
            try:
                fall_batch = self._fall.update(tracks)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the camera
                with self._lock:
                    self._fall_errors += 1
                logger.warning("fall evaluation failed for frame %d: %s", packet.frame_seq, exc)
        decided = time.monotonic()

        detection_ms = batch.timing.total_ms if batch.timing is not None else None
        stages = {
            "pipeline_ms": (tracked - started) * 1000.0,
            "fall_ms": (decided - tracked) * 1000.0,
            "total_ms": (decided - started) * 1000.0,
        }
        if detection_ms is not None:
            stages["detection_ms"] = detection_ms
            stages["tracking_ms"] = max(0.0, stages["pipeline_ms"] - detection_ms)
        self._performance.record(decided, stages)
        self._publish_preview(packet, tracks, fall_batch)

    def _publish_preview(
        self,
        packet: FramePacket,
        tracks: TrackObservations,
        fall_batch: FallBatch | None,
    ) -> None:
        preview = self._preview
        if preview is None or not preview.enabled:
            return
        if packet.stream_session_id != self._preview_session:
            preview.clear()
            self._preview_session = packet.stream_session_id
        annotated = self._pipeline.latest_annotated()
        if annotated is None:
            return
        if not preview.due():
            return
        states = self._fall.current_states()
        annotated = self._caption(annotated, tracks, states)
        try:
            envelope = preview.publish(
                annotated,
                camera_id=packet.camera_id,
                stream_session_id=packet.stream_session_id,
                frame_seq=packet.frame_seq,
                captured_monotonic=packet.receive_monotonic,
                captured_utc=packet.receive_utc,
                source_timestamp=packet.source_timestamp,
            )
        except Exception as exc:  # noqa: BLE001 - a preview must not stop the camera
            with self._lock:
                self._preview_errors += 1
            logger.warning("could not publish a preview for frame %d: %s", packet.frame_seq, exc)
            return
        if envelope is None:
            return
        with self._lock:
            self._previews += 1
        self._publish_snapshot(packet, tracks, states, envelope)

    def _caption(
        self,
        annotated: np.ndarray,
        tracks: TrackObservations,
        states: dict[uuid.UUID, FallDecision],
    ) -> np.ndarray:
        captions: list[tuple[tuple[float, float, float, float], tuple[str, ...]]] = []
        backgrounds: list[tuple[int, int, int] | None] = []
        alarm = False
        for observation in tracks.observations:
            decision = states.get(observation.track_uuid)
            lines = [f"#{observation.display_id}"]
            background = None
            if decision is not None:
                lines[0] += f" {STATE_TEXT[decision.state]}"
                geometry = decision.features.geometry
                parts = []
                if geometry.torso_angle_deg is not None:
                    parts.append(f"sudut {geometry.torso_angle_deg:.0f}")
                if geometry.aspect_ratio is not None:
                    parts.append(f"AR {geometry.aspect_ratio:.2f}")
                if parts:
                    lines.append(" ".join(parts))
                if decision.long_lie_remaining_s is not None:
                    lines.append(f"alarm dalam {decision.long_lie_remaining_s:.1f}s")
                background = _STATE_BACKGROUND.get(decision.state)
                alarm = alarm or decision.state is FallState.FALLEN
            captions.append((observation.bbox, tuple(lines)))
            backgrounds.append(background)
        if not captions and not any(
            decision.state is FallState.FALLEN for decision in states.values()
        ):
            return annotated
        try:
            from fall_detection.detection.annotate import annotate_alert, annotate_captions

            canvas = annotate_captions(annotated.copy(), captions, backgrounds=backgrounds)
            if alarm or any(decision.state is FallState.FALLEN for decision in states.values()):
                canvas = annotate_alert(canvas, "PERINGATAN: TERDETEKSI JATUH")
            return canvas
        except Exception as exc:  # noqa: BLE001 - a caption must not stop the preview
            logger.debug("could not caption the preview: %s", exc)
            return annotated

    def _publish_snapshot(
        self,
        packet: FramePacket,
        tracks: TrackObservations,
        states: dict[uuid.UUID, FallDecision],
        envelope: FrameEnvelope,
    ) -> None:
        broadcaster = self._broadcaster
        if broadcaster is None:
            return
        broadcaster.publish_snapshot(
            {
                "camera_id": str(packet.camera_id),
                "stream_session_id": str(packet.stream_session_id),
                "frame_seq": packet.frame_seq,
                "source_timestamp": packet.source_timestamp,
                "receive_monotonic": packet.receive_monotonic,
                "receive_utc": packet.receive_utc,
                "preview": envelope.as_dict(),
                "tracks": track_payload(tracks, states),
                "active_alarms": sum(
                    1 for decision in states.values() if decision.state is FallState.FALLEN
                ),
                "metrics": {
                    "consumer_updates": self._pipeline.updates,
                    "fall_errors": self.fall_errors,
                    **self._performance.snapshot(),
                },
            }
        )

    def publish_status(self, payload: dict[str, Any]) -> None:
        """Publish a status change immediately."""
        broadcaster = self._broadcaster
        if broadcaster is not None:
            broadcaster.publish_status(payload)

    def snapshot(self) -> dict[str, Any]:
        """Combined status payload for the status endpoint."""
        return {
            **self._pipeline.snapshot(),
            "fall": self._fall.snapshot(),
            "performance": self._performance.snapshot(),
            "previews": self._previews,
            "preview_errors": self._preview_errors,
            "fall_errors": self.fall_errors,
        }


def track_payload(
    tracks: TrackObservations | None,
    states: dict[uuid.UUID, FallDecision],
) -> list[dict[str, Any]]:
    """Per-person live payload: identity, box, and fall state with its measurements.

    A track with no decision yet is still reported with ``fall: null``, because "not
    evaluated yet" and "this person does not exist" are different facts.
    """
    if tracks is None:
        return []
    payload: list[dict[str, Any]] = []
    for summary in tracks.tracks:
        if not summary.is_live:
            continue
        decision = states.get(summary.track_uuid)
        payload.append(
            {
                "track_uuid": str(summary.track_uuid),
                "display_id": summary.display_id,
                "state": summary.state.value,
                "bbox": list(summary.bbox) if summary.bbox is not None else None,
                "age_s": round(summary.age_s, 3),
                "time_since_last_observation_s": round(summary.time_since_last_observation_s, 3),
                "stale": summary.stale,
                "confidence": None if summary.confidence is None else round(summary.confidence, 4),
                "fall": None if decision is None else decision.as_dict(),
            }
        )
    return payload
