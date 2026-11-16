"""Tracking service: the seam between detection and the feature layer.

Thesis plan §9 places person tracking directly after detection, and §3 fixes the
architecture as **one dedicated inference thread that owns model and tracking
state**. This service is what Phase 4 attaches downstream of
:class:`~fall_detection.detection.contracts.DetectionBatch`; it implements the
``DetectionBatch → TrackObservations`` link of the §17.1 contract chain.

Responsibilities, and deliberately nothing more:

* Own exactly one tracker adapter and one lifecycle registry, so tracking state
  has a single owner (§3).
* Convert a frame's person detections into tracked observations with stable
  UUIDs (§11.2).
* Apply the **time-based** rules: loss tolerance, staleness, termination, and the
  rule that a returning person gets a new identity.
* Reset everything when a new stream session begins, which is what §9.2 and the
  Phase 4 acceptance criterion "session reset; no buffer mixing" require.
* Persist track rows through the single database writer, so activity events in
  later phases have a durable identity to reference.
* Report measured counters and the resolved tracker configuration for status
  payloads.

It does **not** extract features, classify activity, associate objects with
zones, render previews, or write activity events. Features are Phase 5 and events
are Phase 6. Non-person detections (phones, laptops, chairs) are ignored here:
§11.2 and §12.4 make person–object association a *feature* concern, and tracking a
single class keeps the integer IDs meaningful.

No accuracy figure is produced here. IDF1, HOTA, and ID-switch counts are
Experiment 2 (§15.2) and need dense trajectory annotation, which the dataset phase
has not produced (``docs/FALL_PILOT_REPORT.md``).
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from fall_detection.config import Settings
from fall_detection.core.clock import Clock, SystemClock
from fall_detection.core.contracts import StreamSessionInfo
from fall_detection.core.errors import ConfigurationError
from fall_detection.core.logging import get_logger
from fall_detection.database import DatabaseWriter, TrackRepository
from fall_detection.detection.contracts import DetectionBatch
from fall_detection.tracking.contracts import (
    TrackObservation,
    TrackObservations,
    TrackState,
    TrackSummary,
    TrackTermination,
)
from fall_detection.tracking.lifecycle import ObserveResult, TrackLifecycle, TrackMatch
from fall_detection.tracking.tracker import RawTrack, TrackerAdapter, TrackerInfo

__all__ = ["TrackingService", "TrackingStats", "ensure_enabled"]

logger = get_logger("tracking.service")

TERMINATION_REASON_LOST_TIMEOUT = "lost_timeout"
TERMINATION_REASON_SESSION_ENDED = "session_ended"
TERMINATION_REASON_STOPPED = "stopped"


@dataclass(slots=True)
class TrackingStats:
    """Counters and measured rates for status payloads.

    Every figure describes this process: how many frames were tracked, how many
    observations were produced, and how many identities were created and ended.
    None of it is an accuracy measurement, and an ID-switch rate in particular
    cannot be derived here — it needs ground truth (§15.2).
    """

    updates: int = 0
    frames_with_observations: int = 0
    observations_total: int = 0
    last_observations: int = 0
    tracks_started: int = 0
    tracks_terminated: int = 0
    terminations_lost_timeout: int = 0
    terminations_session_ended: int = 0
    terminations_stopped: int = 0
    recoveries: int = 0
    empty_frames: int = 0
    sessions_started: int = 0
    invalid_rows: int = 0
    persistence_errors: int = 0
    tracking_errors: int = 0
    last_update_monotonic: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "updates": self.updates,
            "frames_with_observations": self.frames_with_observations,
            "observations_total": self.observations_total,
            "last_observations": self.last_observations,
            "tracks_started": self.tracks_started,
            "tracks_terminated": self.tracks_terminated,
            "terminations_lost_timeout": self.terminations_lost_timeout,
            "terminations_session_ended": self.terminations_session_ended,
            "terminations_stopped": self.terminations_stopped,
            "recoveries": self.recoveries,
            "empty_frames": self.empty_frames,
            "sessions_started": self.sessions_started,
            "invalid_rows": self.invalid_rows,
            "persistence_errors": self.persistence_errors,
            "tracking_errors": self.tracking_errors,
            "last_update_monotonic": self.last_update_monotonic,
        }


class TrackingService:
    """Turn detection batches into tracked observations.

    Args:
        settings: Application settings; ``settings.tracking`` configures the
            tracker and the lifecycle rules.
        writer: The single database writer. Optional, so tests and offline
            experiments can track without persistence.
        clock: Injectable clock; every time-based rule uses it.
        tracker: Optional adapter override, used by the tracker comparison CLI to
            run both trackers against identical detections.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        writer: DatabaseWriter | None = None,
        clock: Clock | None = None,
        tracker: TrackerAdapter | None = None,
    ) -> None:
        self._settings = settings
        self._writer = writer
        self._clock = clock or SystemClock()
        self._tracking = settings.tracking
        self._lock = threading.RLock()
        self._adapter = tracker or TrackerAdapter(self._tracking, clock=self._clock)
        self._lifecycle = self._new_lifecycle()
        self._stats = TrackingStats()
        self._session_id: uuid.UUID | None = None
        self._latest: TrackObservations | None = None

    # -- lifecycle -----------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether configuration enables tracking."""
        return self._tracking.enabled

    @property
    def tracker_name(self) -> str:
        """Configured tracker name."""
        return self._adapter.name

    @property
    def stats(self) -> TrackingStats:
        """Live counters."""
        return self._stats

    @property
    def info(self) -> TrackerInfo:
        """Resolved tracker configuration and measurements."""
        return self._adapter.info

    @property
    def session_id(self) -> uuid.UUID | None:
        """Currently tracked stream session, or ``None``."""
        with self._lock:
            return self._session_id

    def begin_session(self, session: StreamSessionInfo) -> tuple[TrackTermination, ...]:
        """Start a new stream session, clearing tracker and lifecycle state.

        §9.2: a reconnect creates a new session and clears old tracks and buffers.
        This terminates any track still live in the previous session and resets
        the tracker (fresh association state, integer IDs restarting), which is
        the mechanism behind the Phase 4 acceptance criterion "session reset; no
        buffer mixing".

        Returns:
            The terminations produced by closing the previous session.
        """
        with self._lock:
            ended = self._begin_session_locked(session)
        self._after_terminations(ended)
        return ended

    def end_session(
        self, *, reason: str = TERMINATION_REASON_STOPPED
    ) -> tuple[TrackTermination, ...]:
        """Terminate every live track and clear state.

        Called when the pipeline stops or the process shuts down, so no track row
        is left open and no buffer survives into the next run.

        Returns:
            The terminations produced by ending the session.
        """
        with self._lock:
            ended = self._lifecycle.end_session(reason=reason)
            self._latest = None
            self._session_id = None
        self._after_terminations(ended)
        if ended:
            logger.info("tracking session ended reason=%s tracks=%d", reason, len(ended))
        return ended

    def unload(self) -> None:
        """Drop tracker and lifecycle state so shutdown releases it deterministically.

        Mirrors :meth:`DetectionService.unload`: the next :meth:`update` builds a
        fresh tracker, so a partially used one cannot leak into a restart.
        """
        with self._lock:
            self._latest = None
            self._session_id = None
            self._lifecycle = self._new_lifecycle()
            self._adapter = TrackerAdapter(self._tracking, clock=self._clock)

    def validate(self) -> TrackerInfo:
        """Validate the tracker configuration and return its description.

        Called once at startup so an unavailable tracker is reported as readiness
        state rather than discovered mid-stream. Building the tracker instance
        itself stays deferred to :meth:`begin_session`, because a tracker holds
        per-session association state.

        Raises:
            ConfigurationError: The configured tracker is unavailable.
            ModelUnavailableError: The packaged configuration cannot be loaded.
        """
        self._adapter.validate()
        return self._adapter.info

    # -- pipeline seam -------------------------------------------------

    def on_frame(self, packet: object) -> None:
        """Not a frame consumer: tracking consumes detections, not raw frames.

        Exists only to fail loudly. Attaching this service to
        ``PipelineSupervisor(on_frame=...)`` would skip detection entirely, so the
        call raises instead of silently producing nothing. Use
        :class:`~fall_detection.tracking.consumer.DetectionTrackingConsumer`, which
        runs detection and then calls :meth:`update`.
        """
        raise TypeError(
            "TrackingService.on_frame is not a frame consumer; tracking consumes "
            "DetectionBatch. Attach tracking.consumer.DetectionTrackingConsumer to "
            "the pipeline instead."
        )

    def update(self, batch: DetectionBatch) -> TrackObservations:
        """Track one frame's detections and return the resulting observations.

        A new ``stream_session_id`` starts a new session first, so state from two
        decoding attempts can never be mixed (§9.2).

        Args:
            batch: Detections produced from exactly one frame.

        Returns:
            The tracked observations for that frame.

        Raises:
            ConfigurationError: The tracker is unavailable or misconfigured. A
                single malformed tracker row is counted instead, because it must
                not stop the pipeline.
        """
        with self._lock:
            if self._session_id != batch.stream_session_id:
                # A new session ends every track of the previous one. Those
                # terminations belong to this update: reporting them later would
                # let a caller see a live track that had already ended.
                ended_previous = self._begin_session_locked(
                    StreamSessionInfo(
                        camera_id=batch.camera_id,
                        stream_session_id=batch.stream_session_id,
                        started_utc=batch.receive_utc,
                        started_monotonic=batch.receive_monotonic,
                    )
                )
            else:
                ended_previous = ()

            now_monotonic = self._clock.monotonic()
            now_utc = self._clock.utc_timestamp()
            persons = batch.persons()

            try:
                rows = self._adapter.update(
                    persons, frame_shape=batch.frame_shape, now_monotonic=now_monotonic
                )
            except Exception:
                with self._lock:
                    self._stats.tracking_errors += 1
                raise

            # Age the tracks that were not observed *before* accepting this
            # frame's observations: a person who returns after the tolerance
            # elapsed must start a new track, not recover the old one (§11.2).
            terminations = list(ended_previous)
            terminations.extend(
                self._lifecycle.mark_missing(
                    {row.display_id for row in rows}, now_monotonic=now_monotonic
                )
            )

            observations: list[TrackObservation] = []
            for row in rows:
                keypoints = (
                    persons[row.detection_index].keypoints
                    if 0 <= row.detection_index < len(persons)
                    else None
                )
                observed = self._observe(
                    row, batch=batch, now_monotonic=now_monotonic, now_utc=now_utc
                )
                if observed.superseded is not None:
                    # The tracker kept the integer ID, but the person was gone
                    # longer than the tolerance, so the old track ended here and
                    # this observation belongs to a new identity (§11.2).
                    terminations.append(observed.superseded)
                observations.append(
                    self._observation(
                        row,
                        observed.match,
                        batch=batch,
                        now_monotonic=now_monotonic,
                        keypoints=keypoints,
                    )
                )

            result = TrackObservations(
                camera_id=batch.camera_id,
                stream_session_id=batch.stream_session_id,
                frame_seq=batch.frame_seq,
                observations=tuple(observations),
                tracks=self._lifecycle.summaries(now_monotonic=now_monotonic),
                terminated=tuple(terminations),
                frame_shape=batch.frame_shape,
                source_timestamp=batch.source_timestamp,
                receive_monotonic=batch.receive_monotonic,
                receive_utc=batch.receive_utc,
                tracking_monotonic=now_monotonic,
                tracker_name=self._adapter.name,
            )
            self._latest = result
            self._stats.updates += 1
            self._stats.last_update_monotonic = now_monotonic
            self._stats.last_observations = len(observations)
            self._stats.observations_total += len(observations)
            self._stats.invalid_rows = self._adapter.invalid_rows
            self._stats.recoveries += sum(1 for item in observations if item.recovered)
            if observations:
                self._stats.frames_with_observations += 1
            else:
                self._stats.empty_frames += 1

        self._persist_started(result)
        self._after_terminations(tuple(terminations))
        if terminations:
            logger.info(
                "session=%s frame=%d ended %d track(s): %s",
                batch.stream_session_id,
                batch.frame_seq,
                len(terminations),
                ", ".join(f"{item.display_id}:{item.reason}" for item in terminations),
            )
        return result

    # -- read-only accessors -------------------------------------------

    def latest(self) -> TrackObservations | None:
        """Most recent observations, or ``None`` before the first update."""
        with self._lock:
            return self._latest

    def summaries(self) -> tuple[TrackSummary, ...]:
        """Live track summaries, recomputed with the current clock.

        Recomputation matters because ``stale`` depends on elapsed time: a payload
        built from the last update would understate a gap that has grown since.
        """
        return self._lifecycle.summaries(now_monotonic=self._clock.monotonic())

    def live_track_uuids(self) -> tuple[uuid.UUID, ...]:
        """UUIDs of tracks that have not been terminated."""
        return tuple(summary.track_uuid for summary in self.summaries())

    def is_stale(self, track_uuid: uuid.UUID) -> bool:
        """Whether a track's last observation is older than ``stale_after_s``."""
        return self._lifecycle.is_stale(track_uuid, now_monotonic=self._clock.monotonic())

    def tracking_timestamp(self) -> float:
        """Current monotonic time, for callers that must age summaries themselves."""
        return self._clock.monotonic()

    def state_counts(self) -> dict[str, int]:
        """Live track counts per lifecycle state, for status payloads."""
        counts = {state.value: 0 for state in TrackState}
        for summary in self.summaries():
            counts[summary.state.value] += 1
        return counts

    def snapshot(self) -> dict[str, Any]:
        """Status payload for the camera/status endpoints (thesis plan §17.2).

        Reports tracker readiness separately from camera state, exactly as §17.2
        requires, and includes the live tracks so a client could render a track
        list without waiting for the next frame.
        """
        summaries = self.summaries()
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "ready": self._session_id is not None,
            "tracker": self._adapter.describe(),
            "stats": self._stats.as_dict(),
            "live_tracks": len(summaries),
            "stale_tracks": sum(1 for summary in summaries if summary.stale),
            "state_counts": self.state_counts(),
            "tracks": [summary.as_dict() for summary in summaries],
        }
        if self._session_id is not None:
            payload["stream_session_id"] = str(self._session_id)
        return payload

    # -- internals -----------------------------------------------------

    def _begin_session_locked(self, session: StreamSessionInfo) -> tuple[TrackTermination, ...]:
        """Reset tracker and lifecycle state; caller holds the lock.

        Persistence and counters are deliberately left to the caller so a session
        change inside :meth:`update` counts the terminations exactly once, on the
        update that reports them.
        """
        ended = self._lifecycle.begin_session(session)
        self._session_id = session.stream_session_id
        self._latest = None
        self._adapter.reset()
        self._stats.sessions_started += 1
        logger.info(
            "tracking session started session=%s camera=%s tracker=%s ended_previous=%d",
            session.stream_session_id,
            session.camera_id,
            self._adapter.name,
            len(ended),
        )
        return ended

    def _new_lifecycle(self) -> TrackLifecycle:
        return TrackLifecycle(
            loss_tolerance_s=self._tracking.loss_tolerance_s,
            stale_after_s=self._tracking.stale_after_s,
            lock=self._lock,
        )

    def _observe(
        self,
        row: RawTrack,
        *,
        batch: DetectionBatch,
        now_monotonic: float,
        now_utc: float,
    ) -> ObserveResult:
        """Register one tracker row with the lifecycle and return its identity."""
        return self._lifecycle.observe(
            display_id=row.display_id,
            bbox=row.bbox,
            confidence=row.confidence,
            frame_seq=batch.frame_seq,
            now_monotonic=now_monotonic,
            now_utc=now_utc,
        )

    def _observation(
        self,
        row: RawTrack,
        match: TrackMatch,
        *,
        batch: DetectionBatch,
        now_monotonic: float,
        keypoints: tuple[tuple[float, float, float], ...] | None = None,
    ) -> TrackObservation:
        """Build the immutable observation for one matched person."""
        return TrackObservation(
            track_uuid=match.track_uuid,
            display_id=match.display_id,
            camera_id=batch.camera_id,
            stream_session_id=batch.stream_session_id,
            bbox=row.bbox,
            confidence=row.confidence,
            class_name="person",
            frame_seq=batch.frame_seq,
            observed_monotonic=now_monotonic,
            source_timestamp=batch.source_timestamp,
            receive_monotonic=batch.receive_monotonic,
            receive_utc=batch.receive_utc,
            first_seen_monotonic=match.first_seen_monotonic,
            age_s=match.age_s,
            observed_frames=match.observed_frames,
            gap_frames=match.gap_frames,
            recovered=match.recovered,
            keypoints=keypoints,
        )

    def _after_terminations(self, terminations: tuple[TrackTermination, ...]) -> None:
        """Persist and count one batch of terminations."""
        if not terminations:
            return
        self._persist_terminations(terminations)
        for termination in terminations:
            self._stats.tracks_terminated += 1
            if termination.reason == TERMINATION_REASON_LOST_TIMEOUT:
                self._stats.terminations_lost_timeout += 1
            elif termination.reason == TERMINATION_REASON_SESSION_ENDED:
                self._stats.terminations_session_ended += 1
            elif termination.reason == TERMINATION_REASON_STOPPED:
                self._stats.terminations_stopped += 1

    def _persist_started(self, result: TrackObservations) -> None:
        """Insert one row per track observed for the first time in this frame.

        A ``track`` row references its ``stream_session`` row, which the video
        pipeline writes when it opens a session (``video.pipeline.PipelineSupervisor``).
        Tracking therefore never creates session rows itself: the session's
        lifetime is the decoder's, and duplicating it here would let the two
        disagree. A missing session row surfaces as a counted persistence error
        rather than a crash, because tracking must keep working.

        The counter is incremented whether or not a writer is attached: it
        describes tracking, and a run without persistence is still a run.
        """
        fresh = [item for item in result.observations if item.observed_frames == 1]
        if not fresh:
            return
        self._stats.tracks_started += len(fresh)
        if self._writer is None:
            return
        session_id = result.stream_session_id
        seen_at = self._utc_datetime()

        def work(session: Session) -> None:
            repository = TrackRepository(session)
            for observation in fresh:
                repository.open(
                    track_id=observation.track_uuid,
                    stream_session_id=session_id,
                    tracker_id=observation.display_id,
                    first_seen_at=seen_at,
                )

        self._write(work, description="track open")

    def _persist_terminations(self, terminations: tuple[TrackTermination, ...]) -> None:
        """Close the stored row of every terminated track."""
        if self._writer is None:
            return
        ended_at = self._utc_datetime()

        def work(session: Session) -> None:
            repository = TrackRepository(session)
            for termination in terminations:
                repository.close(
                    track_id=termination.track_uuid,
                    termination_reason=termination.reason,
                    last_seen_at=ended_at,
                    terminated_at=ended_at,
                    observed_frames=termination.observed_frames,
                    missing_frames=termination.missing_frames,
                )

        self._write(work, description="track close")

    def _utc_datetime(self) -> Any:
        """Current UTC time as an aware datetime, from the injected clock."""
        from datetime import UTC, datetime

        return datetime.fromtimestamp(self._clock.utc_timestamp(), tz=UTC)

    def _write(self, work: Callable[[Session], None], *, description: str) -> None:
        """Run one writer transaction, never letting persistence break tracking.

        A database problem must not stop the pipeline: tracking state is
        authoritative in memory, and a failure is logged and counted. This mirrors
        the session-row handling in ``video.pipeline``.
        """
        writer = self._writer
        if writer is None:
            return
        try:
            writer.write(work)
        except Exception as exc:  # noqa: BLE001 - persistence is not on the critical path
            self._stats.persistence_errors += 1
            logger.warning("could not persist %s: %s", description, exc)


def ensure_enabled(service: TrackingService) -> None:
    """Raise :class:`ConfigurationError` when tracking is disabled.

    Convenience for CLIs, mirroring ``detection.ensure_loaded``.
    """
    if not service.enabled:
        raise ConfigurationError("tracking is disabled in configuration")
