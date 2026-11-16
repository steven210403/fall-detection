"""Track lifecycle: UUID identity, loss tolerance, and termination.

Thesis plan §11.2 defines the semantics this module implements, and none of it is
delegated to the Ultralytics tracker:

* **Track ID** is a temporary association identifier *within one stream session*.
* **Lost** means no matching observation arrived. A track may recover for
  ``loss_tolerance_s`` (initially two seconds) before it is terminated.
* **Recovery** reuses the same ``track_uuid`` only while the track is still alive.
* **Termination** ends the track, closes its buffer, and consumes its identity: a
  returning person receives a **new** UUID, and the integer ID is retired for the
  rest of the session so a reused integer cannot resurrect an old identity.
* **No merging** of IDs over long periods or across cameras.

Why this is not just bookkeeping around the tracker's own frame buffer: the
tracker counts *frames*, and §9.1 warns that "a frame count is not automatically a
duration in seconds". The rules here are evaluated on the monotonic clock, so the
two-second tolerance holds whatever the processing rate turns out to be. The
adapter keeps the tracker's frame buffer comfortably above the tolerance (see
:mod:`fall_detection.tracking.tracker`) so a recovery inside the tolerance still
works.

Termination ordering matters and is fixed here: tracks are aged *before* the
frame's observations are accepted, and a display ID that already ended is never
resurrected. A person who returns after the tolerance therefore becomes a new
track in the very same update that reports the old one ending.

The class is deliberately free of NumPy, PyTorch, and Ultralytics: it is pure
stdlib plus the tracking contracts, which is what makes the time-based rules
testable with a :class:`~fall_detection.core.clock.ManualClock` instead of sleeps.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from fall_detection.core.contracts import StreamSessionInfo
from fall_detection.tracking.contracts import TrackState, TrackSummary, TrackTermination

__all__ = ["LiveTrack", "ObserveResult", "TrackLifecycle", "TrackMatch"]


@dataclass(frozen=True, slots=True)
class TrackMatch:
    """Identity facts for one observed person, before frame metadata is attached.

    The lifecycle owns identity and time; the service owns frame context. This
    record is the hand-off between the two, so neither has to fake the other's
    fields.

    Attributes:
        track_uuid: Identity of the track the observation belongs to.
        display_id: Integer tracker ID (display/debugging only).
        first_seen_monotonic: Monotonic seconds when the track started.
        age_s: Time since the track started.
        observed_frames: Frames with an observation, including this one.
        gap_frames: Consecutive frames without an observation immediately before
            this one; ``0`` when the observation was uninterrupted.
        recovered: ``True`` when this observation ended a gap.
    """

    track_uuid: uuid.UUID
    display_id: int
    first_seen_monotonic: float
    age_s: float
    observed_frames: int
    gap_frames: int
    recovered: bool


@dataclass(frozen=True, slots=True)
class ObserveResult:
    """Result of registering one observation.

    Attributes:
        match: Identity facts for the observation.
        superseded: Set when the observation arrived *after* the loss tolerance
            elapsed, so the previous track ended and this observation started a
            new one. §11.2 requires the new identity; reporting the old track's
            termination on the same update is what makes the change auditable
            instead of a silent identity swap.
    """

    match: TrackMatch
    superseded: TrackTermination | None = None


@dataclass(slots=True)
class LiveTrack:
    """Mutable runtime record for one track.

    Kept separate from the immutable contracts so no caller can observe a
    half-updated track: summaries and observations are built from this record and
    handed out as frozen dataclasses.

    Attributes:
        track_uuid: Identity minted when the track started.
        display_id: Integer tracker ID (display/debugging only).
        session: Stream session the track belongs to.
        first_seen_monotonic: Monotonic seconds when the track started.
        first_seen_utc: UTC epoch seconds when the track started.
        last_observed_monotonic: Monotonic seconds of the newest observation.
        last_observed_utc: UTC epoch seconds of the newest observation.
        last_bbox: Box of the newest observation (never a prediction).
        last_confidence: Score of the newest observation.
        last_frame_seq: Frame sequence of the newest observation.
        state: Current lifecycle state.
        observed_frames: Frames with a matched observation.
        missing_frames: Frames without one while still alive.
        gap_frames: Consecutive missing frames immediately before the newest
            observation; reset to ``0`` by every observation.
        recovered: Set while the newest observation followed a gap.
    """

    track_uuid: uuid.UUID
    display_id: int
    session: StreamSessionInfo
    first_seen_monotonic: float
    first_seen_utc: float
    last_observed_monotonic: float
    last_observed_utc: float
    last_bbox: tuple[float, float, float, float]
    last_confidence: float
    last_frame_seq: int
    state: TrackState = TrackState.ACTIVE
    observed_frames: int = 1
    missing_frames: int = 0
    gap_frames: int = 0
    recovered: bool = False


class TrackLifecycle:
    """Own the live tracks of one stream session.

    The camera is read from the active session rather than held separately, so
    the internal key ``camera_id + stream_session_uuid + track_uuid`` (§11.2)
    cannot drift out of sync, and one instance can never serve two cameras.

    Args:
        loss_tolerance_s: Seconds a track survives without an observation before
            termination (§11.2).
        stale_after_s: Seconds after which the last observation stops being
            usable evidence (§11.2).
        lock: Optional lock override. The inference thread drives the lifecycle
            while the API thread reads snapshots, so access is serialised.
    """

    def __init__(
        self,
        *,
        loss_tolerance_s: float,
        stale_after_s: float,
        lock: threading.RLock | None = None,
    ) -> None:
        if loss_tolerance_s <= 0:
            raise ValueError("loss_tolerance_s must be positive")
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        self._loss_tolerance_s = loss_tolerance_s
        self._stale_after_s = stale_after_s
        self._lock = lock or threading.RLock()
        self._session: StreamSessionInfo | None = None
        self._tracks: dict[uuid.UUID, LiveTrack] = {}
        self._by_display: dict[int, uuid.UUID] = {}
        self._retired_display_ids: set[int] = set()

    # -- properties ----------------------------------------------------

    @property
    def camera_id(self) -> uuid.UUID | None:
        """Camera of the active session, or ``None`` when none is open."""
        with self._lock:
            return None if self._session is None else self._session.camera_id

    @property
    def loss_tolerance_s(self) -> float:
        """Configured loss tolerance in seconds."""
        return self._loss_tolerance_s

    @property
    def stale_after_s(self) -> float:
        """Configured staleness threshold in seconds."""
        return self._stale_after_s

    @property
    def session(self) -> StreamSessionInfo | None:
        """Session currently tracked, or ``None`` when none is open."""
        with self._lock:
            return self._session

    def live_count(self) -> int:
        """Number of tracks that have not been terminated."""
        with self._lock:
            return len(self._tracks)

    def retired_display_ids(self) -> tuple[int, ...]:
        """Integer IDs consumed by terminated tracks in this session."""
        with self._lock:
            return tuple(sorted(self._retired_display_ids))

    # -- session handling ----------------------------------------------

    def begin_session(self, session: StreamSessionInfo) -> tuple[TrackTermination, ...]:
        """Start tracking a new stream session.

        §9.2 requires a reconnect to clear tracker state, and §11.2 makes a
        session change the end of every track in it. This terminates whatever is
        still live, discards its display IDs, and empties the registry — the
        mechanism that stops buffers from two sessions being mixed.

        Returns:
            The terminations produced by closing the previous session, so the
            caller can persist and report them.
        """
        with self._lock:
            ended: tuple[TrackTermination, ...] = ()
            if self._tracks:
                ended = self._terminate_all_locked(reason="session_ended")
            self._session = session
            self._tracks = {}
            self._by_display = {}
            self._retired_display_ids = set()
            return ended

    # -- updates -------------------------------------------------------

    def observe(
        self,
        *,
        display_id: int,
        bbox: tuple[float, float, float, float],
        confidence: float,
        frame_seq: int,
        now_monotonic: float,
        now_utc: float,
    ) -> ObserveResult:
        """Record one observed person and return its identity facts.

        A display ID that has never been seen mints a new UUID. A display ID whose
        track already ended also mints a new UUID — termination consumes the
        identity, so the tracker may reuse the integer without resurrecting the
        old track (§11.2).

        An observation that arrives more than ``loss_tolerance_s`` after the
        previous one for the same display ID is treated as a new person, even
        though the tracker kept the integer: the tolerance is a *time* rule, and
        §11.2 says a person who has been lost that long gets a new UUID. The
        superseded track is returned so its end can be reported and persisted
        rather than disappearing between updates.

        Args:
            display_id: Integer tracker ID from the adapter.
            bbox: Matched detection box in original-frame pixels.
            confidence: Matched detection score.
            frame_seq: Frame counter within the session.
            now_monotonic: Monotonic seconds of this update.
            now_utc: UTC epoch seconds of this update.

        Returns:
            An :class:`ObserveResult` with the identity facts and, when the
            previous track was superseded, its termination.

        Raises:
            RuntimeError: No session is open; call :meth:`begin_session` first.
        """
        with self._lock:
            session = self._session
            if session is None:
                raise RuntimeError("no stream session is active; call begin_session first")

            existing = self._by_display.get(display_id)
            track = self._tracks.get(existing) if existing is not None else None
            superseded: TrackTermination | None = None

            if track is not None:
                gap = now_monotonic - track.last_observed_monotonic
                if gap > self._loss_tolerance_s:
                    # The tolerance already elapsed (this happens when no frame
                    # was processed during the gap, for example after a stall).
                    # The old track ends here instead of being silently reused.
                    superseded = self._termination_locked(track, reason="lost_timeout")
                    self._tracks.pop(track.track_uuid, None)
                    self._retired_display_ids.add(display_id)
                    track = None

            if track is None:
                # A retired integer ID must not inherit the ended identity.
                self._retired_display_ids.discard(display_id)
                track = LiveTrack(
                    track_uuid=uuid.uuid4(),
                    display_id=display_id,
                    session=session,
                    first_seen_monotonic=now_monotonic,
                    first_seen_utc=now_utc,
                    last_observed_monotonic=now_monotonic,
                    last_observed_utc=now_utc,
                    last_bbox=bbox,
                    last_confidence=confidence,
                    last_frame_seq=frame_seq,
                )
                self._tracks[track.track_uuid] = track
                self._by_display[display_id] = track.track_uuid
            else:
                track.gap_frames = track.missing_frames
                track.recovered = track.missing_frames > 0
                track.missing_frames = 0
                track.state = TrackState.ACTIVE
                track.last_observed_monotonic = now_monotonic
                track.last_observed_utc = now_utc
                track.last_bbox = bbox
                track.last_confidence = confidence
                track.last_frame_seq = frame_seq
                track.observed_frames += 1

            match = TrackMatch(
                track_uuid=track.track_uuid,
                display_id=track.display_id,
                first_seen_monotonic=track.first_seen_monotonic,
                age_s=max(0.0, now_monotonic - track.first_seen_monotonic),
                observed_frames=track.observed_frames,
                gap_frames=track.gap_frames,
                recovered=track.recovered,
            )
            return ObserveResult(match=match, superseded=superseded)

    def mark_missing(
        self,
        observed_display_ids: set[int],
        *,
        now_monotonic: float,
    ) -> tuple[TrackTermination, ...]:
        """Age the tracks that were not observed in this frame.

        Called *before* the frame's observations are accepted, so a person who
        returns after the tolerance elapsed starts a new track instead of
        recovering the old one (§11.2). A track inside the tolerance moves to
        ``lost`` and its missing-frame counter grows; a track beyond it is
        terminated with reason ``lost_timeout``.

        Args:
            observed_display_ids: Display IDs matched in this frame; these are
                never aged or terminated.
            now_monotonic: Monotonic seconds of this update.

        Returns:
            Terminations detected on this update, lowest display ID first.
        """
        with self._lock:
            ended: list[TrackTermination] = []
            for track in self._sorted_tracks_locked():
                if track.display_id in observed_display_ids:
                    continue
                # Counted before the decision so `missing_frames` reports the true
                # gap length, including the frame that exhausted the tolerance.
                track.missing_frames += 1
                gap = now_monotonic - track.last_observed_monotonic
                if gap > self._loss_tolerance_s:
                    self._tracks.pop(track.track_uuid, None)
                    self._by_display.pop(track.display_id, None)
                    self._retired_display_ids.add(track.display_id)
                    ended.append(self._termination_locked(track, reason="lost_timeout"))
                    continue
                track.state = TrackState.LOST
            return tuple(ended)

    # -- teardown ------------------------------------------------------

    def end_session(self, *, reason: str = "stopped") -> tuple[TrackTermination, ...]:
        """Terminate every live track and clear the registry.

        Used when the pipeline stops or the process shuts down. The session is
        dropped too, so a restart begins with a clean slate and cannot mix the
        previous session's buffers.
        """
        with self._lock:
            ended = self._terminate_all_locked(reason=reason)
            self._session = None
            self._tracks = {}
            self._by_display = {}
            self._retired_display_ids = set()
            return ended

    # -- reads ---------------------------------------------------------

    def summaries(self, *, now_monotonic: float) -> tuple[TrackSummary, ...]:
        """Build a summary per live track, ordered by display ID.

        Args:
            now_monotonic: Monotonic seconds used for ages and staleness.
        """
        with self._lock:
            return tuple(
                self._summary_locked(track, now_monotonic=now_monotonic)
                for track in self._sorted_tracks_locked()
            )

    def summary_for(self, track_uuid: uuid.UUID, *, now_monotonic: float) -> TrackSummary | None:
        """Return one track's summary, or ``None`` when it is not live."""
        with self._lock:
            track = self._tracks.get(track_uuid)
            if track is None:
                return None
            return self._summary_locked(track, now_monotonic=now_monotonic)

    def is_stale(self, track_uuid: uuid.UUID, *, now_monotonic: float) -> bool:
        """Whether a track's last observation is older than ``stale_after_s``.

        §11.2: old predictions become stale after 0.5 seconds, which is the point
        where a consumer must stop using the track's last state as evidence. An
        unknown or terminated track reports ``True``, because there is nothing
        usable to consume.
        """
        with self._lock:
            track = self._tracks.get(track_uuid)
            if track is None:
                return True
            return (now_monotonic - track.last_observed_monotonic) > self._stale_after_s

    # -- internals -----------------------------------------------------

    def _sorted_tracks_locked(self) -> list[LiveTrack]:
        """Live tracks ordered by display ID, then UUID for determinism."""
        return sorted(
            self._tracks.values(), key=lambda track: (track.display_id, str(track.track_uuid))
        )

    def _summary_locked(self, track: LiveTrack, *, now_monotonic: float) -> TrackSummary:
        gap = max(0.0, now_monotonic - track.last_observed_monotonic)
        return TrackSummary(
            track_uuid=track.track_uuid,
            display_id=track.display_id,
            camera_id=track.session.camera_id,
            stream_session_id=track.session.stream_session_id,
            state=track.state,
            first_seen_monotonic=track.first_seen_monotonic,
            last_observed_monotonic=track.last_observed_monotonic,
            age_s=max(0.0, now_monotonic - track.first_seen_monotonic),
            time_since_last_observation_s=gap,
            stale=gap > self._stale_after_s,
            observed_frames=track.observed_frames,
            missing_frames=track.missing_frames,
            bbox=track.last_bbox,
            confidence=track.last_confidence,
        )

    def _termination_locked(self, track: LiveTrack, *, reason: str) -> TrackTermination:
        """Build a termination record.

        ``ended_monotonic``/``ended_utc`` are the track's **last observation**,
        not the moment the decision was made: FR-07 requires event times to come
        from observation time, and the loss tolerance is a grace period rather
        than evidence. ``missing_frames`` reports how long that grace period was
        actually used, so the gap is never hidden.
        """
        return TrackTermination(
            track_uuid=track.track_uuid,
            display_id=track.display_id,
            camera_id=track.session.camera_id,
            stream_session_id=track.session.stream_session_id,
            reason=reason,
            ended_monotonic=track.last_observed_monotonic,
            ended_utc=track.last_observed_utc,
            duration_s=max(0.0, track.last_observed_monotonic - track.first_seen_monotonic),
            observed_frames=track.observed_frames,
            missing_frames=track.missing_frames,
        )

    def _terminate_all_locked(self, *, reason: str) -> tuple[TrackTermination, ...]:
        ended: list[TrackTermination] = []
        for track in self._sorted_tracks_locked():
            track.state = TrackState.TERMINATED
            ended.append(self._termination_locked(track, reason=reason))
        return tuple(ended)
