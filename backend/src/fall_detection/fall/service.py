"""Fall service: per-person geometry, velocity, and state machine (THESIS_PLAN §3–§5).

:class:`FallService` is the ``TrackObservations → FallDecision`` link. For every tracked
person in a frame it measures the three fall parameters, advances that person's
:class:`~fall_detection.fall.fsm.FallStateMachine`, and routes alarms to the event writer:

* an alarm is **opened** (persisted and broadcast) on the frame the state machine enters
  ``fallen``;
* it is **closed** when the person recovers (``recovered``), when the track ends
  (``track_lost``), when the session changes (``session_ended``), when the operator stops
  the camera (``stopped``), or when it exceeds ``max_event_s`` (``max_duration``).

Time comes from the source's timestamps when the frame has one, otherwise from the
tracking monotonic clock, and every feature reports which. A person whose time source
changes mid-track starts a fresh velocity history, because differences across two clocks
are meaningless.

The service owns per-person state and is driven by one thread (the inference thread), so
it is guarded by one lock only for the read accessors the API calls.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fall_detection.config import FallSettings, Settings
from fall_detection.core.logging import get_logger
from fall_detection.events.contracts import OpenFallEvent
from fall_detection.fall.contracts import (
    FallBatch,
    FallDecision,
    FallFeatures,
    FallState,
    FallTransition,
    PoseGeometry,
    TimeSource,
)
from fall_detection.fall.fsm import FallStateMachine
from fall_detection.fall.geometry import compute_geometry
from fall_detection.fall.kinematics import HipVelocityEstimator

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from fall_detection.events.writer import FallEventWriter
    from fall_detection.tracking.contracts import TrackObservation, TrackObservations

__all__ = ["FallService", "FallStats"]

logger = get_logger("fall.service")

_TERMINATION_TO_END_REASON: dict[str, str] = {
    "lost_timeout": "track_lost",
    "session_ended": "session_ended",
    "stopped": "stopped",
}


@dataclass(slots=True)
class FallStats:
    """Counters for the status payload."""

    frames: int = 0
    observations: int = 0
    unmeasurable: int = 0
    falls_started: int = 0
    impacts: int = 0
    alerts: int = 0
    recoveries: int = 0
    events_closed: int = 0
    handovers: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable view."""
        return {
            "frames": self.frames,
            "observations": self.observations,
            "unmeasurable": self.unmeasurable,
            "falls_started": self.falls_started,
            "impacts": self.impacts,
            "alerts": self.alerts,
            "recoveries": self.recoveries,
            "events_closed": self.events_closed,
            "handovers": self.handovers,
        }


@dataclass(slots=True)
class _Person:
    machine: FallStateMachine
    velocity: HipVelocityEstimator
    display_id: int
    time_source: TimeSource | None = None
    last_t: float | None = None
    last_utc: float | None = None
    last_decision: FallDecision | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    standing: deque[tuple[float, float]] = field(default_factory=deque)
    transitions: list[FallTransition] = field(default_factory=list)


class FallService:
    """Detect falls from tracked people.

    Args:
        settings: Application settings; ``settings.fall`` provides the thresholds,
            with the frozen calibration applied when one is configured.
        events: Optional event writer. Without one the service still decides states
            (the offline evaluation runs this way) but stores nothing.
        record_transitions: Keep every state transition in :attr:`transition_log`,
            which the offline evaluation reads.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        events: FallEventWriter | None = None,
        record_transitions: bool = False,
        fall_settings: FallSettings | None = None,
    ) -> None:
        self._settings = settings
        self._fall = (fall_settings or settings.fall).with_calibration()
        self._events = events
        self._record = record_transitions
        self._lock = threading.RLock()
        self._people: dict[uuid.UUID, _Person] = {}
        self._session: uuid.UUID | None = None
        self._latest: FallBatch | None = None
        self._stats = FallStats()
        self._log: list[FallTransition] = []
        self._alerts: list[OpenFallEvent] = []

    # -- introspection --------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether fall detection is configured on."""
        return self._fall.enabled

    @property
    def thresholds(self) -> FallSettings:
        """The thresholds actually in use (after calibration)."""
        return self._fall

    @property
    def transition_log(self) -> tuple[FallTransition, ...]:
        """Every transition seen so far, when ``record_transitions`` is on."""
        with self._lock:
            return tuple(self._log)

    @property
    def alert_log(self) -> tuple[OpenFallEvent, ...]:
        """Every alarm opened so far, when ``record_transitions`` is on."""
        with self._lock:
            return tuple(self._alerts)

    def attach_events(self, events: FallEventWriter) -> None:
        """Attach the event writer after construction."""
        self._events = events

    def latest(self) -> FallBatch | None:
        """The most recent frame's batch."""
        with self._lock:
            return self._latest

    def current_states(self) -> dict[uuid.UUID, FallDecision]:
        """Each live person's latest decision."""
        with self._lock:
            return {
                track: person.last_decision
                for track, person in self._people.items()
                if person.last_decision is not None
            }

    def active_alarms(self) -> int:
        """Number of people currently in the ``fallen`` state."""
        with self._lock:
            return sum(
                1 for person in self._people.values() if person.machine.state is FallState.FALLEN
            )

    # -- frame update ---------------------------------------------------

    def update(self, tracks: TrackObservations) -> FallBatch:
        """Advance every observed person by one frame.

        Tracks that ended in this update have their open alarm closed first, so a
        terminated identity can never keep an alarm open.
        """
        with self._lock:
            if tracks.stream_session_id != self._session:
                self._end_all_locked("session_ended", utc=tracks.receive_utc)
                self._session = tracks.stream_session_id

            closed: list[uuid.UUID] = []
            for termination in tracks.terminated:
                reason = _TERMINATION_TO_END_REASON.get(termination.reason, "track_lost")
                if self._close_person_locked(
                    termination.track_uuid, reason, utc=tracks.receive_utc
                ):
                    closed.append(termination.track_uuid)

            decisions: list[FallDecision] = []
            alerts: list[uuid.UUID] = []
            for observation in tracks.observations:
                decision = self._observe_locked(observation, tracks)
                decisions.append(decision)
                if decision.alert_fired:
                    alerts.append(decision.track_uuid)
            observed = {observation.track_uuid for observation in tracks.observations}
            for track_uuid, person in list(self._people.items()):
                if track_uuid in observed:
                    continue
                decision = self._tick_unobserved_locked(track_uuid, person, tracks)
                if decision is not None:
                    decisions.append(decision)
                    alerts.append(track_uuid)

            self._stats.frames += 1
            batch = FallBatch(
                stream_session_id=tracks.stream_session_id,
                frame_seq=tracks.frame_seq,
                decisions=tuple(decisions),
                closed_tracks=tuple(closed),
                alerts=tuple(alerts),
            )
            self._latest = batch
            return batch

    def end_session(self, *, reason: str = "stopped", utc: float | None = None) -> None:
        """Close every open alarm and forget every person (camera stop, shutdown)."""
        with self._lock:
            self._end_all_locked(
                reason if reason in ("stopped", "session_ended") else "stopped", utc=utc
            )
            self._session = None
            self._latest = None

    # -- internals ------------------------------------------------------

    def _observe_locked(
        self, observation: TrackObservation, tracks: TrackObservations
    ) -> FallDecision:
        settings = self._fall
        person = self._people.get(observation.track_uuid)
        if person is None:
            person = self._handover_locked(observation, tracks)
        if person is None:
            person = _Person(
                machine=FallStateMachine(settings, observation.track_uuid),
                velocity=HipVelocityEstimator(
                    window=settings.velocity_window,
                    max_gap_s=settings.max_velocity_gap_s,
                ),
                display_id=observation.display_id,
            )
            self._people[observation.track_uuid] = person

        if observation.source_timestamp is not None:
            t, source = float(observation.source_timestamp), "source"
        else:
            t, source = float(observation.observed_monotonic), "monotonic"
        if person.time_source is not None and person.time_source != source:
            person.velocity.reset()
        person.time_source = source

        geometry = compute_geometry(
            observation.bbox, observation.keypoints, min_conf=settings.keypoint_min_conf
        )
        hip_y = None if geometry.hip_center is None else geometry.hip_center[1]
        vy, vy_norm = person.velocity.update(t, hip_y, geometry.body_length_px)
        height = observation.bbox[3] - observation.bbox[1]
        height_ratio = _height_ratio(person, t, height, geometry, settings)
        features = FallFeatures(
            track_uuid=observation.track_uuid,
            display_id=observation.display_id,
            frame_seq=observation.frame_seq,
            t=t,
            utc=observation.receive_utc,
            time_source=source,
            geometry=geometry,
            vy_px_s=vy,
            vy_norm=vy_norm,
            source_timestamp=observation.source_timestamp,
            height_ratio=height_ratio,
        )
        self._stats.observations += 1
        if geometry.aspect_ratio is None and geometry.torso_angle_deg is None:
            self._stats.unmeasurable += 1

        decision = person.machine.step(features)
        person.last_t = t
        person.last_utc = observation.receive_utc
        person.last_bbox = observation.bbox
        person.display_id = observation.display_id
        person.last_decision = decision
        for transition in decision.transitions:
            self._on_transition_locked(person, transition, tracks)
        if decision.alert_fired:
            self._open_alarm_locked(person, decision, tracks)
        self._check_max_duration_locked(person, decision)
        return decision

    def _tick_unobserved_locked(
        self, track_uuid: uuid.UUID, person: _Person, tracks: TrackObservations
    ) -> FallDecision | None:
        """Run the long-lie timer for a person in ``impact`` who is not visible now."""
        if person.machine.state is not FallState.IMPACT or person.time_source is None:
            return None
        if person.time_source == "source" and tracks.source_timestamp is not None:
            t = float(tracks.source_timestamp)
        elif person.time_source == "monotonic":
            t = float(tracks.tracking_monotonic)
        else:
            return None
        features = FallFeatures(
            track_uuid=track_uuid,
            display_id=person.display_id,
            frame_seq=tracks.frame_seq,
            t=t,
            utc=tracks.receive_utc,
            time_source=person.time_source,
            geometry=_UNOBSERVED,
            vy_px_s=None,
            vy_norm=None,
            source_timestamp=tracks.source_timestamp,
        )
        decision = person.machine.tick(features)
        if decision is None:
            return None
        person.last_decision = decision
        for transition in decision.transitions:
            self._on_transition_locked(person, transition, tracks)
        self._open_alarm_locked(person, decision, tracks)
        return decision

    def _handover_locked(
        self, observation: TrackObservation, tracks: TrackObservations
    ) -> _Person | None:
        """Give a new track the fall state of a person the tracker just lost.

        A falling body changes shape and speed abruptly, and the tracker often loses the
        person and starts a new identity in the middle of the fall. Without this the new
        identity's state machine would never see the velocity spike and the fall would be
        missed. The state moves only from a person who is **not** observed in this frame,
        was last seen within ``handover_s``, and whose last box centre is within
        ``handover_distance`` body lengths of the new box centre; the closest such person
        wins. Moving the state (rather than copying it) means one fall can never become
        two alarms.
        """
        settings = self._fall
        if settings.handover_s <= 0.0:
            return None
        observed = {item.track_uuid for item in tracks.observations}
        t_now = (
            float(observation.source_timestamp)
            if observation.source_timestamp is not None
            else float(observation.observed_monotonic)
        )
        center = _center(observation.bbox)
        best: tuple[float, uuid.UUID] | None = None
        for track_uuid, person in self._people.items():
            if track_uuid in observed or person.last_t is None or person.last_bbox is None:
                continue
            gap = t_now - person.last_t
            if gap < 0.0 or gap > settings.handover_s:
                continue
            previous = _center(person.last_bbox)
            length = max(
                person.velocity.reference_length() or 0.0,
                person.last_bbox[2] - person.last_bbox[0],
                person.last_bbox[3] - person.last_bbox[1],
            )
            if length <= 0.0:
                continue
            distance = ((center[0] - previous[0]) ** 2 + (center[1] - previous[1]) ** 2) ** 0.5
            if distance > settings.handover_distance * length:
                continue
            if best is None or distance < best[0]:
                best = (distance, track_uuid)
        if best is None:
            return None
        person = self._people.pop(best[1])
        self._people[observation.track_uuid] = person
        person.machine.rebind(observation.track_uuid)
        self._stats.handovers += 1
        logger.debug("fall state handed over %s -> %s", best[1], observation.track_uuid)
        return person

    def _on_transition_locked(
        self, person: _Person, transition: FallTransition, tracks: TrackObservations
    ) -> None:
        if self._record:
            self._log.append(transition)
        if transition.to_state is FallState.FALLING:
            self._stats.falls_started += 1
        elif transition.to_state is FallState.IMPACT:
            self._stats.impacts += 1
        elif transition.to_state is FallState.NORMAL and transition.from_state is FallState.FALLEN:
            self._stats.recoveries += 1
            event_id = person.machine.take_closed_event()
            if event_id is not None:
                self._close_event(person, event_id, "recovered", utc=transition.utc)
        logger.debug(
            "track=%s %s -> %s (%s)",
            transition.track_uuid,
            transition.from_state.value,
            transition.to_state.value,
            transition.reason,
        )

    def _open_alarm_locked(
        self, person: _Person, decision: FallDecision, tracks: TrackObservations
    ) -> None:
        machine = person.machine
        event_id = machine.event_id
        if event_id is None:
            return
        utc = decision.features.utc
        event = OpenFallEvent(
            event_id=event_id,
            track_uuid=decision.track_uuid,
            camera_id=tracks.camera_id,
            stream_session_id=tracks.stream_session_id,
            display_id=person.display_id,
            falling_utc=machine.falling_utc if machine.falling_utc is not None else utc,
            impact_utc=machine.impact_utc if machine.impact_utc is not None else utc,
            alert_utc=utc,
            response_time_s=max(
                0.0,
                decision.features.t
                - (machine.impact_t if machine.impact_t is not None else decision.features.t),
            ),
            peak_vy=machine.peak_vy,
            min_torso_angle_deg=machine.min_torso_angle_deg,
            max_aspect_ratio=machine.max_aspect_ratio,
            alert_source_timestamp=decision.features.source_timestamp,
        )
        self._stats.alerts += 1
        if self._record:
            self._alerts.append(event)
        events = self._events
        if events is not None:
            events.open(event)

    def _check_max_duration_locked(self, person: _Person, decision: FallDecision) -> None:
        machine = person.machine
        alert_t = machine.alert_t
        if machine.state is not FallState.FALLEN or alert_t is None:
            return
        if decision.features.t - alert_t < self._fall.max_event_s:
            return
        event_id = machine.event_id
        # Restart the person's machine so a still-lying person can raise a new alarm
        # only after a new fall, rather than holding one row open forever.
        person.machine = FallStateMachine(self._fall, decision.track_uuid)
        if event_id is not None:
            self._close_event(person, event_id, "max_duration", utc=decision.features.utc)

    def _close_person_locked(
        self, track_uuid: uuid.UUID, reason: str, *, utc: float | None
    ) -> bool:
        person = self._people.pop(track_uuid, None)
        if person is None:
            return False
        event_id = person.machine.event_id
        if event_id is not None:
            self._close_event(person, event_id, reason, utc=person.last_utc if utc is None else utc)
        return True

    def _end_all_locked(self, reason: str, *, utc: float | None) -> None:
        for track_uuid in list(self._people):
            self._close_person_locked(track_uuid, reason, utc=utc)

    def _close_event(
        self, person: _Person, event_id: uuid.UUID, reason: str, *, utc: float | None
    ) -> None:
        self._stats.events_closed += 1
        events = self._events
        if events is None:
            return
        ended = utc if utc is not None else (person.last_utc or 0.0)
        machine = person.machine
        events.close(
            event_id,
            ended_utc=ended,
            end_reason=reason,
            peak_vy=machine.peak_vy,
            min_torso_angle_deg=machine.min_torso_angle_deg,
            max_aspect_ratio=machine.max_aspect_ratio,
        )

    def snapshot(self) -> dict[str, Any]:
        """Status payload describing the fall layer."""
        with self._lock:
            counts: dict[str, int] = {state.value: 0 for state in FallState}
            for person in self._people.values():
                counts[person.machine.state.value] += 1
            thresholds = self._fall
            return {
                "enabled": thresholds.enabled,
                "people": len(self._people),
                "states": counts,
                "active_alarms": counts[FallState.FALLEN.value],
                "stats": self._stats.as_dict(),
                "thresholds": {
                    "ar_threshold": thresholds.ar_threshold,
                    "angle_lying_deg": thresholds.angle_lying_deg,
                    "angle_upright_deg": thresholds.angle_upright_deg,
                    "vy_threshold": thresholds.vy_threshold,
                    "impact_window_s": thresholds.impact_window_s,
                    "long_lie_s": thresholds.long_lie_s,
                    "require_both": thresholds.require_both,
                    "recovery_hold_s": thresholds.recovery_hold_s,
                    "calibrated": self._settings.fall.calibration_path is not None,
                    "calibration_path": None
                    if self._settings.fall.calibration_path is None
                    else str(self._settings.fall.calibration_path),
                },
            }


_UNOBSERVED = PoseGeometry(
    aspect_ratio=None,
    torso_angle_deg=None,
    torso_side=None,
    hip_center=None,
    body_length_px=None,
    visible_keypoints=0,
)
"""Geometry of a timer tick for a person who is not visible: nothing was measured."""


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _height_ratio(
    person: _Person,
    t: float,
    height: float,
    geometry: Any,
    settings: FallSettings,
) -> float | None:
    """Box height relative to the person's recent standing height.

    The reference is the tallest box seen while the torso was upright within
    ``standing_window_s``; it is updated *before* the ratio is taken, so a standing
    person reads ~1.0 and a person lying along the camera axis reads well below it.
    """
    angle = geometry.torso_angle_deg
    ratio = geometry.aspect_ratio
    standing_now = (
        angle is not None
        and angle > settings.angle_upright_deg
        and ratio is not None
        and ratio < settings.ar_threshold
    )
    window = person.standing
    if standing_now and height > 0.0:
        window.append((t, height))
    while window and t - window[0][0] > settings.standing_window_s:
        window.popleft()
    if not window:
        return None
    reference = max(value for _, value in window)
    return None if reference <= 0.0 else height / reference
