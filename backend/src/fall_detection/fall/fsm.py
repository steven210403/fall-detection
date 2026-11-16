"""The per-person fall state machine (THESIS_PLAN §5).

The draft models a fall as four states::

    NORMAL ──Vy > V_th──► FALLING ──lying posture──► IMPACT ──down for long_lie_s──► FALLEN
       ▲                     │                          │                             │
       └──no impact in time──┘◄──── recovered ──────────┘◄──────── recovered ─────────┘

* **NORMAL → FALLING** when the normalised downward hip velocity exceeds
  ``vy_threshold``. Lying down slowly on a bed never produces the spike, which is how a
  slow, intentional lie-down is told apart from a fall.
* **FALLING → IMPACT** when, within ``impact_window_s`` of the spike, the body reaches a
  lying posture: ``AR >= ar_threshold`` and/or ``θ < angle_lying_deg`` (``require_both``).
  If the window passes without it (sitting down quickly, bending to pick something up),
  the machine returns to NORMAL.
* **IMPACT → FALLEN** when the person has not recovered an upright torso
  (``θ > angle_upright_deg``) by ``long_lie_s`` after the impact. This is the draft's
  "no vertical-posture recovery for 3 seconds" rule, and it is the moment the alarm fires.
* **IMPACT/FALLEN → NORMAL** when an upright posture has persisted for
  ``recovery_hold_s``. From FALLEN this closes the alarm with reason ``recovered``.

Two rules keep the timers honest:

* **Time is observation time.** Every transition is decided on the timestamp of the
  sample that showed it, never on the wall clock and never by counting frames.
* **An unmeasurable sample is not evidence of recovery.** When neither AR nor the torso
  angle can be measured, the state is held and no recovery is counted. The draft's alarm
  rule is "no recovery for ``long_lie_s`` after the impact", so once the deadline passes
  without an observed recovery the alarm fires — also when the person is not detected at
  all any more (:meth:`FallStateMachine.tick`, reason ``long_lie_unobserved``): a pose
  model often loses someone lying on the floor, and that is exactly when an alarm is due.

The class is pure Python and takes its thresholds from
:class:`~fall_detection.config.FallSettings`, so the same object runs live, in the
offline evaluation, and in the unit tests.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from fall_detection.config import FallSettings
from fall_detection.fall.contracts import FallDecision, FallFeatures, FallState, FallTransition

__all__ = ["FallStateMachine", "PostureReading", "read_posture"]


@dataclass(frozen=True, slots=True)
class PostureReading:
    """What one sample says about posture.

    Attributes:
        measurable: Whether AR or the torso angle could be measured.
        lying: The lying-posture rule held (``False`` when not measurable).
        upright: The torso is upright (``False`` when not measurable).
    """

    measurable: bool
    lying: bool
    upright: bool


def read_posture(features: FallFeatures, settings: FallSettings) -> PostureReading:
    """Apply the posture thresholds to one sample."""
    ratio = features.geometry.aspect_ratio
    angle = features.geometry.torso_angle_deg
    if ratio is None and angle is None:
        return PostureReading(measurable=False, lying=False, upright=False)
    wide = ratio is not None and ratio >= settings.ar_threshold
    flat = angle is not None and angle < settings.angle_lying_deg
    if settings.require_both:
        # With both measures required, a missing angle falls back to the box alone:
        # refusing to decide would hide every fall whose hips the model cannot see.
        lying = (wide and flat) if (ratio is not None and angle is not None) else (wide or flat)
    else:
        lying = wide or flat
    if angle is not None:
        upright = angle > settings.angle_upright_deg and not wide
    else:
        upright = ratio is not None and ratio < settings.ar_threshold * 0.8
    height = features.height_ratio
    if upright and height is not None and height < settings.recovery_height_ratio:
        # A torso that looks vertical in a box much shorter than the person's standing
        # height is someone lying along the camera axis, not someone who stood up.
        upright = False
    return PostureReading(measurable=True, lying=lying, upright=upright)


class FallStateMachine:
    """The fall state of one tracked person.

    Args:
        settings: Thresholds and timers.
        track_uuid: The person this machine belongs to.
    """

    def __init__(self, settings: FallSettings, track_uuid: uuid.UUID) -> None:
        self._settings = settings
        self._track_uuid = track_uuid
        self._state = FallState.UNKNOWN
        self._since_t = 0.0
        self._falling_t: float | None = None
        self._falling_utc: float | None = None
        self._impact_t: float | None = None
        self._impact_utc: float | None = None
        self._upright_since: float | None = None
        self._peak_vy: float | None = None
        self._min_angle: float | None = None
        self._max_ratio: float | None = None
        self._event_id: uuid.UUID | None = None
        self._alert_t: float | None = None
        self._alert_utc: float | None = None

    # -- introspection --------------------------------------------------

    @property
    def state(self) -> FallState:
        """Current state."""
        return self._state

    @property
    def event_id(self) -> uuid.UUID | None:
        """The open fall event while the alarm is active."""
        return self._event_id

    @property
    def falling_utc(self) -> float | None:
        """UTC time the current fall's velocity spike was observed."""
        return self._falling_utc

    @property
    def impact_utc(self) -> float | None:
        """UTC time the current fall's impact posture was observed."""
        return self._impact_utc

    @property
    def impact_t(self) -> float | None:
        """Observation time of the current fall's impact."""
        return self._impact_t

    @property
    def alert_t(self) -> float | None:
        """Observation time the alarm fired, while it is active."""
        return self._alert_t

    @property
    def peak_vy(self) -> float | None:
        """Peak normalised downward velocity of the current fall."""
        return self._peak_vy

    @property
    def min_torso_angle_deg(self) -> float | None:
        """Smallest torso angle seen during the current fall."""
        return self._min_angle

    @property
    def max_aspect_ratio(self) -> float | None:
        """Largest aspect ratio seen during the current fall."""
        return self._max_ratio

    def rebind(self, track_uuid: uuid.UUID) -> None:
        """Continue this machine under a new track identity (tracker handover)."""
        self._track_uuid = track_uuid

    # -- update ---------------------------------------------------------

    def step(self, features: FallFeatures) -> FallDecision:
        """Advance the machine with one sample and return the resulting decision."""
        settings = self._settings
        t = features.t
        posture = read_posture(features, settings)
        transitions: list[FallTransition] = []
        alert_fired = False

        if self._state is FallState.UNKNOWN and posture.measurable:
            self._move(FallState.NORMAL, features, "first_measurement", transitions)

        if self._state in (FallState.FALLING, FallState.IMPACT, FallState.FALLEN):
            self._track_extremes(features)

        if self._state is FallState.NORMAL:
            vy = features.vy_norm
            if vy is not None and vy > settings.vy_threshold:
                self._falling_t = t
                self._falling_utc = features.utc
                self._peak_vy = vy
                self._min_angle = None
                self._max_ratio = None
                self._track_extremes(features)
                self._move(FallState.FALLING, features, "velocity_spike", transitions)

        if self._state is FallState.FALLING:
            if posture.lying:
                self._impact_t = t
                self._impact_utc = features.utc
                self._upright_since = None
                self._move(FallState.IMPACT, features, "lying_posture", transitions)
            elif self._falling_t is not None and t - self._falling_t > settings.impact_window_s:
                self._clear_fall()
                self._move(FallState.NORMAL, features, "no_impact", transitions)

        elif self._state is FallState.IMPACT:
            if self._recovered(posture, t):
                self._clear_fall()
                self._move(FallState.NORMAL, features, "recovered_before_long_lie", transitions)
            elif self._long_lie_elapsed(t):
                self._fire(features, "long_lie", transitions)
                alert_fired = True

        elif self._state is FallState.FALLEN and self._recovered(posture, t):
            self._move(FallState.NORMAL, features, "recovered", transitions)
            # The caller closes the event from the transition; the fall is over.
            self._clear_fall()

        remaining = None
        if self._state is FallState.IMPACT and self._impact_t is not None:
            remaining = max(0.0, settings.long_lie_s - (t - self._impact_t))
        return FallDecision(
            features=features,
            state=self._state,
            state_since_t=self._since_t,
            transitions=tuple(transitions),
            long_lie_remaining_s=remaining,
            event_id=self._event_id,
            alert_fired=alert_fired,
        )

    def hold(self, features: FallFeatures) -> FallDecision:
        """Report the current state for a frame without advancing it."""
        remaining = None
        if self._state is FallState.IMPACT and self._impact_t is not None:
            remaining = max(0.0, self._settings.long_lie_s - (features.t - self._impact_t))
        return FallDecision(
            features=features,
            state=self._state,
            state_since_t=self._since_t,
            long_lie_remaining_s=remaining,
            event_id=self._event_id,
        )

    def tick(self, features: FallFeatures) -> FallDecision | None:
        """Advance the long-lie timer for a person **not observed** on this frame.

        A person who fell and then cannot be seen — the detector often loses someone lying
        on the floor, or furniture hides them — was never seen recovering. The draft's rule
        is "no recovery of vertical posture for ``long_lie_s`` after the impact", so the
        alarm fires on the first frame past the deadline even without a measurement, with
        reason ``long_lie_unobserved``. ``features`` carries the frame's time and no
        geometry. Returns a decision only when the alarm fired.
        """
        if self._state is not FallState.IMPACT or not self._long_lie_elapsed(features.t):
            return None
        transitions: list[FallTransition] = []
        self._fire(features, "long_lie_unobserved", transitions)
        return FallDecision(
            features=features,
            state=self._state,
            state_since_t=self._since_t,
            transitions=tuple(transitions),
            event_id=self._event_id,
            alert_fired=True,
        )

    def take_closed_event(self) -> uuid.UUID | None:
        """Hand over the event id of an alarm that just closed, then forget it."""
        if self._state is FallState.FALLEN:
            return None
        event_id = self._event_id
        self._event_id = None
        self._alert_t = None
        self._alert_utc = None
        return event_id

    # -- helpers --------------------------------------------------------

    def _long_lie_elapsed(self, t: float) -> bool:
        """The long-lie deadline passed and no recovery is in progress."""
        return (
            self._impact_t is not None
            and self._upright_since is None
            and t - self._impact_t >= self._settings.long_lie_s
        )

    def _fire(self, features: FallFeatures, reason: str, transitions: list[FallTransition]) -> None:
        self._event_id = uuid.uuid4()
        self._alert_t = features.t
        self._alert_utc = features.utc
        self._move(FallState.FALLEN, features, reason, transitions)

    def _recovered(self, posture: PostureReading, t: float) -> bool:
        if not posture.measurable:
            # Unmeasurable: neither confirms nor breaks a recovery in progress.
            return False
        if not posture.upright:
            self._upright_since = None
            return False
        if self._upright_since is None:
            self._upright_since = t
        return t - self._upright_since >= self._settings.recovery_hold_s

    def _track_extremes(self, features: FallFeatures) -> None:
        vy = features.vy_norm
        if vy is not None and (self._peak_vy is None or vy > self._peak_vy):
            self._peak_vy = vy
        angle = features.geometry.torso_angle_deg
        if angle is not None and (self._min_angle is None or angle < self._min_angle):
            self._min_angle = angle
        ratio = features.geometry.aspect_ratio
        if ratio is not None and (self._max_ratio is None or ratio > self._max_ratio):
            self._max_ratio = ratio

    def _clear_fall(self) -> None:
        self._falling_t = None
        self._impact_t = None
        self._upright_since = None
        if self._event_id is None:
            self._falling_utc = None
            self._impact_utc = None
            self._peak_vy = None
            self._min_angle = None
            self._max_ratio = None

    def _move(
        self,
        target: FallState,
        features: FallFeatures,
        reason: str,
        transitions: list[FallTransition],
    ) -> None:
        transitions.append(
            FallTransition(
                track_uuid=self._track_uuid,
                from_state=self._state,
                to_state=target,
                t=features.t,
                utc=features.utc,
                frame_seq=features.frame_seq,
                reason=reason,
            )
        )
        self._state = target
        self._since_t = features.t
