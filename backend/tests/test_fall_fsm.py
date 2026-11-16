"""The four-state fall machine (THESIS_PLAN §5).

Each scenario feeds explicit samples on observation time, so the timers are asserted
exactly. The poses are idealised; these tests assert what the rules say, never how
accurate they are.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from tests.helpers_fall import TRACK, bending, features, invisible, lying, sitting, standing

from fall_detection.config import FallSettings
from fall_detection.fall.contracts import FallState
from fall_detection.fall.fsm import FallStateMachine, read_posture

SETTINGS = FallSettings()


def machine(settings: FallSettings = SETTINGS) -> FallStateMachine:
    return FallStateMachine(settings, TRACK)


def run(
    fsm: FallStateMachine,
    samples: Sequence[tuple[float, object, float | None]],
    settings: FallSettings = SETTINGS,
) -> list[FallState]:
    states = []
    for t, pose, vy in samples:
        decision = fsm.step(features(t, pose, vy_norm=vy, settings=settings))  # type: ignore[arg-type]
        states.append(decision.state)
    return states


def fall_until(t_end: float, *, step: float = 0.1) -> list[tuple[float, object, float | None]]:
    """Normal at 0, a velocity spike at 0.1, lying from 0.2 until ``t_end``."""
    samples: list[tuple[float, object, float | None]] = [
        (0.0, standing(), 0.0),
        (0.1, standing(top=200.0), 3.0),
    ]
    t = 0.2
    while t <= t_end + 1e-9:
        samples.append((round(t, 6), lying(), 0.0))
        t += step
    return samples


class TestPosture:
    def test_standing_is_upright(self) -> None:
        reading = read_posture(features(0.0, standing()), SETTINGS)
        assert reading.measurable and reading.upright and not reading.lying

    def test_lying_is_lying(self) -> None:
        reading = read_posture(features(0.0, lying()), SETTINGS)
        assert reading.measurable and reading.lying and not reading.upright

    def test_sitting_is_upright(self) -> None:
        reading = read_posture(features(0.0, sitting()), SETTINGS)
        assert reading.upright and not reading.lying

    def test_require_both_needs_box_and_torso(self) -> None:
        # Bending forward widens the box but the torso is not flat.
        either = read_posture(features(0.0, bending()), FallSettings(require_both=False))
        both = read_posture(features(0.0, bending()), FallSettings(require_both=True))
        assert either.lying is True
        assert both.lying is False

    def test_nothing_measurable(self) -> None:
        sample = features(0.0, ((0.0, 0.0, 50.0, 0.0), invisible((0, 0, 0, 0))[1]))
        assert read_posture(sample, SETTINGS).measurable is False


class TestFallSequence:
    def test_a_fall_reaches_the_alarm_at_exactly_long_lie(self) -> None:
        fsm = machine()
        samples = fall_until(3.5)
        states = run(fsm, samples)
        assert states[:3] == [FallState.NORMAL, FallState.FALLING, FallState.IMPACT]
        # Impact at t=0.2; the alarm fires on the first sample with t - 0.2 >= 3.0.
        first_fallen = samples[states.index(FallState.FALLEN)][0]
        assert abs(first_fallen - 3.2) < 1e-6
        before = [
            samples[index][0] for index, state in enumerate(states) if state is FallState.IMPACT
        ]
        assert max(before) < 3.2

    def test_the_alarm_is_reported_once_with_an_event_id(self) -> None:
        fsm = machine()
        fired = []
        for t, pose, vy in fall_until(5.0):
            decision = fsm.step(features(t, pose, vy_norm=vy))  # type: ignore[arg-type]
            if decision.alert_fired:
                fired.append(decision)
        assert len(fired) == 1
        assert fired[0].event_id is not None
        assert fsm.event_id == fired[0].event_id

    def test_long_lie_countdown_is_reported_during_impact(self) -> None:
        fsm = machine()
        fsm.step(features(0.0, standing(), vy_norm=0.0))
        fsm.step(features(0.1, standing(top=200.0), vy_norm=3.0))
        decision = fsm.step(features(1.2, lying(), vy_norm=0.0))
        assert decision.state is FallState.IMPACT
        assert decision.long_lie_remaining_s is not None

    def test_extremes_are_kept_for_the_event(self) -> None:
        fsm = machine()
        run(fsm, fall_until(3.5))
        assert fsm.peak_vy == 3.0
        assert fsm.min_torso_angle_deg is not None and fsm.min_torso_angle_deg < 30.0
        assert fsm.max_aspect_ratio is not None and fsm.max_aspect_ratio > 1.0


class TestFalsePositiveGuards:
    def test_lying_down_slowly_never_starts_a_fall(self) -> None:
        # No velocity spike: lying on a bed is not a fall.
        fsm = machine()
        samples = [(0.0, standing(), 0.0), (0.5, sitting(), 0.3)]
        samples += [(1.0 + 0.1 * index, lying(), 0.2) for index in range(60)]
        states = run(fsm, samples)
        assert FallState.FALLING not in states
        assert FallState.FALLEN not in states

    def test_sitting_down_quickly_returns_to_normal(self) -> None:
        # A fast sit produces a spike but no lying posture inside the impact window.
        fsm = machine()
        samples = [(0.0, standing(), 0.0), (0.1, sitting(), 2.0)]
        samples += [(0.2 + 0.1 * index, sitting(), 0.0) for index in range(20)]
        states = run(fsm, samples)
        assert states[1] is FallState.FALLING
        assert states[-1] is FallState.NORMAL
        assert FallState.IMPACT not in states

    def test_recovering_before_long_lie_cancels_the_fall(self) -> None:
        fsm = machine()
        samples = fall_until(1.5)
        samples += [(1.6 + 0.1 * index, standing(), 0.0) for index in range(10)]
        states = run(fsm, samples)
        assert FallState.IMPACT in states
        assert FallState.FALLEN not in states
        assert states[-1] is FallState.NORMAL

    def test_one_upright_frame_does_not_count_as_recovery(self) -> None:
        settings = FallSettings(recovery_hold_s=0.5)
        fsm = machine(settings)
        samples = fall_until(1.0)
        samples.append((1.1, standing(), 0.0))  # a single noisy upright reading
        samples += [(1.2 + 0.1 * index, lying(), 0.0) for index in range(25)]
        states = run(fsm, samples, settings)
        assert FallState.FALLEN in states


class TestUnmeasurableSamples:
    def test_an_unmeasured_sample_never_counts_as_recovery(self) -> None:
        fsm = machine()
        run(fsm, fall_until(0.5))
        blind = ((0.0, 0.0, 50.0, 0.0), invisible((0, 0, 0, 0))[1])
        # Before the deadline an unmeasured sample holds the state.
        assert fsm.step(features(1.0, blind, vy_norm=None)).state is FallState.IMPACT
        # Past the deadline with no observed recovery, the alarm fires even though nothing
        # was measured on this sample: the draft's rule is "no recovery for long_lie_s".
        decision = fsm.step(features(4.0, blind, vy_norm=None))
        assert decision.state is FallState.FALLEN
        assert decision.alert_fired

    def test_a_person_who_disappears_after_impact_still_raises_the_alarm(self) -> None:
        fsm = machine()
        run(fsm, fall_until(0.5))
        blind = ((0.0, 0.0, 50.0, 0.0), invisible((0, 0, 0, 0))[1])
        assert fsm.tick(features(2.0, blind, vy_norm=None)) is None
        decision = fsm.tick(features(3.2, blind, vy_norm=None))
        assert decision is not None and decision.alert_fired
        assert decision.transitions[-1].reason == "long_lie_unobserved"
        # A tick outside ``impact`` does nothing.
        assert fsm.tick(features(4.0, blind, vy_norm=None)) is None

    def test_a_recovery_in_progress_holds_the_alarm(self) -> None:
        # Standing back up just before the deadline: the upright hold is still running, so
        # the deadline passing on an unmeasured frame must not fire the alarm.
        fsm = machine(FallSettings(recovery_hold_s=0.5))
        run(fsm, fall_until(2.9))
        fsm.step(features(3.0, standing(), vy_norm=0.0))
        blind = ((0.0, 0.0, 50.0, 0.0), invisible((0, 0, 0, 0))[1])
        assert fsm.step(features(3.3, blind, vy_norm=None)).state is FallState.IMPACT
        assert fsm.step(features(3.6, standing(), vy_norm=0.0)).state is FallState.NORMAL

    def test_unknown_until_first_measurement(self) -> None:
        fsm = machine()
        blind = ((0.0, 0.0, 50.0, 0.0), invisible((0, 0, 0, 0))[1])
        assert fsm.step(features(0.0, blind, vy_norm=None)).state is FallState.UNKNOWN
        assert fsm.step(features(0.1, standing(), vy_norm=None)).state is FallState.NORMAL


class TestRecoveryFromAlarm:
    def test_standing_up_closes_the_alarm(self) -> None:
        fsm = machine()
        run(fsm, fall_until(3.5))
        event_id = fsm.event_id
        assert isinstance(event_id, uuid.UUID)
        states = run(fsm, [(3.6 + 0.1 * index, standing(), 0.0) for index in range(10)])
        assert states[-1] is FallState.NORMAL
        assert fsm.take_closed_event() == event_id
        assert fsm.event_id is None

    def test_a_new_fall_after_recovery_gets_a_new_event(self) -> None:
        fsm = machine()
        run(fsm, fall_until(3.5))
        first = fsm.event_id
        run(fsm, [(3.6 + 0.1 * index, standing(), 0.0) for index in range(10)])
        fsm.take_closed_event()
        later = [(10.0 + t, pose, vy) for t, pose, vy in fall_until(3.5)]
        run(fsm, later)
        assert fsm.event_id is not None and fsm.event_id != first
