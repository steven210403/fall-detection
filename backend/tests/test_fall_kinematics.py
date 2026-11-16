"""Hip vertical velocity on measured elapsed time (THESIS_PLAN §4.3)."""

from __future__ import annotations

import pytest

from fall_detection.fall.kinematics import HipVelocityEstimator


def test_the_first_sample_has_no_velocity() -> None:
    estimator = HipVelocityEstimator(window=1)
    assert estimator.update(0.0, 100.0, 170.0) == (None, None)


def test_velocity_uses_elapsed_time_not_frame_count() -> None:
    # The same 34 px drop over 0.1 s and over 0.2 s are different velocities: a dropped
    # frame lengthens the interval instead of inflating the speed.
    fast = HipVelocityEstimator(window=1)
    fast.update(0.0, 100.0, 170.0)
    slow = HipVelocityEstimator(window=1)
    slow.update(0.0, 100.0, 170.0)
    assert fast.update(0.1, 134.0, 170.0)[0] == pytest.approx(340.0)
    assert slow.update(0.2, 134.0, 170.0)[0] == pytest.approx(170.0)


def test_downward_motion_is_positive() -> None:
    estimator = HipVelocityEstimator(window=1)
    estimator.update(0.0, 100.0, 170.0)
    vy, _ = estimator.update(0.1, 90.0, 170.0)
    assert vy is not None and vy < 0.0


def test_velocity_is_normalised_by_body_length() -> None:
    # The same physical fall seen at twice the distance moves half the pixels and has
    # half the body length, so the normalised velocity is the same.
    near = HipVelocityEstimator(window=1)
    near.update(0.0, 100.0, 200.0)
    far = HipVelocityEstimator(window=1)
    far.update(0.0, 50.0, 100.0)
    _, near_norm = near.update(0.1, 140.0, 200.0)
    _, far_norm = far.update(0.1, 70.0, 100.0)
    assert near_norm == pytest.approx(2.0)
    assert far_norm == pytest.approx(2.0)


def test_the_reference_length_is_the_recent_maximum() -> None:
    # While falling the box shrinks; the reference keeps the upright length so the
    # normalised velocity is not inflated by the shrinking box.
    estimator = HipVelocityEstimator(window=1, reference_window_s=2.0)
    estimator.update(0.0, 100.0, 170.0)
    estimator.update(0.1, 120.0, 120.0)
    assert estimator.reference_length() == 170.0
    estimator.update(2.5, 130.0, 120.0)
    assert estimator.reference_length() == 120.0


def test_no_velocity_across_a_long_gap() -> None:
    estimator = HipVelocityEstimator(window=1, max_gap_s=0.5)
    estimator.update(0.0, 100.0, 170.0)
    assert estimator.update(1.0, 300.0, 170.0) == (None, None)
    # The next sample measures from the post-gap one.
    assert estimator.update(1.1, 310.0, 170.0)[0] == pytest.approx(100.0)


def test_a_timestamp_that_does_not_advance_is_refused() -> None:
    estimator = HipVelocityEstimator(window=1)
    estimator.update(1.0, 100.0, 170.0)
    assert estimator.update(1.0, 120.0, 170.0) == (None, None)
    assert estimator.update(0.9, 120.0, 170.0) == (None, None)
    # The original sample is still the reference.
    assert estimator.update(1.1, 117.0, 170.0)[0] == pytest.approx(170.0)


def test_missing_hips_give_no_velocity() -> None:
    estimator = HipVelocityEstimator(window=1)
    estimator.update(0.0, 100.0, 170.0)
    assert estimator.update(0.1, None, 170.0) == (None, None)


def test_the_median_suppresses_a_single_jitter() -> None:
    estimator = HipVelocityEstimator(window=3)
    estimator.update(0.0, 100.0, 170.0)
    estimator.update(0.1, 101.0, 170.0)
    estimator.update(0.2, 102.0, 170.0)
    vy, _ = estimator.update(0.3, 160.0, 170.0)  # one jumpy keypoint
    assert vy == pytest.approx(10.0)


def test_reset_forgets_history() -> None:
    estimator = HipVelocityEstimator(window=1)
    estimator.update(0.0, 100.0, 170.0)
    estimator.reset()
    assert estimator.reference_length() is None
    assert estimator.update(0.1, 120.0, 170.0) == (None, None)
