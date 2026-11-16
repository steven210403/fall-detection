"""Tests for the reconnection policy (thesis plan §9.2)."""

from __future__ import annotations

import random

import pytest

from fall_detection.video.reconnect import ReconnectPolicy


class TestSchedule:
    def test_matches_the_plan_schedule(self) -> None:
        policy = ReconnectPolicy()
        assert [policy.base_delay(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 30.0]

    def test_caps_at_the_final_delay(self) -> None:
        policy = ReconnectPolicy()
        # The plan requires the backoff to reach 30 seconds and stay there.
        assert policy.base_delay(6) == 30.0
        assert policy.base_delay(100) == 30.0

    def test_rejects_an_attempt_below_one(self) -> None:
        with pytest.raises(ValueError, match="attempt"):
            ReconnectPolicy().base_delay(0)


class TestJitter:
    def test_jitter_only_reduces_the_delay(self) -> None:
        policy = ReconnectPolicy(jitter_ratio=0.5, rng=random.Random(7))
        for attempt in range(1, 6):
            base = policy.base_delay(attempt)
            delay = policy.delay_for(attempt)
            assert 0.0 <= delay <= base

    def test_jitter_is_bounded_by_the_ratio(self) -> None:
        policy = ReconnectPolicy(jitter_ratio=0.2, rng=random.Random(3))
        for _ in range(50):
            delay = policy.delay_for(4)  # base 8.0
            assert 8.0 * 0.8 <= delay <= 8.0

    def test_zero_jitter_is_deterministic(self) -> None:
        policy = ReconnectPolicy(jitter_ratio=0.0)
        assert policy.delay_for(3) == 4.0


class TestRetryLimit:
    def test_retries_forever_by_default(self) -> None:
        policy = ReconnectPolicy()
        assert policy.exhausted_after is None
        assert policy.should_retry(1_000) is True

    def test_stops_after_max_attempts(self) -> None:
        policy = ReconnectPolicy(max_attempts=3)
        assert policy.should_retry(1) is True
        assert policy.should_retry(2) is True
        assert policy.should_retry(3) is False


class TestValidation:
    def test_rejects_empty_schedule(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ReconnectPolicy(backoff_s=())

    def test_rejects_descending_schedule(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            ReconnectPolicy(backoff_s=(4.0, 2.0))

    def test_rejects_non_positive_delay(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ReconnectPolicy(backoff_s=(0.0, 1.0))

    def test_rejects_out_of_range_jitter(self) -> None:
        with pytest.raises(ValueError, match="jitter_ratio"):
            ReconnectPolicy(jitter_ratio=1.5)

    def test_rejects_negative_max_attempts(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            ReconnectPolicy(max_attempts=-1)

    def test_describe_is_serialisable(self) -> None:
        described = ReconnectPolicy(max_attempts=5, jitter_ratio=0.0).describe()
        assert described["backoff_s"] == [1.0, 2.0, 4.0, 8.0, 30.0]
        assert described["max_attempts"] == 5
