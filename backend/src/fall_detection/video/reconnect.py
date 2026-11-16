"""Reconnection policy.

Thesis plan §9.2 specifies exponential backoff of 1, 2, 4, 8, up to 30 seconds
with jitter, and a new stream session on every reconnect. This module owns only
the delay schedule; the session lifecycle lives in
:mod:`fall_detection.video.pipeline`.

Jitter is applied as a *reduction* of at most ``jitter_ratio`` of the delay
(never an increase), so the configured maximum stays a hard ceiling and a fleet
of cameras cannot synchronise their retries.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

__all__ = ["ReconnectPolicy"]


@dataclass(slots=True)
class ReconnectPolicy:
    """Compute retry delays and decide when to give up.

    Args:
        backoff_s: Ascending delays in seconds, applied per consecutive attempt.
        jitter_ratio: Fraction of each delay that may be subtracted as jitter.
        max_attempts: ``0`` retries forever; a positive value stops after that
            many consecutive failures.
        rng: Random source, injectable so tests are deterministic.
    """

    backoff_s: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 30.0)
    jitter_ratio: float = 0.1
    max_attempts: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if not self.backoff_s:
            raise ValueError("backoff_s must not be empty")
        if any(delay <= 0 for delay in self.backoff_s):
            raise ValueError("backoff_s entries must be positive")
        if list(self.backoff_s) != sorted(self.backoff_s):
            raise ValueError("backoff_s must be ascending")
        if not 0.0 <= self.jitter_ratio <= 1.0:
            raise ValueError("jitter_ratio must be within [0, 1]")
        if self.max_attempts < 0:
            raise ValueError("max_attempts must be >= 0")

    @property
    def exhausted_after(self) -> int | None:
        """Number of attempts after which the policy gives up, or ``None``."""
        return self.max_attempts or None

    def base_delay(self, attempt: int) -> float:
        """Return the un-jittered delay for ``attempt`` (1-based).

        Attempts beyond the schedule reuse the final entry, which caps the delay
        at ``backoff_s[-1]`` as required.
        """
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        index = min(attempt, len(self.backoff_s)) - 1
        return self.backoff_s[index]

    def delay_for(self, attempt: int) -> float:
        """Return the jittered delay for ``attempt`` (1-based).

        The result is in ``[base * (1 - jitter_ratio), base]`` and is never
        below zero.
        """
        base = self.base_delay(attempt)
        if self.jitter_ratio == 0.0:
            return base
        return max(0.0, base - base * self.jitter_ratio * self.rng.random())

    def should_retry(self, attempt: int) -> bool:
        """Whether another attempt is allowed after ``attempt`` failures."""
        limit = self.exhausted_after
        return limit is None or attempt < limit

    def describe(self) -> dict[str, object]:
        """Return the policy as a JSON-serialisable mapping."""
        return {
            "backoff_s": list(self.backoff_s),
            "jitter_ratio": self.jitter_ratio,
            "max_attempts": self.max_attempts,
        }
