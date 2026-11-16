"""Clock abstractions.

The pipeline keeps monotonic receive timestamps separate from source
presentation timestamps (thesis plan §9.1). Keeping the clock injectable makes
queue and timeout behaviour testable without sleeping.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Minimal clock interface used by the pipeline."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary, monotonic origin."""
        ...

    def utc_timestamp(self) -> float:
        """Seconds since the Unix epoch in UTC."""
        ...


class AcquisitionClockBase:
    """A clock that reports *acquisition* time instead of wall time.

    Thesis plan §9.1 separates receive time from source time, and the pipeline's time-based
    rules (§11.2's loss tolerance, §12.2's duration conditions, §12.5's EMA) all run on the
    monotonic clock. That is correct for a live camera, where wall time and footage time are
    the same rate. It is **wrong for an offline replay**, which decodes a day of footage in
    minutes: every two-second condition would be measured against a compressed timeline and
    a rule would never accumulate the duration it needs.

    This clock lets a replay supply the acquisition timeline as the monotonic reading. It
    advances only when the caller says a new frame arrived, so the pipeline's time-based
    rules see the footage's own intervals — which is what makes an offline measurement
    comparable with a live one.

    The UTC reading is the acquisition offset added to a fixed origin. It is deliberately
    *not* a wall-clock instant: §13.2 forbids inventing absolute UTC for a source whose time
    zone is unspecified.
    """

    def __init__(self, *, origin_monotonic: float = 0.0, origin_utc: float = 0.0) -> None:
        self._origin_utc = float(origin_utc)
        self._monotonic = float(origin_monotonic)
        self._utc = self._origin_utc + self._monotonic

    def advance_to(self, monotonic: float) -> None:
        """Move the clock forward to ``monotonic`` seconds; never backwards.

        A replay may present frames out of order in principle, and moving time backwards
        would make an elapsed interval negative. Clamping keeps the reading monotonic, which
        is what the protocol promises.
        """
        self._monotonic = max(self._monotonic, float(monotonic))
        self._utc = self._origin_utc + self._monotonic

    def monotonic(self) -> float:
        """Seconds of acquisition time since the replay began."""
        return self._monotonic

    def utc_timestamp(self) -> float:
        """Acquisition offset as a pseudo-epoch, never an absolute instant (§13.2)."""
        return self._utc


class SystemClock:
    """Clock backed by the standard library."""

    def monotonic(self) -> float:
        return time.monotonic()

    def utc_timestamp(self) -> float:
        return time.time()


class ManualClock:
    """A clock the caller advances explicitly.

    Thesis plan §11.2 makes several tracking rules *time-based*: a track survives
    a gap of at most two seconds and its last observation becomes stale after
    0.5 seconds. Sleeping through those intervals would make the tests slow and
    timing-dependent, so this implementation lets a test or a verification CLI
    move time forward by an exact amount and assert the boundary instead of
    hoping the machine was fast enough.

    Args:
        monotonic: Initial reading of the monotonic clock, in seconds.
        utc: Initial reading of the UTC clock, as an epoch timestamp. Both clocks
            advance together so the UTC values stay consistent with monotonic
            differences.
    """

    def __init__(self, *, monotonic: float = 0.0, utc: float = 1_700_000_000.0) -> None:
        self._monotonic = monotonic
        self._utc = utc

    def monotonic(self) -> float:
        """Seconds from an arbitrary, monotonic origin."""
        return self._monotonic

    def utc_timestamp(self) -> float:
        """Seconds since the Unix epoch in UTC."""
        return self._utc

    def advance(self, seconds: float) -> None:
        """Move both clocks forward by ``seconds``.

        Raises:
            ValueError: ``seconds`` is negative; time never moves backwards.
        """
        if seconds < 0:
            raise ValueError("a manual clock cannot move backwards")
        self._monotonic += seconds
        self._utc += seconds


MonotonicClock = SystemClock
"""Alias kept for readability where only monotonic behaviour is relevant."""

__all__ = [
    "AcquisitionClockBase",
    "Clock",
    "ManualClock",
    "MonotonicClock",
    "SystemClock",
]
