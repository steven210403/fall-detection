"""Hip vertical velocity (THESIS_PLAN §4.3).

The draft defines the third fall parameter as the discrete vertical velocity of the hip
midpoint::

    Vy = (Y_hip(t) − Y_hip(t − Δt)) / Δt

Image y grows downward, so a fall gives a **positive** Vy. Three rules keep the value
honest:

* **Δt is measured elapsed time**, never a frame count: the source's timestamps when it
  has them (the dataset's acquisition time, the camera's presentation time), otherwise
  the monotonic clock, and the caller reports which. A dropped frame therefore lengthens
  Δt instead of inflating the velocity.
* **Normalised by body length.** A pixel velocity depends on the distance to the camera
  (a person at 4 m moves fewer pixels than at 2 m). Dividing by the person's reference
  body length — the largest ``max(W, H)`` of their box over the last
  ``reference_window_s`` — gives body lengths per second, so one threshold works at
  every distance. The raw px/s value is still reported, as the draft's formula defines it.
* **Not computed across a gap.** No velocity is reported across a gap longer than
  ``max_gap_s`` or a non-increasing timestamp; a median over the last ``window`` raw
  values suppresses single-frame keypoint jitter.
"""

from __future__ import annotations

import statistics
from collections import deque

__all__ = ["HipVelocityEstimator"]


class HipVelocityEstimator:
    """Per-person hip velocity on measured elapsed time.

    Args:
        window: Median filter length in samples (1 disables filtering).
        max_gap_s: Longest gap a velocity may be computed across.
        reference_window_s: How far back the reference body length looks.
    """

    def __init__(
        self,
        *,
        window: int = 3,
        max_gap_s: float = 0.5,
        reference_window_s: float = 2.0,
    ) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        self._window = window
        self._max_gap_s = max_gap_s
        self._reference_window_s = reference_window_s
        self._last: tuple[float, float] | None = None
        self._raw: deque[float] = deque(maxlen=window)
        self._lengths: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        """Forget all history (a new session or a time-source change)."""
        self._last = None
        self._raw.clear()
        self._lengths.clear()

    def reference_length(self) -> float | None:
        """The reference body length in pixels, or ``None`` before any box."""
        if not self._lengths:
            return None
        return max(length for _, length in self._lengths)

    def update(
        self,
        t: float,
        hip_y: float | None,
        body_length_px: float | None,
    ) -> tuple[float | None, float | None]:
        """Add one observation and return ``(vy_px_s, vy_norm)``.

        Args:
            t: Observation time in seconds.
            hip_y: Hip-midpoint y in pixels, or ``None`` when the hips were not visible.
            body_length_px: ``max(W, H)`` of the person's box this frame.

        Returns:
            The median-filtered downward velocity in px/s and in body lengths per
            second, or ``(None, None)`` when no velocity could be measured.
        """
        if body_length_px is not None and body_length_px > 0.0:
            self._lengths.append((t, body_length_px))
        while self._lengths and t - self._lengths[0][0] > self._reference_window_s:
            self._lengths.popleft()

        if hip_y is None:
            return None, None
        previous = self._last
        if previous is not None and t <= previous[0]:
            # A timestamp that did not advance cannot produce a velocity; keep the
            # earlier sample as the reference rather than dividing by zero.
            return None, None
        self._last = (t, hip_y)
        if previous is None:
            return None, None
        dt = t - previous[0]
        if dt > self._max_gap_s:
            self._raw.clear()
            return None, None
        self._raw.append((hip_y - previous[1]) / dt)
        vy = statistics.median(self._raw)
        reference = self.reference_length()
        vy_norm = None if reference is None or reference <= 0.0 else vy / reference
        return vy, vy_norm
