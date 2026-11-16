"""The labelled-sequence contract every dataset adapter produces (THESIS_PLAN §6).

A :class:`LabelledSequence` is one recorded clip with a sequence-level truth (fall or
activity of daily living), optional per-frame labels, and a way to read its frames with
their **acquisition** timestamps. Adapters never resample, duplicate, or drop frames: a
frame-count mismatch between the video and its timestamp file is refused, not repaired.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    import numpy as np

__all__ = ["FrameLabel", "LabelledSequence", "SequenceFormatError"]

FrameLabel = Literal[-1, 0, 1]
"""URFD's per-frame vocabulary: -1 not lying, 0 transitional (falling), 1 lying."""


class SequenceFormatError(ValueError):
    """A dataset file does not have the documented structure."""


@dataclass(frozen=True, slots=True)
class LabelledSequence:
    """One labelled clip.

    Attributes:
        dataset: ``"urfd"``, ``"caucafall"``, or ``"local"``.
        sequence_id: Unique within the dataset, e.g. ``"fall-01"`` or ``"S1/Fall forward"``.
        is_fall: Sequence-level truth.
        activity: The publisher's activity name.
        group: The unit a bootstrap resamples by (a subject for CAUCAFall, the sequence
            itself for URFD, whose subjects are not published per sequence).
        timestamps_s: Acquisition time of every frame, seconds from the first frame.
        time_source: Where the timestamps came from, in words.
        frame_labels: Optional per-frame labels keyed by zero-based frame index.
        reader: Yields ``(frame_index, BGR frame)`` in order.
        metadata: Anything else worth recording in a report (resolution, rate, files).
    """

    dataset: str
    sequence_id: str
    is_fall: bool
    activity: str
    group: str
    timestamps_s: tuple[float, ...]
    time_source: str
    reader: Callable[[], Iterator[tuple[int, np.ndarray]]] = field(repr=False, compare=False)
    frame_labels: Mapping[int, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        """Number of frames (one timestamp per frame)."""
        return len(self.timestamps_s)

    @property
    def duration_s(self) -> float:
        """Span from the first to the last frame."""
        return 0.0 if not self.timestamps_s else self.timestamps_s[-1] - self.timestamps_s[0]

    def first_label_time(self, label: int) -> float | None:
        """Acquisition time of the first frame carrying ``label``, if any."""
        for index in sorted(self.frame_labels):
            if self.frame_labels[index] == label and index < len(self.timestamps_s):
                return self.timestamps_s[index]
        return None

    def describe(self) -> dict[str, Any]:
        """JSON view for reports (no frames)."""
        return {
            "dataset": self.dataset,
            "sequence_id": self.sequence_id,
            "is_fall": self.is_fall,
            "activity": self.activity,
            "group": self.group,
            "frames": self.frame_count,
            "duration_s": round(self.duration_s, 3),
            "time_source": self.time_source,
            "labelled_frames": len(self.frame_labels),
            **dict(self.metadata),
        }
