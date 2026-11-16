"""Binary classification metrics (THESIS_PLAN §7, the draft's §2.2.7).

The draft defines the fall-classification metrics on the confusion matrix::

    Accuracy    = (TP + TN) / (TP + TN + FP + FN)
    Precision   = TP / (TP + FP)
    Recall      = TP / (TP + FN)          (sensitivity)
    Specificity = TN / (TN + FP)
    F1          = 2 · Precision · Recall / (Precision + Recall)

A ratio whose denominator is zero is **undefined** and reported as ``None``, never as 0
or 1: "no fall was predicted" gives no precision at all, which is a different fact from a
precision of zero.

Stdlib only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = ["BinaryCounts", "confusion"]


@dataclass(frozen=True, slots=True)
class BinaryCounts:
    """A 2×2 confusion matrix with the positive class being "fall"."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        """Number of scored items."""
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float | None:
        return _ratio(self.tp + self.tn, self.total)

    @property
    def precision(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def specificity(self) -> float | None:
        return _ratio(self.tn, self.tn + self.fp)

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0.0:
            return None if precision is None or recall is None else 0.0
        return 2.0 * precision * recall / (precision + recall)

    def __add__(self, other: BinaryCounts) -> BinaryCounts:
        return BinaryCounts(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            tn=self.tn + other.tn,
            fn=self.fn + other.fn,
        )

    def as_dict(self) -> dict[str, Any]:
        """Counts and every derived metric, rounded for a report."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "total": self.total,
            "accuracy": _round(self.accuracy),
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "specificity": _round(self.specificity),
            "f1": _round(self.f1),
        }


def confusion(pairs: Iterable[tuple[bool, bool]]) -> BinaryCounts:
    """Count ``(truth, prediction)`` pairs into a confusion matrix."""
    tp = fp = tn = fn = 0
    for truth, predicted in pairs:
        if truth and predicted:
            tp += 1
        elif truth:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return BinaryCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
