"""Scoring fall detection against labelled sequences (THESIS_PLAN §7).

Three levels are scored, and a report always names which one a figure belongs to:

* **Event level — fall detected** (primary). A sequence is predicted "fall" when any
  person reaches the ``impact`` state: a sudden hip descent followed by a lying posture.
  Truth is the sequence label (fall vs activity of daily living).
* **Event level — alarm** (secondary). Predicted "fall" when the long-lie alarm fires
  (``fallen``). On a dataset whose fall clips end before ``long_lie_s`` has elapsed after
  the impact (URFD), the alarm cannot be observed; the report counts how many fall
  sequences had enough post-impact footage and never presents a missing alarm on a
  too-short clip as a model failure.
* **Frame level.** On datasets with per-frame labels (URFD), a frame is predicted
  "lying" when any person is in ``impact`` or ``fallen``, and scored against label ``1``
  (lying) vs ``-1`` (not lying). Label ``0`` (the transitional falling pose) is excluded,
  as the publisher excludes it.

Delays are reported from the first frame labelled lying: the **detection delay**
(impact state − first lying frame) and the **alarm delay** (alarm − first lying frame).
Confidence intervals resample whole groups (subjects or sequences), never frames.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from fall_detection.research.metrics import BinaryCounts, confusion
from fall_detection.research.replay import SequenceOutcome
from fall_detection.research.sequences import LabelledSequence

__all__ = ["evaluate", "event_counts"]


def event_counts(
    sequences: Sequence[LabelledSequence],
    outcomes: Mapping[str, SequenceOutcome],
    *,
    level: str = "detected",
) -> BinaryCounts:
    """Sequence-level confusion matrix at ``level`` (``detected`` or ``alarmed``)."""
    return confusion(
        (sequence.is_fall, bool(getattr(outcomes[sequence.sequence_id], level)))
        for sequence in sequences
    )


def evaluate(
    sequences: Sequence[LabelledSequence],
    outcomes: Mapping[str, SequenceOutcome],
    *,
    long_lie_s: float,
    bootstrap: bool = True,
) -> dict[str, Any]:
    """Score ``outcomes`` against ``sequences`` at every level."""
    detected = event_counts(sequences, outcomes, level="detected")
    alarmed = event_counts(sequences, outcomes, level="alarmed")

    observable = []
    for sequence in sequences:
        if not sequence.is_fall:
            continue
        first_lying = sequence.first_label_time(1)
        if first_lying is None:
            continue
        footage = sequence.timestamps_s[-1] - first_lying
        observable.append(footage >= long_lie_s)

    frame_pairs: list[tuple[bool, bool]] = []
    for sequence in sequences:
        outcome = outcomes[sequence.sequence_id]
        for index, label in sequence.frame_labels.items():
            if label == 0 or index not in outcome.lying_frames:
                continue
            frame_pairs.append((label == 1, outcome.lying_frames[index]))

    detection_delays, alarm_delays = [], []
    for sequence in sequences:
        outcome = outcomes[sequence.sequence_id]
        first_lying = sequence.first_label_time(1)
        if not sequence.is_fall or first_lying is None:
            continue
        if outcome.impact_t is not None:
            detection_delays.append(outcome.impact_t - first_lying)
        if outcome.alert_t is not None:
            alarm_delays.append(outcome.alert_t - first_lying)

    by_activity: dict[str, list[bool]] = defaultdict(list)
    for sequence in sequences:
        by_activity[sequence.activity].append(outcomes[sequence.sequence_id].detected)

    report: dict[str, Any] = {
        "sequences": len(sequences),
        "falls": sum(1 for sequence in sequences if sequence.is_fall),
        "adls": sum(1 for sequence in sequences if not sequence.is_fall),
        "event_detected": detected.as_dict(),
        "event_alarm": {
            **alarmed.as_dict(),
            "long_lie_s": long_lie_s,
            "falls_with_frame_labels": len(observable),
            "falls_with_enough_post_impact_footage": sum(observable),
            "note": (
                "the alarm needs the person to stay down for long_lie_s after the impact; "
                "a fall clip that ends sooner cannot show it"
            ),
        },
        "frame_level": None
        if not frame_pairs
        else {
            **confusion(frame_pairs).as_dict(),
            "positive": "label 1 (lying)",
            "negative": "label -1 (not lying)",
            "excluded": "label 0 (transitional) and unlabelled frames",
        },
        "detection_delay_s": _summary(detection_delays),
        "alarm_delay_s": _summary(alarm_delays),
        "detection_rate_by_activity": {
            activity: {
                "sequences": len(values),
                "detected": sum(values),
                "rate": round(sum(values) / len(values), 4),
            }
            for activity, values in sorted(by_activity.items())
        },
        "per_sequence": [
            {
                **outcomes[sequence.sequence_id].as_dict(),
                "activity": sequence.activity,
                "group": sequence.group,
                "frames": sequence.frame_count,
                "first_lying_t": sequence.first_label_time(1),
            }
            for sequence in sequences
        ],
    }
    if bootstrap:
        report["bootstrap"] = _bootstrap(sequences, outcomes)
    return report


def _bootstrap(
    sequences: Sequence[LabelledSequence], outcomes: Mapping[str, SequenceOutcome]
) -> dict[str, Any]:
    import numpy as np

    from fall_detection.research.bootstrap import bootstrap_metric

    truth = np.array([sequence.is_fall for sequence in sequences], dtype=bool)
    predicted = np.array(
        [outcomes[sequence.sequence_id].detected for sequence in sequences], dtype=bool
    )
    groups = [sequence.group for sequence in sequences]
    unit = "subject" if len(set(groups)) < len(groups) else "sequence"

    def metric(name: str) -> Any:
        def compute(rows: np.ndarray) -> float | None:
            counts = confusion(zip(truth[rows].tolist(), predicted[rows].tolist(), strict=True))
            return getattr(counts, name)

        return compute

    results = {}
    for name in ("accuracy", "precision", "recall", "specificity", "f1"):
        result = bootstrap_metric(groups, metric(name), name=name, unit=unit)
        results[name] = result.as_dict()
    return {"level": "event_detected", "unit": unit, "metrics": results}


def _summary(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }
