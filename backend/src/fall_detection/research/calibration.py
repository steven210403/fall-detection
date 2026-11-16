"""Threshold calibration on the calibration set only (THESIS_PLAN §6.3).

The protocol keeps the test set untouched: thresholds are chosen by a grid search on
CAUCAFall, frozen into ``datasets/manifests/fall_calibration.json`` with a provenance
hash, and only then scored — once — on URFD. Nothing may be re-selected after the test
results have been seen.

The objective is the event-level F1 of "fall detected" (impact reached). Ties are broken
by specificity (fewer false alarms), then by closeness to the literature defaults, so a
flat optimum does not drift to an arbitrary corner of the grid. ``long_lie_s`` is not
searched: it is the draft's clinical confirmation time (3 s), not a parameter the
calibration set can estimate.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fall_detection.config import FallSettings, Settings
from fall_detection.research.fall_eval import event_counts
from fall_detection.research.replay import ExtractedSequence, simulate
from fall_detection.research.sequences import LabelledSequence

__all__ = ["DEFAULT_GRID", "CalibrationResult", "calibrate", "frozen_payload"]

DEFAULT_GRID: Mapping[str, tuple[Any, ...]] = {
    "vy_threshold": (0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    "ar_threshold": (0.6, 0.7, 0.8, 1.0, 1.2),
    "angle_lying_deg": (20.0, 25.0, 30.0, 35.0, 40.0),
    "impact_window_s": (1.0, 1.5),
    "require_both": (False, True),
}
"""The searched values. Every literature default is one of the grid points."""

LITERATURE_DEFAULTS: Mapping[str, Any] = {
    "vy_threshold": 1.0,
    "ar_threshold": 1.0,
    "angle_lying_deg": 30.0,
    "impact_window_s": 1.0,
    "require_both": False,
}


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """The chosen thresholds and the table they were chosen from."""

    thresholds: dict[str, Any]
    score: dict[str, Any]
    table: list[dict[str, Any]]
    grid: dict[str, list[Any]]


def calibrate(
    sequences: Sequence[LabelledSequence],
    extracted: Mapping[str, ExtractedSequence],
    *,
    settings: Settings,
    grid: Mapping[str, Sequence[Any]] = DEFAULT_GRID,
) -> CalibrationResult:
    """Search ``grid`` and return the best thresholds on ``sequences``."""
    base = settings.fall
    keys = list(grid)
    rows: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[key] for key in keys)):
        candidate = dict(zip(keys, values, strict=True))
        if candidate.get("angle_lying_deg", base.angle_lying_deg) >= base.angle_upright_deg:
            continue
        fall = FallSettings.model_validate({**base.model_dump(), **candidate})
        outcomes = {
            sequence.sequence_id: simulate(
                extracted[sequence.sequence_id],
                is_fall=sequence.is_fall,
                settings=settings,
                fall=fall,
            )
            for sequence in sequences
        }
        counts = event_counts(sequences, outcomes, level="detected")
        rows.append({**candidate, **counts.as_dict(), "distance": _distance(candidate)})

    def rank(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            row["f1"] if row["f1"] is not None else -1.0,
            row["specificity"] if row["specificity"] is not None else -1.0,
            -row["distance"],
        )

    rows.sort(key=rank, reverse=True)
    best = rows[0]
    thresholds = {key: best[key] for key in keys}
    thresholds["long_lie_s"] = base.long_lie_s
    thresholds["angle_upright_deg"] = base.angle_upright_deg
    thresholds["recovery_hold_s"] = base.recovery_hold_s
    thresholds["recovery_height_ratio"] = base.recovery_height_ratio
    score = {
        key: best[key]
        for key in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "specificity", "f1")
    }
    return CalibrationResult(
        thresholds=thresholds,
        score=score,
        table=rows,
        grid={key: list(values) for key, values in grid.items()},
    )


def frozen_payload(
    result: CalibrationResult,
    *,
    model: str,
    weights_sha256: str,
    sequences: Sequence[LabelledSequence],
    extracted: Mapping[str, ExtractedSequence],
) -> dict[str, Any]:
    """The content of the frozen calibration artifact, with its provenance."""
    provenance = {
        "dataset": "caucafall",
        "doi": "10.17632/7w7fccy7ky.4",
        "sequences": len(sequences),
        "falls": sum(1 for sequence in sequences if sequence.is_fall),
        "adls": sum(1 for sequence in sequences if not sequence.is_fall),
        "model": model,
        "weights_sha256": weights_sha256,
        "extraction_digest": _digest(extracted),
        "objective": "event-level F1 of 'fall detected' (impact reached); ties: specificity, "
        "then closeness to the literature defaults",
        "not_searched": {"long_lie_s": "the draft's clinical confirmation time"},
    }
    body = {
        "frozen": True,
        "selected_on": "caucafall (calibration set only)",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "thresholds": result.thresholds,
        "calibration_score": result.score,
        "literature_defaults": dict(LITERATURE_DEFAULTS),
        "grid": result.grid,
        "provenance": provenance,
        "top_candidates": result.table[:10],
        "rule": "never re-select these thresholds after looking at URFD (the test set)",
    }
    body["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in body.items() if k != "created_utc"}, sort_keys=True).encode()
    ).hexdigest()
    return body


def _distance(candidate: Mapping[str, Any]) -> float:
    total = 0.0
    for key, value in candidate.items():
        default = LITERATURE_DEFAULTS.get(key)
        if default is None:
            continue
        if isinstance(value, bool):
            total += 0.0 if value == default else 1.0
        else:
            total += abs(float(value) - float(default)) / max(abs(float(default)), 1e-9)
    return round(total, 6)


def _digest(extracted: Mapping[str, ExtractedSequence]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(extracted):
        item = extracted[key]
        hasher.update(f"{key}:{item.weights_sha256}:{len(item.frames)}".encode())
    return hasher.hexdigest()
