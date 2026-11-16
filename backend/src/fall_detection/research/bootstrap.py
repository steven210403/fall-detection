"""Paired cluster bootstrap for the §15.4 statistics (Phase 11).

§15.4 fixes the method: *"Independent units are sessions and episodes, not overlapping
windows. Use paired session-level bootstrap with 1,000 resamples. Report mean±SD across
three seeds and confidence intervals."* This module is that sentence, once, so every arm
comparison in Phase 11 uses the same arithmetic and the same seed.

Three properties are deliberate:

* **Units are resampled, not windows.** A window is a view of an episode, and windows from
  one recording share people, camera and lighting. Resampling windows would shrink every
  interval by roughly the square root of the window-per-unit ratio, which is the standard
  way an overlapping-window evaluation overstates its own precision.
* **The comparison is paired.** Both arms are evaluated on the *same* resampled units, so
  the interval describes the difference between two models on one dataset rather than two
  independent measurements on two datasets.
* **The metric is recomputed on the resampled pool.** A macro-F1 computed per unit and
  then averaged is a different quantity from a macro-F1 computed on the pooled units,
  because class support differs per unit. The caller passes the metric, so the resample
  measures the same quantity the report quotes.

``None`` is never coerced to ``0.0``: a resample in which a class is absent makes macro-F1
undefined, and those resamples are dropped and counted. If too few resamples remain the
result is refused rather than reported, because an interval over three usable resamples is
not an interval.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "MIN_USABLE_RESAMPLES",
    "BootstrapResult",
    "DifferenceResult",
    "assign_episode_ids",
    "bootstrap_difference",
    "bootstrap_metric",
]

BOOTSTRAP_RESAMPLES = 1000
"""§15.4's declared resample count. Changing it changes the interval's resolution."""

MIN_USABLE_RESAMPLES = 100
"""Fewest usable resamples for a reported interval.

A metric can be undefined on a resample (a class missing from the drawn units). If more
than 90% of the draws are unusable the data cannot support an interval and the result says
so instead of returning the few that survived.
"""


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """A point estimate with a percentile interval over resampled units.

    Attributes:
        metric: Name of the quantity, for a report.
        observed: The metric on the observed units, i.e. the reported point estimate.
        mean: Mean of the resampled estimates.
        ci_low: Lower percentile (``alpha / 2``).
        ci_high: Upper percentile (``1 - alpha / 2``).
        n_units: Number of distinct units.
        n_resamples: Requested resamples.
        usable: Resamples on which the metric was defined.
        seed: Seed, so the interval is reproducible.
        unit: What was resampled, in words.
        refused: Reason the interval is unavailable, or ``None``.
    """

    metric: str
    observed: float | None
    mean: float | None
    ci_low: float | None
    ci_high: float | None
    n_units: int
    n_resamples: int
    usable: int
    seed: int
    unit: str
    refused: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "metric": self.metric,
            "observed": self.observed,
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_units": self.n_units,
            "n_resamples": self.n_resamples,
            "usable": self.usable,
            "seed": self.seed,
            "unit": self.unit,
            "refused": self.refused,
            "method": "percentile bootstrap over resampled units (§15.4)",
        }


@dataclass(frozen=True, slots=True)
class DifferenceResult:
    """The paired difference between two arms on the same resampled units.

    Attributes:
        metric: Name of the compared quantity.
        observed: ``a`` minus ``b`` on the observed units.
        mean: Mean resampled difference.
        ci_low: Lower percentile of the difference.
        ci_high: Upper percentile.
        crosses_zero: Whether the interval contains zero, i.e. the data does not separate
            the arms at this confidence. Reported explicitly so a difference is never
            described as an improvement when the interval says otherwise.
        a: The first arm's own bootstrap.
        b: The second arm's own bootstrap.
        seed: Seed shared by all three intervals.
        usable: Resamples where *both* metrics were defined.
        refused: Reason the difference is unavailable, or ``None``.
    """

    metric: str
    observed: float | None
    mean: float | None
    ci_low: float | None
    ci_high: float | None
    crosses_zero: bool | None
    a: BootstrapResult
    b: BootstrapResult
    seed: int
    usable: int
    refused: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "metric": self.metric,
            "observed": self.observed,
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "crosses_zero": self.crosses_zero,
            "usable": self.usable,
            "seed": self.seed,
            "refused": self.refused,
            "a": self.a.as_dict(),
            "b": self.b.as_dict(),
            "note": (
                "Both arms were scored on the same resampled units (paired), so the "
                "interval describes the difference on this dataset, not two independent "
                "measurements."
            ),
        }


def _percentiles(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Percentile interval bounds for ``values``."""
    low, high = np.percentile(values, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def _resample_indices(
    unit_index: np.ndarray, n_units: int, n_resamples: int, seed: int
) -> list[np.ndarray]:
    """Row indices for each resample, drawn by unit with replacement.

    A unit drawn twice contributes its rows twice, which is what makes the resample a
    bootstrap over units rather than over rows.
    """
    rng = np.random.default_rng(seed)
    members: list[np.ndarray] = [np.nonzero(unit_index == unit)[0] for unit in range(n_units)]
    draws: list[np.ndarray] = []
    for _ in range(n_resamples):
        picked = rng.integers(0, n_units, size=n_units)
        draws.append(np.concatenate([members[unit] for unit in picked]))
    return draws


def bootstrap_metric(
    unit_ids: Sequence[str],
    metric: Callable[[np.ndarray], float | None],
    *,
    name: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
    alpha: float = 0.05,
    unit: str = "recording",
) -> BootstrapResult:
    """Bootstrap ``metric`` by resampling ``unit_ids`` with replacement.

    Args:
        unit_ids: One unit identifier per row; rows sharing an id move together.
        metric: Function from row indices to a float, or ``None`` when the quantity is
            undefined for that draw (e.g. a class absent from the resample).
        name: Metric name for the report.
        n_resamples: Draws; §15.4 declares 1,000.
        seed: Reproducibility seed.
        alpha: Two-sided level; ``0.05`` gives a 95% interval.
        unit: What a unit is, in words (``recording`` or ``episode``).

    Returns:
        The point estimate and its interval, or a refusal when too few draws were usable.
    """
    labels = np.asarray([str(value) for value in unit_ids])
    if labels.size == 0:
        raise ValueError("bootstrap needs at least one row")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    names = sorted(set(labels.tolist()))
    lookup: Mapping[str, int] = {name_: index for index, name_ in enumerate(names)}
    unit_index = np.asarray([lookup[value] for value in labels], dtype=np.int64)
    observed = metric(np.arange(labels.size))
    estimates: list[float] = []
    for rows in _resample_indices(unit_index, len(names), n_resamples, seed):
        value = metric(rows)
        if value is not None and np.isfinite(value):
            estimates.append(float(value))
    refused = None
    if len(names) < 2:
        refused = (
            f"only {len(names)} {unit}(s) available; a bootstrap over one unit has zero "
            "between-unit variance and cannot describe uncertainty"
        )
    elif len(estimates) < MIN_USABLE_RESAMPLES:
        refused = (
            f"{len(estimates)} of {n_resamples} resamples had a defined metric; fewer than "
            f"{MIN_USABLE_RESAMPLES} cannot support an interval"
        )
    if refused is not None or not estimates:
        return BootstrapResult(
            metric=name,
            observed=None if observed is None else float(observed),
            mean=None,
            ci_low=None,
            ci_high=None,
            n_units=len(names),
            n_resamples=n_resamples,
            usable=len(estimates),
            seed=seed,
            unit=unit,
            refused=refused or "no resample produced a defined metric",
        )
    values = np.asarray(estimates, dtype=np.float64)
    low, high = _percentiles(values, alpha)
    return BootstrapResult(
        metric=name,
        observed=None if observed is None else float(observed),
        mean=float(values.mean()),
        ci_low=low,
        ci_high=high,
        n_units=len(names),
        n_resamples=n_resamples,
        usable=len(estimates),
        seed=seed,
        unit=unit,
    )


def bootstrap_difference(
    unit_ids: Sequence[str],
    metric_a: Callable[[np.ndarray], float | None],
    metric_b: Callable[[np.ndarray], float | None],
    *,
    name: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
    alpha: float = 0.05,
    unit: str = "recording",
) -> DifferenceResult:
    """Paired bootstrap of ``metric_a - metric_b`` over the same resampled units.

    Both metrics receive the *same* row selection in each draw, which is what makes the
    comparison paired: the interval describes the difference between the two arms on this
    dataset rather than the difference between two independent estimates.
    """
    labels = np.asarray([str(value) for value in unit_ids])
    if labels.size == 0:
        raise ValueError("bootstrap needs at least one row")
    names = sorted(set(labels.tolist()))
    lookup = {value: index for index, value in enumerate(names)}
    unit_index = np.asarray([lookup[value] for value in labels], dtype=np.int64)
    observed_a = metric_a(np.arange(labels.size))
    observed_b = metric_b(np.arange(labels.size))
    estimates_a: list[float] = []
    estimates_b: list[float] = []
    differences: list[float] = []
    for rows in _resample_indices(unit_index, len(names), n_resamples, seed):
        value_a = metric_a(rows)
        value_b = metric_b(rows)
        if value_a is None or value_b is None:
            continue
        if not (np.isfinite(value_a) and np.isfinite(value_b)):
            continue
        estimates_a.append(float(value_a))
        estimates_b.append(float(value_b))
        differences.append(float(value_a) - float(value_b))
    a = _summarise(name, observed_a, estimates_a, len(names), n_resamples, seed, alpha, unit)
    b = _summarise(name, observed_b, estimates_b, len(names), n_resamples, seed, alpha, unit)
    refused = a.refused or b.refused
    if refused is not None or not differences:
        return DifferenceResult(
            metric=name,
            observed=(
                None
                if observed_a is None or observed_b is None
                else float(observed_a) - float(observed_b)
            ),
            mean=None,
            ci_low=None,
            ci_high=None,
            crosses_zero=None,
            a=a,
            b=b,
            seed=seed,
            usable=len(differences),
            refused=refused or "no resample produced both metrics",
        )
    values = np.asarray(differences, dtype=np.float64)
    low, high = _percentiles(values, alpha)
    observed = (
        None if observed_a is None or observed_b is None else float(observed_a) - float(observed_b)
    )
    return DifferenceResult(
        metric=name,
        observed=observed,
        mean=float(values.mean()),
        ci_low=low,
        ci_high=high,
        crosses_zero=bool(low <= 0.0 <= high),
        a=a,
        b=b,
        seed=seed,
        usable=len(values),
    )


def _summarise(
    name: str,
    observed: float | None,
    estimates: Sequence[float],
    n_units: int,
    n_resamples: int,
    seed: int,
    alpha: float,
    unit: str,
) -> BootstrapResult:
    """Build one arm's :class:`BootstrapResult` from its collected estimates."""
    refused = None
    if n_units < 2:
        refused = (
            f"only {n_units} {unit}(s) available; a bootstrap over one unit has zero "
            "between-unit variance and cannot describe uncertainty"
        )
    elif len(estimates) < MIN_USABLE_RESAMPLES:
        refused = (
            f"{len(estimates)} of {n_resamples} resamples had a defined metric; fewer than "
            f"{MIN_USABLE_RESAMPLES} cannot support an interval"
        )
    if refused is not None or not estimates:
        return BootstrapResult(
            metric=name,
            observed=None if observed is None else float(observed),
            mean=None,
            ci_low=None,
            ci_high=None,
            n_units=n_units,
            n_resamples=n_resamples,
            usable=len(estimates),
            seed=seed,
            unit=unit,
            refused=refused or "no resample produced a defined metric",
        )
    values = np.asarray(estimates, dtype=np.float64)
    low, high = _percentiles(values, alpha)
    return BootstrapResult(
        metric=name,
        observed=None if observed is None else float(observed),
        mean=float(values.mean()),
        ci_low=low,
        ci_high=high,
        n_units=n_units,
        n_resamples=n_resamples,
        usable=len(estimates),
        seed=seed,
        unit=unit,
    )


def assign_episode_ids(
    recording_ids: Sequence[str],
    track_keys: Sequence[str],
    targets: Sequence[int],
) -> list[str]:
    """Episode identifier per row: a contiguous run of one target inside one slot run.

    §15.4's independent unit is the *episode*, and an episode ends at a label change, a
    missing observation or a gap — which is exactly where ``track_key`` (the
    segment/slot-continuity proxy) or the target changes. Rows are assumed to be in
    acquisition order per recording, as ``build_samples`` emits them.

    Returns:
        One identifier per row of the form ``<track_key>#e<n>``. Masked targets still get
        an identifier, so a caller can count them; they simply are not scored.
    """
    if not (len(recording_ids) == len(track_keys) == len(targets)):
        raise ValueError("recording_ids, track_keys and targets must have the same length")
    out: list[str] = []
    previous_key: str | None = None
    previous_target: int | None = None
    run = -1
    for key, target in zip(track_keys, targets, strict=True):
        value = int(target)
        if key != previous_key or value != previous_target:
            run += 1
        out.append(f"{key}#e{run}")
        previous_key = key
        previous_target = value
    return out
