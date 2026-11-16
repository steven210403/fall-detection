"""The §15.4 paired cluster bootstrap (Phase 11).

What is asserted is the *method*, never a metric value:

* units are resampled, not rows — a dataset of two units cannot produce a tight interval
  however many rows it has, and a test that expected one would be testing the fixture;
* the comparison is paired — both metrics see the same draw;
* ``None`` is never coerced to ``0.0``, and a metric that is undefined on most draws is
  refused rather than reported over the survivors;
* the same seed reproduces the same interval.
"""

from __future__ import annotations

import numpy as np
import pytest

from fall_detection.research import bootstrap


def _rows(units: dict[str, int]) -> list[str]:
    """One unit label per row, repeated ``count`` times."""
    return [unit for unit, count in units.items() for _ in range(count)]


class TestBootstrapMetric:
    def test_interval_brackets_the_observed_value(self) -> None:
        labels = _rows({"a": 20, "b": 20, "c": 20})
        values = np.asarray([0.9] * 20 + [0.5] * 20 + [0.7] * 20)
        result = bootstrap.bootstrap_metric(
            labels, lambda rows: float(values[rows].mean()), name="mean", seed=7
        )
        assert result.observed == pytest.approx(0.7)
        assert result.ci_low is not None and result.ci_high is not None
        assert result.ci_low <= result.observed <= result.ci_high  # type: ignore[operator]
        assert result.usable == bootstrap.BOOTSTRAP_RESAMPLES
        assert result.n_units == 3

    def test_resampling_is_by_unit_not_by_row(self) -> None:
        # Every row inside a unit is identical, so a row-level bootstrap would produce a
        # zero-width interval. A unit-level one must not.
        labels = _rows({"a": 50, "b": 50})
        values = np.asarray([0.0] * 50 + [1.0] * 50)
        result = bootstrap.bootstrap_metric(
            labels, lambda rows: float(values[rows].mean()), name="mean", seed=11
        )
        assert result.ci_low is not None and result.ci_high is not None
        assert result.ci_high > result.ci_low

    def test_a_single_unit_is_refused_with_a_reason(self) -> None:
        labels = _rows({"only": 30})
        result = bootstrap.bootstrap_metric(labels, lambda rows: 0.5, name="mean", seed=1)
        assert result.ci_low is None
        assert result.refused is not None
        assert "one unit" in result.refused

    def test_an_undefined_metric_on_most_draws_is_refused(self) -> None:
        labels = _rows({"a": 10, "b": 10})
        calls = {"n": 0}

        def flaky(_rows_selection: np.ndarray) -> float | None:
            calls["n"] += 1
            return 0.5 if calls["n"] == 1 else None

        result = bootstrap.bootstrap_metric(labels, flaky, name="flaky", seed=3)
        assert result.ci_low is None
        assert result.refused is not None
        assert "cannot support an interval" in result.refused

    def test_nan_is_treated_as_undefined_rather_than_zero(self) -> None:
        # Unit ``c`` alone makes the metric undefined; a draw that omits ``c`` is usable.
        # NaN must drop that draw, not become 0.0 and drag the interval down.
        labels = _rows({"a": 10, "b": 10, "c": 10})
        values = np.asarray([0.5] * 10 + [0.5] * 10 + [0.5] * 10)

        def sometimes_nan(rows: np.ndarray) -> float | None:
            drawn = {labels[index] for index in rows.tolist()}
            return float("nan") if drawn == {"c"} else float(values[rows].mean())

        result = bootstrap.bootstrap_metric(labels, sometimes_nan, name="nan", seed=5)
        # Some draws contain only ``c`` and are dropped; the rest still support an interval.
        assert result.usable < bootstrap.BOOTSTRAP_RESAMPLES
        assert result.ci_low == pytest.approx(0.5)

    def test_an_all_undefined_metric_is_refused_rather_than_zeroed(self) -> None:
        labels = _rows({"a": 10, "b": 10})

        def always_nan(_rows: np.ndarray) -> float | None:
            return float("nan")

        result = bootstrap.bootstrap_metric(labels, always_nan, name="nan", seed=5)
        assert result.ci_low is None
        assert result.refused is not None

    def test_the_same_seed_reproduces_the_interval(self) -> None:
        labels = _rows({"a": 10, "b": 10, "c": 10})
        rng = np.random.default_rng(0)
        values = rng.random(30)
        first = bootstrap.bootstrap_metric(
            labels, lambda rows: float(values[rows].mean()), name="mean", seed=99
        )
        second = bootstrap.bootstrap_metric(
            labels, lambda rows: float(values[rows].mean()), name="mean", seed=99
        )
        assert first.ci_low == second.ci_low
        assert first.ci_high == second.ci_high

    def test_a_three_unit_interval_is_not_degenerate(self) -> None:
        labels = _rows({"a": 10, "b": 10, "c": 10})
        values = np.asarray([0.0] * 10 + [0.5] * 10 + [1.0] * 10)
        result = bootstrap.bootstrap_metric(
            labels, lambda rows: float(values[rows].mean()), name="mean", seed=1
        )
        assert result.ci_low is not None and result.ci_high is not None
        assert result.ci_high > result.ci_low

    def test_an_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            bootstrap.bootstrap_metric([], lambda rows: 0.0, name="mean")

    def test_a_non_positive_resample_count_raises(self) -> None:
        with pytest.raises(ValueError):
            bootstrap.bootstrap_metric(["a", "b"], lambda rows: 0.0, name="mean", n_resamples=0)

    def test_the_payload_names_its_unit_and_method(self) -> None:
        labels = _rows({"a": 5, "b": 5})
        payload = bootstrap.bootstrap_metric(
            labels, lambda rows: 0.5, name="mean", unit="episode"
        ).as_dict()
        assert payload["unit"] == "episode"
        assert "percentile bootstrap" in payload["method"]


class TestBootstrapDifference:
    def test_a_constant_offset_is_detected(self) -> None:
        labels = _rows({"a": 20, "b": 20, "c": 20})
        base = np.asarray([0.5] * 20 + [0.6] * 20 + [0.7] * 20)
        better = base + 0.1
        result = bootstrap.bootstrap_difference(
            labels,
            lambda rows: float(better[rows].mean()),
            lambda rows: float(base[rows].mean()),
            name="macro_f1",
            seed=4,
        )
        assert result.observed == pytest.approx(0.1, abs=1e-9)
        assert result.crosses_zero is False
        assert result.ci_low is not None and result.ci_low > 0.0

    def test_identical_arms_cross_zero(self) -> None:
        labels = _rows({"a": 20, "b": 20, "c": 20})
        base = np.asarray([0.5] * 20 + [0.6] * 20 + [0.7] * 20)
        result = bootstrap.bootstrap_difference(
            labels,
            lambda rows: float(base[rows].mean()),
            lambda rows: float(base[rows].mean()),
            name="macro_f1",
            seed=4,
        )
        assert result.observed == pytest.approx(0.0)
        assert result.crosses_zero is True

    def test_both_arms_see_the_same_draw(self) -> None:
        labels = _rows({"a": 10, "b": 10, "c": 10})
        seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def metric_a(rows: np.ndarray) -> float:
            seen.append((tuple(sorted(rows.tolist())), ()))
            return 1.0

        def metric_b(rows: np.ndarray) -> float:
            seen[-1] = (seen[-1][0], tuple(sorted(rows.tolist())))
            return 0.5

        bootstrap.bootstrap_difference(labels, metric_a, metric_b, name="m", seed=8)
        assert seen
        for left, right in seen:
            assert left == right

    def test_the_difference_reports_both_arms(self) -> None:
        labels = _rows({"a": 10, "b": 10, "c": 10})
        result = bootstrap.bootstrap_difference(
            labels, lambda rows: 0.8, lambda rows: 0.6, name="m", seed=2
        )
        payload = result.as_dict()
        assert payload["a"]["observed"] == pytest.approx(0.8)
        assert payload["b"]["observed"] == pytest.approx(0.6)
        assert "paired" in payload["note"]


class TestEpisodeIds:
    def test_a_label_change_starts_a_new_episode(self) -> None:
        ids = bootstrap.assign_episode_ids(["day16"] * 3, ["day16#s0#0"] * 3, [0, 0, 1])
        assert ids[0] == ids[1]
        assert ids[2] != ids[1]

    def test_a_new_track_key_starts_a_new_episode(self) -> None:
        ids = bootstrap.assign_episode_ids(["day16", "day16"], ["day16#s0#0", "day16#s0#1"], [0, 0])
        assert ids[0] != ids[1]

    def test_masked_rows_still_receive_an_identifier(self) -> None:
        ids = bootstrap.assign_episode_ids(["day16"] * 2, ["day16#s0#0"] * 2, [-1, -1])
        assert len(ids) == 2
        assert ids[0] == ids[1]

    def test_a_run_interrupted_by_a_returning_label_is_two_episodes(self) -> None:
        ids = bootstrap.assign_episode_ids(["day16"] * 4, ["day16#s0#0"] * 4, [0, 0, 1, 0])
        assert ids[0] == ids[1]
        assert ids[2] not in {ids[0], ids[3]}
        assert ids[3] != ids[0]

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            bootstrap.assign_episode_ids(["day16"], ["day16#s0#0", "day16#s0#1"], [0])
