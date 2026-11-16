"""Latency, throughput, and memory reporting (§15.2 Experiment 5, §23.1).

Two rules are asserted here, and both are about honesty rather than speed:

* an empty sample reports ``None``, never ``0.0`` — a percentile of nothing is not a
  measurement, and a zero would read as "instantaneous";
* on Apple Silicon the accelerator figure is labelled *unified memory*, because §23.1
  forbids presenting it as discrete VRAM.

A per-frame total is only reported when every stage ran the same number of times: summing
a partially instrumented run would silently omit a stage's cost.
"""

from __future__ import annotations

import pytest

from fall_detection.research import pipeline_benchmark as bench


class TestPercentile:
    def test_the_median_of_an_odd_sample_is_the_middle_value(self) -> None:
        assert bench.percentile([3.0, 1.0, 2.0], 50.0) == pytest.approx(2.0)

    def test_p95_of_a_known_sample_interpolates(self) -> None:
        values = [float(value) for value in range(1, 101)]
        assert bench.percentile(values, 95.0) == pytest.approx(95.05, abs=0.02)

    def test_one_sample_is_its_own_percentile(self) -> None:
        assert bench.percentile([7.5], 95.0) == 7.5

    def test_an_empty_sample_is_none(self) -> None:
        assert bench.percentile([], 50.0) is None

    def test_an_out_of_range_quantile_raises(self) -> None:
        with pytest.raises(ValueError):
            bench.percentile([1.0], 101.0)


class TestLatencySummary:
    def test_an_empty_summary_reports_none_and_says_why(self) -> None:
        summary = bench.summarise_latency("detect", [])
        assert summary.samples == 0
        assert summary.p50 is None and summary.p95 is None
        assert "not reported as 0.0" in summary.note

    def test_a_summary_carries_its_sample_count(self) -> None:
        summary = bench.summarise_latency("detect", [1.0, 2.0, 3.0])
        assert summary.samples == 3
        assert summary.mean == pytest.approx(2.0)
        assert summary.minimum == 1.0 and summary.maximum == 3.0

    def test_the_note_refuses_a_capture_to_display_claim(self) -> None:
        payload = bench.summarise_latency("detect", [1.0]).as_dict()
        assert "not a capture-to-display" in payload["note"]


class TestStageTimer:
    def test_stages_are_summarised_independently(self) -> None:
        timer = bench.StageTimer()
        timer.extend("detect", [10.0, 12.0])
        timer.extend("track", [1.0, 1.5])
        assert timer.summary("detect").samples == 2
        assert timer.summary("track").mean == pytest.approx(1.25)

    def test_the_total_is_the_per_frame_sum(self) -> None:
        timer = bench.StageTimer()
        timer.record("detect", 10.0)
        timer.record("track", 2.0)
        timer.record("detect", 20.0)
        timer.record("track", 4.0)
        total = timer.total()
        assert total.samples == 2
        # Per-frame sums in recording order: 10+2 and 20+4.
        assert total.minimum == pytest.approx(12.0)
        assert total.maximum == pytest.approx(24.0)

    def test_a_ragged_run_refuses_a_total(self) -> None:
        timer = bench.StageTimer()
        timer.record("detect", 10.0)
        timer.record("detect", 11.0)
        timer.record("track", 1.0)
        total = timer.total()
        assert total.samples == 0
        assert "omit a stage's cost" in total.note

    def test_a_recorded_total_is_not_added_to_the_stage_sum(self) -> None:
        """A caller that times the whole frame records it; the total is not that plus itself.

        The GRU replay records every stage *and* the frame span covering them, so adding
        the recorded total to the stage samples reported 41.0 ms for a frame whose stages
        summed to 20.5 ms — every figure doubled.
        """
        timer = bench.StageTimer()
        timer.record("detect", 18.0)
        timer.record("track", 1.0)
        timer.record("features", 0.5)
        timer.record("activity", 0.5)
        # The literal stage name the replays use, so this test fails on the old
        # arithmetic rather than on a missing constant.
        timer.record("frame_total", 20.0)
        total = timer.total()
        assert total.samples == 1
        assert total.mean == pytest.approx(20.0)
        assert "measured per-frame total" in total.note

    def test_the_recorded_total_survives_the_payload(self) -> None:
        timer = bench.StageTimer()
        timer.record("detect", 18.0)
        timer.record("frame_total", 20.0)
        payload = timer.as_dict()
        assert payload["frame_total"]["mean"] == pytest.approx(20.0)
        # The recorded total is reported once, as the total, never as a stage beside it.
        assert payload["counts"]["frame_total"] == 1
        assert payload["detect"]["mean"] == pytest.approx(18.0)

    def test_a_total_only_caller_keeps_its_measurement(self) -> None:
        """The soak records only a total; that measurement must not be discarded."""
        timer = bench.StageTimer()
        timer.extend("frame_total", [30.0, 34.0, 32.0])
        total = timer.total()
        assert total.samples == 3
        assert total.mean == pytest.approx(32.0)
        assert timer.as_dict()["frame_total"]["samples"] == 3

    def test_stage_means_and_a_recorded_total_cannot_double_count(self) -> None:
        """The regression in one assertion: a 20 ms frame stays 20 ms."""
        timer = bench.StageTimer()
        for _ in range(3):
            timer.record("detect", 10.0)
            timer.record("track", 6.0)
            timer.record("features", 4.0)
            timer.record("frame_total", 20.0)
        stage_sum = sum(timer.summary(name).mean or 0.0 for name in ("detect", "track", "features"))
        assert stage_sum == pytest.approx(20.0)
        assert timer.total().mean == pytest.approx(20.0)

    def test_an_unmeasured_stage_is_absent_not_zero(self) -> None:
        timer = bench.StageTimer()
        timer.record("detect", 5.0)
        payload = timer.as_dict()
        assert "track" not in payload
        assert payload["detect"]["samples"] == 1

    def test_the_payload_counts_every_call(self) -> None:
        timer = bench.StageTimer()
        timer.record("detect", 5.0)
        timer.record("detect", 6.0)
        assert timer.as_dict()["counts"]["detect"] == 2


class TestMemory:
    def test_a_cpu_run_reports_no_accelerator_allocation(self) -> None:
        payload = bench.memory_snapshot(device="cpu")
        assert payload["accelerator"]["kind"] == "none"
        assert "no accelerator memory" in payload["accelerator"]["reason"]

    def test_the_unified_memory_statement_is_always_present(self) -> None:
        payload = bench.memory_snapshot(device="cpu")
        assert "unified-memory" in payload["note"] or "unified memory" in payload["note"]
        assert "not discrete VRAM" in payload["note"]

    def test_an_mps_run_is_labelled_unified(self) -> None:
        payload = bench.memory_snapshot(device="mps")
        accelerator = payload["accelerator"]
        assert accelerator["kind"] == "mps"
        if accelerator.get("available"):
            assert accelerator["unified_memory"] is True
            assert "not discrete VRAM" in accelerator["note"]

    def test_process_figures_are_reported(self) -> None:
        payload = bench.memory_snapshot(device="cpu")
        assert payload["process_peak_rss_bytes"] > 0
        assert payload["threads"] >= 1


class TestThroughput:
    def test_frames_per_second_is_frames_over_time(self) -> None:
        payload = bench.summarise_throughput(300, 10.0, people=450)
        assert payload["frames_per_second"] == pytest.approx(30.0)
        assert payload["people_per_frame"] == pytest.approx(1.5)

    def test_a_non_positive_elapsed_time_is_refused(self) -> None:
        payload = bench.summarise_throughput(10, 0.0)
        assert payload["status"] == "refused"

    def test_the_note_refuses_a_live_camera_claim(self) -> None:
        payload = bench.summarise_throughput(10, 1.0)
        assert "not a claim about a live camera" in payload["note"]
