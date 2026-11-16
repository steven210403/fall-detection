"""Tracker adapter tests: explicit configuration and measured frame-buffer conversion.

Thesis plan §11.1 requires the tracker to be selected explicitly and warns that
Ultralytics defaults can change, so these tests assert that the values our
configuration declares are the values the tracker runs with — including GMC being
disabled for a fixed camera and ReID being off.

The adapter is driven with synthetic detection boxes, which is exactly what
Experiment 2 (§15.2) will do with cached detections: no detector is loaded, so no
accuracy claim is possible or intended here. What is asserted is behaviour that is
ours to define — how identities survive a gap, when the tracker reissues an
integer, and how a frame count is converted into a duration.
"""

from __future__ import annotations

import pytest

from fall_detection.config import Settings, TrackerName, TrackingSettings
from fall_detection.core.clock import ManualClock
from fall_detection.core.errors import ConfigurationError
from fall_detection.detection.contracts import Detection
from fall_detection.tracking import TrackerAdapter, buffer_frames_for

FRAME_SHAPE = (480, 640, 3)


def tracking_settings(**overrides: object) -> TrackingSettings:
    """Build a tracking settings group without touching the environment."""
    return TrackingSettings(**overrides)  # type: ignore[arg-type]


def make_adapter(
    *,
    tracker: TrackerName = "bytetrack",
    clock: ManualClock | None = None,
    **overrides: object,
) -> TrackerAdapter:
    settings = tracking_settings(tracker=tracker, **overrides)
    return TrackerAdapter(settings, clock=clock)


def person(
    x: float,
    y: float = 100.0,
    *,
    width: float = 60.0,
    height: float = 200.0,
    confidence: float = 0.9,
) -> Detection:
    """A person detection at ``x``; only geometry matters to the tracker."""
    return Detection(
        bbox=(x, y, x + width, y + height),
        confidence=confidence,
        class_name="person",
        class_id=0,
    )


def other(kind: str, x: float, class_id: int) -> Detection:
    return Detection(
        bbox=(x, 10.0, x + 30.0, 40.0), confidence=0.7, class_name=kind, class_id=class_id
    )


class TestBufferConversion:
    def test_frames_convert_from_the_assumed_rate(self) -> None:
        # 2 s at 10 Hz is 20 frames. §9.1 requires this conversion to be explicit
        # because the tracker counts frames, not seconds.
        settings = tracking_settings(loss_tolerance_s=2.0, assumed_update_rate_hz=10.0)
        assert settings.buffer_frames == 20

    def test_frames_round_up_so_the_tolerance_is_never_shortened(self) -> None:
        settings = tracking_settings(loss_tolerance_s=2.0, assumed_update_rate_hz=3.0)
        # 6 frames exactly; a fractional case must round up.
        assert settings.buffer_frames == 6
        settings = tracking_settings(loss_tolerance_s=1.0, assumed_update_rate_hz=3.0)
        assert settings.buffer_frames == 3
        settings = tracking_settings(loss_tolerance_s=0.5, assumed_update_rate_hz=3.0)
        assert settings.buffer_frames == 2  # ceil(1.5)

    def test_frames_never_drop_below_one(self) -> None:
        # A sub-frame tolerance still needs a one-frame buffer, because the
        # tracker cannot hold a track for less than one frame.
        settings = tracking_settings(
            loss_tolerance_s=0.05, stale_after_s=0.02, assumed_update_rate_hz=2.0
        )
        assert settings.buffer_frames == 1

    def test_derived_buffer_never_shrinks_with_a_faster_measured_rate(self) -> None:
        # A faster measured rate needs *more* frames for the same duration, and an
        # unknown/zero rate must not shorten the configured assumption either.
        assert buffer_frames_for(2.0, 30.0, minimum_frames=20, maximum_frames=120) == 60
        assert buffer_frames_for(2.0, 0.0, minimum_frames=20, maximum_frames=120) == 20

    def test_derived_buffer_is_capped(self) -> None:
        # A very fast stream must not grow the lost-track pool without bound.
        assert buffer_frames_for(2.0, 1000.0, minimum_frames=20, maximum_frames=120) == 120

    def test_configured_minimum_wins_over_a_slow_measured_rate(self) -> None:
        assert buffer_frames_for(2.0, 1.0, minimum_frames=20, maximum_frames=120) == 20


class TestExplicitConfiguration:
    def test_thresholds_come_from_settings(self) -> None:
        adapter = make_adapter(
            track_high_thresh=0.4,
            track_low_thresh=0.2,
            new_track_thresh=0.3,
            match_thresh=0.7,
            fuse_score=False,
        )
        thresholds = adapter.info.thresholds
        assert thresholds["track_high_thresh"] == 0.4
        assert thresholds["track_low_thresh"] == 0.2
        assert thresholds["new_track_thresh"] == 0.3
        assert thresholds["match_thresh"] == 0.7
        assert thresholds["fuse_score"] is False

    def test_both_trackers_are_selectable(self) -> None:
        for name in ("bytetrack", "botsort"):
            adapter = make_adapter(tracker=name)
            adapter.reset()
            assert adapter.name == name
            assert adapter.info.tracker_type == name

    def test_botsort_runs_without_gmc_or_reid(self) -> None:
        # §11.1: the camera is fixed, so GMC is disabled, and ReID stays off so a
        # BoT-SORT result cannot be described as evidence of a ReID benefit.
        adapter = make_adapter(tracker="botsort")
        adapter.reset()
        info = adapter.info
        assert info.gmc_method == "none"
        assert info.with_reid is False

    def test_botsort_gmc_is_actually_disabled(self) -> None:
        # The configuration value is not enough: §11.1's decision is that no
        # global motion compensation runs at all.
        adapter = make_adapter(tracker="botsort")
        adapter.reset()
        gmc = getattr(adapter._tracker, "gmc", None)  # noqa: SLF001 - asserting the real object
        assert gmc is not None
        assert gmc.method is None

    def test_botsort_has_no_appearance_encoder(self) -> None:
        adapter = make_adapter(tracker="botsort")
        adapter.reset()
        assert getattr(adapter._tracker, "encoder", None) is None  # noqa: SLF001

    def test_reid_is_rejected_rather_than_ignored(self) -> None:
        # §11.1 defers ReID; silently ignoring the flag would let a run be
        # mislabelled as a ReID comparison.
        with pytest.raises(ValueError, match="with_reid"):
            tracking_settings(with_reid=True)

    def test_unavailable_tracker_is_rejected_at_parse_time(self) -> None:
        # The Literal type rejects a tracker this project does not implement, so a
        # typo (or an unimplemented tracker such as DeepSORT from §11.1) surfaces
        # as configuration validation rather than a mid-stream failure.
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="bytetrack"):
            Settings(
                _env_file=None,  # type: ignore[call-arg]
                tracking={"tracker": "deepsort"},
            )

    def test_validate_reports_a_missing_tracker_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = make_adapter()
        # Simulate a release that dropped the tracker while keeping the YAML.
        import ultralytics.trackers.track as track_module

        monkeypatch.setattr(track_module, "TRACKER_MAP", {})
        with pytest.raises(ConfigurationError, match="not available"):
            adapter.validate()

    def test_missing_packaged_config_is_a_model_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = make_adapter()

        def boom(_name: str) -> str:
            raise OSError("no such yaml")

        import ultralytics.utils.checks as checks_module

        monkeypatch.setattr(checks_module, "check_yaml", boom)
        from fall_detection.core.errors import ModelUnavailableError

        with pytest.raises(ModelUnavailableError, match="tracker configuration"):
            adapter.validate()

    def test_update_before_reset_raises(self) -> None:
        adapter = make_adapter()
        with pytest.raises(ConfigurationError, match="before reset"):
            adapter.update([person(100.0)], frame_shape=FRAME_SHAPE, now_monotonic=0.0)


class TestTrackingBehaviour:
    def _steady(
        self,
        adapter: TrackerAdapter,
        clock: ManualClock,
        *,
        frames: int = 6,
        ids: tuple[int, ...] = (1,),
    ):
        rows = ()
        for _ in range(frames):
            detections = [person(100.0 + 4.0 * index) for index in range(len(ids))]
            rows = adapter.update(
                detections, frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()
            )
            clock.advance(0.1)
        return rows

    def test_empty_frame_returns_no_rows(self) -> None:
        clock = ManualClock()
        adapter = make_adapter(clock=clock)
        adapter.reset()
        assert adapter.update([], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()) == ()

    def test_steady_person_keeps_one_integer_id(self) -> None:
        clock = ManualClock()
        adapter = make_adapter(clock=clock)
        adapter.reset()
        identifiers = set()
        for _ in range(6):
            rows = adapter.update(
                [person(100.0)], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()
            )
            identifiers.update(row.display_id for row in rows)
            clock.advance(0.1)
        assert len(identifiers) == 1

    def test_multiple_people_get_multiple_ids(self) -> None:
        clock = ManualClock()
        adapter = make_adapter(clock=clock)
        adapter.reset()
        rows = ()
        for _ in range(4):
            rows = adapter.update(
                [person(100.0), person(300.0), person(500.0)],
                frame_shape=FRAME_SHAPE,
                now_monotonic=clock.monotonic(),
            )
            clock.advance(0.1)
        assert len({row.display_id for row in rows}) == 3

    def test_lost_track_emits_no_row(self) -> None:
        # §11.2: a motion prediction is not activity evidence, so a track without
        # a matching detection must not appear in the output at all.
        clock = ManualClock()
        adapter = make_adapter(clock=clock)
        adapter.reset()
        self._steady(adapter, clock)
        for _ in range(3):
            rows = adapter.update([], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic())
            assert rows == ()
            clock.advance(0.1)

    def test_recovery_within_the_buffer_reuses_the_integer(self) -> None:
        clock = ManualClock()
        adapter = make_adapter(clock=clock, loss_tolerance_s=2.0)
        adapter.reset()
        rows = self._steady(adapter, clock)
        first = {row.display_id for row in rows}
        for _ in range(5):  # 0.5 s, inside the tolerance
            adapter.update([], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic())
            clock.advance(0.1)
        rows = adapter.update(
            [person(120.0)], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()
        )
        assert {row.display_id for row in rows} == first

    def test_bbox_and_confidence_come_from_the_matched_detection(self) -> None:
        clock = ManualClock()
        adapter = make_adapter(clock=clock)
        adapter.reset()
        detection = person(100.0, confidence=0.77)
        rows = adapter.update([detection], frame_shape=FRAME_SHAPE, now_monotonic=0.0)
        assert len(rows) == 1
        # The adapter reads the score back through the detection index instead of
        # trusting the tracker's own score column.
        assert rows[0].confidence == pytest.approx(0.77)
        assert rows[0].detection_index == 0
        assert rows[0].class_id == 0
        x1, y1, x2, y2 = rows[0].bbox
        assert x1 == pytest.approx(100.0, abs=2.0)
        assert y2 == pytest.approx(300.0, abs=2.0)

    def test_detection_index_is_taken_from_the_person_list(self) -> None:
        # The index is what recovers the matched evidence, so it must refer to the
        # sequence the caller passed, not to the whole frame's detections.
        clock = ManualClock()
        adapter = make_adapter(clock=clock)
        adapter.reset()
        first = adapter.update(
            [person(100.0, confidence=0.6), person(400.0, confidence=0.8)],
            frame_shape=FRAME_SHAPE,
            now_monotonic=0.0,
        )
        second = adapter.update(
            [person(400.0, confidence=0.8), person(100.0, confidence=0.6)],
            frame_shape=FRAME_SHAPE,
            now_monotonic=0.1,
        )
        scores = {row.display_id: row.confidence for row in first}
        for row in second:
            assert row.confidence == pytest.approx(scores[row.display_id])

    def test_low_confidence_detection_is_still_associated(self) -> None:
        # §11.1: ByteTrack associates high *and* low confidence detections. A low
        # score must not silently drop a person who is already tracked.
        clock = ManualClock()
        adapter = make_adapter(clock=clock, track_low_thresh=0.1, track_high_thresh=0.5)
        adapter.reset()
        rows = ()
        for _ in range(4):
            rows = adapter.update(
                [person(100.0, confidence=0.9)],
                frame_shape=FRAME_SHAPE,
                now_monotonic=clock.monotonic(),
            )
            clock.advance(0.1)
        assert len(rows) == 1
        rows = adapter.update(
            [person(104.0, confidence=0.3)],
            frame_shape=FRAME_SHAPE,
            now_monotonic=clock.monotonic(),
        )
        assert len(rows) == 1

    def test_very_low_confidence_detection_starts_no_track(self) -> None:
        # Below `track_low_thresh`, a detection is not evidence of a track.
        clock = ManualClock()
        adapter = make_adapter(clock=clock, track_low_thresh=0.2, new_track_thresh=0.5)
        adapter.reset()
        for _ in range(3):
            rows = adapter.update(
                [person(100.0, confidence=0.05)],
                frame_shape=FRAME_SHAPE,
                now_monotonic=clock.monotonic(),
            )
            assert rows == ()
            clock.advance(0.1)


class TestMeasuredRate:
    def test_rate_is_unknown_before_two_updates(self) -> None:
        adapter = make_adapter()
        adapter.reset()
        assert adapter.measured_rate_hz is None
        assert adapter.effective_buffer_s is None

    def test_rate_follows_the_update_interval(self) -> None:
        clock = ManualClock()
        adapter = make_adapter(clock=clock, assumed_update_rate_hz=10.0)
        adapter.reset()
        for _ in range(20):
            adapter.update(
                [person(100.0)], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()
            )
            clock.advance(0.05)  # 20 Hz
        assert adapter.measured_rate_hz == pytest.approx(20.0, rel=0.05)

    def test_buffer_grows_when_the_measured_rate_is_higher(self) -> None:
        # §9.1: the frame count must be re-derived from the measured rate, or a
        # 25 fps stream would give a 2 s tolerance only 8 frames of buffer.
        clock = ManualClock()
        adapter = make_adapter(clock=clock, loss_tolerance_s=2.0, assumed_update_rate_hz=10.0)
        adapter.reset()
        assert adapter.buffer_frames == 20
        for _ in range(30):
            adapter.update(
                [person(100.0)], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()
            )
            clock.advance(1.0 / 25.0)  # 25 Hz
        assert adapter.buffer_frames >= 50
        assert adapter.info.buffer_sync_supported
        assert adapter.effective_buffer_s == pytest.approx(2.0, rel=0.1)

    def test_buffer_never_drops_below_the_configured_assumption(self) -> None:
        clock = ManualClock()
        adapter = make_adapter(clock=clock, loss_tolerance_s=2.0, assumed_update_rate_hz=30.0)
        adapter.reset()
        assert adapter.buffer_frames == 60
        for _ in range(30):
            adapter.update(
                [person(100.0)], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()
            )
            clock.advance(0.2)  # 5 Hz, far slower than assumed
        assert adapter.buffer_frames == 60

    def test_buffer_respects_the_configured_ceiling(self) -> None:
        clock = ManualClock()
        adapter = make_adapter(clock=clock, loss_tolerance_s=2.0, max_buffer_frames=25)
        adapter.reset()
        for _ in range(30):
            adapter.update(
                [person(100.0)], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()
            )
            clock.advance(0.02)  # 50 Hz
        assert adapter.buffer_frames == 25

    def test_reset_clears_the_measurement(self) -> None:
        clock = ManualClock()
        adapter = make_adapter(clock=clock)
        adapter.reset()
        for _ in range(5):
            adapter.update(
                [person(100.0)], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()
            )
            clock.advance(0.1)
        assert adapter.measured_rate_hz is not None
        adapter.reset()
        assert adapter.measured_rate_hz is None
        assert adapter.info.frames_processed == 0
        assert adapter.info.reset_count == 2


class TestSessionReset:
    def test_reset_restarts_integer_ids(self) -> None:
        # §9.2: a reconnect clears old tracks. Integer IDs restarting is safe
        # because the lifecycle retires terminated IDs and a new session has no
        # live tracks; UUIDs remain the identities.
        clock = ManualClock()
        adapter = make_adapter(clock=clock)
        adapter.reset()
        for _ in range(4):
            adapter.update(
                [person(100.0)], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()
            )
            clock.advance(0.1)
        adapter.reset()
        rows = ()
        for _ in range(4):
            rows = adapter.update(
                [person(100.0)], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()
            )
            clock.advance(0.1)
        assert {row.display_id for row in rows} == {1}

    def test_reset_does_not_reuse_association_state(self) -> None:
        # A fresh instance must not remember the previous session's positions, so a
        # far-away detection becomes a new track instead of being matched.
        clock = ManualClock()
        adapter = make_adapter(clock=clock)
        adapter.reset()
        for _ in range(4):
            adapter.update(
                [person(100.0)], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic()
            )
            clock.advance(0.1)
        adapter.reset()
        rows = adapter.update([person(500.0)], frame_shape=FRAME_SHAPE, now_monotonic=0.0)
        assert len(rows) == 1
        assert rows[0].detection_index == 0

    def test_frames_processed_counts_per_session(self) -> None:
        clock = ManualClock()
        adapter = make_adapter(clock=clock)
        adapter.reset()
        for _ in range(3):
            adapter.update([], frame_shape=FRAME_SHAPE, now_monotonic=clock.monotonic())
            clock.advance(0.1)
        assert adapter.info.frames_processed == 3
        adapter.reset()
        assert adapter.info.frames_processed == 0


class TestDescribe:
    def test_describe_is_json_safe(self) -> None:
        import json

        adapter = make_adapter()
        adapter.reset()
        payload = adapter.describe()
        assert payload["name"] == "bytetrack"
        assert payload["with_reid"] is False
        # 3 s loss tolerance at the assumed 15 Hz.
        assert payload["buffer_frames"] == 45
        json.dumps(payload)

    def test_describe_reports_the_configured_assumption(self) -> None:
        # The configured frame count is reported next to the effective one, so a
        # wrong assumption is visible instead of hidden.
        adapter = make_adapter(loss_tolerance_s=2.0, assumed_update_rate_hz=5.0)
        adapter.reset()
        assert adapter.info.configured_buffer_frames == 10
        assert adapter.info.buffer_frames == 10
