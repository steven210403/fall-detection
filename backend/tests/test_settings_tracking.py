"""Settings tests for the Phase 4 tracking configuration.

Thesis plan §11.1 requires the tracker to be selected explicitly and §11.2 fixes
the time-based lifecycle rules, so these tests pin the defaults an operator
inherits, the validation that keeps a misconfiguration from reaching the tracker,
and the ``FALL_TRACKING__*`` environment surface.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fall_detection.config import Settings, TrackingSettings


class TestTrackingDefaults:
    def test_tracking_is_enabled_by_default(self) -> None:
        # Tracking needs no weights and no warmup, so unlike detection it can be on
        # without making an import or a start expensive.
        assert Settings().tracking.enabled is True

    def test_bytetrack_is_the_default_tracker(self) -> None:
        # §11.1: ByteTrack is the initial primary tracker.
        assert TrackingSettings().tracker == "bytetrack"

    def test_loss_tolerance_defaults_to_three_seconds(self) -> None:
        # A person lying still after a fall can drop out of detection briefly, so the
        # tolerance is longer than a typical walking-scene default.
        assert TrackingSettings().loss_tolerance_s == 3.0

    def test_staleness_defaults_to_half_a_second(self) -> None:
        # §11.2: "Old predictions become stale after 0.5 seconds".
        assert TrackingSettings().stale_after_s == 0.5

    def test_assumed_rate_matches_the_detection_sampling_target(self) -> None:
        # Pose runs on every decoded frame; 15 Hz is the planned live decode rate.
        assert TrackingSettings().assumed_update_rate_hz == 15.0

    def test_frame_buffer_is_derived_from_the_tolerance_and_rate(self) -> None:
        # §9.1: a frame count is not automatically a duration, so the conversion
        # is explicit and reported.
        assert TrackingSettings().buffer_frames == 45

    def test_reid_is_off_by_default(self) -> None:
        # §11.1: BoT-SORT runs without ReID as the primary comparison.
        assert TrackingSettings().with_reid is False

    def test_thresholds_are_declared_not_inherited(self) -> None:
        settings = TrackingSettings()
        assert settings.track_high_thresh == 0.25
        assert settings.track_low_thresh == 0.10
        assert settings.new_track_thresh == 0.25
        assert settings.match_thresh == 0.8
        assert settings.fuse_score is True


class TestTrackingValidation:
    def test_tracker_must_be_one_of_the_implemented_adapters(self) -> None:
        # DeepSORT is a literature-review item in §11.1 with implementation
        # deferred, so it must not be selectable.
        with pytest.raises(ValidationError, match="bytetrack"):
            TrackingSettings(tracker="deepsort")  # type: ignore[arg-type]

    def test_loss_tolerance_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TrackingSettings(loss_tolerance_s=0.0)

    def test_staleness_must_not_exceed_the_loss_tolerance(self) -> None:
        # Staleness says "stop trusting the last observation"; the tolerance ends
        # the track. A longer stale window would be meaningless.
        with pytest.raises(ValidationError, match="stale_after_s"):
            TrackingSettings(loss_tolerance_s=1.0, stale_after_s=2.0)

    def test_staleness_may_equal_the_tolerance(self) -> None:
        settings = TrackingSettings(loss_tolerance_s=1.0, stale_after_s=1.0)
        assert settings.stale_after_s == 1.0

    def test_low_threshold_must_be_below_high_threshold(self) -> None:
        with pytest.raises(ValidationError, match="track_low_thresh"):
            TrackingSettings(track_low_thresh=0.9, track_high_thresh=0.5)

    def test_new_track_threshold_must_not_be_below_the_low_threshold(self) -> None:
        with pytest.raises(ValidationError, match="new_track_thresh"):
            TrackingSettings(track_low_thresh=0.2, new_track_thresh=0.1)

    def test_reid_is_rejected_rather_than_ignored(self) -> None:
        with pytest.raises(ValidationError, match="with_reid"):
            TrackingSettings(with_reid=True)

    def test_thresholds_are_bounded(self) -> None:
        with pytest.raises(ValidationError):
            TrackingSettings(track_high_thresh=1.5)
        with pytest.raises(ValidationError):
            TrackingSettings(match_thresh=0.0)

    def test_assumed_rate_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            TrackingSettings(assumed_update_rate_hz=0.0)
        with pytest.raises(ValidationError):
            TrackingSettings(assumed_update_rate_hz=1000.0)

    def test_buffer_ceiling_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            TrackingSettings(max_buffer_frames=0)


class TestTrackingEnvironment:
    def test_nested_tracking_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FALL_TRACKING__TRACKER", "botsort")
        monkeypatch.setenv("FALL_TRACKING__LOSS_TOLERANCE_S", "3.5")
        monkeypatch.setenv("FALL_TRACKING__ASSUMED_UPDATE_RATE_HZ", "5")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.tracking.tracker == "botsort"
        assert settings.tracking.loss_tolerance_s == 3.5
        # The derived frame buffer follows the environment, not a constant.
        assert settings.tracking.buffer_frames == 18

    def test_tracking_can_be_disabled_by_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FALL_TRACKING__ENABLED", "false")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.tracking.enabled is False

    def test_tracking_group_is_part_of_settings(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert isinstance(settings.tracking, TrackingSettings)

    def test_tracking_and_detection_groups_are_independent(self) -> None:
        # Detection stays off by default (loading a model is opt-in) while tracking
        # is on, so the two switches must not be coupled.
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.detection.enabled is False
        assert settings.tracking.enabled is True
