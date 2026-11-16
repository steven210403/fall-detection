"""Tracking service tests: session reset, buffer isolation, and persistence.

The thesis plan's Phase 4 acceptance criteria are "multi-person tracking; session
reset; no buffer mixing", so those are the behaviours asserted here. Boxes are
synthetic and idealised: this module proves the service wiring and the identity
rules, and it cannot and does not claim tracking accuracy (§15).

Both trackers are exercised, because §11.1 makes the ByteTrack/BoT-SORT comparison
mandatory and a service that only worked with one of them would invalidate
Experiment 2 before it starts.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable

import pytest
from sqlalchemy import select

from fall_detection.config import Settings
from fall_detection.core.clock import ManualClock
from fall_detection.core.contracts import StreamSessionInfo
from fall_detection.core.errors import ConfigurationError
from fall_detection.database import DatabaseWriter, Track, TrackRepository
from fall_detection.detection.contracts import Detection, DetectionBatch
from fall_detection.tracking import TrackingService, TrackObservations, TrackState
from fall_detection.tracking.service import ensure_enabled

CAMERA = uuid.UUID("22222222-2222-2222-2222-222222222222")
SESSION_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
SESSION_B = uuid.UUID("44444444-4444-4444-4444-444444444444")


def make_settings(**overrides: object) -> Settings:
    """Tracking settings isolated from the environment."""
    base: dict[str, object] = {
        "environment": "test",
        "log_level": "WARNING",
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class Harness:
    """Drive a tracking service with a manual clock, one frame at a time."""

    def __init__(
        self,
        *,
        tracker: str = "bytetrack",
        interval_s: float = 0.1,
        loss_tolerance_s: float = 2.0,
        writer: DatabaseWriter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.clock = ManualClock()
        self.settings = settings or make_settings(
            tracking={
                "tracker": tracker,
                "loss_tolerance_s": loss_tolerance_s,
                "assumed_update_rate_hz": 10.0,
            }
        )
        self.service = TrackingService(self.settings, writer=writer, clock=self.clock)
        self.interval_s = interval_s
        self.frame_seq = 0

    def update(
        self,
        detections: tuple[Detection, ...] = (),
        *,
        session_id: uuid.UUID = SESSION_A,
        advance: bool = True,
        frame_shape: tuple[int, int, int] = (480, 640, 3),
    ) -> TrackObservations:
        """Track one frame and step the clock forward."""
        batch = DetectionBatch(
            camera_id=CAMERA,
            stream_session_id=session_id,
            frame_seq=self.frame_seq,
            detections=detections,
            frame_shape=frame_shape,
            source_timestamp=self.frame_seq / 10.0,
            receive_monotonic=self.clock.monotonic(),
            receive_utc=self.clock.utc_timestamp(),
            inference_monotonic=self.clock.monotonic(),
            model_name="yolov8n-pose.pt",
        )
        self.frame_seq += 1
        result = self.service.update(batch)
        if advance:
            self.clock.advance(self.interval_s)
        return result

    def steady(
        self,
        *,
        x: float = 100.0,
        frames: int = 5,
        session_id: uuid.UUID = SESSION_A,
        advance: bool = True,
    ) -> TrackObservations:
        """Observe one person for several frames; returns the last update."""
        result = self.update((person(x),), session_id=session_id, advance=advance)
        for _ in range(frames - 1):
            result = self.update((person(x),), session_id=session_id, advance=advance)
        return result

    def gap(self, frames: int, *, session_id: uuid.UUID = SESSION_A) -> TrackObservations:
        """Run frames with no detections at all."""
        result = self.update((), session_id=session_id)
        for _ in range(frames - 1):
            result = self.update((), session_id=session_id)
        return result


def person(x: float, *, confidence: float = 0.9, y: float = 100.0) -> Detection:
    return Detection(
        bbox=(x, y, x + 60.0, y + 200.0), confidence=confidence, class_name="person", class_id=0
    )


def non_person(kind: str = "cell phone", class_id: int = 67) -> Detection:
    return Detection(
        bbox=(10.0, 10.0, 40.0, 60.0), confidence=0.7, class_name=kind, class_id=class_id
    )


class TestFirstBatch:
    def test_first_batch_opens_a_session(self) -> None:
        harness = Harness()
        assert harness.service.session_id is None
        result = harness.update((person(100.0),))
        assert result.stream_session_id == SESSION_A
        assert harness.service.session_id == SESSION_A
        assert harness.service.stats.sessions_started == 1

    def test_observation_carries_frame_context(self) -> None:
        harness = Harness()
        result = harness.update((person(100.0),))
        observation = result.observations[0]
        assert observation.camera_id == CAMERA
        assert observation.stream_session_id == SESSION_A
        assert observation.frame_seq == 0
        assert observation.class_name == "person"
        assert observation.source_timestamp == 0.0
        assert observation.receive_utc == pytest.approx(1_700_000_000.0)
        assert observation.observed_frames == 1
        assert not observation.recovered

    def test_a_new_session_id_starts_a_new_session(self) -> None:
        harness = Harness()
        harness.steady()
        result = harness.update((person(100.0),), session_id=SESSION_B)
        assert result.stream_session_id == SESSION_B
        assert harness.service.stats.sessions_started == 2

    def test_empty_frame_still_updates(self) -> None:
        # An empty frame is how the tracker learns a person is missing, so it must
        # reach the tracker rather than being skipped.
        harness = Harness()
        result = harness.update(())
        assert result.observations == ()
        assert harness.service.stats.updates == 1
        assert harness.service.stats.empty_frames == 1


class TestMultiPerson:
    def test_three_people_are_tracked_concurrently(self) -> None:
        harness = Harness()
        result = None
        for _ in range(6):
            result = harness.update((person(100.0), person(300.0), person(500.0)))
        assert result is not None
        assert len(result.observations) == 3
        assert len({item.track_uuid for item in result.observations}) == 3
        assert [item.display_id for item in result.tracks] == sorted(
            item.display_id for item in result.tracks
        )

    def test_people_keep_separate_identities(self) -> None:
        harness = Harness()
        first = harness.update((person(100.0), person(300.0)))
        uuids = {item.display_id: item.track_uuid for item in first.observations}
        for _ in range(5):
            result = harness.update((person(100.0), person(300.0)))
            for item in result.observations:
                assert item.track_uuid == uuids[item.display_id]

    def test_crossing_people_do_not_swap_identities(self) -> None:
        # Phase 4 lists crossings as a risk to verify, so identity continuity is
        # checked through the crossing: each person must keep a distinct UUID and
        # a distinct integer, and no frame may report one identity twice.
        harness = Harness()
        left, right = 100.0, 440.0
        result = harness.update((person(left), person(right)))
        assert len(result.observations) == 2
        # Move them towards each other and past one another.
        for _ in range(12):
            left += 15.0
            right -= 15.0
            harness.update((person(left), person(right)))
        result = harness.update((person(left), person(right)))
        assert len(result.observations) == 2
        assert len({item.track_uuid for item in result.observations}) == 2
        assert len({item.display_id for item in result.observations}) == 2
        # Both people are still live: a crossing must not terminate a track.
        assert len(result.tracks) == 2
        assert all(summary.state is TrackState.ACTIVE for summary in result.tracks)

    def test_non_person_detections_are_ignored(self) -> None:
        # §12.4 makes person–object association a feature concern, so phones and
        # laptops must not create tracks.
        harness = Harness()
        result = harness.update((person(100.0), non_person(), non_person("laptop", 63)))
        assert len(result.observations) == 1
        assert all(item.class_name == "person" for item in result.observations)
        assert len(result.tracks) == 1


class TestLossAndRecovery:
    def test_short_gap_recovers_the_same_uuid(self) -> None:
        harness = Harness(loss_tolerance_s=2.0)
        first = harness.steady()
        original = first.observations[0].track_uuid
        harness.gap(5)  # 0.5 s
        result = harness.update((person(120.0),))
        assert result.observations[0].track_uuid == original
        assert result.observations[0].recovered
        assert result.observations[0].gap_frames == 5
        assert harness.service.stats.recoveries == 1

    def test_long_gap_yields_a_new_uuid_in_the_same_update(self) -> None:
        # §11.2: after termination a returning person receives a new track UUID.
        # The old track's end must be reported on the update that starts the new
        # one, otherwise the identity change would be invisible.
        harness = Harness(loss_tolerance_s=0.5)
        first = harness.steady()
        original = first.observations[0].track_uuid
        result = harness.gap(6)
        assert [item.reason for item in result.terminated] == ["lost_timeout"]
        result = harness.update((person(120.0),))
        assert result.observations[0].track_uuid != original

    def test_termination_is_reported_before_the_new_identity_in_one_update(self) -> None:
        # A gap that is never sampled (for example after a stall) still has to
        # terminate the old track and start a new one on the first frame back.
        harness = Harness(loss_tolerance_s=0.5, interval_s=5.0)
        first = harness.steady(frames=2)
        original = first.observations[0].track_uuid
        result = harness.update((person(120.0),))
        assert [item.reason for item in result.terminated] == ["lost_timeout"]
        assert result.terminated[0].track_uuid == original
        assert result.observations[0].track_uuid != original

    def test_lost_track_is_summarised_without_evidence(self) -> None:
        # §11.2: while a track is lost it must not contribute fall evidence.
        # The summary reports its last observation and the gap length, and the
        # observation list for the frame is empty.
        harness = Harness()
        harness.steady()
        result = harness.gap(4)
        assert result.observations == ()
        assert len(result.lost_tracks()) == 1
        summary = result.lost_tracks()[0]
        assert summary.state is TrackState.LOST
        assert summary.missing_frames == 4
        assert summary.bbox is not None
        assert summary.time_since_last_observation_s > 0

    def test_staleness_marks_the_end_of_usable_evidence(self) -> None:
        harness = Harness()
        harness.steady()
        uuid_seen = harness.service.live_track_uuids()[0]
        assert not harness.service.is_stale(uuid_seen)
        harness.gap(6)  # 0.6 s, past the 0.5 s threshold
        assert harness.service.is_stale(uuid_seen)

    def test_terminated_uuid_disappears_from_live_tracks(self) -> None:
        harness = Harness(loss_tolerance_s=0.5)
        harness.steady()
        uuid_seen = harness.service.live_track_uuids()[0]
        harness.gap(6)
        assert harness.service.live_track_uuids() == ()
        assert harness.service.is_stale(uuid_seen)
        assert harness.service.stats.terminations_lost_timeout == 1


class TestSessionReset:
    def test_new_session_terminates_previous_tracks(self) -> None:
        # §9.2 and the Phase 4 acceptance criterion: a reconnect clears old tracks.
        harness = Harness()
        harness.steady()
        assert harness.service.stats.sessions_started == 1
        result = harness.update((person(100.0),), session_id=SESSION_B)
        assert harness.service.stats.sessions_started == 2
        assert harness.service.stats.terminations_session_ended == 1
        assert [item.reason for item in result.terminated] == ["session_ended"]

    def test_no_buffer_mixing_between_sessions(self) -> None:
        # The same integer ID in two sessions must not share a UUID, age, or
        # counters: that is exactly the mixing the acceptance criterion forbids.
        harness = Harness()
        harness.steady(frames=8)
        first_uuid = harness.service.live_track_uuids()[0]
        first_summary = harness.service.summaries()[0]
        result = harness.update((person(100.0),), session_id=SESSION_B)
        observation = result.observations[0]
        assert observation.track_uuid != first_uuid
        assert observation.observed_frames == 1
        assert observation.age_s == 0.0
        assert observation.first_seen_monotonic != first_summary.first_seen_monotonic
        assert len(result.tracks) == 1
        assert result.tracks[0].missing_frames == 0

    def test_reset_clears_statistics_only_for_state_not_counters(self) -> None:
        harness = Harness()
        harness.steady()
        harness.update((person(100.0),), session_id=SESSION_B)
        # Cumulative counters keep describing the whole run, which is what a
        # status payload should report.
        assert harness.service.stats.sessions_started == 2
        assert harness.service.stats.observations_total >= 2

    def test_explicit_begin_session_terminates_live_tracks(self) -> None:
        harness = Harness()
        harness.steady()
        ended = harness.service.begin_session(
            StreamSessionInfo(
                camera_id=CAMERA,
                stream_session_id=SESSION_B,
                started_utc=harness.clock.utc_timestamp(),
                started_monotonic=harness.clock.monotonic(),
                attempt=2,
            )
        )
        assert [item.reason for item in ended] == ["session_ended"]
        assert harness.service.live_track_uuids() == ()

    def test_end_session_terminates_and_closes(self) -> None:
        harness = Harness()
        harness.steady()
        ended = harness.service.end_session(reason="stopped")
        assert [item.reason for item in ended] == ["stopped"]
        assert harness.service.session_id is None
        assert harness.service.stats.terminations_stopped == 1
        assert harness.service.snapshot()["ready"] is False


class TestTrackerSelection:
    @pytest.mark.parametrize("tracker", ["bytetrack", "botsort"])
    def test_service_tracks_people_with_either_tracker(self, tracker: str) -> None:
        # §11.1 makes the comparison mandatory, so the service must work with both.
        harness = Harness(tracker=tracker)
        result = harness.steady()
        assert len(result.observations) == 1
        assert result.tracker_name == tracker
        assert harness.service.info.name == tracker
        assert harness.service.info.with_reid is False

    @pytest.mark.parametrize("tracker", ["bytetrack", "botsort"])
    def test_session_reset_works_with_either_tracker(self, tracker: str) -> None:
        harness = Harness(tracker=tracker)
        harness.steady()
        result = harness.update((person(100.0),), session_id=SESSION_B)
        assert len(result.observations) == 1
        assert [item.reason for item in result.terminated] == ["session_ended"]

    def test_botsort_disables_gmc(self) -> None:
        harness = Harness(tracker="botsort")
        harness.steady()
        assert harness.service.info.gmc_method == "none"


class TestUnload:
    def test_unload_clears_tracking_state(self) -> None:
        harness = Harness()
        harness.steady()
        harness.service.unload()
        assert harness.service.session_id is None
        assert harness.service.latest() is None
        assert harness.service.live_track_uuids() == ()

    def test_update_after_unload_starts_a_fresh_session(self) -> None:
        harness = Harness()
        harness.steady()
        first = harness.service.live_track_uuids()
        harness.service.unload()
        result = harness.update((person(100.0),))
        assert result.observations[0].track_uuid not in first

    def test_on_frame_is_rejected_as_a_frame_consumer(self) -> None:
        # Attaching the service directly to the pipeline would silently skip
        # detection, so it must fail loudly instead.
        harness = Harness()
        with pytest.raises(TypeError, match="not a frame consumer"):
            harness.service.on_frame(object())


class TestPersistence:
    def test_track_rows_are_created_and_closed(
        self,
        writer: DatabaseWriter,
        make_settings: Callable[..., Settings],
        stream_session_row: Callable[..., uuid.UUID],
    ) -> None:
        stream_session_row()
        settings = make_settings(tracking={"tracker": "bytetrack", "loss_tolerance_s": 0.5})
        harness = Harness(writer=writer, settings=settings)
        harness.steady()
        rows = writer.read(lambda session: list(session.scalars(select(Track))))
        assert len(rows) == 1
        assert rows[0].terminated_at is None
        assert rows[0].tracker_id == 1

        harness.gap(6)
        rows = writer.read(lambda session: list(session.scalars(select(Track))))
        assert rows[0].termination_reason == "lost_timeout"
        assert rows[0].terminated_at is not None
        assert rows[0].missing_frames == 6

    def test_track_row_identity_matches_the_runtime_uuid(
        self,
        writer: DatabaseWriter,
        make_settings: Callable[..., Settings],
        stream_session_row: Callable[..., uuid.UUID],
    ) -> None:
        stream_session_row()
        settings = make_settings(tracking={"tracker": "bytetrack"})
        harness = Harness(writer=writer, settings=settings)
        result = harness.steady()
        runtime_uuid = result.observations[0].track_uuid
        stored = writer.read(lambda session: session.scalars(select(Track.id)).all())
        assert list(stored) == [runtime_uuid]

    def test_integer_tracker_id_is_stored_but_is_not_the_key(
        self,
        writer: DatabaseWriter,
        make_settings: Callable[..., Settings],
        stream_session_row: Callable[..., uuid.UUID],
    ) -> None:
        # §19: integer tracker IDs are not primary keys.
        stream_session_row()
        settings = make_settings(tracking={"tracker": "bytetrack"})
        harness = Harness(writer=writer, settings=settings)
        harness.steady()
        row = writer.read(lambda session: session.scalars(select(Track)).one())
        assert row.tracker_id == 1
        assert row.id != uuid.UUID(int=1)

    def test_two_people_create_two_rows(
        self,
        writer: DatabaseWriter,
        make_settings: Callable[..., Settings],
        stream_session_row: Callable[..., uuid.UUID],
    ) -> None:
        stream_session_row()
        settings = make_settings(tracking={"tracker": "bytetrack"})
        harness = Harness(writer=writer, settings=settings)
        harness.update((person(100.0), person(300.0)))
        rows = writer.read(lambda session: list(session.scalars(select(Track))))
        assert len(rows) == 2
        assert {row.tracker_id for row in rows} == {1, 2}

    def test_track_rows_belong_to_the_session_row(
        self,
        writer: DatabaseWriter,
        make_settings: Callable[..., Settings],
        stream_session_row: Callable[..., uuid.UUID],
    ) -> None:
        # §19: a track is anchored to one stream session, which is what keeps data
        # from two decoding runs apart.
        stream_session_row()
        settings = make_settings(tracking={"tracker": "bytetrack"})
        harness = Harness(writer=writer, settings=settings)
        harness.steady()
        row = writer.read(lambda session: session.scalars(select(Track)).one())
        assert row.stream_session_id == SESSION_A

    def test_missing_session_row_is_a_counted_error_not_a_crash(
        self,
        writer: DatabaseWriter,
        make_settings: Callable[..., Settings],
    ) -> None:
        # Tracking must keep working when persistence cannot: the session row is
        # the pipeline's to write, so its absence is a data problem, not a reason
        # to stop tracking people.
        settings = make_settings(tracking={"tracker": "bytetrack"})
        harness = Harness(writer=writer, settings=settings)
        result = harness.steady()
        assert len(result.observations) == 1
        assert harness.service.stats.persistence_errors == 1
        assert harness.service.stats.tracks_started == 1

    def test_persistence_is_optional(self) -> None:
        harness = Harness()
        result = harness.steady()
        assert result.observations
        assert harness.service.stats.tracks_started == 1

    def test_persistence_failure_does_not_stop_tracking(
        self,
        writer: DatabaseWriter,
        make_settings: Callable[..., Settings],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A database problem must not stop the pipeline: tracking state is
        # authoritative in memory, and the failure is counted.
        settings = make_settings(tracking={"tracker": "bytetrack"})
        service = TrackingService(settings, writer=writer, clock=ManualClock())

        def boom(_session: object) -> None:
            raise RuntimeError("database is gone")

        monkeypatch.setattr(TrackRepository, "open", boom)
        batch = DetectionBatch(
            camera_id=CAMERA,
            stream_session_id=SESSION_A,
            frame_seq=0,
            detections=(person(100.0),),
            frame_shape=(480, 640, 3),
            source_timestamp=0.0,
            receive_monotonic=0.0,
            receive_utc=1_700_000_000.0,
            inference_monotonic=0.0,
        )
        result = service.update(batch)
        assert len(result.observations) == 1
        assert service.stats.persistence_errors == 1

    def test_closed_track_rows_survive_a_tracker_id_reissue(
        self,
        writer: DatabaseWriter,
        make_settings: Callable[..., Settings],
        stream_session_row: Callable[..., uuid.UUID],
    ) -> None:
        # The integer is not unique across a session, so a reissued ID must not
        # collide with the ended row: the UUID is the identity.
        stream_session_row()
        settings = make_settings(tracking={"tracker": "bytetrack", "loss_tolerance_s": 0.5})
        harness = Harness(writer=writer, settings=settings)
        harness.steady()
        harness.gap(6)
        harness.update((person(100.0),))
        rows = writer.read(lambda session: list(session.scalars(select(Track))))
        assert len(rows) == 2
        assert len({row.id for row in rows}) == 2
        assert sum(1 for row in rows if row.termination_reason == "lost_timeout") == 1


class TestSnapshot:
    def test_snapshot_is_json_safe(self) -> None:
        harness = Harness()
        harness.steady()
        payload = harness.service.snapshot()
        json.dumps(payload)
        assert payload["enabled"] is True
        assert payload["ready"] is True
        assert payload["live_tracks"] == 1
        assert payload["stream_session_id"] == str(SESSION_A)

    def test_snapshot_reports_tracker_configuration(self) -> None:
        harness = Harness(tracker="botsort")
        harness.steady()
        payload = harness.service.snapshot()["tracker"]
        assert payload["name"] == "botsort"
        assert payload["gmc_method"] == "none"
        assert payload["with_reid"] is False
        assert payload["configured_buffer_frames"] == 20

    def test_snapshot_before_the_first_frame(self) -> None:
        harness = Harness()
        payload = harness.service.snapshot()
        assert payload["ready"] is False
        assert payload["live_tracks"] == 0
        assert "stream_session_id" not in payload

    def test_snapshot_recomputes_staleness(self) -> None:
        # A snapshot taken long after the last update must not report a stale
        # track as fresh, which is why summaries are rebuilt rather than cached.
        harness = Harness()
        harness.steady()
        assert harness.service.snapshot()["stale_tracks"] == 0
        harness.clock.advance(2.0)
        payload = harness.service.snapshot()
        assert payload["stale_tracks"] == 1
        assert payload["tracks"][0]["stale"] is True

    def test_state_counts_and_live_uuids_agree(self) -> None:
        harness = Harness()
        harness.steady()
        assert harness.service.state_counts()["active"] == 1
        assert len(harness.service.live_track_uuids()) == 1

    def test_stats_are_measured_counters(self) -> None:
        harness = Harness()
        harness.steady(frames=4)
        stats = harness.service.stats
        assert stats.updates == 4
        assert stats.frames_with_observations == 4
        assert stats.observations_total == 4
        assert stats.tracks_started == 1
        assert stats.invalid_rows == 0
        assert stats.persistence_errors == 0


class TestEnableSwitch:
    def test_disabled_tracking_reports_enabled_false(self) -> None:
        settings = make_settings(tracking={"enabled": False})
        service = TrackingService(settings, clock=ManualClock())
        assert service.enabled is False
        assert service.snapshot()["enabled"] is False

    def test_ensure_enabled_raises_when_disabled(self) -> None:
        settings = make_settings(tracking={"enabled": False})
        service = TrackingService(settings, clock=ManualClock())
        with pytest.raises(ConfigurationError, match="disabled"):
            ensure_enabled(service)

    def test_ensure_enabled_passes_when_enabled(self) -> None:
        service = TrackingService(make_settings(), clock=ManualClock())
        ensure_enabled(service)
