"""Track lifecycle tests: identity, loss tolerance, staleness, and termination.

Thesis plan §11.2 makes these rules time-based — a two-second loss tolerance and a
0.5-second staleness threshold — so they are driven here by a
:class:`~fall_detection.core.clock.ManualClock`. Asserting an exact boundary is only
possible if time is controlled: sleeping for two seconds would test the machine,
not the rule.

No tracker is involved, so these tests are deterministic and make no claim about
tracking accuracy.
"""

from __future__ import annotations

import uuid

import pytest

from fall_detection.core.clock import ManualClock
from fall_detection.core.contracts import StreamSessionInfo
from fall_detection.tracking import TrackLifecycle, TrackState

CAMERA = uuid.UUID("22222222-2222-2222-2222-222222222222")
SESSION_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
SESSION_B = uuid.UUID("44444444-4444-4444-4444-444444444444")

BOX = (100.0, 100.0, 160.0, 300.0)
"""A fixed box: identity continuity is what is under test, not geometry."""


def make_session(session_id: uuid.UUID = SESSION_A, *, started: float = 0.0) -> StreamSessionInfo:
    return StreamSessionInfo(
        camera_id=CAMERA,
        stream_session_id=session_id,
        started_utc=1_700_000_000.0 + started,
        started_monotonic=started,
        attempt=1,
    )


class Harness:
    """Drive a lifecycle with a manual clock, one frame at a time."""

    def __init__(
        self,
        *,
        loss_tolerance_s: float = 2.0,
        stale_after_s: float = 0.5,
        interval_s: float = 0.1,
        session_id: uuid.UUID = SESSION_A,
    ) -> None:
        self.clock = ManualClock()
        self.lifecycle = TrackLifecycle(
            loss_tolerance_s=loss_tolerance_s,
            stale_after_s=stale_after_s,
            lock=None,
        )
        self.interval_s = interval_s
        self.frame_seq = 0
        self.superseded: list = []
        self.lifecycle.begin_session(make_session(session_id))

    def observe(self, display_id: int, *, x: float = 100.0, advance: bool = True):
        """Observe one person in this frame and step the clock forward.

        A superseded track (the observation arrived after the loss tolerance
        elapsed) is collected in :attr:`superseded`, so a test can read all ends
        from one place regardless of which path produced them.
        """
        result = self.lifecycle.observe(
            display_id=display_id,
            bbox=(x, BOX[1], x + 60.0, BOX[3]),
            confidence=0.9,
            frame_seq=self.frame_seq,
            now_monotonic=self.clock.monotonic(),
            now_utc=self.clock.utc_timestamp(),
        )
        self.frame_seq += 1
        if advance:
            self.clock.advance(self.interval_s)
        if result.superseded is not None:
            self.superseded.append(result.superseded)
        return result.match

    def missing(self, *, observed: set[int] | None = None, advance: bool = True):
        """Record a frame with no observation for the given tracks."""
        ended = self.lifecycle.mark_missing(observed or set(), now_monotonic=self.clock.monotonic())
        self.frame_seq += 1
        if advance:
            self.clock.advance(self.interval_s)
        return ended

    def gap(self, frames: int, *, observed: set[int] | None = None) -> list:
        """Run ``frames`` frames without an observation, collecting terminations."""
        ended: list = []
        for _ in range(frames):
            ended.extend(self.missing(observed=observed))
        return ended

    def summaries(self):
        return self.lifecycle.summaries(now_monotonic=self.clock.monotonic())


class TestIdentity:
    def test_first_observation_mints_a_uuid(self) -> None:
        harness = Harness()
        match = harness.observe(1)
        assert isinstance(match.track_uuid, uuid.UUID)
        assert match.display_id == 1
        assert match.observed_frames == 1
        assert match.gap_frames == 0
        assert not match.recovered

    def test_continuous_observation_keeps_one_uuid(self) -> None:
        harness = Harness()
        first = harness.observe(1)
        last = first
        for _ in range(20):
            last = harness.observe(1, x=100.0)
            assert last.track_uuid == first.track_uuid
        assert harness.lifecycle.live_count() == 1
        assert last.observed_frames == 21

    def test_distinct_display_ids_get_distinct_uuids(self) -> None:
        harness = Harness()
        first = harness.observe(1)
        second = harness.observe(2, x=400.0)
        assert first.track_uuid != second.track_uuid
        assert harness.lifecycle.live_count() == 2

    def test_age_grows_with_time_not_frame_count(self) -> None:
        # Age is measured on the clock, not by counting frames, so a slower
        # update rate must not make a track look younger than it is (§9.1). The
        # tolerance is raised so this test isolates age from termination.
        harness = Harness(interval_s=0.5, loss_tolerance_s=30.0)
        first = harness.observe(1)
        harness.clock.advance(3.0)
        match = harness.observe(1)
        assert match.track_uuid == first.track_uuid
        assert match.age_s == pytest.approx(3.5)

    def test_observe_without_a_session_raises(self) -> None:
        lifecycle = TrackLifecycle(loss_tolerance_s=2.0, stale_after_s=0.5)
        with pytest.raises(RuntimeError, match="no stream session is active"):
            lifecycle.observe(
                display_id=1,
                bbox=BOX,
                confidence=0.9,
                frame_seq=0,
                now_monotonic=0.0,
                now_utc=0.0,
            )

    def test_invalid_thresholds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="loss_tolerance_s"):
            TrackLifecycle(loss_tolerance_s=0.0, stale_after_s=0.5)
        with pytest.raises(ValueError, match="stale_after_s"):
            TrackLifecycle(loss_tolerance_s=2.0, stale_after_s=0.0)


class TestLossAndRecovery:
    def test_lost_state_and_missing_frames_are_tracked(self) -> None:
        harness = Harness()
        harness.observe(1)
        harness.gap(3)
        summary = harness.summaries()[0]
        assert summary.state is TrackState.LOST
        assert summary.missing_frames == 3
        assert summary.observed_frames == 1
        # The observation was made at t=0 and the clock then advanced one
        # interval, so after three further frames the gap is four intervals.
        assert summary.time_since_last_observation_s == pytest.approx(0.4)

    def test_recovery_inside_the_tolerance_keeps_the_uuid(self) -> None:
        harness = Harness()
        first = harness.observe(1)
        harness.gap(5)  # 0.5 s, well inside the 2 s tolerance
        match = harness.observe(1)
        assert match.track_uuid == first.track_uuid
        assert match.recovered
        assert match.gap_frames == 5
        assert harness.summaries()[0].state is TrackState.ACTIVE
        assert harness.summaries()[0].missing_frames == 0

    def test_recovery_at_the_exact_boundary_still_recovers(self) -> None:
        # §11.2 allows recovery "before termination", and termination is decided
        # by `gap > tolerance`, so a gap of exactly the tolerance survives. The
        # observation is at t=0 and each frame advances one interval, so nine
        # frames after it reaches 1.0 s.
        harness = Harness(loss_tolerance_s=1.0, interval_s=0.1)
        first = harness.observe(1)
        harness.gap(9)  # exactly 1.0 s
        assert harness.clock.monotonic() == pytest.approx(1.0)
        match = harness.observe(1)
        assert match.track_uuid == first.track_uuid
        assert match.recovered

    def test_recovery_after_the_tolerance_creates_a_new_track(self) -> None:
        harness = Harness(loss_tolerance_s=1.0, interval_s=0.1)
        first = harness.observe(1)
        ended = harness.gap(11)  # 1.1 s, past the tolerance
        assert [item.reason for item in ended] == ["lost_timeout"]
        match = harness.observe(1)
        assert match.track_uuid != first.track_uuid
        assert not match.recovered
        assert match.observed_frames == 1

    def test_observed_track_is_never_aged_away(self) -> None:
        harness = Harness()
        first = harness.observe(1)
        for _ in range(50):
            harness.missing(observed={1})
            match = harness.observe(1)
            assert match.track_uuid == first.track_uuid
        assert harness.summaries()[0].missing_frames == 0

    def test_one_lost_track_does_not_affect_another(self) -> None:
        harness = Harness()
        first = harness.observe(1)
        second = harness.observe(2, x=400.0)
        # Person 1 is gone; person 2 keeps being observed.
        ended = harness.gap(5, observed={2})
        assert ended == []
        # The gap() frames observed only person 2, so person 1 was aged.
        harness.observe(2, x=400.0)
        summary_one = next(s for s in harness.summaries() if s.track_uuid == first.track_uuid)
        summary_two = next(s for s in harness.summaries() if s.track_uuid == second.track_uuid)
        assert summary_one.state is TrackState.LOST
        assert summary_two.state is TrackState.ACTIVE
        assert summary_two.missing_frames == 0


class TestTermination:
    def test_termination_reports_last_observation_time(self) -> None:
        # FR-07: stored times come from observation time, so the grace period
        # must not appear as if the person had been visible during it.
        harness = Harness(loss_tolerance_s=1.0, interval_s=0.1)
        harness.observe(1)
        last_seen = harness.clock.monotonic() - harness.interval_s
        ended = harness.gap(11)
        assert len(ended) == 1
        assert ended[0].ended_monotonic == pytest.approx(last_seen)
        # Every frame of the gap is counted, including the one that exhausted the
        # tolerance, so the reported gap is never understated.
        assert ended[0].missing_frames == 11

    def test_termination_consumes_the_identity(self) -> None:
        harness = Harness(loss_tolerance_s=0.5, interval_s=0.1)
        first = harness.observe(1)
        harness.gap(6)
        assert first.track_uuid not in {s.track_uuid for s in harness.summaries()}
        assert harness.lifecycle.live_count() == 0

    def test_retired_display_id_is_never_resurrected(self) -> None:
        # §11.2: after termination a returning person receives a new UUID, and the
        # integer the tracker may reissue must not bring the old identity back.
        harness = Harness(loss_tolerance_s=0.5, interval_s=0.1)
        first = harness.observe(1)
        harness.gap(6)
        assert harness.lifecycle.retired_display_ids() == (1,)
        again = harness.observe(1)
        assert again.track_uuid != first.track_uuid
        # The reissued integer is live again, so it is no longer retired.
        assert harness.lifecycle.retired_display_ids() == ()

    def test_termination_of_several_tracks_is_ordered_by_display_id(self) -> None:
        harness = Harness(loss_tolerance_s=0.5, interval_s=0.1)
        # Both tracks are observed in the same frame, so both are terminated by
        # the same update and the ordering within that update is observable.
        harness.observe(3, x=400.0, advance=False)
        harness.observe(1)
        ended = harness.gap(6)
        # Ordered by display ID, not by insertion order, so a payload is stable.
        assert [item.display_id for item in ended] == [1, 3]

    def test_tracks_that_end_in_different_updates_are_reported_separately(self) -> None:
        # A track whose last observation is older reaches the tolerance first, so
        # the two ends must not be merged into one update.
        harness = Harness(loss_tolerance_s=0.5, interval_s=0.1)
        harness.observe(3, x=400.0)
        harness.observe(1)
        ended = harness.gap(6)
        assert [item.display_id for item in ended] == [3, 1]

    def test_duration_spans_first_to_last_observation(self) -> None:
        harness = Harness(loss_tolerance_s=0.5, interval_s=0.1)
        harness.observe(1)
        for _ in range(4):
            harness.observe(1)
        ended = harness.gap(6)
        assert len(ended) == 1
        assert ended[0].duration_s == pytest.approx(0.4)
        assert ended[0].observed_frames == 5


class TestStaleness:
    def test_stale_after_the_threshold(self) -> None:
        # §11.2: old predictions become stale after 0.5 seconds. The observation
        # is made at t=0 and the harness advances one interval per frame, so the
        # gap equals the number of frames times the interval.
        harness = Harness(stale_after_s=0.5, interval_s=0.1)
        harness.observe(1)
        assert not harness.summaries()[0].stale
        harness.gap(4)  # 0.5 s exactly: not yet stale
        assert harness.summaries()[0].time_since_last_observation_s == pytest.approx(0.5)
        assert not harness.summaries()[0].stale
        harness.gap(1)  # 0.6 s
        assert harness.summaries()[0].stale

    def test_is_stale_matches_the_summary(self) -> None:
        harness = Harness(stale_after_s=0.5, interval_s=0.1)
        match = harness.observe(1)
        harness.gap(6)
        now = harness.clock.monotonic()
        assert harness.lifecycle.is_stale(match.track_uuid, now_monotonic=now) is True

    def test_unknown_track_is_stale(self) -> None:
        # Nothing usable exists for an unknown or terminated track, so reporting
        # "stale" is the safe answer rather than "fresh".
        harness = Harness()
        assert harness.lifecycle.is_stale(uuid.uuid4(), now_monotonic=0.0) is True

    def test_a_fresh_observation_clears_staleness(self) -> None:
        harness = Harness(stale_after_s=0.5, interval_s=0.1)
        harness.observe(1)
        harness.gap(8)
        assert harness.summaries()[0].stale
        harness.observe(1)
        assert not harness.summaries()[0].stale


class TestSessions:
    def test_new_session_terminates_live_tracks(self) -> None:
        # §9.2: a reconnect creates a new stream session and clears old tracks and
        # buffers. Every live track must therefore end as `session_ended`.
        harness = Harness()
        harness.observe(1)
        harness.observe(2, x=400.0)
        ended = harness.lifecycle.begin_session(make_session(SESSION_B, started=10.0))
        assert [item.reason for item in ended] == ["session_ended", "session_ended"]
        assert harness.lifecycle.live_count() == 0

    def test_new_session_resets_identity_and_counters(self) -> None:
        harness = Harness()
        first = harness.observe(1)
        harness.lifecycle.begin_session(make_session(SESSION_B, started=10.0))
        second = harness.observe(1)
        assert second.track_uuid != first.track_uuid
        assert second.first_seen_monotonic == pytest.approx(harness.clock.monotonic() - 0.1)
        assert second.observed_frames == 1
        # Retired IDs belong to the session that retired them.
        assert harness.lifecycle.retired_display_ids() == ()

    def test_no_terminations_when_nothing_was_live(self) -> None:
        harness = Harness()
        assert harness.lifecycle.begin_session(make_session(SESSION_B, started=10.0)) == ()

    def test_session_identifies_the_camera(self) -> None:
        harness = Harness()
        assert harness.lifecycle.camera_id == CAMERA

    def test_end_session_terminates_and_clears(self) -> None:
        harness = Harness()
        harness.observe(1)
        ended = harness.lifecycle.end_session(reason="stopped")
        assert [item.reason for item in ended] == ["stopped"]
        assert harness.lifecycle.live_count() == 0
        assert harness.lifecycle.session is None
        assert harness.lifecycle.camera_id is None

    def test_summaries_are_ordered_by_display_id(self) -> None:
        harness = Harness()
        for display_id in (7, 3, 11):
            harness.observe(display_id, x=float(display_id) * 40.0)
        assert [summary.display_id for summary in harness.summaries()] == [3, 7, 11]

    def test_summary_for_unknown_track_is_none(self) -> None:
        harness = Harness()
        assert harness.lifecycle.summary_for(uuid.uuid4(), now_monotonic=0.0) is None

    def test_summary_carries_the_session_and_camera(self) -> None:
        harness = Harness()
        match = harness.observe(1)
        summary = harness.lifecycle.summary_for(match.track_uuid, now_monotonic=0.0)
        assert summary is not None
        assert summary.stream_session_id == SESSION_A
        assert summary.camera_id == CAMERA
        assert summary.bbox == (100.0, 100.0, 160.0, 300.0)
        assert summary.confidence == 0.9
