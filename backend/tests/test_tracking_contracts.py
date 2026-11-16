"""Tracking contract tests: geometry, states, and serialised views.

These tests use no tracker and no model. They pin the shape of the contracts the
rest of the project depends on, and they assert the *absences* thesis plan §17.1
requires — a contract module that imported NumPy or SQLAlchemy would defeat the
point of keeping typed records between layers.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from fall_detection.tracking import (
    TRACK_TERMINATION_REASONS,
    TrackObservation,
    TrackObservations,
    TrackState,
    TrackSummary,
    TrackTermination,
)

CAMERA = uuid.UUID("22222222-2222-2222-2222-222222222222")
SESSION = uuid.UUID("11111111-1111-1111-1111-111111111111")
TRACK = uuid.UUID("33333333-3333-3333-3333-333333333333")


def make_observation(**overrides: object) -> TrackObservation:
    values: dict[str, object] = {
        "track_uuid": TRACK,
        "display_id": 7,
        "camera_id": CAMERA,
        "stream_session_id": SESSION,
        "bbox": (10.0, 20.0, 70.0, 220.0),
        "confidence": 0.83,
        "class_name": "person",
        "frame_seq": 5,
        "observed_monotonic": 12.0,
        "source_timestamp": 0.4,
        "receive_monotonic": 11.9,
        "receive_utc": 1_700_000_000.0,
        "first_seen_monotonic": 10.0,
        "age_s": 2.0,
        "observed_frames": 6,
        "gap_frames": 0,
        "recovered": False,
    }
    values.update(overrides)
    return TrackObservation(**values)  # type: ignore[arg-type]


def make_summary(**overrides: object) -> TrackSummary:
    values: dict[str, object] = {
        "track_uuid": TRACK,
        "display_id": 7,
        "camera_id": CAMERA,
        "stream_session_id": SESSION,
        "state": TrackState.ACTIVE,
        "first_seen_monotonic": 10.0,
        "last_observed_monotonic": 12.0,
        "age_s": 2.0,
        "time_since_last_observation_s": 0.0,
        "stale": False,
        "observed_frames": 6,
        "missing_frames": 0,
        "bbox": (10.0, 20.0, 70.0, 220.0),
        "confidence": 0.83,
    }
    values.update(overrides)
    return TrackSummary(**values)  # type: ignore[arg-type]


def make_termination(**overrides: object) -> TrackTermination:
    values: dict[str, object] = {
        "track_uuid": TRACK,
        "display_id": 7,
        "camera_id": CAMERA,
        "stream_session_id": SESSION,
        "reason": "lost_timeout",
        "ended_monotonic": 12.0,
        "ended_utc": 1_700_000_000.0,
        "duration_s": 2.0,
        "observed_frames": 6,
        "missing_frames": 20,
    }
    values.update(overrides)
    return TrackTermination(**values)  # type: ignore[arg-type]


def make_all(
    observations: tuple[TrackObservation, ...] = (),
    tracks: tuple[TrackSummary, ...] = (),
    terminated: tuple[TrackTermination, ...] = (),
) -> TrackObservations:
    return TrackObservations(
        camera_id=CAMERA,
        stream_session_id=SESSION,
        frame_seq=5,
        observations=observations,
        tracks=tracks,
        terminated=terminated,
        frame_shape=(480, 640, 3),
        source_timestamp=0.4,
        receive_monotonic=11.9,
        receive_utc=1_700_000_000.0,
        tracking_monotonic=12.0,
        tracker_name="bytetrack",
    )


class TestTrackObservation:
    def test_geometry_helpers(self) -> None:
        observation = make_observation()
        assert observation.width == 60.0
        assert observation.height == 200.0
        assert observation.center == (40.0, 120.0)

    def test_inverted_box_reports_zero_width(self) -> None:
        observation = make_observation(bbox=(70.0, 20.0, 10.0, 220.0))
        assert observation.width == 0.0
        assert observation.height == 200.0

    def test_serialised_view_is_json_safe(self) -> None:
        payload = make_observation().as_dict()
        # The UUID must be a string: a status payload is serialised to JSON, and
        # a UUID object would raise at that point rather than here.
        assert payload["track_uuid"] == str(TRACK)
        json.dumps(payload)

    def test_serialised_view_reports_gap_and_recovery(self) -> None:
        payload = make_observation(gap_frames=4, recovered=True).as_dict()
        assert payload["gap_frames"] == 4
        assert payload["recovered"] is True

    def test_missing_source_timestamp_is_preserved(self) -> None:
        # §9.1: a missing source timestamp is never replaced with a synthetic one.
        assert make_observation(source_timestamp=None).source_timestamp is None


class TestTrackSummary:
    def test_live_states(self) -> None:
        assert make_summary(state=TrackState.ACTIVE).is_live
        assert make_summary(state=TrackState.LOST).is_live
        assert not make_summary(state=TrackState.TERMINATED).is_live

    def test_bbox_may_be_absent_before_the_first_observation(self) -> None:
        summary = make_summary(bbox=None, confidence=None)
        assert summary.as_dict()["bbox"] is None
        assert summary.as_dict()["confidence"] is None

    def test_serialised_view_surfaces_state_and_staleness(self) -> None:
        # §18.1 forbids encoding unknown/stale state in colour alone, so both
        # must be explicit fields.
        payload = make_summary(state=TrackState.LOST, stale=True).as_dict()
        assert payload["state"] == "lost"
        assert payload["stale"] is True

    def test_termination_reason_is_omitted_when_absent(self) -> None:
        assert "termination_reason" not in make_summary().as_dict()
        terminated = make_summary(state=TrackState.TERMINATED, termination_reason="lost_timeout")
        assert terminated.as_dict()["termination_reason"] == "lost_timeout"


class TestTrackTermination:
    def test_reason_vocabulary_matches_the_database_constraint(self) -> None:
        # The database check constraint repeats the vocabulary rather than
        # importing this module, so the two must be kept identical deliberately.
        from fall_detection.database import TRACK_TERMINATION_REASONS as db_reasons

        assert db_reasons == TRACK_TERMINATION_REASONS

    def test_serialised_view_is_json_safe(self) -> None:
        payload = make_termination().as_dict()
        assert payload["reason"] == "lost_timeout"
        json.dumps(payload)


class TestTrackObservations:
    def test_empty_frame_has_no_observations(self) -> None:
        update = make_all()
        assert update.observations == ()
        assert update.observed_uuids() == ()
        assert update.lost_tracks() == ()

    def test_lost_tracks_are_reported_from_summaries(self) -> None:
        update = make_all(
            observations=(make_observation(),),
            tracks=(make_summary(), make_summary(state=TrackState.LOST, display_id=8)),
        )
        lost = update.lost_tracks()
        assert len(lost) == 1
        assert lost[0].state is TrackState.LOST

    def test_by_uuid_indexes_the_frame(self) -> None:
        observation = make_observation()
        update = make_all(observations=(observation,))
        assert update.by_uuid()[TRACK] is observation

    def test_serialised_view_is_json_safe(self) -> None:
        update = make_all(
            observations=(make_observation(),),
            tracks=(make_summary(),),
            terminated=(make_termination(),),
        )
        payload = update.as_dict()
        assert payload["tracker"] == "bytetrack"
        assert len(payload["observations"]) == 1
        assert len(payload["terminated"]) == 1
        json.dumps(payload)


class TestModuleBoundaries:
    def test_contracts_stay_free_of_cv_and_orm_dependencies(self) -> None:
        # §17.1: CV modules exchange dataclasses and never touch ORM models. A
        # contract that imported NumPy or SQLAlchemy would pull the ML stack into
        # every API import, so the absence is asserted against the source — the
        # only way to check an absence of imports.
        import fall_detection.tracking.contracts as contracts_module

        assert contracts_module.__file__ is not None
        source = Path(contracts_module.__file__).read_text(encoding="utf-8")
        assert "import numpy" not in source
        assert "import torch" not in source
        assert "import cv2" not in source
        assert "sqlalchemy" not in source
        assert "fall_detection.database" not in source

    def test_lifecycle_module_is_free_of_tracker_and_orm_imports(self) -> None:
        # The time-based rules must stay testable without Ultralytics, which is
        # what lets a ManualClock drive them instead of sleeping.
        import fall_detection.tracking.lifecycle as lifecycle_module

        assert lifecycle_module.__file__ is not None
        source = Path(lifecycle_module.__file__).read_text(encoding="utf-8")
        assert "import ultralytics" not in source
        assert "import numpy" not in source
        assert "import torch" not in source
        assert "sqlalchemy" not in source
        assert "fall_detection.database" not in source

    def test_tracker_imports_ml_stack_lazily(self) -> None:
        # Same rule detection.detector follows: an ML import at module scope would
        # make importing the application expensive even with tracking disabled.
        import fall_detection.tracking.tracker as tracker_module

        assert tracker_module.__file__ is not None
        source = Path(tracker_module.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            if "ultralytics" in stripped or "numpy" in stripped:
                assert line[:1].isspace(), f"module-level ML import: {line!r}"

    def test_service_and_lifecycle_do_not_import_the_api(self) -> None:
        import fall_detection.tracking.service as service_module

        for module in (service_module,):
            assert module.__file__ is not None
            source = Path(module.__file__).read_text(encoding="utf-8")
            assert "from fall_detection.api" not in source
            assert "import fastapi" not in source

    def test_tracking_does_not_import_the_video_package(self) -> None:
        # Tracking consumes DetectionBatch, never a decoded frame, so it must not
        # depend on the video layer even though it sits next to it in §9.
        import fall_detection.tracking.lifecycle as lifecycle_module
        import fall_detection.tracking.service as service_module
        import fall_detection.tracking.tracker as tracker_module

        for module in (lifecycle_module, service_module, tracker_module):
            assert module.__file__ is not None
            source = Path(module.__file__).read_text(encoding="utf-8")
            assert "from fall_detection.video" not in source


class TestStateVocabulary:
    def test_states_are_strings_so_payloads_stay_readable(self) -> None:
        # §19: string enums, portable across SQLite and PostgreSQL.
        assert [state.value for state in TrackState] == ["active", "lost", "terminated"]

    def test_termination_reasons_are_the_documented_set(self) -> None:
        assert TRACK_TERMINATION_REASONS == ("lost_timeout", "session_ended", "stopped")

    def test_contracts_are_frozen(self) -> None:
        with pytest.raises(AttributeError):
            make_observation().display_id = 9  # type: ignore[misc]
