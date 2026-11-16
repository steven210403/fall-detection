"""Evaluation bookkeeping: metrics, URFD parsing, replay simulation, scoring, calibration.

These tests use synthetic files and idealised poses. They assert the protocol — what is
counted, what is excluded, what is refused — and never a dataset statistic or an accuracy
figure.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from tests.helpers_fall import fall_sequence, lying, standing

from fall_detection.config import FallSettings, Settings
from fall_detection.research.calibration import calibrate, frozen_payload
from fall_detection.research.fall_eval import evaluate, event_counts
from fall_detection.research.metrics import BinaryCounts, confusion
from fall_detection.research.replay import ExtractedFrame, ExtractedSequence, simulate
from fall_detection.research.sequences import LabelledSequence, SequenceFormatError
from fall_detection.research.urfd import (
    assets_for,
    load_sequence,
    load_valid_sequences,
    parse_label_csv,
    parse_timestamps,
    post_impact_seconds,
)


class TestMetrics:
    def test_the_draft_formulas(self) -> None:
        counts = BinaryCounts(tp=8, fp=2, tn=9, fn=1)
        assert counts.accuracy == pytest.approx(17 / 20)
        assert counts.precision == pytest.approx(0.8)
        assert counts.recall == pytest.approx(8 / 9)
        assert counts.specificity == pytest.approx(9 / 11)
        assert counts.f1 == pytest.approx(2 * 0.8 * (8 / 9) / (0.8 + 8 / 9))

    def test_an_undefined_ratio_is_none_not_zero(self) -> None:
        counts = BinaryCounts(tn=5)
        assert counts.precision is None
        assert counts.recall is None
        assert counts.f1 is None
        assert counts.specificity == 1.0

    def test_confusion_counts_pairs(self) -> None:
        pairs = [(True, True), (True, False), (False, True), (False, False), (False, False)]
        assert confusion(pairs) == BinaryCounts(tp=1, fn=1, fp=1, tn=2)


class TestUrfdFiles:
    def test_timestamps_are_relative_seconds(self) -> None:
        assert parse_timestamps("1,1000,1.0\n2,1033,1.0\n3,1066\n") == (0.0, 0.033, 0.066)

    def test_out_of_order_frames_are_refused(self) -> None:
        with pytest.raises(SequenceFormatError, match="out of order"):
            parse_timestamps("1,0\n3,66\n")

    def test_time_going_backwards_is_refused(self) -> None:
        with pytest.raises(SequenceFormatError, match="backwards"):
            parse_timestamps("1,100\n2,50\n")

    def test_labels_are_zero_based_and_validated(self) -> None:
        labels = parse_label_csv("fall-01,1,-1,3.1\nfall-01,2,0,3.0\nadl-01,6,-1,1.0\n")
        assert labels == {"fall-01": {0: -1, 1: 0}, "adl-01": {5: -1}}
        with pytest.raises(SequenceFormatError, match="unknown label"):
            parse_label_csv("fall-01,1,2,3.0\n")

    def test_assets_include_both_label_files(self) -> None:
        names = [asset.name for asset in assets_for(["fall-01"], source="mp4")]
        assert names == [
            "urfall-cam0-falls.csv",
            "urfall-cam0-adls.csv",
            "fall-01-cam0.mp4",
            "fall-01-data.csv",
        ]

    def test_a_frame_count_mismatch_is_refused(self, tmp_path: Path) -> None:
        cv2 = pytest.importorskip("cv2")
        import zipfile

        import numpy as np

        archive = tmp_path / "fall-01-cam0-rgb.zip"
        ok, encoded = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))
        assert ok
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("fall-01-cam0-rgb-001.png", encoded.tobytes())
        (tmp_path / "fall-01-data.csv").write_text("1,0\n2,33\n", encoding="utf-8")
        with pytest.raises(SequenceFormatError, match="refusing to pair"):
            load_sequence(tmp_path, "fall-01", labels={}, source="rgb")

    def test_a_matching_archive_loads_with_acquisition_time(self, tmp_path: Path) -> None:
        cv2 = pytest.importorskip("cv2")
        import zipfile

        import numpy as np

        archive = tmp_path / "adl-01-cam0-rgb.zip"
        ok, encoded = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))
        assert ok
        with zipfile.ZipFile(archive, "w") as handle:
            for index in range(3):
                handle.writestr(f"adl-01-cam0-rgb-{index + 1:03d}.png", encoded.tobytes())
        (tmp_path / "adl-01-data.csv").write_text("1,500\n2,533\n3,600\n", encoding="utf-8")
        sequence = load_sequence(tmp_path, "adl-01", labels={"adl-01": {2: 1}}, source="rgb")
        assert sequence.is_fall is False
        assert sequence.timestamps_s == (0.0, 0.033, 0.1)
        assert [index for index, _ in sequence.reader()] == [0, 1, 2]
        assert post_impact_seconds(sequence) == pytest.approx(0.0)

    def test_a_malformed_sequence_is_excluded_with_its_reason(self, tmp_path: Path) -> None:
        cv2 = pytest.importorskip("cv2")
        import zipfile

        import numpy as np

        ok, encoded = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))
        assert ok
        for sequence_id, frames in (("adl-01", 2), ("adl-02", 3)):
            with zipfile.ZipFile(tmp_path / f"{sequence_id}-cam0-rgb.zip", "w") as handle:
                for index in range(frames):
                    handle.writestr(f"{sequence_id}-cam0-rgb-{index + 1:03d}.png", encoded)
            (tmp_path / f"{sequence_id}-data.csv").write_text("1,0\n2,33\n", encoding="utf-8")
        for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
            (tmp_path / name).write_text("", encoding="utf-8")

        sequences, excluded = load_valid_sequences(tmp_path, ["adl-01", "adl-02"])
        assert [sequence.sequence_id for sequence in sequences] == ["adl-01"]
        assert list(excluded) == ["adl-02"]
        assert "refusing to pair" in excluded["adl-02"]


def _sequence(
    sequence_id: str, *, is_fall: bool, frames: int, labels: dict[int, int]
) -> LabelledSequence:
    return LabelledSequence(
        dataset="synthetic",
        sequence_id=sequence_id,
        is_fall=is_fall,
        activity="fall" if is_fall else "adl",
        group=sequence_id,
        timestamps_s=tuple(index / 30.0 for index in range(frames)),
        time_source="synthetic",
        reader=lambda: iter(()),
        frame_labels=labels,
    )


def _extracted(sequence_id: str, poses: list[tuple[float, object]]) -> ExtractedSequence:
    track = str(uuid.uuid4())
    frames = tuple(
        ExtractedFrame(
            index=index,
            t=t,
            people=(
                {
                    "uuid": track,
                    "id": 1,
                    "bbox": list(pose[0]),  # type: ignore[index]
                    "conf": 0.9,
                    "kp": [list(point) for point in pose[1]],  # type: ignore[index]
                },
            ),
            terminated=(),
            detection_ms=10.0,
            persons_detected=1,
        )
        for index, (t, pose) in enumerate(poses)
    )
    return ExtractedSequence(
        dataset="synthetic",
        sequence_id=sequence_id,
        model="test",
        weights_sha256="0" * 64,
        frames=frames,
        wall_s=0.0,
    )


@pytest.fixture
def corpus() -> tuple[list[LabelledSequence], dict[str, ExtractedSequence]]:
    fall = fall_sequence(lie_seconds=4.0)
    stand = [(index / 30.0, standing()) for index in range(150)]
    slow_lie = [(index / 30.0, standing() if index < 60 else lying()) for index in range(150)]
    fall_labels = {index: (1 if t >= 1.4 else -1) for index, (t, _) in enumerate(fall)}
    sequences = [
        _sequence("fall", is_fall=True, frames=len(fall), labels=fall_labels),
        _sequence("stand", is_fall=False, frames=len(stand), labels={0: -1, 1: -1}),
        _sequence("bed", is_fall=False, frames=len(slow_lie), labels={}),
    ]
    extracted = {
        "fall": _extracted("fall", fall),  # type: ignore[arg-type]
        "stand": _extracted("stand", stand),  # type: ignore[arg-type]
        "bed": _extracted("bed", slow_lie),  # type: ignore[arg-type]
    }
    return sequences, extracted


def test_simulate_reports_detection_and_alarm(
    make_settings: Callable[..., Settings],
    corpus: tuple[list[LabelledSequence], dict[str, ExtractedSequence]],
) -> None:
    sequences, extracted = corpus
    outcome = simulate(extracted["fall"], is_fall=True, settings=make_settings())
    assert outcome.detected and outcome.alarmed
    assert outcome.falling_t is not None and outcome.impact_t is not None
    assert outcome.alert_t is not None and outcome.alert_t - outcome.impact_t >= 3.0
    assert len(outcome.lying_frames) == sequences[0].frame_count


def test_evaluate_scores_every_level(
    make_settings: Callable[..., Settings],
    corpus: tuple[list[LabelledSequence], dict[str, ExtractedSequence]],
) -> None:
    sequences, extracted = corpus
    settings = make_settings()
    outcomes = {
        sequence.sequence_id: simulate(
            extracted[sequence.sequence_id], is_fall=sequence.is_fall, settings=settings
        )
        for sequence in sequences
    }
    report = evaluate(sequences, outcomes, long_lie_s=3.0, bootstrap=False)
    assert report["event_detected"]["tp"] == 1
    assert report["event_detected"]["fn"] == 0
    assert report["event_alarm"]["falls_with_enough_post_impact_footage"] == 1
    assert report["frame_level"]["excluded"].startswith("label 0")
    assert report["detection_delay_s"]["n"] == 1
    # Lying down slowly (no velocity spike) is never counted as a detected fall.
    assert report["detection_rate_by_activity"]["adl"]["detected"] == 0
    assert event_counts(sequences, outcomes, level="alarmed").fp == 0


def test_calibration_picks_from_the_grid_and_freezes_provenance(
    make_settings: Callable[..., Settings],
    corpus: tuple[list[LabelledSequence], dict[str, ExtractedSequence]],
) -> None:
    sequences, extracted = corpus
    grid = {"vy_threshold": (1.0, 15.0), "ar_threshold": (1.0,)}
    result = calibrate(sequences, extracted, settings=make_settings(), grid=grid)
    # A 15 body-lengths-per-second threshold misses the fall; 1.0 catches it.
    assert result.thresholds["vy_threshold"] == 1.0
    assert result.thresholds["long_lie_s"] == FallSettings().long_lie_s
    payload = frozen_payload(
        result, model="test", weights_sha256="0" * 64, sequences=sequences, extracted=extracted
    )
    assert payload["frozen"] is True
    assert payload["selected_on"].startswith("caucafall")
    assert len(payload["sha256"]) == 64
    assert "never re-select" in payload["rule"]
