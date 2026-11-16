"""Replay a labelled sequence through the real pipeline (THESIS_PLAN §6–§7).

Evaluation runs the **same** detector, tracker, and fall service the live application
runs, in two stages so that threshold calibration does not repeat the expensive part:

1. :func:`extract` runs YOLO pose and the tracker over every frame, on the sequence's
   acquisition timestamps (the tracker's clock is an
   :class:`~fall_detection.core.clock.AcquisitionClockBase`, so the loss tolerance is
   measured on footage time, not on how fast the replay runs). The result — each frame's
   tracked boxes and keypoints, and the measured detection time — is cached on disk keyed
   by the model's weights hash and the sequence.
2. :func:`simulate` feeds the cached observations to a fresh
   :class:`~fall_detection.fall.service.FallService` with a given set of thresholds and
   returns what happened: whether a fall was detected (impact reached), whether the
   long-lie alarm fired, when, and the per-frame state.

Stage 2 is pure Python and fast, so a calibration grid can try hundreds of threshold sets
over cached pose output. Frames are never duplicated or skipped.
"""

from __future__ import annotations

import gzip
import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fall_detection.config import FallSettings, Settings
from fall_detection.core.clock import AcquisitionClockBase
from fall_detection.fall.contracts import FallState
from fall_detection.fall.service import FallService
from fall_detection.research.sequences import LabelledSequence
from fall_detection.tracking.contracts import (
    TrackObservation,
    TrackObservations,
    TrackTermination,
)

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from fall_detection.detection.service import DetectionService

__all__ = [
    "CACHE_VERSION",
    "ExtractedFrame",
    "ExtractedSequence",
    "SequenceOutcome",
    "extract",
    "load_or_extract",
    "simulate",
]

CACHE_VERSION = 1
REPLAY_CAMERA = uuid.UUID("00000000-0000-0000-0000-00000000fa11")
REPLAY_ORIGIN_UTC = 1_700_000_000.0
_LYING_STATES = (FallState.IMPACT, FallState.FALLEN)


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    """One frame's tracked people, as cached."""

    index: int
    t: float
    people: tuple[dict[str, Any], ...]
    terminated: tuple[dict[str, Any], ...]
    detection_ms: float | None
    persons_detected: int


@dataclass(frozen=True, slots=True)
class ExtractedSequence:
    """Pose + tracking output for one sequence and one model."""

    dataset: str
    sequence_id: str
    model: str
    weights_sha256: str
    frames: tuple[ExtractedFrame, ...]
    wall_s: float

    def frames_with_person(self) -> int:
        """Frames on which the detector found at least one person."""
        return sum(1 for frame in self.frames if frame.persons_detected > 0)

    def detection_ms(self) -> list[float]:
        """Measured detector wall time per frame."""
        return [frame.detection_ms for frame in self.frames if frame.detection_ms is not None]


@dataclass(slots=True)
class SequenceOutcome:
    """What the fall service decided on one sequence."""

    sequence_id: str
    is_fall: bool
    detected: bool
    alarmed: bool
    falling_t: float | None
    impact_t: float | None
    alert_t: float | None
    lying_frames: dict[int, bool] = field(default_factory=dict)
    states: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "is_fall": self.is_fall,
            "detected": self.detected,
            "alarmed": self.alarmed,
            "falling_t": _round(self.falling_t),
            "impact_t": _round(self.impact_t),
            "alert_t": _round(self.alert_t),
        }


def extract(
    sequence: LabelledSequence,
    detection: DetectionService,
    settings: Settings,
    *,
    model: str,
    weights_sha256: str,
) -> ExtractedSequence:
    """Run pose detection and tracking over every frame of ``sequence``."""
    from fall_detection.core.contracts import FramePacket
    from fall_detection.tracking.service import TrackingService

    clock = AcquisitionClockBase(origin_utc=REPLAY_ORIGIN_UTC)
    tracking = TrackingService(settings, clock=clock)
    session = uuid.uuid5(uuid.NAMESPACE_URL, f"{sequence.dataset}:{sequence.sequence_id}")
    frames: list[ExtractedFrame] = []
    started = time.perf_counter()
    for index, image in sequence.reader():
        if index >= sequence.frame_count:
            break
        t = sequence.timestamps_s[index]
        clock.advance_to(t)
        packet = FramePacket(
            camera_id=REPLAY_CAMERA,
            stream_session_id=session,
            frame_seq=index,
            frame=image,
            source_timestamp=t,
            receive_monotonic=t,
            receive_utc=REPLAY_ORIGIN_UTC + t,
        )
        batch = detection.process(packet)
        tracks = tracking.update(batch)
        frames.append(
            ExtractedFrame(
                index=index,
                t=t,
                people=tuple(_observation_dict(item) for item in tracks.observations),
                terminated=tuple(
                    {"uuid": str(item.track_uuid), "reason": item.reason}
                    for item in tracks.terminated
                ),
                detection_ms=None if batch.timing is None else batch.timing.total_ms,
                persons_detected=len(batch.persons()),
            )
        )
    wall = time.perf_counter() - started
    return ExtractedSequence(
        dataset=sequence.dataset,
        sequence_id=sequence.sequence_id,
        model=model,
        weights_sha256=weights_sha256,
        frames=tuple(frames),
        wall_s=wall,
    )


def load_or_extract(
    sequence: LabelledSequence,
    *,
    cache_dir: Path,
    model: str,
    weights_sha256: str,
    detection: DetectionService | None,
    settings: Settings,
) -> ExtractedSequence:
    """Read the cached extraction, or run and cache it.

    The cache key includes the weights hash, the detector's image size and confidence,
    and the tracker configuration, so changing any of them re-extracts.
    """
    key = _cache_key(sequence, model=model, weights_sha256=weights_sha256, settings=settings)
    path = cache_dir / model / sequence.dataset / f"{_safe(sequence.sequence_id)}-{key}.json.gz"
    if path.is_file():
        return _read(path)
    if detection is None:
        raise FileNotFoundError(f"no cached extraction at {path} and no detector to run")
    extracted = extract(sequence, detection, settings, model=model, weights_sha256=weights_sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, extracted)
    return extracted


def simulate(
    extracted: ExtractedSequence,
    *,
    is_fall: bool,
    settings: Settings,
    fall: FallSettings | None = None,
) -> SequenceOutcome:
    """Run the fall service over cached observations with the given thresholds."""
    service = FallService(settings, record_transitions=True, fall_settings=fall)
    session = uuid.uuid5(uuid.NAMESPACE_URL, f"{extracted.dataset}:{extracted.sequence_id}")
    outcome = SequenceOutcome(
        sequence_id=extracted.sequence_id,
        is_fall=is_fall,
        detected=False,
        alarmed=False,
        falling_t=None,
        impact_t=None,
        alert_t=None,
    )
    for frame in extracted.frames:
        batch = service.update(_track_observations(frame, session))
        lying = any(decision.state in _LYING_STATES for decision in batch.decisions)
        outcome.lying_frames[frame.index] = lying
        top = max(
            (decision.state for decision in batch.decisions),
            key=_state_rank,
            default=FallState.UNKNOWN,
        )
        outcome.states.append(top.value)
    for transition in service.transition_log:
        if transition.to_state is FallState.FALLING and outcome.falling_t is None:
            outcome.falling_t = transition.t
        elif transition.to_state is FallState.IMPACT and outcome.impact_t is None:
            outcome.impact_t = transition.t
            outcome.detected = True
        elif transition.to_state is FallState.FALLEN and outcome.alert_t is None:
            outcome.alert_t = transition.t
            outcome.alarmed = True
    return outcome


def _state_rank(state: FallState) -> int:
    order = (
        FallState.UNKNOWN,
        FallState.NORMAL,
        FallState.FALLING,
        FallState.IMPACT,
        FallState.FALLEN,
    )
    return order.index(state)


def _track_observations(frame: ExtractedFrame, session: uuid.UUID) -> TrackObservations:
    utc = REPLAY_ORIGIN_UTC + frame.t
    observations = tuple(
        TrackObservation(
            track_uuid=uuid.UUID(person["uuid"]),
            display_id=int(person["id"]),
            camera_id=REPLAY_CAMERA,
            stream_session_id=session,
            bbox=tuple(person["bbox"]),  # type: ignore[arg-type]
            confidence=float(person["conf"]),
            class_name="person",
            frame_seq=frame.index,
            observed_monotonic=frame.t,
            source_timestamp=frame.t,
            receive_monotonic=frame.t,
            receive_utc=utc,
            first_seen_monotonic=0.0,
            age_s=0.0,
            observed_frames=1,
            keypoints=None
            if person["kp"] is None
            else tuple((float(x), float(y), float(c)) for x, y, c in person["kp"]),
        )
        for person in frame.people
    )
    terminated = tuple(
        TrackTermination(
            track_uuid=uuid.UUID(item["uuid"]),
            display_id=0,
            camera_id=REPLAY_CAMERA,
            stream_session_id=session,
            reason=str(item["reason"]),
            ended_monotonic=frame.t,
            ended_utc=utc,
            duration_s=0.0,
            observed_frames=0,
            missing_frames=0,
        )
        for item in frame.terminated
    )
    return TrackObservations(
        camera_id=REPLAY_CAMERA,
        stream_session_id=session,
        frame_seq=frame.index,
        observations=observations,
        tracks=(),
        terminated=terminated,
        frame_shape=(0, 0, 3),
        source_timestamp=frame.t,
        receive_monotonic=frame.t,
        receive_utc=utc,
        tracking_monotonic=frame.t,
        tracker_name="replay",
    )


def _observation_dict(observation: TrackObservation) -> dict[str, Any]:
    return {
        "uuid": str(observation.track_uuid),
        "id": observation.display_id,
        "bbox": [round(value, 2) for value in observation.bbox],
        "conf": round(observation.confidence, 4),
        "kp": None
        if observation.keypoints is None
        else [[round(x, 2), round(y, 2), round(c, 4)] for x, y, c in observation.keypoints],
    }


def _cache_key(
    sequence: LabelledSequence, *, model: str, weights_sha256: str, settings: Settings
) -> str:
    import hashlib

    payload = json.dumps(
        {
            "version": CACHE_VERSION,
            "model": model,
            "weights": weights_sha256,
            "imgsz": settings.detection.imgsz,
            "conf": settings.detection.conf,
            "iou": settings.detection.iou,
            "tracking": settings.tracking.model_dump(mode="json"),
            "frames": sequence.frame_count,
            "first_t": sequence.timestamps_s[0] if sequence.timestamps_s else None,
            "last_t": sequence.timestamps_s[-1] if sequence.timestamps_s else None,
            "source": dict(sequence.metadata).get("frame_source"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _write(path: Path, extracted: ExtractedSequence) -> None:
    payload = {
        "version": CACHE_VERSION,
        "dataset": extracted.dataset,
        "sequence_id": extracted.sequence_id,
        "model": extracted.model,
        "weights_sha256": extracted.weights_sha256,
        "wall_s": extracted.wall_s,
        "frames": [
            {
                "i": frame.index,
                "t": frame.t,
                "people": list(frame.people),
                "terminated": list(frame.terminated),
                "det_ms": frame.detection_ms,
                "n": frame.persons_detected,
            }
            for frame in extracted.frames
        ],
    }
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    tmp.replace(path)


def _read(path: Path) -> ExtractedSequence:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("version") != CACHE_VERSION:
        raise ValueError(f"{path} was written by cache version {payload.get('version')}")
    return ExtractedSequence(
        dataset=payload["dataset"],
        sequence_id=payload["sequence_id"],
        model=payload["model"],
        weights_sha256=payload["weights_sha256"],
        wall_s=float(payload["wall_s"]),
        frames=tuple(
            ExtractedFrame(
                index=int(frame["i"]),
                t=float(frame["t"]),
                people=tuple(frame["people"]),
                terminated=tuple(frame["terminated"]),
                detection_ms=frame["det_ms"],
                persons_detected=int(frame["n"]),
            )
            for frame in payload["frames"]
        ),
    )


def _safe(name: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in name
    )


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def outcomes_by_id(outcomes: Sequence[SequenceOutcome]) -> dict[str, SequenceOutcome]:
    """Index outcomes by sequence id."""
    return {outcome.sequence_id: outcome for outcome in outcomes}
