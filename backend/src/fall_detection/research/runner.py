"""Shared plumbing for the evaluation scripts: models, settings, extraction.

The scripts under ``scripts/`` are thin entrypoints; everything they share lives here so
the type checker covers it and a test can import it.
"""

from __future__ import annotations

import hashlib
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fall_detection.config import Settings
from fall_detection.research.replay import ExtractedSequence, load_or_extract
from fall_detection.research.sequences import LabelledSequence

__all__ = [
    "MODELS",
    "REPO_ROOT",
    "evaluation_settings",
    "extract_all",
    "performance_summary",
    "weights_for",
]

REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS: dict[str, str] = {
    "yolov8n-pose": "yolov8n-pose.pt",
    "yolo11n-pose": "yolo11n-pose.pt",
    "yolo26n-pose": "yolo26n-pose.pt",
}
"""The compared pose models. YOLOv8n-pose is the thesis's primary model."""


def weights_for(model: str) -> Path:
    """Path of a model's weights under ``models/`` (downloaded by the smoke script)."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose one of {sorted(MODELS)}")
    return REPO_ROOT / "models" / MODELS[model]


def evaluation_settings(model: str, *, device: str, imgsz: int = 640) -> Settings:
    """Settings for an offline run: no database, no ``.env``, the chosen model."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment="test",
        device=device,  # type: ignore[arg-type]
        detection={
            "enabled": True,
            "model_path": str(weights_for(model)),
            "imgsz": imgsz,
            "warmup_runs": 1,
        },
    )


def sha256_of(path: Path) -> str:
    """SHA-256 of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_all(
    sequences: Sequence[LabelledSequence],
    *,
    model: str,
    settings: Settings,
    cache_dir: Path,
) -> dict[str, ExtractedSequence]:
    """Run (or read cached) pose + tracking for every sequence."""
    weights = weights_for(model)
    if not weights.is_file():
        raise FileNotFoundError(
            f"{weights} is missing; run scripts/phase0_ml_smoke.py to download the weights"
        )
    digest = sha256_of(weights)
    detection = None
    results: dict[str, ExtractedSequence] = {}
    for position, sequence in enumerate(sequences, start=1):
        try:
            results[sequence.sequence_id] = load_or_extract(
                sequence,
                cache_dir=cache_dir,
                model=model,
                weights_sha256=digest,
                detection=None,
                settings=settings,
            )
            continue
        except FileNotFoundError:
            pass
        if detection is None:
            from fall_detection.detection import DetectionService

            detection = DetectionService(settings)
            detection.load()
        print(
            f"[{position}/{len(sequences)}] {model} {sequence.dataset}:{sequence.sequence_id} "
            f"({sequence.frame_count} frames)",
            file=sys.stderr,
        )
        results[sequence.sequence_id] = load_or_extract(
            sequence,
            cache_dir=cache_dir,
            model=model,
            weights_sha256=digest,
            detection=detection,
            settings=settings,
        )
    if detection is not None:
        detection.unload()
    return results


def performance_summary(extracted: dict[str, ExtractedSequence]) -> dict[str, Any]:
    """Detector latency and the person-detection rate over the replayed frames.

    The latency is the detector's measured wall time per frame on this machine; FPS is
    ``frames / total detector time``. Neither is an accuracy figure.
    """
    latencies = [value for item in extracted.values() for value in item.detection_ms()]
    frames = sum(len(item.frames) for item in extracted.values())
    with_person = sum(item.frames_with_person() for item in extracted.values())
    if not latencies:
        return {"frames": frames}
    ordered = sorted(latencies)
    total_s = sum(latencies) / 1000.0
    return {
        "frames": frames,
        "frames_with_person": with_person,
        "person_detection_rate": round(with_person / frames, 4) if frames else None,
        "detector_ms_mean": round(statistics.fmean(latencies), 3),
        "detector_ms_p50": round(ordered[len(ordered) // 2], 3),
        "detector_ms_p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "detector_fps": round(frames / total_s, 2) if total_s > 0 else None,
        "note": "detector wall time per frame on this machine (pose inference incl. pre/post)",
    }
