#!/usr/bin/env python
"""Score fall detection on the UR Fall Detection Dataset (THESIS_PLAN §7).

URFD is the held-out test set. Two threshold sets are scored, each once:

* ``frozen`` — the thresholds frozen on CAUCAFall (``datasets/manifests/fall_calibration.json``);
* ``literature`` — the draft's literature defaults (AR >= 1.0, lying < 30 deg,
  Vy > 1 body length/s, 3 s long-lie), as the baseline.

The report holds the event-level confusion matrix and accuracy / precision / recall /
specificity / F1 ("fall detected" and "alarm"), frame-level scores against URFD's
per-frame labels, detection and alarm delays, bootstrap intervals by sequence, and the
detector's measured latency and FPS. Run it once per model to compare YOLOv8n-pose,
YOLO11n-pose, and YOLO26n-pose.

Example::

    uv run --project backend python scripts/urfd_evaluate.py --model yolov8n-pose --device mps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fall_detection.config import FallSettings
from fall_detection.research.fall_eval import evaluate
from fall_detection.research.replay import simulate
from fall_detection.research.runner import (
    MODELS,
    REPO_ROOT,
    evaluation_settings,
    extract_all,
    performance_summary,
)
from fall_detection.research.urfd import load_valid_sequences, post_impact_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="yolov8n-pose")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--source", choices=("rgb", "mp4"), default="rgb")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "datasets" / "raw" / "urfd")
    parser.add_argument(
        "--cache", type=Path, default=REPO_ROOT / "datasets" / "cache" / "fall_observations"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "datasets/manifests/fall_calibration.json",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    args.calibration = args.calibration.resolve()

    try:
        sequences, excluded = load_valid_sequences(args.root, source=args.source)
    except FileNotFoundError as exc:
        print(f"missing {exc}; run scripts/urfd_download.py --source {args.source} first")
        return 2

    for sequence_id, reason in excluded.items():
        print(f"excluded {sequence_id}: {reason}")

    settings = evaluation_settings(args.model, device=args.device)
    extracted = extract_all(sequences, model=args.model, settings=settings, cache_dir=args.cache)

    threshold_sets: dict[str, tuple[FallSettings, dict[str, Any]]] = {
        "literature": (FallSettings(), {"source": "the draft's literature defaults"}),
    }
    if args.calibration.is_file():
        frozen = json.loads(args.calibration.read_text(encoding="utf-8"))
        if not frozen.get("frozen"):
            print(f"{args.calibration} is not frozen; refusing to score the test set with it")
            return 1
        fall = FallSettings(calibration_path=args.calibration).with_calibration()
        threshold_sets["frozen"] = (
            fall,
            {
                "source": str(args.calibration.relative_to(REPO_ROOT)),
                "sha256": frozen.get("sha256"),
                "selected_on": frozen.get("selected_on"),
                "calibration_model": frozen.get("provenance", {}).get("model"),
            },
        )
    else:
        print(f"note: {args.calibration} not found; scoring the literature defaults only")

    results: dict[str, Any] = {}
    for name, (fall, provenance) in threshold_sets.items():
        outcomes = {
            sequence.sequence_id: simulate(
                extracted[sequence.sequence_id],
                is_fall=sequence.is_fall,
                settings=settings,
                fall=fall,
            )
            for sequence in sequences
        }
        results[name] = {
            "thresholds": fall.model_dump(mode="json", exclude={"calibration_path"}),
            "provenance": provenance,
            **evaluate(sequences, outcomes, long_lie_s=fall.long_lie_s),
        }

    footage = [post_impact_seconds(sequence) for sequence in sequences if sequence.is_fall]
    report = {
        "check": "urfd_evaluation",
        "status": "PASS",
        "detail": "URFD scored once per threshold set; no threshold was selected on URFD",
        "dataset": "urfd",
        "camera": "cam0",
        "frame_source": args.source,
        "model": args.model,
        "device": args.device,
        "sequences_scored": {
            "falls": sum(sequence.is_fall for sequence in sequences),
            "adls": sum(not sequence.is_fall for sequence in sequences),
        },
        "excluded_sequences": excluded,
        "post_impact_footage_s": sorted(round(value, 3) for value in footage if value is not None),
        "results": results,
        "performance": performance_summary(extracted),
        "limitations": [
            "URFD actors are young adults simulating falls, not elderly people",
            "cam0 is mounted about 1 m high, parallel to the floor, not as elevated CCTV",
            "every URFD fall clip ends within about 2.1 s of the first lying frame, so the "
            "3 s long-lie alarm cannot be observed on URFD; 'fall detected' is the primary "
            "event-level result",
        ],
    }
    out = args.json or REPO_ROOT / "experiments" / "reports" / f"urfd_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in results.items():
        print(
            name,
            json.dumps(
                {
                    "event_detected": result["event_detected"],
                    "frame_level": result["frame_level"],
                }
            ),
        )
    print(json.dumps(report["performance"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
