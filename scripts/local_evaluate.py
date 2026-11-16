#!/usr/bin/env python
"""Score the system on locally recorded videos (THESIS_PLAN §6.4).

Put each video in one folder with a JSON sidecar of the same name (see
``fall_detection.research.local``): whether it is a fall, the activity, the distance to the
camera, the lighting, and optionally when the person is first on the floor. The report
gives the same metrics as the URFD evaluation, plus a breakdown by distance and lighting,
so the draft's distance (2/3/4 m) and false-positive scenarios can be reported.

Example::

    uv run --project backend python scripts/local_evaluate.py \\
        --root datasets/raw/local --model yolov8n-pose --device mps
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fall_detection.config import FallSettings
from fall_detection.research.fall_eval import evaluate
from fall_detection.research.local import load_local_sequences
from fall_detection.research.metrics import confusion
from fall_detection.research.replay import simulate
from fall_detection.research.runner import (
    MODELS,
    REPO_ROOT,
    evaluation_settings,
    extract_all,
    performance_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "datasets" / "raw" / "local")
    parser.add_argument("--model", choices=sorted(MODELS), default="yolov8n-pose")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--calibration", type=Path, default=REPO_ROOT / "datasets/manifests/fall_calibration.json"
    )
    parser.add_argument(
        "--cache", type=Path, default=REPO_ROOT / "datasets" / "cache" / "fall_observations"
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"{args.root} does not exist; record the videos and their JSON sidecars first")
        return 2
    sequences = load_local_sequences(args.root)
    if not sequences:
        print(f"no video with a JSON sidecar in {args.root}")
        return 3

    fall = (
        FallSettings(calibration_path=args.calibration).with_calibration()
        if args.calibration.is_file()
        else FallSettings()
    )
    settings = evaluation_settings(args.model, device=args.device)
    extracted = extract_all(sequences, model=args.model, settings=settings, cache_dir=args.cache)
    outcomes = {
        sequence.sequence_id: simulate(
            extracted[sequence.sequence_id], is_fall=sequence.is_fall, settings=settings, fall=fall
        )
        for sequence in sequences
    }
    report: dict[str, Any] = {
        "check": "local_evaluation",
        "status": "PASS",
        "model": args.model,
        "thresholds": fall.model_dump(mode="json", exclude={"calibration_path"}),
        "calibrated": args.calibration.is_file(),
        **evaluate(sequences, outcomes, long_lie_s=fall.long_lie_s),
        "performance": performance_summary(extracted),
    }
    for key in ("distance_m", "lighting"):
        groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        for sequence in sequences:
            value = dict(sequence.metadata).get(key)
            if value is not None:
                groups[str(value)].append(
                    (sequence.is_fall, outcomes[sequence.sequence_id].detected)
                )
        report[f"by_{key}"] = {value: confusion(pairs).as_dict() for value, pairs in groups.items()}

    out = args.json or REPO_ROOT / "experiments" / "reports" / f"local_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["event_detected"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
