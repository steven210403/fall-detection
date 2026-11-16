#!/usr/bin/env python
"""Calibrate the fall thresholds on CAUCAFall and freeze them (THESIS_PLAN §6.3).

Runs the pose model and tracker over the 100 CAUCAFall videos (cached after the first
run), grid-searches the state-machine thresholds for the best event-level F1, and writes
the frozen artifact ``datasets/manifests/fall_calibration.json`` plus a report. URFD is
never read here: it is the held-out test set.

Example::

    uv run --project backend python scripts/fall_calibrate.py --model yolov8n-pose --device mps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fall_detection.research.calibration import calibrate, frozen_payload
from fall_detection.research.caucafall import load_manifest, load_sequences
from fall_detection.research.runner import (
    MODELS,
    REPO_ROOT,
    evaluation_settings,
    extract_all,
    performance_summary,
    sha256_of,
    weights_for,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="yolov8n-pose")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "datasets" / "raw" / "caucafall")
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "datasets/manifests/caucafall_videos.json"
    )
    parser.add_argument(
        "--cache", type=Path, default=REPO_ROOT / "datasets" / "cache" / "fall_observations"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "datasets/manifests/fall_calibration.json"
    )
    parser.add_argument(
        "--report", type=Path, default=REPO_ROOT / "experiments/reports/fall_calibration.json"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing frozen calibration"
    )
    args = parser.parse_args()

    if args.out.is_file() and not args.force:
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if existing.get("frozen"):
            print(f"{args.out} is frozen; pass --force only if the test set was never scored")
            return 1

    if not args.manifest.is_file():
        print(f"missing {args.manifest}; run scripts/caucafall_download.py first")
        return 2
    videos = load_manifest(args.manifest)
    try:
        sequences = load_sequences(args.root, videos)
    except FileNotFoundError as exc:
        print(f"missing video {exc}; run scripts/caucafall_download.py first")
        return 2

    settings = evaluation_settings(args.model, device=args.device)
    extracted = extract_all(sequences, model=args.model, settings=settings, cache_dir=args.cache)
    result = calibrate(sequences, extracted, settings=settings)
    payload = frozen_payload(
        result,
        model=args.model,
        weights_sha256=sha256_of(weights_for(args.model)),
        sequences=sequences,
        extracted=extracted,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = {
        "check": "fall_calibration",
        "status": "PASS",
        "detail": "thresholds selected on CAUCAFall only and frozen",
        "model": args.model,
        "device": args.device,
        "thresholds": result.thresholds,
        "calibration_score": result.score,
        "grid_size": len(result.table),
        "table": result.table,
        "sequences": [sequence.describe() for sequence in sequences],
        "performance": performance_summary(extracted),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"thresholds": result.thresholds, "score": result.score}, indent=2))
    print(f"wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
