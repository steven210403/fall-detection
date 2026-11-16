#!/usr/bin/env python
"""Person-detection and pose accuracy of the compared models (Research Question 1).

URFD and CAUCAFall have no keypoint ground truth, so detector quality is measured the
standard way: Ultralytics' validator on the COCO-Pose validation set (val2017, persons
with 17 keypoints). It reports box precision, recall, mAP50 and mAP50-95, and the same for
the keypoints (OKS-based pose mAP). ``--data coco8-pose.yaml`` runs an 8-image smoke test
instead of the full ~1 GB download.

The stock ``coco-pose.yaml`` also lists train2017 (~19 GB), and Ultralytics downloads every
split it cannot find. Validation never reads the training split, so the script writes a
val-only copy of the dataset file (``datasets/cache/coco-pose-val.yaml``) and uses that.

Example::

    uv run --project backend python scripts/pose_benchmark.py --device mps
    uv run --project backend python scripts/pose_benchmark.py --data coco8-pose.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fall_detection.research.runner import MODELS, REPO_ROOT, sha256_of, weights_for


def _metrics(result: Any) -> dict[str, Any]:
    box = result.box
    pose = result.pose
    speed = getattr(result, "speed", {}) or {}
    return {
        "box": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
        },
        "pose": {
            "precision": round(float(pose.mp), 4),
            "recall": round(float(pose.mr), 4),
            "map50": round(float(pose.map50), 4),
            "map50_95": round(float(pose.map), 4),
        },
        "speed_ms_per_image": {key: round(float(value), 3) for key, value in speed.items()},
    }


def _val_only(data: str) -> str:
    """Return a dataset file that names only the validation split.

    Only the stock ``coco-pose.yaml`` is rewritten; anything else is passed through.
    """
    if data != "coco-pose.yaml":
        return data
    import yaml
    from ultralytics.utils.checks import check_yaml

    config = yaml.safe_load(Path(str(check_yaml(data))).read_text(encoding="utf-8"))
    config["train"] = config["val"]
    config.pop("test", None)
    out = REPO_ROOT / "datasets" / "cache" / "coco-pose-val.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=sorted(MODELS), choices=sorted(MODELS))
    parser.add_argument("--data", default="coco-pose.yaml")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    # Keep Ultralytics' settings and dataset downloads inside this repository's gitignored
    # ``datasets/`` instead of the user's global Ultralytics configuration.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / "datasets" / "cache" / "ultralytics"))
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS.update({"datasets_dir": str(REPO_ROOT / "datasets" / "raw" / "ultralytics")})

    data = _val_only(args.data)
    results: dict[str, Any] = {}
    for model in args.models:
        weights = weights_for(model)
        if not weights.is_file():
            print(f"{weights} is missing; run scripts/phase0_ml_smoke.py first")
            return 2
        validator = YOLO(str(weights)).val(
            data=data,
            imgsz=args.imgsz,
            device=args.device,
            plots=False,
            verbose=False,
            project=str(REPO_ROOT / "experiments" / "runs" / "pose_benchmark"),
            name=model,
            exist_ok=True,
        )
        results[model] = {"weights_sha256": sha256_of(weights), **_metrics(validator)}
        print(model, json.dumps(results[model]))

    report = {
        "check": "pose_benchmark",
        "status": "PASS",
        "dataset": args.data,
        "imgsz": args.imgsz,
        "device": args.device,
        "results": results,
        "note": "measured with the Ultralytics validator; COCO-Pose val2017 unless --data says "
        "otherwise. coco8-pose is a smoke test, not a benchmark.",
    }
    stem = Path(args.data).stem.replace("-", "_")
    out = args.json or REPO_ROOT / "experiments" / "reports" / f"pose_benchmark_{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
