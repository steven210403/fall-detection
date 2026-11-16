#!/usr/bin/env python
"""Build the YOLOv8n / YOLO11n / YOLO26n comparison table (THESIS_PLAN §7).

Reads the reports the other scripts wrote and puts one row per model:

* COCO-Pose box and pose mAP (``pose_benchmark_coco_pose.json``);
* CAUCAFall calibration F1 (``fall_calibration*.json``);
* URFD event-level and frame-level scores with the model's frozen thresholds
  (``urfd_<model>.json``);
* the detector's measured FPS and latency.

A missing report leaves its cells empty rather than failing, so the table can be built as
the runs finish. Writes ``experiments/reports/model_comparison.json`` and a Markdown table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fall_detection.research.runner import MODELS, REPO_ROOT

REPORTS = REPO_ROOT / "experiments" / "reports"


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _cell(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=REPORTS)
    args = parser.parse_args()

    pose = _load(args.reports / "pose_benchmark_coco_pose.json") or {}
    rows: list[dict[str, Any]] = []
    for model in sorted(MODELS, key=lambda name: (name != "yolov8n-pose", name)):
        calibration_name = (
            "fall_calibration.json" if model == "yolov8n-pose" else f"fall_calibration_{model}.json"
        )
        calibration = _load(args.reports / calibration_name) or {}
        urfd = _load(args.reports / f"urfd_{model}.json") or {}
        frozen = urfd.get("results", {}).get("frozen", {})
        literature = urfd.get("results", {}).get("literature", {})
        coco = pose.get("results", {}).get(model, {})
        rows.append(
            {
                "model": model,
                "coco_box_map50_95": coco.get("box", {}).get("map50_95"),
                "coco_pose_map50_95": coco.get("pose", {}).get("map50_95"),
                "caucafall_f1": calibration.get("calibration_score", {}).get("f1"),
                "urfd_f1_frozen": frozen.get("event_detected", {}).get("f1"),
                "urfd_accuracy_frozen": frozen.get("event_detected", {}).get("accuracy"),
                "urfd_recall_frozen": frozen.get("event_detected", {}).get("recall"),
                "urfd_specificity_frozen": frozen.get("event_detected", {}).get("specificity"),
                "urfd_frame_f1_frozen": (frozen.get("frame_level") or {}).get("f1"),
                "urfd_f1_literature": literature.get("event_detected", {}).get("f1"),
                "detector_fps": urfd.get("performance", {}).get("detector_fps"),
                "detector_ms_mean": urfd.get("performance", {}).get("detector_ms_mean"),
            }
        )

    headers = [
        ("Model", "model"),
        ("COCO box mAP50-95", "coco_box_map50_95"),
        ("COCO pose mAP50-95", "coco_pose_map50_95"),
        ("CAUCAFall F1 (calib.)", "caucafall_f1"),
        ("URFD F1", "urfd_f1_frozen"),
        ("URFD Acc", "urfd_accuracy_frozen"),
        ("URFD Recall", "urfd_recall_frozen"),
        ("URFD Spec.", "urfd_specificity_frozen"),
        ("URFD frame F1", "urfd_frame_f1_frozen"),
        ("URFD F1 (literature)", "urfd_f1_literature"),
        ("Detector FPS", "detector_fps"),
        ("Detector ms", "detector_ms_mean"),
    ]
    lines = [
        "| " + " | ".join(title for title, _ in headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row[key]) for _, key in headers) + " |")
    table = "\n".join(lines)
    (args.reports / "model_comparison.json").write_text(
        json.dumps({"rows": rows, "markdown": table}, indent=2) + "\n", encoding="utf-8"
    )
    (args.reports / "model_comparison.md").write_text(
        "# Model comparison\n\nURFD columns use each model's thresholds frozen on CAUCAFall; "
        "'literature' uses the draft's default thresholds. Event level = 'fall detected'.\n\n"
        + table
        + "\n",
        encoding="utf-8",
    )
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
