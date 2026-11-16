#!/usr/bin/env python
"""Phase 0 YOLO26 training-path smoke check (thesis plan §4.3 item 3).

The plan requires YOLO26n/s *inference* and *one training step*. Inference is
covered by ``scripts/phase0_ml_smoke.py``. This script validates the training
path (dataloader, loss, optimizer, checkpoint write) using a tiny **synthetic**
dataset because no authorized annotated CCTV footage exists yet.

The dataset is deliberately trivial: solid-colour rectangles on a neutral
background with a single ``person``-class box. It proves the training code path
runs on CPU with the pinned PyTorch/Ultralytics versions. It proves **nothing**
about detection accuracy, and its loss values must never be reported as a
result. Phase 8 replaces this with real fine-tuning on audited footage.

The synthetic dataset is written outside version control (``datasets/synthetic``)
and the output directory is a temporary run directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "synthetic"
RUN_ROOT = REPO_ROOT / "experiments" / "runs" / "phase0_yolo_train"


def build_synthetic_dataset(root: Path, *, images: int, size: int) -> Path:
    """Write a minimal single-class YOLO dataset; returns the dataset YAML path."""
    import cv2

    images_dir = root / "images" / "train"
    labels_dir = root / "labels" / "train"
    for directory in (images_dir, labels_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(17)
    for index in range(images):
        frame = np.full((size, size, 3), rng.integers(0, 40), dtype=np.uint8)
        box_w = int(size * 0.25)
        box_h = int(size * 0.55)
        x = int(rng.integers(0, size - box_w))
        y = int(rng.integers(0, size - box_h))
        frame[y : y + box_h, x : x + box_w] = rng.integers(120, 255, 3)

        cv2.imwrite(str(images_dir / f"synth_{index:03d}.jpg"), frame)
        cx = (x + box_w / 2) / size
        cy = (y + box_h / 2) / size
        labels_dir.joinpath(f"synth_{index:03d}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {box_w / size:.6f} {box_h / size:.6f}\n",
            encoding="utf-8",
        )

    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "# Synthetic Phase 0 fixture. Not research data.",
                f"path: {root}",
                "train: images/train",
                "val: images/train",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    yaml_path = build_synthetic_dataset(DATASET_ROOT, images=args.images, size=args.size)
    run_dir = args.out or RUN_ROOT

    model = YOLO(args.weights)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        verbose=False,
        plots=False,
        val=True,
        workers=0,
        # Deterministic and small; this is a path check, not an experiment.
        seed=17,
        deterministic=True,
        pretrained=True,
        # Keep the fixture fast: no augmentation, no early stopping complexity.
        optimizer="SGD",
        lr0=0.001,
        warmup_epochs=0,
        patience=0,
    )

    trainer = getattr(model, "trainer", None)
    metrics = getattr(trainer, "metrics", None)
    save_dir = Path(getattr(trainer, "save_dir", run_dir))
    best_weights = list(save_dir.glob("weights/*.pt"))

    payload = {
        "note": (
            "Synthetic single-class fixture. Validates the training code path only. "
            "Loss/metric values are not results and must not be reported as accuracy."
        ),
        "weights": args.weights,
        "device": args.device,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "dataset": str(yaml_path),
        "save_dir": str(save_dir),
        "checkpoints_written": [str(p) for p in best_weights],
        "metrics_available": bool(metrics),
        "final_loss_names": sorted(metrics.keys()) if isinstance(metrics, dict) else None,
        "results_object": bool(results),
    }
    print(json.dumps(payload, indent=2))

    if not best_weights:
        print("FAIL: no checkpoint written by the training run", file=sys.stderr)
        return 1
    print("PASS: YOLO training path completed and wrote a checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
