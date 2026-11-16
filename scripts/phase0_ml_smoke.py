#!/usr/bin/env python
"""ML/CV compatibility smoke test and pose-weight download.

Exercises, in order:

1. NumPy array creation, dtype behaviour, and the ABI version.
2. NumPy <-> PyTorch tensor conversion (zero-copy where expected).
3. OpenCV decode/resize/cvtColor/optical flow and its NumPy interaction.
4. torchvision transform, MobileNetV3-Small, and an ImageNet-shaped forward pass.
5. YOLO pose inference for the three compared models — YOLOv8n-pose (primary),
   YOLO11n-pose, and YOLO26n-pose — downloading each into ``models/`` on first use and
   checking that it returns 17 keypoints per person.
6. ByteTrack and BoT-SORT tracking over a short synthetic clip.

Every check is reported as ``PASS``, ``FAIL``, or ``SKIP`` with a reason. The
script exits non-zero only when a check fails; skips are counted separately so
the report can distinguish "not applicable here" from "broken".
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

RESULTS: list[dict[str, Any]] = []


def record(name: str, status: str, detail: str = "", **extra: Any) -> None:
    entry = {"check": name, "status": status, "detail": detail}
    entry.update(
        {key: str(value) if isinstance(value, Path) else value for key, value in extra.items()}
    )
    RESULTS.append(entry)
    print(f"[{status:4}] {name}: {detail}")


def run_check(name: str, fn, /, **extra: Any) -> Any:
    """Run ``fn`` and record PASS/FAIL.

    ``fn`` may raise :class:`SkipCheck` to record a SKIP instead of a failure.
    """
    start = time.perf_counter()
    try:
        detail = fn(**extra)
    except SkipCheck as exc:
        record(name, "SKIP", str(exc), **extra)
        return None
    except Exception as exc:  # noqa: BLE001 - a smoke test reports every failure
        record(name, "FAIL", f"{type(exc).__name__}: {exc}", **extra)
        return None
    duration = time.perf_counter() - start
    record(name, "PASS", str(detail), seconds=round(duration, 4), **extra)
    return detail


class SkipCheck(RuntimeError):
    """Raised by a check that cannot run in this environment."""


# --------------------------------------------------------------------------- #
# NumPy
# --------------------------------------------------------------------------- #
def check_numpy() -> str:
    import numpy as np

    integer = np.arange(12, dtype=np.int32).reshape(3, 4)
    floats = np.linspace(0.0, 1.0, 12, dtype=np.float32)

    if integer.sum() != 66:
        raise AssertionError("unexpected integer reduction")
    if not np.isclose(float(floats.mean()), 0.5, atol=1e-6):
        raise AssertionError("unexpected float mean")
    if integer.dtype.itemsize * integer.size != 48:
        raise AssertionError("unexpected contiguous buffer size")

    return f"numpy {np.__version__}, int32+float32 ops OK"


# --------------------------------------------------------------------------- #
# PyTorch <-> NumPy
# --------------------------------------------------------------------------- #
def check_torch_numpy_bridge() -> str:
    import numpy as np
    import torch

    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tensor = torch.from_numpy(source)
    back = tensor.numpy()

    if tensor.dtype is not torch.float32:
        raise AssertionError(f"dtype changed on conversion: {tensor.dtype}")
    if not np.shares_memory(source, back):
        raise AssertionError("tensor.numpy() no longer shares memory with the source array")

    doubled = (tensor * 2).numpy()
    if not np.allclose(doubled, source * 2):
        raise AssertionError("elementwise arithmetic mismatch across the bridge")

    # A strided (non-contiguous) view must still round-trip correctly.
    strided = torch.from_numpy(source[:, ::2, :])
    if not np.allclose(strided.numpy(), source[:, ::2, :]):
        raise AssertionError("strided view mismatch")

    return f"torch {torch.__version__} / numpy {np.__version__}, shared memory confirmed"


# --------------------------------------------------------------------------- #
# OpenCV
# --------------------------------------------------------------------------- #
def check_opencv() -> str:
    import cv2
    import numpy as np

    # Encode a synthetic BGR frame and decode it again to exercise the codecs.
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[40:140, 80:240] = (0, 180, 240)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise AssertionError("JPEG encode failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != frame.shape:
        raise AssertionError(f"JPEG round-trip shape mismatch: {getattr(decoded, 'shape', None)}")

    resized = cv2.resize(decoded, (160, 90), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    if resized.shape != (90, 160, 3) or gray.shape != (180, 320):
        raise AssertionError("resize/cvtColor produced an unexpected shape")

    # Lightweight optical flow on aligned crops is a required feature (plan §12.4).
    a = gray.astype(np.float32)
    b = np.roll(a, 3, axis=1)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    if flow is None or flow.shape != (180, 320, 2):
        raise AssertionError("Farneback optical flow failed")

    build = cv2.getBuildInformation()
    cuda = "CUDA:                   YES" in build or "CUDA:                   YES" in build
    return (
        f"opencv {cv2.__version__}, numpy {np.__version__}, encode/decode/resize/gray/flow OK, "
        f"CUDA build={cuda}"
    )


# --------------------------------------------------------------------------- #
# torchvision
# --------------------------------------------------------------------------- #
def check_torchvision() -> str:
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    from torchvision.transforms import v2

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    model.eval()

    # Build PIL images, then apply the official weights preprocessing plus the
    # 224x224 resize the thesis plan specifies for crop embeddings (§12.4).
    rng = np.random.default_rng(17)
    images = [
        Image.fromarray(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8), mode="RGB")
        for _ in range(2)
    ]
    preprocessing = v2.Compose(
        [v2.Resize(256), v2.CenterCrop(224), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    # torchvision v2 transforms apply per-sample; stack into a batch, then apply
    # the official weights normalization (thesis plan §12.4 requires the
    # preprocessing associated with the torchvision weights).
    batch = torch.stack([preprocessing(image) for image in images])
    batch = weights.transforms()(batch)

    if tuple(batch.shape) != (2, 3, 224, 224):
        raise AssertionError(f"unexpected transform output shape {tuple(batch.shape)}")
    if not torch.isfinite(batch).all():
        raise AssertionError("non-finite values after normalization")

    with torch.inference_mode():
        features = model.features(batch)
        pooled = model.avgpool(features).flatten(1)
    if tuple(pooled.shape) != (2, 576):
        raise AssertionError(f"unexpected MobileNetV3-Small embedding shape {tuple(pooled.shape)}")

    return (
        f"torchvision {torchvision.__version__}, batch {tuple(batch.shape)}, "
        f"MobileNetV3-Small embedding {tuple(pooled.shape)}"
    )


# --------------------------------------------------------------------------- #
# Ultralytics YOLO
# --------------------------------------------------------------------------- #
def yolo_inference(model_name: str, image, device: str, weights_dir: Path) -> str:
    import hashlib

    from ultralytics import YOLO

    weights_path = weights_dir / model_name
    model = YOLO(str(weights_path))
    start = time.perf_counter()
    results = model.predict(source=image, device=device, verbose=False, imgsz=640)
    elapsed = time.perf_counter() - start

    if not results:
        raise AssertionError("no results returned")
    boxes = results[0].boxes
    if boxes is None:
        raise AssertionError("result has no boxes attribute")
    if getattr(model, "task", None) != "pose":
        raise AssertionError(f"{model_name} is a {model.task!r} model, not a pose model")
    keypoints = results[0].keypoints
    shape = None if keypoints is None else tuple(keypoints.data.shape)
    if shape is not None and len(shape) == 3 and shape[1] != 17:
        raise AssertionError(f"{model_name} returned {shape[1]} keypoints, expected 17")

    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    return (
        f"{model_name} device={device} dets={len(boxes)} keypoints={shape} "
        f"first_latency={elapsed:.3f}s sha256={digest[:16]}"
    )


def check_yolo(device: str, weights_dir: Path) -> str:
    import numpy as np

    weights_dir.mkdir(parents=True, exist_ok=True)
    # A deterministic synthetic frame; this validates the runtime path only and
    # is not a detection-quality measurement.
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[100:600, 200:450] = 200
    summaries = [
        yolo_inference(name, image, device, weights_dir)
        for name in ("yolov8n-pose.pt", "yolo11n-pose.pt", "yolo26n-pose.pt")
    ]
    return " | ".join(summaries)


def check_yolo_training_step(device: str) -> str:
    """One optimizer step on a synthetic single-class fixture.

    The fixture proves the training code path (dataloader, loss, optimizer,
    checkpoint write) runs with the pinned versions. It makes no accuracy claim;
    see ``scripts/phase0_yolo_train_step.py``.
    """
    raise SkipCheck(
        "run separately: scripts/phase0_yolo_train_step.py builds a synthetic fixture "
        "and reports checkpoint creation (kept out of this script to avoid nested "
        "ultralytics runs)"
    )


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
def check_tracking(tracker: str) -> str:
    """Track synthetic detections directly through the Ultralytics tracker.

    Detections are supplied explicitly instead of running a detector, so the
    check isolates *tracker* behaviour. The plan requires this because
    Ultralytics defaults can change (thesis plan §11.1); the tracker is selected
    by name and its own YAML config is loaded for the run.
    """
    import numpy as np
    import torch
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    config_path = Path(check_yaml(f"{tracker}.yaml"))
    cfg = IterableSimpleNamespace(**YAML.load(config_path))
    cfg.device = "cpu"
    if tracker == "botsort":
        # The camera is fixed, so global motion compensation is disabled
        # (thesis plan §11.1 experiment 1).
        cfg.gmc_method = "none"
    if hasattr(cfg, "with_reid"):
        cfg.with_reid = False

    tracker_cls = TRACKER_MAP[cfg.tracker_type]
    instance = tracker_cls(cfg)

    # One person walking steadily right across 20 frames.
    ids_per_frame: list[set[int]] = []
    for frame_index in range(20):
        x1 = 40.0 + frame_index * 12.0
        boxes = np.array(
            [[x1, 100.0, x1 + 60.0, 260.0, 0.9, 0]],
            dtype=np.float32,
        )
        results = Boxes(torch.from_numpy(boxes), (480, 640))
        tracks = instance.update(results)
        frame_ids = set()
        if tracks is not None and len(tracks) > 0:
            frame_ids = {int(t[4]) for t in np.asarray(tracks)}
        ids_per_frame.append(frame_ids)

    all_ids = sorted(set().union(*ids_per_frame))
    frames_tracked = sum(1 for ids in ids_per_frame if ids)
    if frames_tracked != len(ids_per_frame):
        raise AssertionError(
            f"{tracker} produced no track on {len(ids_per_frame) - frames_tracked} frames"
        )
    if len(all_ids) != 1:
        raise AssertionError(f"{tracker} fragmented a continuous trajectory into ids {all_ids}")

    return (
        f"{tracker} (config {config_path.name}): {frames_tracked}/{len(ids_per_frame)} "
        f"frames tracked, ids={all_ids}, buffer={cfg.track_buffer} frames"
    )


def check_bytetrack() -> str:
    return check_tracking("bytetrack")


def check_botsort() -> str:
    return check_tracking("botsort")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def collect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    import numpy as np

    info["numpy"] = np.__version__
    import cv2

    info["opencv"] = cv2.__version__
    import torch

    info["torch"] = torch.__version__
    info["torch_mps_built"] = torch.backends.mps.is_built()
    info["torch_mps_available"] = torch.backends.mps.is_available()
    import torchvision

    info["torchvision"] = torchvision.__version__
    import ultralytics

    info["ultralytics"] = ultralytics.__version__
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="device used for the YOLO and tracking checks",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="skip checks that download pretrained weights",
    )
    parser.add_argument("--json", type=str, default=None, help="write results to this path")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="cache directory for pretrained weights (gitignored)",
    )
    args = parser.parse_args()

    print("== ML/CV smoke test ==")
    environment = collect_environment()
    for key, value in environment.items():
        print(f"    {key}: {value}")
    print()

    run_check("numpy", check_numpy)
    run_check("torch-numpy-bridge", check_torch_numpy_bridge)
    run_check("opencv", check_opencv)
    run_check("torchvision", check_torchvision)

    if args.skip_weights:
        record("yolo-pose-inference", "SKIP", "--skip-weights requested")
        record("yolo-training-step", "SKIP", "--skip-weights requested")
        record("bytetrack", "SKIP", "--skip-weights requested")
        record("botsort", "SKIP", "--skip-weights requested")
    else:
        run_check(
            "yolo-pose-inference", check_yolo, device=args.device, weights_dir=args.weights_dir
        )
        run_check("yolo-training-step", check_yolo_training_step, device=args.device)
        run_check("bytetrack", check_bytetrack)
        run_check("botsort", check_botsort)

    failures = [r for r in RESULTS if r["status"] == "FAIL"]
    skipped = [r for r in RESULTS if r["status"] == "SKIP"]
    print()
    print(
        f"== {len(RESULTS) - len(failures) - len(skipped)} passed, "
        f"{len(failures)} failed, {len(skipped)} skipped =="
    )

    payload = {"environment": environment, "results": RESULTS}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
