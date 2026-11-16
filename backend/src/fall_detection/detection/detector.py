"""Ultralytics YOLO detector wrapper.

Thesis plan §10 selects **YOLO26s** as the primary model and **YOLO26n** as the
lightweight baseline; §9.2 requires that the model is **loaded once and warmed up**
before the pipeline is declared ready, that inference runs under
``torch.inference_mode()``, and that detection batch size is one frame.

This module is the *only* place the ML stack (``ultralytics``, ``torch``,
``numpy``, ``cv2``) is imported, and it imports them lazily inside
:meth:`Detector.load` so that importing ``fall_detection.detection`` — or the
FastAPI application when detection is disabled — stays cheap.

Nothing here reports accuracy. The pilot that would justify any accuracy claim
has not run (see ``docs/FALL_PILOT_REPORT.md``); the only figures produced are
measured stage latencies for *this* machine, which the thesis plan forbids
presenting as FPS targets or results (§8, §15).
"""

from __future__ import annotations

import hashlib
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fall_detection.config import DetectionClass, DetectionSettings, Settings
from fall_detection.core.clock import Clock, SystemClock
from fall_detection.core.contracts import FramePacket
from fall_detection.core.errors import CctvError, ConfigurationError, ModelUnavailableError
from fall_detection.core.logging import get_logger
from fall_detection.detection.classes import class_id_map, resolve_classes
from fall_detection.detection.contracts import Detection, DetectionBatch, DetectionTiming, Keypoint

__all__ = ["Detector", "DetectorInfo", "build_detector", "resolve_model_path"]

logger = get_logger("detection.detector")

_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class DetectorInfo:
    """Immutable description of a loaded detector, for status payloads.

    Every field is safe to serialise and log. ``weights_sha256`` exists because
    thesis plan §8 requires a run to be traceable to exact weights; it is the
    same value the Phase 0 report recorded for the downloaded checkpoint.

    Attributes:
        model_name: Weight file name, e.g. ``yolo26n.pt``.
        model_path: Resolved absolute path of the weight file.
        weights_sha256: SHA-256 of the weight file.
        parameters: Parameter count reported by the model.
        device: The device actually used (``cpu``/``mps``/``cuda``/``intel:gpu``/
            ``intel:cpu``), which differs from the requested device when that one
            is unavailable.
        requested_device: The device asked for in configuration.
        imgsz: Inference size in pixels.
        conf: Confidence threshold handed to the model.
        iou: NMS IoU threshold.
        max_detections: Maximum detections per frame.
        classes: Detection classes this detector reports.
        half: Whether half precision was requested.
        warmup_runs: Warmup forward passes performed at load time.
        loaded_monotonic: Monotonic seconds when loading finished.
        runtime: ``pytorch``, or ``openvino`` when an ``intel:*`` device runs the
            OpenVINO export of the same weights (``weights_sha256`` names the
            ``.pt`` the export was made from).
    """

    model_name: str
    model_path: str
    weights_sha256: str
    parameters: int
    device: str
    requested_device: str
    imgsz: int
    conf: float
    iou: float
    max_detections: int
    classes: tuple[str, ...]
    half: bool
    warmup_runs: int
    loaded_monotonic: float
    runtime: str = "pytorch"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, without the full weight hash."""
        return {
            "model_name": self.model_name,
            "weights_sha256": self.weights_sha256,
            "parameters": self.parameters,
            "device": self.device,
            "runtime": self.runtime,
            "requested_device": self.requested_device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "max_detections": self.max_detections,
            "classes": list(self.classes),
            "half": self.half,
            "warmup_runs": self.warmup_runs,
        }


def resolve_model_path(path: Path) -> Path:
    """Resolve a configured weight path to an existing file.

    A relative path is interpreted against the current working directory first
    and then against the repository root, so the documented default
    (``models/phase0/yolo26n.pt``) works both from ``backend/`` and from the
    repository root without the operator rewriting configuration.

    Raises:
        ModelUnavailableError: No candidate path exists.
    """
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        candidates.append(_REPO_ROOT / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    looked = ", ".join(str(candidate) for candidate in candidates)
    raise ModelUnavailableError(
        f"detector weights not found at {path}; looked in: {looked}. "
        "Run 'uv run --project backend python scripts/phase0_ml_smoke.py' to "
        "download the Phase 0 weights, or set FALL_DETECTION__MODEL_PATH."
    )


def _openvino_device_available(device: str) -> bool:
    """Whether OpenVINO is installed and reports ``device`` (``intel:gpu`` → ``GPU``).

    Ultralytics itself would silently fall back to ``AUTO`` for a missing device,
    which would make the reported device wrong; checking here keeps it truthful.
    """
    try:
        import openvino as ov
    except ImportError:
        return False
    name = device.split(":", 1)[1].upper()
    available = ov.Core().available_devices
    return any(d == name or d.startswith(f"{name}.") for d in available)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Detector:
    """One loaded YOLO model, warmed up and reusable.

    The class owns a single model instance for the whole process (thesis plan
    §9.2/§11: "load each model once"). It is safe to call :meth:`detect` from the
    one inference thread the supervisor uses; a lock additionally protects
    against accidental concurrent use, because the Ultralytics predictor is not
    documented as re-entrant.

    Construct with :func:`build_detector` rather than directly so the configured
    device and class mapping are applied in one place.
    """

    def __init__(
        self,
        settings: DetectionSettings,
        *,
        requested_device: str = "cpu",
        clock: Clock | None = None,
    ) -> None:
        self._settings = settings
        self._requested_device = requested_device
        self._clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._model: Any = None
        self._info: DetectorInfo | None = None
        self._class_indices: list[int] | None = None
        self._raw_to_class: dict[int, DetectionClass] = {}

    # -- lifecycle -----------------------------------------------------

    def load(self) -> DetectorInfo:
        """Load the weights, resolve the class map, and warm up.

        Idempotent: a second call returns the cached info without reloading, so
        a duplicated start cannot load a second model (FR-11).

        Returns:
            The immutable :class:`DetectorInfo` for the loaded model.

        Raises:
            ModelUnavailableError: Weights are missing, unreadable, or expose no
                usable classes; or the requested device is unavailable.
            ConfigurationError: The class configuration is invalid.
        """
        with self._lock:
            if self._info is not None:
                return self._info
            try:
                self._info = self._load_locked()
            except CctvError:
                raise
            except Exception as exc:  # noqa: BLE001 - any unexpected load failure
                # Warmup, hashing, or an Ultralytics/device failure must surface as
                # a typed error so the lifespan can start the process and report
                # the model as degraded instead of crashing (thesis plan §17.2).
                raise ModelUnavailableError(
                    f"detector failed to load from {self._settings.model_path}: {exc}"
                ) from exc
            return self._info

    def _load_locked(self) -> DetectorInfo:
        import torch
        from ultralytics import YOLO

        weights = resolve_model_path(self._settings.model_path)
        device = self._select_device(torch)

        try:
            model = YOLO(str(weights))
        except Exception as exc:  # noqa: BLE001 - Ultralytics raises varied types
            raise ModelUnavailableError(
                f"could not load detector weights from {weights}: {exc}"
            ) from exc

        requested = tuple(self._settings.classes)
        try:
            raw_to_class = resolve_classes(model.names)
            selected = class_id_map(model.names, requested)
        except ModelUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - defensive: vocabulary format drift
            raise ConfigurationError(f"could not resolve detector classes: {exc}") from exc

        self._raw_to_class = raw_to_class
        self._class_indices = sorted(selected)
        reported = tuple(dict.fromkeys(selected[index] for index in self._class_indices))

        # Read from the PyTorch model: an exported model exposes no parameters.
        parameters = int(sum(part.numel() for part in model.model.parameters()))  # type: ignore[union-attr]
        runtime = "pytorch"
        if device.startswith("intel:"):
            model = YOLO(str(self._openvino_export(model, weights)), task=str(model.task))
            runtime = "openvino"

        if self._settings.warmup_runs:
            self._warmup(model, device, torch)

        info = DetectorInfo(
            model_name=weights.name,
            model_path=str(weights),
            weights_sha256=_sha256_file(weights),
            parameters=parameters,
            device=device,
            requested_device=self._requested_device,
            imgsz=self._settings.imgsz,
            conf=self._settings.conf,
            iou=self._settings.iou,
            max_detections=self._settings.max_detections,
            classes=reported,
            half=self._settings.half,
            warmup_runs=self._settings.warmup_runs,
            loaded_monotonic=self._clock.monotonic(),
            runtime=runtime,
        )
        self._model = model
        logger.info(
            "detector loaded model=%s device=%s imgsz=%d classes=%s sha256=%s",
            info.model_name,
            info.device,
            info.imgsz,
            ",".join(info.classes),
            info.weights_sha256[:16],
        )
        if info.device != info.requested_device:
            # §8 portability: CPU is always available, but a silent fallback would
            # make a benchmark claim misleading, so it is logged explicitly.
            logger.warning(
                "requested device %s unavailable; using %s",
                info.requested_device,
                info.device,
            )
        return info

    def _predict_kwargs(self) -> dict[str, Any]:
        """Return the Ultralytics ``predict`` keyword arguments for this config.

        ``half`` is only passed when it is actually enabled: the argument is
        deprecated in the pinned Ultralytics release and emits a warning on every
        call, which would otherwise flood the pipeline log. Half precision stays
        off by default because MPS support is not guaranteed (thesis plan §27).
        """
        kwargs: dict[str, Any] = {
            "verbose": False,
            "imgsz": self._settings.imgsz,
            "conf": self._settings.conf,
            "iou": self._settings.iou,
            "max_det": self._settings.max_detections,
            "classes": self._class_indices,
        }
        if self._settings.half:
            kwargs["half"] = True
        return kwargs

    def _warmup(self, model: Any, device: str, torch: Any) -> None:
        """Run a few forward passes so the first real frame is not a cold start."""
        import numpy as np

        frame = np.zeros((self._settings.imgsz, self._settings.imgsz, 3), dtype=np.uint8)
        with torch.inference_mode():
            for _ in range(self._settings.warmup_runs):
                model.predict(source=frame, device=device, **self._predict_kwargs())

    def _select_device(self, torch: Any) -> str:
        """Pick a usable device, falling back to PyTorch on CPU when it is absent."""
        requested = self._requested_device
        if requested.startswith("intel:"):
            return requested if _openvino_device_available(requested) else "cpu"
        if requested == "mps":
            if not torch.backends.mps.is_available():
                return "cpu"
            return "mps"
        if requested == "cuda":
            if not torch.cuda.is_available():
                return "cpu"
            return "cuda"
        return "cpu"

    def _openvino_export(self, model: Any, weights: Path) -> Path:
        """Return the OpenVINO export of ``weights`` at the configured size.

        The export has a static input shape, so its directory is keyed by ``imgsz``
        and reused across runs. It is written to a temporary name first and moved
        into place, so an interrupted export is never mistaken for a finished one.
        """
        imgsz = self._settings.imgsz
        # Ultralytics recognises an OpenVINO directory by its ``_openvino_model`` suffix.
        target = weights.parent / f"{weights.stem}_{imgsz}_openvino_model"
        if any(target.glob("*.xml")):
            return target
        exported = Path(model.export(format="openvino", imgsz=imgsz, dynamic=False, verbose=False))
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(exported), str(target))
        logger.info("exported %s to OpenVINO at %s", weights.name, target)
        return target

    # -- properties ----------------------------------------------------

    @property
    def info(self) -> DetectorInfo | None:
        """Description of the loaded model, or ``None`` before :meth:`load`."""
        return self._info

    @property
    def loaded(self) -> bool:
        """Whether the weights are loaded and warm."""
        return self._info is not None

    @property
    def device(self) -> str:
        """Device in use, or the requested device before loading."""
        return self._info.device if self._info is not None else self._requested_device

    @property
    def model_name(self) -> str:
        """Weight file name."""
        return self._info.model_name if self._info is not None else self._settings.model_path.name

    # -- inference -----------------------------------------------------

    def detect(self, packet: FramePacket) -> DetectionBatch:
        """Run the detector on one frame and return an immutable batch.

        The batch repeats the packet's identity and timing so no caller has to
        keep the frame array alive to interpret the boxes.

        Args:
            packet: The decoded frame to process.

        Returns:
            A :class:`DetectionBatch` for this exactly one frame.

        Raises:
            ModelUnavailableError: Called before :meth:`load`.
        """
        if self._model is None or self._info is None:
            raise ModelUnavailableError("detector is not loaded; call load() first")

        import torch

        frame = packet.frame
        started = self._clock.monotonic()
        with self._lock, torch.inference_mode():
            results = self._model.predict(
                source=frame,
                device=self._info.device,
                **self._predict_kwargs(),
            )
        finished = self._clock.monotonic()

        detections = self._extract(results, packet)
        timing = self._timing(results, finished - started)
        return DetectionBatch(
            camera_id=packet.camera_id,
            stream_session_id=packet.stream_session_id,
            frame_seq=packet.frame_seq,
            detections=detections,
            frame_shape=packet.shape,
            source_timestamp=packet.source_timestamp,
            receive_monotonic=packet.receive_monotonic,
            receive_utc=packet.receive_utc,
            inference_monotonic=finished,
            timing=timing,
            model_name=self._info.model_name,
        )

    def _extract(self, results: list[Any], packet: FramePacket) -> tuple[Detection, ...]:
        """Convert Ultralytics tensors into clipped, class-mapped detections."""
        if not results:
            return ()
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return ()

        height, width = packet.shape[0], packet.shape[1]
        coordinates = boxes.xyxy.tolist()
        confidences = boxes.conf.tolist()
        class_ids = boxes.cls.tolist()
        keypoint_rows = _keypoint_rows(results[0], len(coordinates))

        detections: list[Detection] = []
        for (x1, y1, x2, y2), confidence, raw_id, keypoints in zip(
            coordinates, confidences, class_ids, keypoint_rows, strict=True
        ):
            class_name = self._raw_to_class.get(int(raw_id))
            if class_name is None:
                continue
            detections.append(
                Detection(
                    bbox=(
                        float(min(max(x1, 0.0), width)),
                        float(min(max(y1, 0.0), height)),
                        float(min(max(x2, 0.0), width)),
                        float(min(max(y2, 0.0), height)),
                    ),
                    confidence=float(confidence),
                    class_name=class_name,
                    class_id=int(raw_id),
                    keypoints=keypoints,
                )
            )
        return tuple(detections)

    def _timing(self, results: list[Any], total_s: float) -> DetectionTiming | None:
        """Read stage timings from the Ultralytics result, when present."""
        if not results:
            return DetectionTiming(
                preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0, total_ms=total_s * 1000
            )
        speed = getattr(results[0], "speed", None) or {}
        return DetectionTiming(
            preprocess_ms=float(speed.get("preprocess", 0.0)),
            inference_ms=float(speed.get("inference", 0.0)),
            postprocess_ms=float(speed.get("postprocess", 0.0)),
            total_ms=total_s * 1000,
        )


def _keypoint_rows(result: Any, count: int) -> list[tuple[Keypoint, ...] | None]:
    """Read each detection's keypoints from a pose result; ``None`` for a box model.

    Ultralytics returns ``keypoints.data`` with shape ``(N, K, 3)`` holding
    ``(x, y, conf)`` in original-frame pixels, in the same order as ``boxes``.
    """
    keypoints = getattr(result, "keypoints", None)
    data = None if keypoints is None else getattr(keypoints, "data", None)
    if data is None:
        return [None] * count
    rows = data.tolist()
    if len(rows) != count:
        return [None] * count
    parsed: list[tuple[Keypoint, ...] | None] = []
    for row in rows:
        points: list[Keypoint] = []
        for point in row:
            x, y = float(point[0]), float(point[1])
            conf = float(point[2]) if len(point) > 2 else 1.0
            points.append((x, y, conf))
        parsed.append(tuple(points))
    return parsed


def build_detector(settings: Settings, *, clock: Clock | None = None) -> Detector:
    """Construct a :class:`Detector` from application settings.

    Does **not** load the model; the lifespan calls :meth:`Detector.load` so a
    load failure is reported as readiness state rather than at import time.
    """
    return Detector(
        settings.detection,
        requested_device=settings.device,
        clock=clock,
    )
