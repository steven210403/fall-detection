"""Typed application settings.

This is the single configuration path (thesis plan §20). Phase 1 adds the
logging, media-root, and video-ingestion settings Phase 2 consumes; later phases
extend this class rather than introducing a second mechanism.

Nested groups (``video``, ``reconnect``) are sub-models rather than flat fields
so the environment variables read naturally, e.g. ``FALL_VIDEO__MODE=rtsp``
uses the ``__`` delimiter that pydantic-settings defines for nesting.
"""

from __future__ import annotations

import math
import os
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DeviceName = Literal["cpu", "mps", "cuda", "intel:gpu", "intel:cpu"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
VideoMode = Literal["mp4", "rtsp", "webcam"]
"""Declared source modes: a video file, an RTSP stream, or a local webcam read
through the platform's FFmpeg capture device (see :func:`webcam_backend_for`)."""

WebcamBackend = Literal["avfoundation", "v4l2", "dshow"]
"""FFmpeg capture demuxers: AVFoundation (macOS), Video4Linux2 (Linux), DirectShow
(Windows)."""


def webcam_backend_for(platform: str) -> WebcamBackend | None:
    """The FFmpeg capture demuxer for ``platform`` (a ``sys.platform`` value), if any."""
    if platform == "darwin":
        return "avfoundation"
    if platform.startswith("linux"):
        return "v4l2"
    if platform == "win32":
        return "dshow"
    return None


TrackerName = Literal["bytetrack", "botsort"]
"""Trackers implemented by the Phase 4 adapter (thesis plan §11.1).

ByteTrack is the initial primary tracker and BoT-SORT the mandatory comparison.
DeepSORT stays a literature-review item with no implementation, so it is absent
from this vocabulary on purpose.
"""

DetectionClass = Literal["person"]
"""Classes the detector may report.

The fall detector is a YOLO *pose* model, whose vocabulary is the single class
``person`` with 17 COCO keypoints. No other class is evidence for a fall.
"""

DEFAULT_DETECTION_CLASSES: tuple[DetectionClass, ...] = ("person",)

COCO_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
"""The 17 COCO keypoints a YOLO pose model emits, in the model's own order."""

FALL_STATES: tuple[str, ...] = ("unknown", "normal", "falling", "impact", "fallen")
"""States of the per-person fall state machine (THESIS_PLAN §5)."""

FALL_EVENT_END_REASONS: tuple[str, ...] = (
    "recovered",  # the person stood up again after the alert
    "track_lost",  # the tracked identity ended while the alert was active
    "session_ended",  # a reconnect or restart began a new stream session
    "stopped",  # the operator stopped the pipeline or the process shut down
    "max_duration",  # the configured ceiling closed a very long alert
)
"""Why a fall event closed."""


class VideoSettings(BaseModel):
    """Decoding and queue behaviour (thesis plan §9.1, §9.2).

    Live mode keeps a two-frame queue and drops the oldest frame under load;
    offline mode never drops samples because the machine is slow. The queue
    capacity fields encode that difference explicitly instead of relying on a
    mode check at every call site.
    """

    mode: VideoMode = "mp4"
    source_ref: str | None = None
    """Committed, credential-free reference: a path under ``media_roots`` for
    MP4, or a host/stream identifier for RTSP. Credentials come from a separate
    environment variable (``FALL_VIDEO__RTSP_URL``) at runtime. Unused for a
    webcam, where it only names the camera if set."""

    webcam_device: str = "0"
    """Camera for ``mode=webcam``; its form depends on the platform:

    * macOS (AVFoundation): an index or a device name;
      ``ffmpeg -f avfoundation -list_devices true -i ""`` lists them, and ``0`` is
      normally the built-in camera.
    * Linux (V4L2): an index ``N`` for ``/dev/videoN``, or a ``/dev/...`` path;
      ``v4l2-ctl --list-devices`` lists them.
    * Windows (DirectShow): the device *name*, never an index;
      ``ffmpeg -list_devices true -f dshow -i dummy`` lists them."""

    webcam_width: int = Field(default=1280, gt=0)
    webcam_height: int = Field(default=720, gt=0)
    """Requested capture size. The camera must offer it; frames are scaled to it."""

    webcam_fps: float = Field(default=30.0, gt=0)
    """Requested capture rate. The capture device refuses a rate the camera does
    not offer; the built-in MacBook camera offers 30."""

    fallback_fps: float = Field(default=25.0, gt=0)
    """Frames per second assumed only when neither the container nor the
    requested rate is known. Loop speed is never used to derive time."""

    read_timeout_s: float = Field(default=15.0, gt=0)
    """Per-frame read timeout. Exceeding it raises ``DecoderStallError`` so a
    hung decoder reconnects instead of blocking the pipeline forever (§22)."""

    live_queue_capacity: int = Field(default=2, ge=2, le=16)
    offline_queue_capacity: int = Field(default=64, ge=4, le=4096)

    reconnect_backoff_s: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 30.0)
    """Backoff schedule in seconds (thesis plan §9.2: 1, 2, 4, 8, up to 30)."""

    reconnect_jitter_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    """Random jitter added to each backoff delay, as a fraction of the delay."""

    max_reconnect_attempts: int = Field(default=0, ge=0)
    """``0`` retries forever (a camera may return days later); a positive value
    caps attempts and then reports ``offline``."""

    @field_validator("reconnect_backoff_s")
    @classmethod
    def _backoff_is_ascending(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("reconnect_backoff_s must not be empty")
        if any(delay <= 0 for delay in value):
            raise ValueError("reconnect_backoff_s entries must be positive")
        if list(value) != sorted(value):
            raise ValueError("reconnect_backoff_s must be ascending")
        return value


class MediaRootsSettings(BaseModel):
    """Filesystem allow-lists (thesis plan §24.2).

    A source path must resolve inside one of these roots, which prevents the API
    from reading arbitrary files. ``Path.resolve()`` is used so ``..`` and
    symlinks cannot escape a root.
    """

    # ``NoDecode`` stops pydantic-settings from trying to JSON-parse the raw
    # environment string before the validator runs; without it a
    # path-list ``FALL_MEDIA_ROOTS__ROOTS`` value fails with a SettingsError
    # instead of reaching ``_coerce_roots``.
    roots: Annotated[tuple[Path, ...], NoDecode] = ()
    """Empty means *no* MP4 path is allowed. The operator must opt in by listing
    authorized directories; this fails closed rather than open."""

    @field_validator("roots", mode="before")
    @classmethod
    def _coerce_roots(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, str):
            # ``os.pathsep``, like ``PATH``: ``:`` on macOS/Linux, ``;`` on Windows,
            # where ``:`` belongs to the drive letter (``C:\\videos``).
            return tuple(part for part in value.split(os.pathsep) if part)
        return value


class DetectionSettings(BaseModel):
    """YOLO detector configuration (thesis plan §10).

    ``enabled`` is ``False`` by default so importing this project never triggers
    a model download or a multi-second weight load. Operationally the detector
    is on (``.env.example`` sets ``FALL_DETECTION__ENABLED=true``); leaving it
    off is what lets the Phase 2 ingestion tests keep running with no weights.

    ``model_path`` points at the Ultralytics weight file. The Phase 0 smoke test
    already populated ``models/phase0``; Phase 8 fine-tuning will replace these
    weights, and the resolved file's SHA-256 is reported in every detection
    status payload so a run can be traced to exact weights (thesis plan §8).
    """

    enabled: bool = False
    model_path: Path = Path("models/yolov8n-pose.pt")
    """YOLOv8n-pose: the thesis's primary model (THESIS_PLAN §3). YOLO11n-pose and
    YOLO26n-pose are the comparison models, selected by pointing this path at their
    weights. The model must be a *pose* model, because the fall features need the
    17 body keypoints."""

    imgsz: int = Field(default=640, ge=320, le=1280)
    """Inference size. 640 is the baseline; 960 is the small-object alternative
    and 1280 is allowed only if the audit shows it is needed (§10.2)."""

    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    """Minimum detection confidence handed to the tracker. Reported as a
    configuration value, never presented as an accuracy result."""

    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    """NMS IoU threshold."""

    max_detections: int = Field(default=64, ge=1, le=1024)
    """Upper bound on detections per frame; rooms hold few people and objects."""

    classes: Annotated[tuple[DetectionClass, ...], NoDecode] = DEFAULT_DETECTION_CLASSES
    """Reported classes. A pose model only knows ``person``."""

    warmup_runs: int = Field(default=1, ge=0, le=8)
    """Forward passes on a synthetic frame before the pipeline is declared
    ready, which materialises the MPS kernels and avoids a slow first frame
    (thesis plan §9.2)."""

    half: bool = False
    """Half precision is off by default: MPS support is not guaranteed and the
    initial decision is eager FP32 (thesis plan §27)."""

    @field_validator("classes", mode="before")
    @classmethod
    def _coerce_classes(cls, value: object) -> object:
        # ``NoDecode`` keeps pydantic-settings from JSON-parsing the env value, so
        # a colon-separated ``FALL_DETECTION__CLASSES=person:laptop`` arrives here
        # as a plain string. Class names contain spaces but no colon, so a colon
        # is an unambiguous delimiter.
        if value is None:
            return DEFAULT_DETECTION_CLASSES
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(":") if part.strip())
        return value

    @field_validator("classes")
    @classmethod
    def _classes_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("detection.classes must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("detection.classes must not contain duplicates")
        return value


class TrackingSettings(BaseModel):
    """Multi-object tracking configuration (thesis plan §11).

    §11.1 requires the tracker to be selected **explicitly** because Ultralytics
    defaults can change between releases; every value the tracker actually runs
    with is therefore written into this group and recorded in the status payload
    rather than inherited from a packaged YAML file.

    §11.2 fixes the lifecycle rules these fields encode: a track survives a gap
    of at most ``loss_tolerance_s`` (initially two seconds), its last observation
    becomes stale after ``stale_after_s`` (0.5 seconds), and a person who returns
    after termination receives a new track UUID.

    BoT-SORT is configured the way §11.1 prescribes for a **fixed camera**: GMC
    is disabled (there is no camera motion to compensate) and ReID stays off, so
    a BoT-SORT result can never be described as evidence of a ReID benefit.
    """

    enabled: bool = True
    """Tracking needs no weights and no warmup, so it is on by default; the
    switch exists for a detection-only ablation run."""

    tracker: TrackerName = "bytetrack"
    """§11.1: ByteTrack is the initial primary tracker, BoT-SORT the mandatory
    comparison. Both are implemented by the same adapter."""

    loss_tolerance_s: float = Field(default=3.0, gt=0.0, le=60.0)
    """§11.2: initial loss tolerance, validated against the measured update rate
    at runtime because a frame count is not a duration in seconds (§9.1)."""

    stale_after_s: float = Field(default=0.5, gt=0.0, le=60.0)
    """§11.2: after this long without an observation, the last predicted state is
    stale and must not be used as activity evidence."""

    assumed_update_rate_hz: float = Field(default=15.0, gt=0.0, le=120.0)
    """§8 detection/tracking sampling target, used only to convert
    ``loss_tolerance_s`` into the tracker's initial frame buffer. The measured
    rate replaces it at runtime, so a wrong assumption cannot silently shorten
    the tolerance."""

    track_high_thresh: float = Field(default=0.25, ge=0.0, le=1.0)
    """First-stage association threshold. Matching the detector's ``conf`` keeps
    the tracker and the detector looking at the same evidence."""

    track_low_thresh: float = Field(default=0.10, ge=0.0, le=1.0)
    """Second-stage threshold for low-score recovery (§11.1: ByteTrack associates
    high *and* low confidence detections)."""

    new_track_thresh: float = Field(default=0.25, ge=0.0, le=1.0)
    """A detection below this score does not start a new track, which suppresses
    one-frame false positives."""

    match_thresh: float = Field(default=0.8, gt=0.0, le=1.0)
    """Association similarity threshold."""

    fuse_score: bool = True
    """Fuse detection score into the association cost, as ByteTrack does."""

    with_reid: bool = False
    """§11.1 defers ReID to an extra experiment; enabling it here is rejected
    rather than silently ignored, so a run can never be mislabelled as a ReID
    comparison."""

    max_buffer_frames: int = Field(default=120, ge=1, le=4096)
    """Hard ceiling for the frame buffer derived from the measured rate, so a
    fast stream cannot grow the tracker's lost-track pool without bound."""

    @property
    def buffer_frames(self) -> int:
        """Initial ``track_buffer`` in frames, converted from the tolerance.

        §9.1 requires this conversion to be explicit: the Ultralytics tracker
        counts *frames*, so ``ceil(loss_tolerance_s × assumed_update_rate_hz)``
        is the frame count that corresponds to the configured duration at the
        assumed rate. The adapter re-derives it from the measured rate and
        reports the effective duration in seconds.
        """
        return max(1, math.ceil(self.loss_tolerance_s * self.assumed_update_rate_hz))

    @model_validator(mode="after")
    def _check_thresholds_and_windows(self) -> TrackingSettings:
        if self.track_low_thresh >= self.track_high_thresh:
            raise ValueError("track_low_thresh must be below track_high_thresh")
        if self.new_track_thresh < self.track_low_thresh:
            raise ValueError("new_track_thresh must be at least track_low_thresh")
        if self.stale_after_s > self.loss_tolerance_s:
            # Staleness marks "stop trusting the last observation"; the loss
            # tolerance ends the track. The stale window can only be shorter.
            raise ValueError("stale_after_s must not exceed loss_tolerance_s")
        if self.with_reid:
            raise ValueError(
                "with_reid is deferred by thesis plan §11.1; BoT-SORT runs "
                "without ReID so its results cannot be reported as a ReID benefit"
            )
        return self


class FallSettings(BaseModel):
    """Fall-detection features and state machine (THESIS_PLAN §4–§5).

    Every threshold is a configuration value. The defaults are the literature
    values the draft cites (AR >= 1.0, upright torso > 45 deg, lying < 30 deg,
    a 3 s long-lie confirmation); ``calibration_path`` replaces them with the
    thresholds frozen on the calibration dataset, so a run reports which set it used.
    """

    enabled: bool = True

    keypoint_min_conf: float = Field(default=0.3, ge=0.0, le=1.0)
    """A keypoint below this confidence is treated as not visible."""

    ar_threshold: float = Field(default=1.0, gt=0.0, le=10.0)
    """Bounding-box aspect ratio W/H at or above which the body is horizontal."""

    angle_lying_deg: float = Field(default=30.0, ge=0.0, le=90.0)
    """Torso angle to the horizontal below which the torso is lying."""

    angle_upright_deg: float = Field(default=45.0, ge=0.0, le=90.0)
    """Torso angle above which the person counts as upright (recovered)."""

    vy_threshold: float = Field(default=1.0, gt=0.0, le=20.0)
    """Downward hip velocity, in body heights per second, that starts a fall."""

    impact_window_s: float = Field(default=1.0, gt=0.0, le=10.0)
    """How long after the velocity spike a lying posture may still confirm impact."""

    long_lie_s: float = Field(default=3.0, gt=0.0, le=120.0)
    """How long the lying posture must persist before the alarm fires."""

    require_both: bool = False
    """Impact needs AR *and* torso angle (True) or either one (False)."""

    recovery_height_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    """A recovery also needs the box height back to this fraction of the person's
    standing height. Without it, a person lying along the camera axis — whose torso looks
    vertical in the image — would count as standing up and cancel their own alarm."""

    standing_window_s: float = Field(default=10.0, gt=0.0, le=120.0)
    """How far back the standing-height reference looks."""

    recovery_hold_s: float = Field(default=0.5, ge=0.0, le=10.0)
    """An upright posture must persist this long to count as a recovery, so one
    noisy keypoint frame cannot cancel a fall or close an alarm."""

    handover_s: float = Field(default=1.0, ge=0.0, le=10.0)
    """A new track that appears within this time of losing a person nearby inherits that
    person's fall state (0 disables the handover)."""

    handover_distance: float = Field(default=1.0, ge=0.0, le=5.0)
    """How far, in body lengths, the new track may be from the lost person's last box."""

    velocity_window: int = Field(default=3, ge=1, le=15)
    """Median filter length, in samples, applied to the hip velocity."""

    max_velocity_gap_s: float = Field(default=0.5, gt=0.0, le=10.0)
    """A velocity is not computed across a longer gap between two observations."""

    max_event_s: float = Field(default=3600.0, gt=0.0, le=86_400.0)
    """An alert open longer than this is closed with ``max_duration``."""

    calibration_path: Path | None = None
    """Optional frozen calibration JSON whose ``thresholds`` override the defaults."""

    @model_validator(mode="after")
    def _check_angles(self) -> FallSettings:
        if self.angle_lying_deg >= self.angle_upright_deg:
            raise ValueError("angle_lying_deg must be below angle_upright_deg")
        return self

    def with_calibration(self) -> FallSettings:
        """Return a copy with the frozen calibration applied, when one is configured."""
        if self.calibration_path is None:
            return self
        import json

        path = self.calibration_path
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).resolve().parents[4] / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        thresholds = payload.get("thresholds", payload)
        allowed = {
            "ar_threshold",
            "angle_lying_deg",
            "angle_upright_deg",
            "vy_threshold",
            "impact_window_s",
            "long_lie_s",
            "require_both",
            "recovery_hold_s",
            "recovery_height_ratio",
        }
        update = {key: value for key, value in thresholds.items() if key in allowed}
        return self.model_validate({**self.model_dump(), **update, "calibration_path": None})


class RetentionSettings(BaseModel):
    """FR-12's data-retention policy (§19, §24.1).

    §24.1 proposes "30 days for application events" and leaves research data to the
    institutional protocol. FR-12's acceptance criterion is "Expired events are deleted;
    aggregate results follow research policy", so this model covers exactly the first
    half and states the second.

    The switch is **off by default**. Phase 6's ``delete_before`` existed and was
    deliberately never called: a policy that deleted history before the dataset phases
    finished would have been data loss rather than cleanup. Turning it on is an operator
    decision, and an enabled policy must name a window.
    """

    enabled: bool = False
    """Whether expired events are deleted. Off unless an operator asks for it."""

    event_days: int | None = Field(default=30, ge=0, le=3650)
    """Days an event is kept after it ended. ``None`` keeps everything.

    ``0`` deletes every closed event on the next pass, which is only useful in a test.
    """

    backup_dir: Path | None = None
    """Where ``scripts/phase12_retention.py --backup`` writes copies.

    ``None`` means the caller must pass a destination explicitly. Nothing in the
    application writes a backup on its own: §19's rule is that a backup is taken
    deliberately with the SQLite backup API, not that the process snapshots itself.
    """

    @model_validator(mode="after")
    def _require_window_when_enabled(self) -> RetentionSettings:
        if self.enabled and self.event_days is None:
            raise ValueError(
                "retention.enabled requires retention.event_days; an enabled policy with no "
                "window would either delete everything or nothing depending on how it is read"
            )
        return self


class StreamingSettings(BaseModel):
    """Preview encoding and client backpressure (thesis plan §8, §17.2, §18.2).

    §8 caps the preview at 5 FPS and 960x540, §18.2 chooses MJPEG for the MVP, and
    §17.2 fixes what happens to a slow client: "Slow clients receive the latest
    snapshot, not a backlog." The client queue is therefore bounded and drops its
    oldest entry, and the drop count is reported so the loss is visible rather than
    silent.
    """

    enabled: bool = True

    preview_fps: float = Field(default=5.0, gt=0.0, le=60.0)
    """§8: "Preview: 5 FPS, initially capped at 960x540"."""

    preview_max_width: int = Field(default=960, ge=16, le=7680)
    preview_max_height: int = Field(default=540, ge=16, le=4320)

    jpeg_quality: int = Field(default=80, ge=1, le=100)
    """JPEG quality. §18.2 gives the bandwidth arithmetic
    (``bytes x FPS x 8 / 1e6``), so the measured size is reported in the status
    payload rather than assumed."""

    client_queue_capacity: int = Field(default=2, ge=1, le=64)
    """Messages a client may have pending before the oldest is dropped.

    Two is deliberate: one message can be in flight while the next waits, and a
    client that falls further behind loses the *oldest* message instead of growing
    a backlog. A dropped event is recoverable through the event cursor (§17.2), and
    the registry reports how many were dropped.
    """

    snapshot_interval_s: float = Field(default=0.5, gt=0.0, le=10.0)
    """§17.2: "Send track snapshots at 2 Hz; send events/status immediately."

    Only the track snapshot is throttled; events and status changes are published
    the moment they happen.
    """

    max_clients: int = Field(default=8, ge=1, le=256)
    """Upper bound on concurrent WebSocket clients, so a runaway browser cannot
    grow the process without limit."""


class Settings(BaseSettings):
    """Environment-driven settings, prefixed with ``FALL_``."""

    app_name: str = "fall-detection"
    environment: Literal["development", "test", "production"] = "development"

    log_level: LogLevel = "INFO"

    device: DeviceName = "cpu"
    """Requested inference device. ``cpu`` is always available.

    ``intel:gpu`` and ``intel:cpu`` run an OpenVINO export of the configured weights
    (Intel integrated graphics such as Iris Xe, or the OpenVINO CPU runtime). The
    export is created once next to the weights. An unavailable device falls back to
    PyTorch on ``cpu``, and the detector reports the device it actually used."""

    database_url: str = "sqlite+pysqlite:///./data/fall_detection.db"
    sqlite_busy_timeout_ms: int = Field(default=5_000, ge=0)
    sqlite_wal: bool = True

    camera_id: str | None = None
    """Explicit camera UUID, when the operator wants to fix one.

    Without it the camera is derived deterministically from the configured source
    reference (see :meth:`camera_uuid`), so restarting the process continues the
    same camera's history instead of creating a new one per run.
    """

    video: VideoSettings = Field(default_factory=VideoSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    media_roots: MediaRootsSettings = Field(default_factory=MediaRootsSettings)
    fall: FallSettings = Field(default_factory=FallSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)
    streaming: StreamingSettings = Field(default_factory=StreamingSettings)

    rtsp_url: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    """Runtime-only credential-bearing RTSP URL.

    It is read from the environment (``FALL_RTSP_URL``) and never written to
    configuration files or logs; the video layer redacts it before use
    (thesis plan §24.2). ``exclude=True`` keeps it out of ``model_dump()`` and
    ``repr=False`` keeps it out of tracebacks and debug output.
    """

    model_config = SettingsConfigDict(
        env_prefix="FALL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    @model_validator(mode="after")
    def _require_source_for_video(self) -> Settings:
        if self.video.mode == "rtsp" and not self.rtsp_url:
            # A missing RTSP URL is a configuration error, not a camera failure:
            # the process cannot start the pipeline at all. Recorded here so the
            # failure is reported once, at startup, instead of per reconnect.
            raise ValueError(
                "video.mode is 'rtsp' but FALL_RTSP_URL is unset; "
                "set the environment variable or switch video.mode to 'mp4'"
            )
        return self

    @property
    def database_path(self) -> Path | None:
        """Filesystem path for a SQLite URL, otherwise ``None``."""
        prefix = "sqlite+pysqlite:///"
        if self.database_url.startswith(prefix):
            return Path(self.database_url[len(prefix) :])
        return None

    @property
    def media_root_paths(self) -> tuple[Path, ...]:
        """Media roots as absolute, resolved paths."""
        return tuple(root.expanduser().resolve() for root in self.media_roots.roots)

    @property
    def preview_fps(self) -> float:
        """Preview frame rate (§8: 5 FPS).

        Kept as a read-only view of ``streaming.preview_fps`` rather than a second
        field, so the preview rate has exactly one source of truth.
        """
        return self.streaming.preview_fps

    @property
    def preview_max_width(self) -> int:
        """Preview width cap (§8: 960x540)."""
        return self.streaming.preview_max_width

    @property
    def preview_max_height(self) -> int:
        """Preview height cap (§8: 960x540)."""
        return self.streaming.preview_max_height

    def camera_uuid(self) -> uuid.UUID:
        """The camera identity this process runs.

        An explicit ``FALL_CAMERA_ID`` wins. Otherwise the UUID is *derived* from
        the configured source reference with UUID5, which makes it stable across
        restarts: the MVP has one fixed camera (thesis plan §6), and a random
        per-process UUID would put every run in a separate camera's history and
        break FR-08's timeline after a restart.
        """
        if self.camera_id:
            return uuid.UUID(self.camera_id)
        return uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"fall-detection:{self.video.mode}:{self._source_identity}",
        )

    @property
    def _source_identity(self) -> str:
        if self.video.source_ref:
            return self.video.source_ref
        if self.video.mode == "webcam":
            # ``avfoundation:0`` on macOS, as before, so an existing camera row keeps
            # its UUID; the same device string means a different camera on another OS.
            backend = webcam_backend_for(sys.platform) or "webcam"
            return f"{backend}:{self.video.webcam_device}"
        return "unconfigured"

    @property
    def camera_name(self) -> str:
        """Human-readable camera name: ``source_ref``, the webcam device, or the UUID."""
        if self.video.source_ref:
            return self.video.source_ref
        if self.video.mode == "webcam":
            return f"Webcam {self.video.webcam_device}"
        return str(self.camera_uuid())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
