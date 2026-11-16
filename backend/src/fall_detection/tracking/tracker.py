"""Tracker adapters for ByteTrack and BoT-SORT.

Thesis plan §11.1 selects **ByteTrack** as the initial primary tracker and
**BoT-SORT** as the mandatory comparison, and states the implementation rules this
module follows:

* "Select the tracker explicitly. Do not rely on Ultralytics defaults, which can
  change." Every behavioural value therefore comes from
  :class:`~fall_detection.config.TrackingSettings`, and the fully resolved
  configuration is reported in the status payload so a run can be described
  exactly.
* BoT-SORT runs with **GMC disabled** (the camera is fixed, there is no global
  motion to compensate) and **without ReID**, so a BoT-SORT result can never be
  presented as evidence of a ReID benefit. ``settings.with_reid`` is rejected
  rather than ignored.
* §9.1: "Validate ``track_buffer`` conversion and FPS assumptions in the
  Ultralytics implementation; a frame count is not automatically a duration in
  seconds." The adapter converts the configured loss tolerance into frames using
  the **measured** update interval, re-syncs the tracker's buffer when the rate
  changes, and reports ``effective_buffer_s`` — the duration the frame count
  actually covers at the measured rate.

Only this module imports ``ultralytics``/``numpy``, and it does so lazily inside
:meth:`TrackerAdapter.reset` and :meth:`TrackerAdapter.update`, so importing
``fall_detection.tracking`` (or the application with tracking disabled) stays cheap
— the same rule ``detection.detector`` follows.

Nothing here reports tracking accuracy. IDF1, HOTA, and ID-switch counts are
Experiment 2 (§15.2) and need dense trajectory annotation (Phase 7); the numbers
this module produces are configuration values and measured update rates.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fall_detection.config import TrackerName, TrackingSettings
from fall_detection.core.clock import Clock, SystemClock
from fall_detection.core.errors import ConfigurationError, ModelUnavailableError
from fall_detection.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from collections.abc import Sequence

    from fall_detection.detection.contracts import Detection

__all__ = ["RawTrack", "TrackerAdapter", "TrackerInfo", "buffer_frames_for"]

logger = get_logger("tracking.tracker")

#: Bounds for one measured update interval. A single very long stall must not
#: collapse the frame buffer, and a bogus near-zero interval must not inflate it
#: without limit; the configured ``max_buffer_frames`` still caps the result.
_MIN_INTERVAL_S = 1.0 / 240.0
_MAX_INTERVAL_S = 5.0

_SMOOTHING_ALPHA = 0.2
"""Weight of the newest interval in the exponential moving average.

Low enough that one dropped or slow frame does not move the derived frame buffer
by much, high enough to follow a real rate change within a few frames.
"""


def buffer_frames_for(
    loss_tolerance_s: float,
    rate_hz: float,
    *,
    minimum_frames: int,
    maximum_frames: int,
) -> int:
    """Convert a loss tolerance into a tracker frame buffer.

    The Ultralytics trackers count *frames*, so the buffer that corresponds to a
    duration depends on the update rate (§9.1). The result is never below
    ``minimum_frames`` (the value implied by the configured assumed rate) and
    never above ``maximum_frames``, which bounds the tracker's lost-track pool.

    Args:
        loss_tolerance_s: Duration a track may survive without an observation.
        rate_hz: Observed update rate in hertz.
        minimum_frames: Lower bound, so an over-estimated rate cannot shorten the
            tolerance below the configured assumption.
        maximum_frames: Upper bound, so a fast stream cannot grow the pool without
            limit.

    Returns:
        Frame count, at least one.
    """
    if rate_hz <= 0:
        return max(1, minimum_frames)
    frames = math.ceil(loss_tolerance_s * rate_hz)
    return max(1, min(max(frames, minimum_frames), maximum_frames))


@dataclass(frozen=True, slots=True)
class RawTrack:
    """One tracker output row, before identity and lifecycle are applied.

    Thesis plan §11.2: this integer ID is a **temporary association identifier**
    used for display and debugging. It is never an identity; the lifecycle mints
    the UUID that downstream code uses.

    Attributes:
        display_id: Integer tracker ID.
        bbox: ``(x1, y1, x2, y2)`` in original-frame pixels. For a row emitted by
            this adapter the box is the tracker's measurement update from the
            matched detection — a lost track emits no row at all, so a
            prediction can never appear here.
        confidence: Score of the matched detection, read back through the
            detection index rather than from the tracker's own score column.
        class_id: Raw detection class index of the matched detection.
        detection_index: Index of the matched detection in the frame's person
            detection list, which is how the matched evidence is recovered.
    """

    display_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    detection_index: int


@dataclass(frozen=True, slots=True)
class TrackerInfo:
    """Resolved tracker configuration and measured behaviour, for status payloads.

    Attributes:
        name: Configured tracker name (``bytetrack`` or ``botsort``).
        tracker_type: ``tracker_type`` value the Ultralytics tracker reported.
        frames_processed: Tracker updates performed in the current session.
        buffer_frames: Frame buffer currently applied to the tracker.
        configured_buffer_frames: Frame buffer implied by the configured assumed
            update rate, kept so the assumption can be checked against reality.
        buffer_sync_supported: Whether the tracker exposes the frame buffer the
            adapter re-syncs. Reported rather than assumed; when ``False`` the
            time-based lifecycle rules still apply.
        measured_rate_hz: Measured update rate, or ``None`` before the second
            update.
        effective_buffer_s: Duration ``buffer_frames`` covers at the measured
            rate, or ``None`` before it is known.
        gmc_method: Global motion compensation method (``none`` for a fixed
            camera).
        with_reid: Always ``False`` in Phase 4 (§11.1).
        thresholds: The association thresholds actually passed to the tracker.
        reset_count: Number of tracker resets, one per stream session.
    """

    name: TrackerName
    tracker_type: str
    frames_processed: int
    buffer_frames: int
    configured_buffer_frames: int
    buffer_sync_supported: bool
    measured_rate_hz: float | None
    effective_buffer_s: float | None
    gmc_method: str
    with_reid: bool
    thresholds: dict[str, float | bool]
    reset_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for status payloads."""
        return {
            "name": self.name,
            "tracker_type": self.tracker_type,
            "frames_processed": self.frames_processed,
            "buffer_frames": self.buffer_frames,
            "configured_buffer_frames": self.configured_buffer_frames,
            "buffer_sync_supported": self.buffer_sync_supported,
            "measured_rate_hz": (
                round(self.measured_rate_hz, 3) if self.measured_rate_hz is not None else None
            ),
            "effective_buffer_s": (
                round(self.effective_buffer_s, 3) if self.effective_buffer_s is not None else None
            ),
            "gmc_method": self.gmc_method,
            "with_reid": self.with_reid,
            "thresholds": {key: value for key, value in self.thresholds.items()},
            "reset_count": self.reset_count,
        }


class TrackerAdapter:
    """Drive one Ultralytics tracker for one stream session at a time.

    Args:
        settings: Tracking configuration; supplies the tracker name, the loss
            tolerance, the assumed update rate, and every threshold.
        clock: Injectable clock, used only for the fallback interval estimate.
        tracker_name: Optional override, used by the tracker comparison CLI to run
            both trackers against identical detections.
    """

    def __init__(
        self,
        settings: TrackingSettings,
        *,
        clock: Clock | None = None,
        tracker_name: TrackerName | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or SystemClock()
        self._name: TrackerName = tracker_name or settings.tracker
        self._tracker: Any = None
        self._tracker_type = ""
        self._gmc_method = "none"
        self._monotonic_lock = threading.RLock()
        self._frames_processed = 0
        self._reset_count = 0
        self._interval_ema: float | None = None
        self._last_update_monotonic: float | None = None
        self._invalid_rows = 0
        self._buffer_frames = settings.buffer_frames
        self._buffer_sync_supported = True

    # -- properties ----------------------------------------------------

    @property
    def name(self) -> TrackerName:
        """Configured tracker name."""
        return self._name

    @property
    def buffer_frames(self) -> int:
        """Frame buffer currently applied to the tracker."""
        return self._buffer_frames

    @property
    def invalid_rows(self) -> int:
        """Rows the tracker returned that could not be matched to a detection."""
        return self._invalid_rows

    @property
    def info(self) -> TrackerInfo:
        """Resolved configuration and measured behaviour."""
        return TrackerInfo(
            name=self._name,
            tracker_type=self._tracker_type or self._name,
            frames_processed=self._frames_processed,
            buffer_frames=self._buffer_frames,
            configured_buffer_frames=self._settings.buffer_frames,
            buffer_sync_supported=self._buffer_sync_supported,
            measured_rate_hz=self.measured_rate_hz,
            effective_buffer_s=self.effective_buffer_s,
            gmc_method=self._gmc_method,
            with_reid=False,
            thresholds={
                "track_high_thresh": self._settings.track_high_thresh,
                "track_low_thresh": self._settings.track_low_thresh,
                "new_track_thresh": self._settings.new_track_thresh,
                "match_thresh": self._settings.match_thresh,
                "fuse_score": self._settings.fuse_score,
            },
            reset_count=self._reset_count,
        )

    @property
    def measured_rate_hz(self) -> float | None:
        """Measured update rate, or ``None`` until two updates have occurred."""
        with self._monotonic_lock:
            if self._interval_ema is None or self._interval_ema <= 0:
                return None
            return 1.0 / self._interval_ema

    @property
    def effective_buffer_s(self) -> float | None:
        """Duration the frame buffer covers at the measured rate."""
        rate = self.measured_rate_hz
        if rate is None:
            return None
        return self._buffer_frames / rate

    # -- lifecycle -----------------------------------------------------

    def validate(self) -> None:
        """Check that the configured tracker can be built, without building it.

        Called once at startup so a bad tracker name fails immediately instead of
        mid-stream, and so the readiness payload can describe the tracker the
        process would actually use. Construction itself is deferred to
        :meth:`reset` because a tracker instance holds per-session state.

        Raises:
            ConfigurationError: The configured tracker is not available in this
                Ultralytics release.
            ModelUnavailableError: The packaged tracker configuration cannot be
                loaded.
        """
        config = self._build_config()
        from ultralytics.trackers.track import TRACKER_MAP

        if str(getattr(config, "tracker_type", "")) not in TRACKER_MAP:
            raise ConfigurationError(
                f"tracker {self._name!r} is not available in this Ultralytics release; "
                f"available trackers are {sorted(TRACKER_MAP)}"
            )

    def reset(self) -> None:
        """Discard tracker state and build a fresh tracker instance.

        Called at the start of every stream session, so §9.2's "a reconnect
        creates a new stream session and clears old tracks and buffers" is
        enforced at the tracker itself and not only in the lifecycle. Integer IDs
        restart, which is safe because the lifecycle retires terminated IDs and a
        new session has no live tracks.

        Raises:
            ConfigurationError: The configured tracker is unavailable in this
                Ultralytics release.
            ModelUnavailableError: The Ultralytics tracker configuration cannot be
                loaded.
        """
        with self._monotonic_lock:
            config = self._build_config()
            self._tracker = self._construct(config)
            self._tracker_type = str(getattr(config, "tracker_type", self._name))
            self._gmc_method = str(getattr(config, "gmc_method", "none"))
            self._frames_processed = 0
            self._reset_count += 1
            self._interval_ema = None
            self._last_update_monotonic = None
            self._invalid_rows = 0
            self._buffer_frames = self._settings.buffer_frames
            self._sync_buffer_locked()
            logger.info(
                "tracker ready name=%s type=%s buffer_frames=%d assumed_rate=%.2fHz "
                "loss_tolerance=%.2fs gmc=%s reid=%s",
                self._name,
                self._tracker_type,
                self._buffer_frames,
                self._settings.assumed_update_rate_hz,
                self._settings.loss_tolerance_s,
                self._gmc_method,
                False,
            )

    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_shape: tuple[int, int, int],
        now_monotonic: float,
    ) -> tuple[RawTrack, ...]:
        """Advance the tracker by one frame and return its output rows.

        Args:
            detections: Person detections for this frame, in model order. The
                detection index reported in :class:`RawTrack` refers to this
                sequence.
            frame_shape: ``(height, width, channels)`` of the source frame, used
                for the tracker's coordinate space.
            now_monotonic: Monotonic seconds of this update, used to measure the
                update rate that the frame buffer is derived from.

        Returns:
            One :class:`RawTrack` per currently tracked person. A track without a
            matching detection emits no row, so its motion prediction is never
            returned as evidence (thesis plan §11.2).

        Raises:
            ConfigurationError: Called before :meth:`reset`.
        """
        tracker = self._tracker
        if tracker is None:
            raise ConfigurationError("tracker adapter used before reset(); call reset() first")

        import numpy as np
        from ultralytics.engine.results import Boxes

        rows = np.asarray(
            [
                (
                    detection.bbox[0],
                    detection.bbox[1],
                    detection.bbox[2],
                    detection.bbox[3],
                    float(detection.confidence),
                    float(detection.class_id),
                )
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape(-1, 6)

        height, width = int(frame_shape[0]), int(frame_shape[1])
        with self._monotonic_lock:
            self._record_interval_locked(now_monotonic)
            self._sync_buffer_locked()
            self._frames_processed += 1

        output = tracker.update(Boxes(rows, (height, width)), img=None)
        return self._parse(output, detections)

    def describe(self) -> dict[str, Any]:
        """Return the resolved configuration and measurements for status payloads."""
        return self.info.as_dict()

    # -- internals -----------------------------------------------------

    def _build_config(self) -> Any:
        """Build the tracker configuration namespace.

        The packaged YAML is read first so a newer Ultralytics release that adds a
        required key still works; every field whose value affects tracking is then
        overwritten from our settings, so no behavioural default is inherited.
        """
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        try:
            resolved = check_yaml(f"{self._name}.yaml")
            # ``check_yaml`` is typed as returning a path or a list of paths; the
            # packaged tracker configuration is always a single file.
            config_path = resolved[0] if isinstance(resolved, list) else resolved
            loaded = YAML.load(config_path)
            values: dict[str, Any] = dict(loaded) if loaded else {}
        except Exception as exc:  # noqa: BLE001 - any failure means "cannot configure"
            raise ModelUnavailableError(
                f"could not load the packaged {self._name} tracker configuration: {exc}"
            ) from exc

        values.update(
            {
                # Explicit selection (§11.1): never inherit `tracker_type`.
                "tracker_type": self._name,
                "track_high_thresh": self._settings.track_high_thresh,
                "track_low_thresh": self._settings.track_low_thresh,
                "new_track_thresh": self._settings.new_track_thresh,
                "track_buffer": self._settings.buffer_frames,
                "match_thresh": self._settings.match_thresh,
                "fuse_score": self._settings.fuse_score,
            }
        )
        if self._name == "botsort":
            # §11.1 experiments 1 and 2: fixed camera, no GMC, no ReID.
            values["gmc_method"] = "none"
            values["with_reid"] = False
            values["model"] = "auto"
        return IterableSimpleNamespace(**values)

    def _construct(self, config: Any) -> Any:
        """Instantiate the tracker class for this configuration."""
        from ultralytics.trackers.track import TRACKER_MAP

        tracker_cls = TRACKER_MAP.get(str(getattr(config, "tracker_type", "")))
        if tracker_cls is None:
            raise ConfigurationError(
                f"tracker {self._name!r} is not available in this Ultralytics release; "
                f"available trackers are {sorted(TRACKER_MAP)}"
            )
        try:
            return tracker_cls(config)
        except Exception as exc:  # noqa: BLE001 - construction failure must be typed
            raise ConfigurationError(f"could not construct tracker {self._name!r}: {exc}") from exc

    def _record_interval_locked(self, now_monotonic: float) -> None:
        """Update the measured update interval from consecutive updates."""
        previous = self._last_update_monotonic
        self._last_update_monotonic = now_monotonic
        if previous is None:
            return
        interval = now_monotonic - previous
        if interval <= 0:
            return
        clamped = min(max(interval, _MIN_INTERVAL_S), _MAX_INTERVAL_S)
        if self._interval_ema is None:
            self._interval_ema = clamped
            return
        self._interval_ema = (
            _SMOOTHING_ALPHA * clamped + (1.0 - _SMOOTHING_ALPHA) * self._interval_ema
        )

    def _sync_buffer_locked(self) -> None:
        """Re-derive the tracker's frame buffer from the measured update rate.

        §9.1: a frame count is not automatically a duration. The buffer is
        therefore derived from the *measured* rate and never dropped below the
        value implied by the configured assumed rate, so the configured loss
        tolerance stays achievable even if the assumption was optimistic.
        """
        tracker = self._tracker
        if tracker is None:
            return
        rate = self.measured_rate_hz or self._settings.assumed_update_rate_hz
        frames = buffer_frames_for(
            self._settings.loss_tolerance_s,
            rate,
            minimum_frames=self._settings.buffer_frames,
            maximum_frames=self._settings.max_buffer_frames,
        )
        if frames == self._buffer_frames:
            return
        self._buffer_frames = frames
        try:
            tracker.max_frames_lost = frames
        except AttributeError:
            # Reported through `buffer_sync_supported`; the time-based lifecycle
            # rules still terminate tracks on the configured tolerance.
            self._buffer_sync_supported = False
            logger.warning(
                "tracker %s does not expose max_frames_lost; the frame buffer stays at %d",
                self._name,
                self._settings.buffer_frames,
            )
            return
        args = getattr(tracker, "args", None)
        if args is not None and hasattr(args, "track_buffer"):
            # Belt and braces: a release that re-reads `args.track_buffer` every
            # frame should see the same value as the attribute-based path.
            args.track_buffer = frames

    def _parse(self, output: Any, detections: Sequence[Detection]) -> tuple[RawTrack, ...]:
        """Convert tracker output rows into :class:`RawTrack` records."""
        import numpy as np

        rows = np.asarray(output, dtype=np.float32)
        if rows.size == 0:
            return ()
        if rows.ndim != 2 or rows.shape[1] < 8:
            logger.warning(
                "tracker %s returned an unexpected row shape %s; ignoring the frame",
                self._name,
                tuple(rows.shape),
            )
            self._invalid_rows += rows.shape[0] if rows.ndim else 1
            return ()

        tracks: list[RawTrack] = []
        for row in rows:
            index = int(row[7])
            if index < 0 or index >= len(detections):
                # The row cannot be traced back to a detection, so it cannot be
                # used as evidence. Counted rather than raised: one malformed row
                # must not stop the pipeline, and `invalid_rows` makes it visible.
                self._invalid_rows += 1
                logger.warning(
                    "tracker %s returned detection index %d outside the frame's %d detections",
                    self._name,
                    index,
                    len(detections),
                )
                continue
            detection = detections[index]
            tracks.append(
                RawTrack(
                    display_id=int(row[4]),
                    bbox=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                    confidence=float(detection.confidence),
                    class_id=int(detection.class_id),
                    detection_index=index,
                )
            )
        return tuple(tracks)
