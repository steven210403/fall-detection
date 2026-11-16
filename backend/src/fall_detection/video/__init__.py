"""Video ingestion: source validation, FFmpeg decoding, queues, and lifecycle.

Opens one source (a video file, an RTSP stream, or the laptop's built-in webcam),
reports the camera state, and starts and stops without duplicating decoders. Live
sources use a small drop-oldest queue and reconnect with backoff; files never drop
a frame.

Detection, tracking, and feature extraction are deliberately absent — they are
Phase 3 and later. A consumer callback is the seam where the detector attaches.
"""

from fall_detection.video.decoder import DecoderMetadata, FfmpegDecoder, probe_source
from fall_detection.video.pipeline import (
    FrameConsumer,
    PipelineStats,
    PipelineSupervisor,
    build_supervisor,
    wait_for_state,
)
from fall_detection.video.queue import FrameQueue, QueueClosedError, QueueStats
from fall_detection.video.reconnect import ReconnectPolicy
from fall_detection.video.sources import (
    Mp4Source,
    RtspSource,
    VideoSource,
    WebcamSource,
    describe_source,
    parse_source,
    resolve_media_path,
)

__all__ = [
    "DecoderMetadata",
    "FfmpegDecoder",
    "FrameConsumer",
    "FrameQueue",
    "Mp4Source",
    "PipelineStats",
    "PipelineSupervisor",
    "QueueClosedError",
    "QueueStats",
    "ReconnectPolicy",
    "RtspSource",
    "VideoSource",
    "WebcamSource",
    "build_supervisor",
    "describe_source",
    "parse_source",
    "probe_source",
    "resolve_media_path",
    "wait_for_state",
]
