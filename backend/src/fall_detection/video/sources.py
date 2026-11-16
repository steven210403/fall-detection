"""Video source definition and validation.

Media paths are restricted to authorized directories, every source is validated
before FFmpeg is launched, and credentials are never written to configuration
files or logs. This module is the single place where a configured source is
turned into an FFmpeg input, so those rules cannot be bypassed by a caller.

Three source kinds are supported (KF-01):

* :class:`Mp4Source` — a file path that must resolve inside a configured media
  root. Rejecting paths outside the roots replaces the need for a shell-escaping
  strategy: FFmpeg is always invoked with an argument list and ``shell=False``.
* :class:`RtspSource` — a URL whose credentials are supplied at runtime. The
  committed reference is credential-free; the credential-bearing URL is stored
  separately and never exposed through :meth:`RtspSource.describe`.
* :class:`WebcamSource` — a local camera, read by FFmpeg through the platform's
  capture demuxer: AVFoundation (macOS), V4L2 (Linux) or DirectShow (Windows).
  The capture size and rate are requested explicitly because a camera has no
  container for FFprobe to read.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from fall_detection.config.settings import WebcamBackend, webcam_backend_for
from fall_detection.core.errors import ConfigurationError, InvalidSourceError
from fall_detection.core.logging import redact_url

__all__ = [
    "Mp4Source",
    "RtspSource",
    "VideoSource",
    "WebcamSource",
    "describe_source",
    "parse_source",
    "resolve_media_path",
]

# Schemes FFmpeg would happily accept but this application must never open:
# ``concat:``, ``subfile:``, ``http:``, and similar would let a crafted source
# reference read arbitrary local files or reach arbitrary hosts.
_ALLOWED_RTSP_SCHEMES = frozenset({"rtsp", "rtsps"})
_MEDIA_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".m4v", ".avi", ".ts"})
_DANGEROUS_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")
# An AVFoundation device is an index or a device name, and a DirectShow device is a
# name. ``:`` separates the video and audio devices in both demuxers' ``-i`` forms
# (``<video>:<audio>``, ``video=<name>:audio=<name>``), so it is refused along with
# control characters; anything else is passed as one argv element.
_WEBCAM_DEVICE_RE = re.compile(r"^[^:\x00-\x1f]{1,128}$")
# A V4L2 device is an index ``N`` (``/dev/videoN``) or a path under ``/dev``.
# ``/dev/v4l/by-path/...`` names contain ``:``, which V4L2 does not treat specially.
_V4L2_DEVICE_RE = re.compile(r"^(?:\d{1,3}|/dev/[^\x00-\x1f]{1,123})$")


class VideoSource(Protocol):
    """A validated input that FFmpeg can open."""

    @property
    def kind(self) -> str:
        """``"mp4"``, ``"rtsp"`` or ``"webcam"``."""
        ...

    @property
    def is_live(self) -> bool:
        """Whether frames arrive in real time (drop-oldest queue) rather than from a file."""
        ...

    def ffmpeg_input_args(self) -> tuple[str, ...]:
        """Input options placed before ``-i`` (demuxer selection, capture size)."""
        ...

    @property
    def safe_ref(self) -> str:
        """Credential-free reference safe for storage and logging."""
        ...

    def ffmpeg_input(self) -> str:
        """Return the exact string passed to FFmpeg as ``-i``.

        May include credentials for RTSP. Never log the return value directly;
        use :meth:`safe_ref` or :func:`describe_source`.
        """
        ...

    def describe(self) -> dict[str, object]:
        """Return a JSON-serialisable, credential-free description."""
        ...


@dataclass(frozen=True, slots=True)
class Mp4Source:
    """A local media file inside an authorized root."""

    path: Path

    @property
    def kind(self) -> str:
        return "mp4"

    @property
    def is_live(self) -> bool:
        return False

    @property
    def safe_ref(self) -> str:
        return str(self.path)

    def ffmpeg_input_args(self) -> tuple[str, ...]:
        return ()

    def ffmpeg_input(self) -> str:
        return str(self.path)

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "path": str(self.path)}


@dataclass(frozen=True, slots=True)
class RtspSource:
    """An RTSP endpoint with credentials resolved at runtime."""

    url: str
    """The credential-bearing URL. Excluded from every serialised form."""

    @property
    def kind(self) -> str:
        return "rtsp"

    @property
    def is_live(self) -> bool:
        return True

    @property
    def safe_ref(self) -> str:
        return redact_url(self.url)

    def ffmpeg_input_args(self) -> tuple[str, ...]:
        # UDP can silently drop frames and reorder timestamps; TCP is the
        # predictable choice for a single camera.
        return ("-rtsp_transport", "tcp")

    def ffmpeg_input(self) -> str:
        return self.url

    def describe(self) -> dict[str, object]:
        """Return the redacted host, port, and path — never the userinfo."""
        parts = urlsplit(self.url)
        return {
            "kind": self.kind,
            "scheme": parts.scheme,
            "host": parts.hostname,
            "port": parts.port,
            "path": parts.path or "/",
            "has_credentials": parts.username is not None,
        }


@dataclass(frozen=True, slots=True)
class WebcamSource:
    """A local camera, captured through the platform's FFmpeg demuxer.

    Each demuxer stamps every frame with a presentation timestamp from the capture
    clock, so the fall layer measures time from the source, not from frame counts.
    """

    device: str
    """Device as configured: an index or name (AVFoundation), an index or ``/dev``
    path (V4L2), or a name (DirectShow)."""

    width: int
    height: int
    fps: float
    """Requested capture rate. The demuxer refuses a rate the camera does not offer."""

    backend: WebcamBackend = "avfoundation"
    """FFmpeg capture demuxer; :func:`parse_source` picks it from the platform."""

    @property
    def kind(self) -> str:
        return "webcam"

    @property
    def is_live(self) -> bool:
        return True

    @property
    def safe_ref(self) -> str:
        return f"{self.backend}:{self.device}"

    def ffmpeg_input_args(self) -> tuple[str, ...]:
        return (
            "-f",
            self.backend,
            "-framerate",
            f"{self.fps:g}",
            "-video_size",
            f"{self.width}x{self.height}",
        )

    def ffmpeg_input(self) -> str:
        if self.backend == "v4l2":
            return self.device if self.device.startswith("/") else f"/dev/video{self.device}"
        if self.backend == "dshow":
            return f"video={self.device}"
        # ``<video>:none`` opens the camera without a microphone.
        return f"{self.device}:none"

    def describe(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "backend": self.backend,
            "device": self.device,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


def resolve_media_path(path: str | Path, roots: tuple[Path, ...]) -> Path:
    """Resolve ``path`` and assert it lies inside one of ``roots``.

    Args:
        path: Candidate media path, absolute or relative to the process CWD.
        roots: Authorized directories, already expanded and resolved.

    Returns:
        The resolved absolute path.

    Raises:
        InvalidSourceError: If the path is outside every root, is not a regular
            file, or has a non-media suffix.
    """
    if not roots:
        raise ConfigurationError(
            "no media roots are configured, so no local video path may be opened; "
            "set FALL_MEDIA_ROOTS__ROOTS to a list of authorized directories separated by "
            "':' (macOS/Linux) or ';' (Windows)"
        )

    resolved = Path(path).expanduser().resolve()
    if not any(_is_within(resolved, root) for root in roots):
        raise InvalidSourceError(
            "media path is outside every authorized root",
            detail=f"path={resolved} roots={[str(root) for root in roots]}",
        )
    if not resolved.is_file():
        raise InvalidSourceError("media file does not exist", detail=f"path={resolved}")
    if resolved.suffix.lower() not in _MEDIA_SUFFIXES:
        raise InvalidSourceError(
            "media file does not have a supported extension",
            detail=f"suffix={resolved.suffix} allowed={sorted(_MEDIA_SUFFIXES)}",
        )
    return resolved


def _is_within(candidate: Path, root: Path) -> bool:
    """Return whether ``candidate`` is ``root`` or below it."""
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def parse_source(
    *,
    mode: str,
    source_ref: str | None,
    media_roots: tuple[Path, ...] = (),
    rtsp_url: str | None = None,
    webcam_device: str = "0",
    webcam_width: int = 1280,
    webcam_height: int = 720,
    webcam_fps: float = 30.0,
    platform: str = sys.platform,
) -> VideoSource:
    """Validate configuration and return a usable source.

    Args:
        mode: ``"mp4"``, ``"rtsp"`` or ``"webcam"``.
        source_ref: Committed reference. An MP4 path, or an RTSP host/stream
            identifier when ``rtsp_url`` is supplied. Unused for a webcam.
        media_roots: Authorized directories for MP4 sources.
        rtsp_url: Runtime credential-bearing URL for RTSP mode.
        webcam_device: Webcam device for the platform's demuxer (see
            :attr:`fall_detection.config.VideoSettings.webcam_device`).
        webcam_width: Requested capture width for webcam mode.
        webcam_height: Requested capture height for webcam mode.
        webcam_fps: Requested capture rate for webcam mode.
        platform: ``sys.platform``; injectable so every platform's webcam demuxer is
            testable on one host.

    Raises:
        ConfigurationError: The combination of arguments cannot identify a source.
        InvalidSourceError: The source is identifiable but unusable.
    """
    if mode == "mp4":
        if not source_ref:
            raise ConfigurationError(
                "video.mode is 'mp4' but video.source_ref is unset; "
                "set FALL_VIDEO__SOURCE_REF to a path inside a media root"
            )
        if _looks_like_scheme(source_ref):
            # A URL passed as an MP4 reference would make FFmpeg open the network
            # (or a local file through ``file:``). Refuse rather than guess.
            raise InvalidSourceError(
                "source_ref looks like a URL; use video.mode 'rtsp' for network sources",
                detail=redact_url(source_ref),
            )
        return Mp4Source(path=resolve_media_path(source_ref, media_roots))

    if mode == "rtsp":
        url = rtsp_url or source_ref
        if not url:
            raise ConfigurationError(
                "video.mode is 'rtsp' but no RTSP URL is available; "
                "set FALL_RTSP_URL in the environment"
            )
        parts = urlsplit(url)
        if parts.scheme.lower() not in _ALLOWED_RTSP_SCHEMES:
            raise InvalidSourceError(
                "RTSP source must use the rtsp:// or rtsps:// scheme",
                detail=f"scheme={parts.scheme!r}",
            )
        if not parts.hostname:
            raise InvalidSourceError("RTSP source has no host", detail=redact_url(url))
        return RtspSource(url=url)

    if mode == "webcam":
        backend = webcam_backend_for(platform)
        if backend is None:
            raise InvalidSourceError(
                "webcam mode supports macOS (AVFoundation), Linux (V4L2) and "
                "Windows (DirectShow) only",
                detail=f"platform={platform}",
            )
        device = webcam_device.strip()
        _check_webcam_device(backend, device)
        if webcam_width <= 0 or webcam_height <= 0 or webcam_fps <= 0:
            raise InvalidSourceError(
                "webcam capture size and rate must be positive",
                detail=f"size={webcam_width}x{webcam_height} fps={webcam_fps}",
            )
        return WebcamSource(
            device=device,
            width=webcam_width,
            height=webcam_height,
            fps=webcam_fps,
            backend=backend,
        )

    raise ConfigurationError(f"unsupported video mode {mode!r}")


def describe_source(source: VideoSource) -> dict[str, object]:
    """Return a credential-free description suitable for logs and API payloads."""
    return {"kind": source.kind, "safe_ref": source.safe_ref, **source.describe()}


def _check_webcam_device(backend: WebcamBackend, device: str) -> None:
    """Refuse a device string the platform's demuxer cannot open as one camera."""
    if backend == "v4l2":
        if not _V4L2_DEVICE_RE.match(device):
            raise InvalidSourceError(
                "webcam device must be a V4L2 index (0 for /dev/video0) or a /dev path; "
                "`v4l2-ctl --list-devices` lists them",
                detail=f"device={device!r}",
            )
        return
    if not _WEBCAM_DEVICE_RE.match(device):
        raise InvalidSourceError(
            f"webcam device for {backend} must be 1-128 characters without ':' or "
            "control characters",
            detail=f"device={device!r}",
        )
    if backend == "dshow" and device.isdigit():
        # DirectShow opens a camera by name only; an index would be read as a name.
        raise InvalidSourceError(
            "webcam device must be a DirectShow device name on Windows, not an index; "
            "`ffmpeg -list_devices true -f dshow -i dummy` lists them, then set "
            "FALL_VIDEO__WEBCAM_DEVICE to the name",
            detail=f"device={device!r}",
        )


def _looks_like_scheme(value: str) -> bool:
    """Return whether ``value`` starts with something that looks like a URL scheme."""
    return bool(_DANGEROUS_SCHEME_RE.match(value)) and not Path(value).is_absolute()


def wrap_without_credentials(url: str) -> str:
    """Return ``url`` with its userinfo stripped (used when building probes)."""
    parts = urlsplit(url)
    if parts.hostname is None:
        return redact_url(url)
    netloc = parts.hostname
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit(parts._replace(netloc=netloc))
