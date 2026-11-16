#!/usr/bin/env python
"""Phase 0 FFmpeg decoding validation (thesis plan §4.3 item 8, §9.1).

Validates the decoder path the pipeline will use:

1. ``ffmpeg -version`` and the linked library list are captured verbatim.
2. ``ffprobe`` reports container, stream, codec, resolution, and timestamps.
3. Raw BGR24 frames are piped out of FFmpeg and decoded with NumPy, which is the
   exact subprocess contract described in thesis plan §3.
4. Presentation timestamps (PTS) are read from ``showinfo`` so the pipeline can
   use source time rather than loop speed.
5. An optional RTSP URL is probed with a bounded timeout; when no authorized
   source is configured the check is reported as SKIP, never as a pass.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

RESULTS: list[dict[str, Any]] = []


def record(name: str, status: str, detail: str = "", **extra: Any) -> None:
    RESULTS.append({"check": name, "status": status, "detail": detail, **extra})
    print(f"[{status:4}] {name}: {detail}")


def run(cmd: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` with an argument list and ``shell=False`` (thesis plan §24.2)."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def check_binaries() -> str:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError(f"ffmpeg={ffmpeg!r} ffprobe={ffprobe!r}")

    version = run(["ffmpeg", "-version"]).stdout.splitlines()[0]
    configuration = next(
        (line for line in run(["ffmpeg", "-version"]).stdout.splitlines() if line.startswith("configuration:")),
        "",
    )
    has_videotoolbox = "--enable-videotoolbox" in configuration
    # RTSP is provided by the `rtsp` demuxer (over tcp/udp) rather than a
    # separately listed protocol, so check the demuxer table.
    demuxers = run(["ffmpeg", "-hide_banner", "-demuxers"]).stdout
    has_rtsp = any("rtsp" in line.lower() for line in demuxers.splitlines())
    return (
        f"{version} | ffmpeg={ffmpeg} ffprobe={ffprobe} | "
        f"videotoolbox={has_videotoolbox} rtsp_demuxer={has_rtsp}"
    )


def probe(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def check_container(path: Path) -> str:
    data = probe(path)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    fmt = data["format"]
    summary = {
        "format": fmt["format_name"],
        "duration_s": round(float(fmt["duration"]), 3),
        "size_bytes": int(fmt["size"]),
        "codec": video["codec_name"],
        "profile": video.get("profile"),
        "pix_fmt": video.get("pix_fmt"),
        "width": video["width"],
        "height": video["height"],
        "avg_frame_rate": video.get("avg_frame_rate"),
        "nb_frames": video.get("nb_frames"),
        "bit_rate": fmt.get("bit_rate"),
    }
    return json.dumps(summary, separators=(",", ":"))


def check_rawvideo_pipe(path: Path, width: int, height: int, frames: int = 30) -> str:
    """Decode ``frames`` frames as BGR24 through the subprocess pipe contract."""
    import numpy as np

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-frames:v",
        str(frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())

    frame_bytes = width * height * 3
    if len(result.stdout) < frame_bytes:
        raise RuntimeError(f"short read: {len(result.stdout)} < {frame_bytes}")

    count = len(result.stdout) // frame_bytes
    buffer = np.frombuffer(result.stdout[: count * frame_bytes], dtype=np.uint8)
    frames_array = buffer.reshape(count, height, width, 3)

    if frames_array.shape[1:] != (height, width, 3):
        raise AssertionError(f"unexpected frame shape {frames_array.shape}")
    # A real frame must not be uniformly black; this catches a broken pipe
    # without asserting anything about content or detection quality.
    mean = float(frames_array[0].mean())
    if mean <= 0.0:
        raise AssertionError("decoded frame is uniformly zero")

    return (
        f"{count} frames, shape={frames_array.shape}, dtype={frames_array.dtype}, "
        f"first-frame mean={mean:.2f}"
    )


def check_pts_monotonic(path: Path) -> str:
    """Confirm presentation timestamps are available and monotonic."""
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    stamps = [
        float(value)
        for line in result.stdout.splitlines()
        for value in line.strip().rstrip(",").split(",")
        if value.strip()
    ]
    if len(stamps) < 2:
        raise AssertionError(f"only {len(stamps)} timestamps available")
    if any(b < a for a, b in itertools.pairwise(stamps)):
        raise AssertionError("presentation timestamps are not monotonic")
    if stamps[-1] <= 0.0:
        raise AssertionError("presentation timestamps do not advance")

    return (
        f"{len(stamps)} timestamps, first={stamps[0]:.3f}s last={stamps[-1]:.3f}s, monotonic"
    )


def check_rtsp(url: str | None) -> str:
    if not url:
        raise RTSPNotConfigured(
            "no authorized RTSP URL supplied; set --rtsp-url or FALL_RTSP_URL. "
            "RTSP access must be validated against the institution's camera."
        )

    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-timeout",
            "5000000",
            "-print_format",
            "json",
            "-show_streams",
            url,
        ],
        timeout=20.0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"RTSP probe failed: {result.stderr.strip()[:400]}")

    streams = json.loads(result.stdout)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    return f"rtsp reachable, codec={video['codec_name']} {video['width']}x{video['height']}"


class RTSPNotConfigured(RuntimeError):
    """Raised when no authorized RTSP source is configured."""


def run_check(name: str, fn, /, meta: dict[str, Any] | None = None, **call_kwargs: Any) -> None:
    """Run ``fn(**call_kwargs)``; ``meta`` is attached to the result record only."""
    extra = meta or {}
    try:
        detail = fn(**call_kwargs)
    except RTSPNotConfigured as exc:
        record(name, "SKIP", str(exc), **extra)
    except Exception as exc:  # noqa: BLE001 - a smoke test reports every failure
        record(name, "FAIL", f"{type(exc).__name__}: {exc}", **extra)
    else:
        record(name, "PASS", str(detail), **extra)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="MP4/MOV file to decode")
    parser.add_argument("--rtsp-url", default=None, help="optional authorized RTSP URL")
    parser.add_argument("--json", type=Path, default=None, help="write results to this path")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 2

    print("== Phase 0 FFmpeg decode validation ==")
    print(f"    video: {args.video}")

    run_check("ffmpeg-binaries", check_binaries)
    run_check("container-and-streams", check_container, meta={"video": str(args.video)}, path=args.video)

    data = probe(args.video)
    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    width, height = int(video_stream["width"]), int(video_stream["height"])

    run_check(
        "rawvideo-bgr24-pipe",
        check_rawvideo_pipe,
        meta={"width": width, "height": height},
        path=args.video,
        width=width,
        height=height,
    )
    run_check(
        "presentation-timestamps",
        check_pts_monotonic,
        meta={"video": str(args.video)},
        path=args.video,
    )
    run_check("rtsp-probe", lambda: check_rtsp(args.rtsp_url))

    failures = [r for r in RESULTS if r["status"] == "FAIL"]
    skipped = [r for r in RESULTS if r["status"] == "SKIP"]
    print()
    print(
        f"== {len(RESULTS) - len(failures) - len(skipped)} passed, "
        f"{len(failures)} failed, {len(skipped)} skipped =="
    )

    if args.json:
        args.json.write_text(
            json.dumps({"video": str(args.video), "results": RESULTS}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
