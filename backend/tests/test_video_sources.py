"""Tests for source validation and credential handling (thesis plan §24.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fall_detection.core.errors import ConfigurationError, InvalidSourceError
from fall_detection.video.sources import (
    Mp4Source,
    RtspSource,
    WebcamSource,
    describe_source,
    parse_source,
    resolve_media_path,
)


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "authorized"
    root.mkdir()
    (root / "clip.mp4").write_bytes(b"not really a video")
    return root


class TestResolveMediaPath:
    def test_accepts_a_file_inside_a_root(self, media_root: Path) -> None:
        resolved = resolve_media_path(media_root / "clip.mp4", (media_root,))
        assert resolved == (media_root / "clip.mp4").resolve()

    def test_rejects_paths_outside_every_root(self, media_root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "elsewhere.mp4"
        outside.write_bytes(b"x")
        with pytest.raises(InvalidSourceError, match="outside every authorized root"):
            resolve_media_path(outside, (media_root,))

    def test_rejects_parent_traversal(self, media_root: Path) -> None:
        sneaky = media_root / ".." / "escape.mp4"
        (media_root.parent / "escape.mp4").write_bytes(b"x")
        with pytest.raises(InvalidSourceError, match="outside every authorized root"):
            resolve_media_path(sneaky, (media_root,))

    def test_fails_closed_when_no_root_is_configured(self, media_root: Path) -> None:
        # An empty allow-list must mean "nothing is authorized", not "everything".
        with pytest.raises(ConfigurationError, match="no media roots"):
            resolve_media_path(media_root / "clip.mp4", ())

    def test_rejects_missing_file(self, media_root: Path) -> None:
        with pytest.raises(InvalidSourceError, match="does not exist"):
            resolve_media_path(media_root / "absent.mp4", (media_root,))

    def test_rejects_unsupported_extension(self, media_root: Path) -> None:
        (media_root / "secrets.txt").write_text("x")
        with pytest.raises(InvalidSourceError, match="supported extension"):
            resolve_media_path(media_root / "secrets.txt", (media_root,))


class TestParseSourceMp4:
    def test_returns_an_mp4_source(self, media_root: Path) -> None:
        source = parse_source(
            mode="mp4",
            source_ref=str(media_root / "clip.mp4"),
            media_roots=(media_root,),
        )
        assert isinstance(source, Mp4Source)
        assert source.kind == "mp4"
        assert source.ffmpeg_input() == str((media_root / "clip.mp4").resolve())

    def test_requires_a_source_ref(self) -> None:
        with pytest.raises(ConfigurationError, match="source_ref is unset"):
            parse_source(mode="mp4", source_ref=None)

    def test_rejects_a_url_as_an_mp4_reference(self, media_root: Path) -> None:
        # Otherwise FFmpeg would happily open the network, or ``file:`` would
        # read an arbitrary local path.
        with pytest.raises(InvalidSourceError, match="looks like a URL"):
            parse_source(
                mode="mp4",
                source_ref="http://evil.example/x.mp4",
                media_roots=(media_root,),
            )


class TestParseSourceRtsp:
    def test_returns_an_rtsp_source(self) -> None:
        source = parse_source(
            mode="rtsp",
            source_ref=None,
            rtsp_url="rtsp://user:pw@camera.example:554/stream1",
        )
        assert isinstance(source, RtspSource)
        assert source.kind == "rtsp"

    def test_requires_a_url(self) -> None:
        with pytest.raises(ConfigurationError, match="no RTSP URL"):
            parse_source(mode="rtsp", source_ref=None)

    @pytest.mark.parametrize("url", ["http://cam.example/x", "file:///etc/passwd", "cam.example"])
    def test_rejects_non_rtsp_schemes(self, url: str) -> None:
        with pytest.raises(InvalidSourceError):
            parse_source(mode="rtsp", source_ref=None, rtsp_url=url)

    def test_rejects_a_url_without_a_host(self) -> None:
        with pytest.raises(InvalidSourceError, match="no host"):
            parse_source(mode="rtsp", source_ref=None, rtsp_url="rtsp:///stream")

    def test_uses_source_ref_when_no_runtime_url_is_given(self) -> None:
        source = parse_source(
            mode="rtsp",
            source_ref="rtsp://camera.example/stream1",
        )
        assert source.safe_ref == "rtsp://camera.example/stream1"


class TestCredentialContainment:
    def test_safe_ref_redacts_credentials(self) -> None:
        source = RtspSource(url="rtsp://admin:hunter2@camera.example/s1")
        assert source.safe_ref == "rtsp://***@camera.example/s1"
        assert "hunter2" not in source.safe_ref

    def test_describe_never_exposes_the_password(self) -> None:
        source = RtspSource(url="rtsp://admin:hunter2@camera.example:554/s1")
        payload = describe_source(source)
        rendered = repr(payload)
        assert "hunter2" not in rendered
        assert payload["host"] == "camera.example"
        assert payload["port"] == 554
        assert payload["has_credentials"] is True

    def test_ffmpeg_input_keeps_credentials_for_the_subprocess_only(self) -> None:
        url = "rtsp://admin:hunter2@camera.example/s1"
        assert RtspSource(url=url).ffmpeg_input() == url

    def test_describe_mp4_reports_the_path(self, media_root: Path) -> None:
        source = Mp4Source(path=media_root / "clip.mp4")
        described = describe_source(source)
        assert str(described["path"]).endswith("clip.mp4")


class TestUnsupportedMode:
    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ConfigurationError, match="unsupported video mode"):
            parse_source(mode="usb", source_ref="0")


class TestWebcamSource:
    def test_parses_the_built_in_camera(self) -> None:
        source = parse_source(mode="webcam", source_ref=None, platform="darwin")
        assert source == WebcamSource(device="0", width=1280, height=720, fps=30.0)
        assert source.kind == "webcam"
        assert source.is_live

    def test_ffmpeg_input_selects_avfoundation_without_audio(self) -> None:
        source = WebcamSource(device="0", width=640, height=480, fps=30.0)
        assert source.ffmpeg_input_args() == (
            "-f",
            "avfoundation",
            "-framerate",
            "30",
            "-video_size",
            "640x480",
        )
        assert source.ffmpeg_input() == "0:none"

    def test_accepts_a_device_name(self) -> None:
        source = parse_source(
            mode="webcam", source_ref=None, webcam_device="MacBook Air Camera", platform="darwin"
        )
        assert isinstance(source, WebcamSource)
        assert source.ffmpeg_input() == "MacBook Air Camera:none"

    @pytest.mark.parametrize("device", ["", "0:0", "cam\nera"])
    def test_rejects_unusable_device(self, device: str) -> None:
        with pytest.raises(InvalidSourceError, match="webcam device"):
            parse_source(mode="webcam", source_ref=None, webcam_device=device, platform="darwin")

    def test_rejects_a_platform_without_a_capture_demuxer(self) -> None:
        with pytest.raises(InvalidSourceError, match="macOS .*Linux .*Windows"):
            parse_source(mode="webcam", source_ref=None, platform="freebsd14")

    @pytest.mark.parametrize(
        ("device", "expected_input"),
        [("0", "/dev/video0"), ("2", "/dev/video2"), ("/dev/video4", "/dev/video4")],
    )
    def test_linux_reads_the_camera_through_v4l2(self, device: str, expected_input: str) -> None:
        source = parse_source(
            mode="webcam",
            source_ref=None,
            webcam_device=device,
            webcam_width=640,
            webcam_height=480,
            platform="linux",
        )
        assert isinstance(source, WebcamSource)
        assert source.backend == "v4l2"
        assert source.ffmpeg_input_args() == (
            "-f",
            "v4l2",
            "-framerate",
            "30",
            "-video_size",
            "640x480",
        )
        assert source.ffmpeg_input() == expected_input
        assert source.safe_ref == f"v4l2:{device}"

    def test_linux_accepts_a_by_path_device_with_colons(self) -> None:
        device = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:1:1.0-video-index0"
        source = parse_source(
            mode="webcam", source_ref=None, webcam_device=device, platform="linux"
        )
        assert isinstance(source, WebcamSource)
        assert source.ffmpeg_input() == device

    @pytest.mark.parametrize("device", ["Integrated Camera", "video0", "../video0", "1234"])
    def test_linux_rejects_a_device_that_is_not_an_index_or_dev_path(self, device: str) -> None:
        with pytest.raises(InvalidSourceError, match="V4L2"):
            parse_source(mode="webcam", source_ref=None, webcam_device=device, platform="linux")

    def test_windows_reads_the_camera_through_dshow_by_name(self) -> None:
        source = parse_source(
            mode="webcam",
            source_ref=None,
            webcam_device="Integrated Camera",
            platform="win32",
        )
        assert isinstance(source, WebcamSource)
        assert source.backend == "dshow"
        assert source.ffmpeg_input_args()[:2] == ("-f", "dshow")
        assert source.ffmpeg_input() == "video=Integrated Camera"
        assert describe_source(source)["safe_ref"] == "dshow:Integrated Camera"

    def test_windows_refuses_an_index_because_dshow_opens_cameras_by_name(self) -> None:
        with pytest.raises(InvalidSourceError, match="DirectShow device name"):
            parse_source(mode="webcam", source_ref=None, webcam_device="0", platform="win32")

    @pytest.mark.parametrize("device", ["", "cam:mic", "cam\nera"])
    def test_windows_rejects_unusable_device(self, device: str) -> None:
        with pytest.raises(InvalidSourceError, match="webcam device"):
            parse_source(mode="webcam", source_ref=None, webcam_device=device, platform="win32")

    def test_rejects_non_positive_size(self) -> None:
        with pytest.raises(InvalidSourceError, match="positive"):
            parse_source(mode="webcam", source_ref=None, webcam_width=0, platform="darwin")

    def test_description_names_the_device(self) -> None:
        source = WebcamSource(device="0", width=1280, height=720, fps=30.0)
        description = describe_source(source)
        assert description["kind"] == "webcam"
        assert description["safe_ref"] == "avfoundation:0"
        assert description["backend"] == "avfoundation"

    def test_files_are_not_live(self, media_root: Path) -> None:
        source = parse_source(
            mode="mp4", source_ref=str(media_root / "clip.mp4"), media_roots=(media_root,)
        )
        assert not source.is_live
        assert source.ffmpeg_input_args() == ()
