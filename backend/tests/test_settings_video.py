"""Settings tests for the Phase 1 additions: logging, media roots, and video.

Configuration is environment-driven through one class (thesis plan §20), so
these tests assert the env-var surface an operator will actually use, including
the nested ``__`` delimiter and the fail-closed media allow-list.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from fall_detection.config import Settings
from fall_detection.config.settings import MediaRootsSettings, VideoSettings


class TestDefaults:
    def test_video_defaults(self) -> None:
        video = Settings().video
        assert video.mode == "mp4"
        assert video.source_ref is None
        assert video.live_queue_capacity == 2  # thesis plan §9.1
        assert video.offline_queue_capacity >= 4
        assert video.reconnect_backoff_s == (1.0, 2.0, 4.0, 8.0, 30.0)

    def test_media_roots_fail_closed_by_default(self) -> None:
        assert Settings().media_roots.roots == ()

    def test_log_level_default(self) -> None:
        assert Settings(log_level="WARNING").log_level == "WARNING"

    def test_rtsp_url_defaults_to_none(self) -> None:
        assert Settings().rtsp_url is None


class TestEnvironmentOverrides:
    def test_nested_video_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FALL_VIDEO__MODE", "rtsp")
        monkeypatch.setenv("FALL_RTSP_URL", "rtsp://user:pw@host/s1")
        monkeypatch.setenv("FALL_VIDEO__READ_TIMEOUT_S", "3.5")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.video.mode == "rtsp"
        assert settings.video.read_timeout_s == 3.5

    def test_media_roots_split_on_the_os_path_separator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``;`` on Windows, where ``:`` is part of a drive letter such as ``C:\\videos``.
        monkeypatch.setattr("fall_detection.config.settings.os.pathsep", ";")
        monkeypatch.setenv("FALL_MEDIA_ROOTS__ROOTS", "C:/a/video;D:/b/video")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert [root.as_posix() for root in settings.media_roots.roots] == [
            "C:/a/video",
            "D:/b/video",
        ]

    def test_colon_separated_media_roots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A colon-separated list is the documented macOS/Linux format; it must not
        # be JSON-parsed into a failure.
        monkeypatch.setattr("fall_detection.config.settings.os.pathsep", ":")
        monkeypatch.setenv("FALL_MEDIA_ROOTS__ROOTS", "/a/video:/b/video")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert [str(root) for root in settings.media_roots.roots] == ["/a/video", "/b/video"]

    def test_media_root_paths_are_resolved(self, tmp_path: Path) -> None:
        settings = Settings(
            media_roots=MediaRootsSettings(roots=(tmp_path / "media",)),
            _env_file=None,  # type: ignore[call-arg]
        )
        resolved = settings.media_root_paths
        assert len(resolved) == 1
        assert resolved[0].is_absolute()

    def test_rtsp_url_is_excluded_from_dumps(self) -> None:
        # The credential-bearing URL must never be serialised (§24.2).
        settings = Settings(
            rtsp_url="rtsp://user:pw@host/s1",
            video=VideoSettings(mode="rtsp"),
            _env_file=None,  # type: ignore[call-arg]
        )
        assert "pw" not in settings.model_dump_json()


class TestValidation:
    def test_rtsp_mode_requires_a_url(self) -> None:
        with pytest.raises(ValidationError, match="FALL_RTSP_URL is unset"):
            Settings(video=VideoSettings(mode="rtsp"), _env_file=None)  # type: ignore[call-arg]

    def test_rejects_unknown_log_level(self) -> None:
        with pytest.raises(ValidationError):
            Settings(log_level="LOUD", _env_file=None)  # type: ignore[call-arg]

    def test_rejects_unknown_video_mode(self) -> None:
        with pytest.raises(ValidationError):
            Settings(video={"mode": "usb"}, _env_file=None)  # type: ignore[call-arg]

    def test_webcam_mode_needs_no_source_ref(self) -> None:
        settings = Settings(video={"mode": "webcam"}, _env_file=None)  # type: ignore[call-arg]
        assert settings.video.webcam_device == "0"
        assert (settings.video.webcam_width, settings.video.webcam_height) == (1280, 720)
        assert settings.video.webcam_fps == 30.0
        assert settings.camera_name == "Webcam 0"

    def test_webcam_camera_id_is_stable_and_device_specific(self) -> None:
        first = Settings(video={"mode": "webcam"}, _env_file=None)  # type: ignore[call-arg]
        again = Settings(video={"mode": "webcam"}, _env_file=None)  # type: ignore[call-arg]
        other = Settings(
            video={"mode": "webcam", "webcam_device": "1"},
            _env_file=None,  # type: ignore[call-arg]
        )
        assert first.camera_uuid() == again.camera_uuid()
        assert first.camera_uuid() != other.camera_uuid()

    def test_macos_webcam_keeps_its_avfoundation_camera_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Existing databases recorded the macOS webcam as ``avfoundation:0``; the
        # per-platform demuxer must not give that camera a new UUID.
        monkeypatch.setattr("fall_detection.config.settings.sys.platform", "darwin")
        mac = Settings(video={"mode": "webcam"}, _env_file=None)  # type: ignore[call-arg]
        mac_uuid = mac.camera_uuid()
        assert mac_uuid == uuid.uuid5(uuid.NAMESPACE_URL, "fall-detection:webcam:avfoundation:0")
        monkeypatch.setattr("fall_detection.config.settings.sys.platform", "linux")
        linux = Settings(video={"mode": "webcam"}, _env_file=None)  # type: ignore[call-arg]
        assert linux.camera_uuid() != mac_uuid

    def test_rejects_non_positive_webcam_rate(self) -> None:
        with pytest.raises(ValidationError):
            Settings(video={"mode": "webcam", "webcam_fps": 0}, _env_file=None)  # type: ignore[call-arg]

    def test_rejects_non_ascending_backoff(self) -> None:
        with pytest.raises(ValidationError, match="ascending"):
            Settings(
                video={"reconnect_backoff_s": [4.0, 2.0]},
                _env_file=None,  # type: ignore[call-arg]
            )

    def test_rejects_tiny_live_queue(self) -> None:
        # The live queue is two frames by construction; a capacity below two
        # would defeat the drop-oldest policy.
        with pytest.raises(ValidationError):
            Settings(video={"live_queue_capacity": 1}, _env_file=None)  # type: ignore[call-arg]

    def test_rejects_non_positive_read_timeout(self) -> None:
        with pytest.raises(ValidationError):
            Settings(video={"read_timeout_s": 0}, _env_file=None)  # type: ignore[call-arg]


class TestVideoSettingsModel:
    def test_backoff_validator_runs_on_direct_construction(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            VideoSettings(reconnect_backoff_s=())

    def test_jitter_ratio_bounds(self) -> None:
        with pytest.raises(ValidationError):
            VideoSettings(reconnect_jitter_ratio=2.0)
