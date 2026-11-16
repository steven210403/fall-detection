"""Tests for the shared core layer: errors, logging redaction, and contracts."""

from __future__ import annotations

import logging
import uuid

import numpy as np
import pytest

from fall_detection.core import (
    CameraState,
    CameraUnavailableError,
    CctvError,
    DecoderStallError,
    FramePacket,
    InvalidSourceError,
    StreamSessionInfo,
    configure_logging,
    get_logger,
    redact_text,
    redact_url,
)


class TestRedaction:
    def test_strips_rtsp_credentials(self) -> None:
        assert redact_url("rtsp://admin:hunter2@10.0.0.5:554/stream1") == (
            "rtsp://***@10.0.0.5:554/stream1"
        )

    def test_leaves_credential_free_urls_untouched(self) -> None:
        url = "rtsp://10.0.0.5:554/stream1"
        assert redact_url(url) == url

    def test_redacts_credential_query_parameters(self) -> None:
        redacted = redact_url("http://cam.local/snap?token=abc123&x=1")
        assert "abc123" not in redacted
        assert "token=***" in redacted

    def test_redact_text_handles_embedded_urls(self) -> None:
        text = "ffprobe failed for rtsp://user:pw@host/stream (exit 1)"
        result = redact_text(text)
        assert "pw" not in result
        assert "rtsp://***@host/stream" in result


class TestLoggingConfiguration:
    def test_redaction_filter_rewrites_records(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("tests.redaction")
        with caplog.at_level(logging.INFO, logger=logger.name):
            logger.info("connecting to %s", "rtsp://user:secret@host/s1")

        assert "secret" not in caplog.text
        assert "rtsp://***@host/s1" in caplog.text

    def test_configure_logging_is_idempotent(self) -> None:
        configure_logging("DEBUG", stream=False)
        configure_logging("DEBUG", stream=False)
        logger = logging.getLogger("fall_detection")
        assert logger.level == logging.DEBUG
        assert any(isinstance(f, logging.Filter) for f in logger.filters)

    def test_get_logger_namespaces(self) -> None:
        assert get_logger("video.pipeline").name == "fall_detection.video.pipeline"
        assert get_logger().name == "fall_detection"
        assert get_logger("fall_detection.x").name == "fall_detection.x"


class TestErrorHierarchy:
    def test_all_errors_share_the_base(self) -> None:
        for error in (
            InvalidSourceError("x"),
            CameraUnavailableError("x"),
            DecoderStallError("x"),
        ):
            assert isinstance(error, CctvError)

    def test_codes_are_distinct_and_stable(self) -> None:
        assert InvalidSourceError.code == "invalid_source"
        assert CameraUnavailableError.code == "camera_unavailable"
        assert DecoderStallError.code == "decoder_stall"
        assert DecoderStallError.code != CctvError.code

    def test_as_dict_omits_absent_detail(self) -> None:
        assert InvalidSourceError("bad path").as_dict() == {
            "code": "invalid_source",
            "message": "bad path",
        }
        assert InvalidSourceError("bad path", detail="p").as_dict()["detail"] == "p"


class TestFramePacket:
    def _packet(self, **overrides: object) -> FramePacket:
        defaults: dict[str, object] = {
            "camera_id": uuid.uuid4(),
            "stream_session_id": uuid.uuid4(),
            "frame_seq": 0,
            "frame": np.zeros((48, 64, 3), dtype=np.uint8),
            "source_timestamp": 0.0,
            "receive_monotonic": 1.0,
            "receive_utc": 1_700_000_000.0,
        }
        defaults.update(overrides)
        return FramePacket(**defaults)  # type: ignore[arg-type]

    def test_shape_helpers(self) -> None:
        packet = self._packet()
        assert packet.shape == (48, 64, 3)
        assert packet.height == 48
        assert packet.width == 64
        assert packet.nbytes() == 48 * 64 * 3

    def test_missing_source_timestamp_stays_none(self) -> None:
        # Thesis plan §9.1: absent source time must not be replaced by a
        # synthetic value, so a live source without PTS keeps ``None``.
        packet = self._packet(source_timestamp=None)
        assert packet.source_timestamp is None

    def test_packet_is_frozen(self) -> None:
        packet = self._packet()
        with pytest.raises(AttributeError):
            packet.frame_seq = 1  # type: ignore[misc]


class TestStreamSessionInfo:
    def test_defaults_to_first_attempt(self) -> None:
        info = StreamSessionInfo(
            camera_id=uuid.uuid4(),
            stream_session_id=uuid.uuid4(),
            started_utc=0.0,
            started_monotonic=0.0,
        )
        assert info.attempt == 1


class TestCameraState:
    def test_states_are_distinct_strings(self) -> None:
        values = {state.value for state in CameraState}
        assert values == {
            "idle",
            "connecting",
            "live",
            "reconnecting",
            "offline",
            "ended",
        }
        assert len(values) == len(CameraState)

    def test_is_str_comparable(self) -> None:
        assert CameraState.LIVE == "live"


class TestAcquisitionClock:
    """The clock an offline replay supplies instead of wall time (§9.1).

    The pipeline's time-based rules all read the monotonic clock. On a live camera that is
    the footage's own rate; in a replay it is the decode's rate, which is many times faster
    and would leave every duration condition unmet. This clock exists to close that gap, so
    its own behaviour is what the measurement rests on.
    """

    def test_it_starts_at_the_origin(self) -> None:
        from fall_detection.core.clock import AcquisitionClockBase

        clock = AcquisitionClockBase(origin_monotonic=5.0, origin_utc=1000.0)
        assert clock.monotonic() == 5.0
        assert clock.utc_timestamp() == 1005.0

    def test_advancing_moves_both_readings_together(self) -> None:
        from fall_detection.core.clock import AcquisitionClockBase

        clock = AcquisitionClockBase()
        clock.advance_to(12.5)
        assert clock.monotonic() == 12.5
        assert clock.utc_timestamp() == 12.5

    def test_it_never_moves_backwards(self) -> None:
        # A frame presented out of order must not make an elapsed interval negative.
        from fall_detection.core.clock import AcquisitionClockBase

        clock = AcquisitionClockBase()
        clock.advance_to(10.0)
        clock.advance_to(3.0)
        assert clock.monotonic() == 10.0

    def test_the_utc_reading_is_an_offset_not_an_absolute_instant(self) -> None:
        # §13.2 forbids inventing absolute UTC for a source whose time zone is unknown: the
        # default origin is zero, so the reading is a duration rather than a date.
        from fall_detection.core.clock import AcquisitionClockBase

        clock = AcquisitionClockBase()
        clock.advance_to(90.0)
        assert clock.utc_timestamp() == 90.0

    def test_it_satisfies_the_clock_protocol(self) -> None:
        from fall_detection.core.clock import AcquisitionClockBase, Clock

        assert isinstance(AcquisitionClockBase(), Clock)
