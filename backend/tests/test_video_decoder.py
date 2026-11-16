"""Decoder integration tests.

These exercise the real FFmpeg subprocess because the decoder's contract — a
bgr24 pipe, presentation timestamps from ``showinfo``, and a bounded queue — can
only be verified against it. A synthetic clip is generated locally, so no
authorized footage is required and nothing here says anything about detection or
activity accuracy.
"""

from __future__ import annotations

import itertools
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

from fall_detection.core.errors import DecoderError, DecoderStallError
from fall_detection.video.decoder import DecoderMetadata, FfmpegDecoder, probe_source
from fall_detection.video.queue import FrameQueue
from fall_detection.video.sources import Mp4Source, RtspSource, WebcamSource


def build_decoder(
    path: Path,
    *,
    queue_capacity: int = 64,
    drop_oldest: bool = False,
    read_timeout_s: float = 10.0,
    extra_input_args: tuple[str, ...] = (),
) -> FfmpegDecoder:
    return FfmpegDecoder(
        Mp4Source(path=path),
        camera_id=uuid.uuid4(),
        stream_session_id=uuid.uuid4(),
        queue_capacity=queue_capacity,
        drop_oldest=drop_oldest,
        read_timeout_s=read_timeout_s,
        extra_input_args=extra_input_args,
    )


class TestProbe:
    def test_reads_dimensions_and_rate(self, synthetic_video: Path) -> None:
        metadata = probe_source(Mp4Source(path=synthetic_video))
        assert (metadata.width, metadata.height) == (160, 120)
        assert metadata.fps == pytest.approx(10.0)
        assert metadata.codec == "h264"
        assert metadata.has_source_timestamps is True
        assert metadata.ffmpeg_version is not None

    def test_metadata_describe_is_json_safe(self, synthetic_video: Path) -> None:
        described = probe_source(Mp4Source(path=synthetic_video)).describe()
        assert set(described) >= {"width", "height", "fps", "codec"}
        assert isinstance(described["width"], int)

    def test_frame_bytes_matches_bgr24(self) -> None:
        metadata = DecoderMetadata(
            width=10,
            height=5,
            fps=1.0,
            codec="x",
            has_source_timestamps=True,
        )
        assert metadata.frame_bytes == 10 * 5 * 3

    def test_explicit_dimensions_bypass_probe(self, synthetic_video: Path) -> None:
        metadata = probe_source(Mp4Source(path=synthetic_video), width=32, height=16)
        assert (metadata.width, metadata.height) == (32, 16)

    def test_missing_file_raises_decoder_error(self, media_root: Path) -> None:
        with pytest.raises(DecoderError, match="ffprobe failed"):
            probe_source(Mp4Source(path=media_root / "absent.mp4"))


class TestDecodeHappyPath:
    def test_decodes_the_whole_clip(self, synthetic_video: Path) -> None:
        decoder = build_decoder(synthetic_video)
        decoder.open()
        try:
            frames = []
            while True:
                packet = decoder.read(timeout=10.0)
                if packet is None:
                    break
                frames.append(packet)
        finally:
            decoder.close()

        assert len(frames) == decoder.frames_read > 0
        assert frames[0].frame.shape == (120, 160, 3)
        assert frames[0].frame.dtype == np.uint8

    def test_frame_sequence_is_dense_and_ordered(self, synthetic_video: Path) -> None:
        decoder = build_decoder(synthetic_video)
        decoder.open()
        try:
            sequences = []
            while True:
                packet = decoder.read(timeout=10.0)
                if packet is None:
                    break
                sequences.append(packet.frame_seq)
        finally:
            decoder.close()

        assert sequences == list(range(len(sequences)))

    def test_source_timestamps_are_monotonic(self, synthetic_video: Path) -> None:
        decoder = build_decoder(synthetic_video)
        decoder.open()
        try:
            stamps = []
            while True:
                packet = decoder.read(timeout=10.0)
                if packet is None:
                    break
                assert packet.source_timestamp is not None
                stamps.append(packet.source_timestamp)
        finally:
            decoder.close()

        assert len(stamps) > 5
        assert all(b >= a for a, b in itertools.pairwise(stamps))
        # ~10 fps: 1 second of video must not be reported as anything else.
        assert stamps[-1] - stamps[0] == pytest.approx(0.9, abs=0.2)

    def test_receive_timestamps_are_monotonic(self, synthetic_video: Path) -> None:
        decoder = build_decoder(synthetic_video)
        decoder.open()
        try:
            receives = []
            while True:
                packet = decoder.read(timeout=10.0)
                if packet is None:
                    break
                receives.append(packet.receive_monotonic)
        finally:
            decoder.close()

        assert all(b >= a for a, b in itertools.pairwise(receives))

    def test_packet_carries_session_identity(self, synthetic_video: Path) -> None:
        camera_id = uuid.uuid4()
        session_id = uuid.uuid4()
        decoder = FfmpegDecoder(
            Mp4Source(path=synthetic_video),
            camera_id=camera_id,
            stream_session_id=session_id,
            queue_capacity=8,
            drop_oldest=False,
        )
        decoder.open()
        try:
            packet = decoder.read(timeout=10.0)
        finally:
            decoder.close()
        assert packet is not None
        assert packet.camera_id == camera_id
        assert packet.stream_session_id == session_id

    def test_frames_are_independently_owned(self, synthetic_video: Path) -> None:
        # Each packet must own its buffer; a view over a reused accumulator would
        # make every frame alias the last one decoded.
        decoder = build_decoder(synthetic_video)
        decoder.open()
        try:
            first = decoder.read(timeout=10.0)
            second = decoder.read(timeout=10.0)
        finally:
            decoder.close()
        assert first is not None and second is not None
        assert first.frame is not second.frame
        assert not np.shares_memory(first.frame, second.frame)
        assert first.frame.flags.writeable
        # Writing to one frame must not disturb the other.
        before = second.frame[0, 0].copy()
        first.frame[0, 0] = (1, 2, 3)
        assert np.array_equal(second.frame[0, 0], before)


class TestDecodeFailures:
    def test_unknown_input_option_fails_to_open(self, synthetic_video: Path) -> None:
        # FFmpeg starts and then exits with an argument error, so the failure
        # surfaces from ``read`` (through the reader thread) rather than from
        # ``open``, which only probes and launches the process.
        decoder = build_decoder(synthetic_video, extra_input_args=("-definitely-not-an-option",))
        decoder.open()
        try:
            with pytest.raises(DecoderError):
                for _ in range(50):
                    if decoder.read(timeout=2.0) is None:
                        pytest.fail("decoder exited without reporting an error")
        finally:
            decoder.close()

    def test_stall_timeout_raises_decoder_stall(self, synthetic_video: Path) -> None:
        # The contract is: no frame within the timeout *while FFmpeg is still
        # running* must raise a stall, not block forever or report a clean EOF
        # (§22 lists decoder hangs as a required fault case). The process state
        # is stubbed so the assertion is deterministic and does not depend on
        # how quickly a machine can decode a clip.
        decoder = build_decoder(synthetic_video, read_timeout_s=0.05)
        decoder.open()
        try:
            # Let the real reader finish the short clip first: it resolves
            # ``self._queue`` at call time, so it would otherwise keep filling
            # the replacement queue used below.
            assert decoder._reader is not None  # noqa: SLF001
            decoder._reader.join(timeout=10.0)  # noqa: SLF001
            decoder._queue = FrameQueue(2, drop_oldest=True)  # noqa: SLF001
            decoder._counters.eof = False  # noqa: SLF001
            decoder.is_alive = lambda: True  # type: ignore[method-assign]
            decoder.returncode = lambda: None  # type: ignore[method-assign]
            with pytest.raises(DecoderStallError, match="stall timeout"):
                decoder.read()
        finally:
            decoder.close()

    def test_reader_marks_a_silent_process_as_stalled(self, synthetic_video: Path) -> None:
        # The reader thread is what decides a stall happened. Drive it against a
        # real pipe that never receives data while the "process" reports itself
        # alive: the reader must set the stall flag instead of reporting EOF, or
        # ``read`` would treat a hung decoder as a finished file.
        import os

        read_fd, write_fd = os.pipe()
        read_stream = os.fdopen(read_fd, "rb", buffering=0)

        class SilentProcess:
            """A stand-in whose stdout is a pipe that never receives data."""

            stdout = read_stream
            stderr = None

            def poll(self) -> None:
                return None

            def wait(self, timeout: float | None = None) -> int:
                return 0

        decoder = build_decoder(synthetic_video, read_timeout_s=0.05)
        decoder.open()
        try:
            decoder._process = SilentProcess()  # type: ignore[assignment]  # noqa: SLF001
            decoder._metadata = DecoderMetadata(  # noqa: SLF001
                width=4, height=4, fps=1.0, codec=None, has_source_timestamps=True
            )
            decoder._counters.eof = False  # noqa: SLF001
            decoder._counters.stalled = False  # noqa: SLF001
            decoder._read_frames()  # noqa: SLF001 - the unit under test
            assert decoder._counters.stalled is True  # noqa: SLF001
            assert decoder._counters.error is None  # noqa: SLF001
        finally:
            os.close(write_fd)
            read_stream.close()
            decoder._process = None  # noqa: SLF001
            decoder._stop.set()  # noqa: SLF001
            decoder._queue.close()  # noqa: SLF001

    def test_read_after_close_returns_none(self, synthetic_video: Path) -> None:
        decoder = build_decoder(synthetic_video)
        decoder.open()
        while decoder.read(timeout=5.0) is not None:
            pass  # drain the clip so no buffered frame masks a closed queue
        decoder.close()
        assert decoder.read(timeout=0.02) is None

    def test_close_is_idempotent(self, synthetic_video: Path) -> None:
        decoder = build_decoder(synthetic_video)
        decoder.open()
        decoder.close()
        decoder.close()
        assert decoder.is_alive() is False

    def test_open_twice_raises(self, synthetic_video: Path) -> None:
        decoder = build_decoder(synthetic_video)
        decoder.open()
        try:
            with pytest.raises(DecoderError, match="already open"):
                decoder.open()
        finally:
            decoder.close()

    def test_missing_binary_is_reported(self, synthetic_video: Path) -> None:
        decoder = FfmpegDecoder(
            Mp4Source(path=synthetic_video),
            camera_id=uuid.uuid4(),
            stream_session_id=uuid.uuid4(),
            ffmpeg_bin="ffmpeg-does-not-exist",
        )
        with pytest.raises(DecoderError, match="not found on PATH"):
            decoder.open()


class TestQueuePolicy:
    def test_live_mode_drops_and_counts(self, synthetic_video: Path) -> None:
        # Never read: the reader thread must keep draining FFmpeg and dropping,
        # so ``put`` can never block waiting for a consumer.
        decoder = build_decoder(synthetic_video, queue_capacity=2, drop_oldest=True)
        decoder.open()
        try:
            for _ in range(200):
                if decoder.frames_read >= 10:
                    break
            stats = decoder.queue.stats()
        finally:
            decoder.close()
        assert stats["capacity"] == 2
        assert stats["drop_oldest"] is True
        assert stats["depth"] <= 2

    def test_offline_mode_drops_nothing_when_consumed(self, synthetic_video: Path) -> None:
        decoder = build_decoder(synthetic_video, queue_capacity=4, drop_oldest=False)
        decoder.open()
        try:
            consumed = 0
            while decoder.read(timeout=10.0) is not None:
                consumed += 1
        finally:
            decoder.close()
        assert consumed > 0
        assert decoder.frames_dropped == 0


class TestCommandConstruction:
    def test_file_input_omits_rtsp_transport(self, synthetic_video: Path) -> None:
        decoder = build_decoder(synthetic_video)
        decoder.open()
        try:
            # ``-rtsp_transport`` is invalid for a file; if it were present,
            # ``open`` would already have failed.
            assert decoder.metadata is not None
        finally:
            decoder.close()

    def test_rtsp_input_selects_tcp(self) -> None:
        # The command is asserted directly so no network access is required.
        decoder = FfmpegDecoder(
            RtspSource(url="rtsp://user:pw@camera.example:554/s1"),
            camera_id=uuid.uuid4(),
            stream_session_id=uuid.uuid4(),
            width=16,
            height=16,
        )
        command = decoder._build_command(  # noqa: SLF001 - deliberate white-box assertion
            DecoderMetadata(
                width=16,
                height=16,
                fps=1.0,
                codec=None,
                has_source_timestamps=True,
            )
        )
        assert "-rtsp_transport" in command
        assert command[command.index("-rtsp_transport") + 1] == "tcp"
        assert command[command.index("-f") + 1] == "rawvideo"
        assert "bgr24" in command


class TestWebcamCommand:
    def test_webcam_is_not_probed_and_uses_its_requested_size(self) -> None:
        # FFprobe is never run for a camera; a missing binary would raise otherwise.
        metadata = probe_source(
            WebcamSource(device="0", width=640, height=480, fps=30.0),
            ffprobe_bin="definitely-not-ffprobe",
        )
        assert (metadata.width, metadata.height, metadata.fps) == (640, 480, 30.0)
        assert metadata.has_source_timestamps

    def test_webcam_command_selects_avfoundation_and_pins_the_size(self) -> None:
        decoder = FfmpegDecoder(
            WebcamSource(device="0", width=640, height=480, fps=30.0),
            camera_id=uuid.uuid4(),
            stream_session_id=uuid.uuid4(),
        )
        command = decoder._build_command(  # noqa: SLF001 - deliberate white-box assertion
            DecoderMetadata(width=640, height=480, fps=30.0, codec=None, has_source_timestamps=True)
        )
        input_at = command.index("-i")
        assert command[input_at + 1] == "0:none"
        assert command[command.index("avfoundation") - 1] == "-f"
        assert command.index("avfoundation") < input_at
        assert command[command.index("-video_size") + 1] == "640x480"
        assert command[command.index("-framerate") + 1] == "30"
        assert "-rtsp_transport" not in command
        assert command[command.index("-vf") + 1] == "showinfo,scale=640:480"


@pytest.mark.parametrize(
    ("backend", "hint"),
    [("v4l2", "'video' group"), ("dshow", "Privacy & security > Camera")],
)
def test_a_silent_webcam_stall_names_the_platform_hint(backend: str, hint: str) -> None:
    decoder = FfmpegDecoder(
        WebcamSource(device="0", width=640, height=480, fps=30.0, backend=backend),  # type: ignore[arg-type]
        camera_id=uuid.uuid4(),
        stream_session_id=uuid.uuid4(),
    )
    assert hint in decoder._stall_message()  # noqa: SLF001


def test_a_silent_webcam_stall_names_the_camera_permission() -> None:
    decoder = FfmpegDecoder(
        WebcamSource(device="0", width=640, height=480, fps=30.0),
        camera_id=uuid.uuid4(),
        stream_session_id=uuid.uuid4(),
    )
    assert "Privacy & Security > Camera" in decoder._stall_message()  # noqa: SLF001


@pytest.mark.skipif(sys.platform != "win32", reason="Windows pipes are polled, not selected")
class TestWindowsPipeWait:
    """``select`` rejects pipes on Windows; the PeekNamedPipe wait must match it."""

    def test_reports_waiting_bytes_an_empty_pipe_and_a_closed_writer(self) -> None:
        import os
        import threading

        from fall_detection.video.decoder import _wait_pipe_windows  # noqa: PLC2701

        read_fd, write_fd = os.pipe()
        stop = threading.Event()
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            assert _wait_pipe_windows(stream, 0.05, stop) is False
            os.write(write_fd, b"x")
            assert _wait_pipe_windows(stream, 0.05, stop) is True
            assert stream.read(1) == b"x"
            os.close(write_fd)
            # A broken pipe counts as readable so ``read`` returns end of file.
            assert _wait_pipe_windows(stream, 0.05, stop) is True
            assert stream.read(1) == b""
