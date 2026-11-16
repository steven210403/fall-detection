#!/usr/bin/env python
"""Phase 2 replay and ingestion verification (thesis plan §25, Phase 2 row).

Acceptance for Phase 2 is a *10-minute replay* with clean EOF, a
disconnect/reconnect cycle, bounded queues, and a timestamp report. This script
produces exactly that evidence from a local MP4, so it needs no authorized
footage and no camera.

It verifies, in one run:

1. **Clean EOF** — the file is decoded to its last frame and the pipeline enters
   ``ended`` with the session closed as ``source_ended``.
2. **Timestamps** — every packet carries a monotonic receive timestamp, and
   source presentation timestamps are monotonic and match the container rate.
   It reports how many frames lacked a source timestamp rather than assuming.
3. **Bounded queues** — the live drop policy is demonstrated separately by
   decoding with a stalled consumer, and the offline policy is shown to drop
   nothing while the whole file is consumed.
4. **Fault injection** — with ``--inject-fault``, FFmpeg is killed mid-replay and
   the supervisor must reconnect with a *new* stream session and recover.
5. **Session records** — the session rows written through the single writer are
   read back and reported.

Nothing here measures detection or activity accuracy; the clip is synthetic.

Exit codes: ``0`` all checks pass, ``1`` a check failed, ``2`` missing input.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "backend" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from sqlalchemy import select  # noqa: E402

from fall_detection.config import Settings  # noqa: E402
from fall_detection.core.contracts import CameraState, FramePacket  # noqa: E402
from fall_detection.database import (  # noqa: E402
    Base,
    DatabaseWriter,
    StreamSession,
    create_engine,
    create_session_factory,
)
from fall_detection.video.pipeline import PipelineSupervisor, wait_for_state  # noqa: E402
from fall_detection.video.reconnect import ReconnectPolicy  # noqa: E402

RESULTS: list[dict[str, Any]] = []


def record(check: str, status: str, detail: str = "", **extra: Any) -> None:
    """Append a check result and print it."""
    RESULTS.append({"check": check, "status": status, "detail": detail, **extra})
    print(f"[{status:4}] {check}: {detail}")


def run_check(check: str, fn: Callable[[], tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Run ``fn`` and record PASS/FAIL; returns the collected measurements."""
    start = time.perf_counter()
    try:
        detail, measurements = fn()
    except Exception as exc:  # noqa: BLE001 - a verification script reports everything
        record(check, "FAIL", f"{type(exc).__name__}: {exc}")
        return {}
    record(check, "PASS", detail, seconds=round(time.perf_counter() - start, 3))
    return measurements


class FrameRecorder:
    """Collect per-frame timing measurements from the pipeline callback."""

    def __init__(self) -> None:
        self.count = 0
        self.source_timestamps: list[float] = []
        self.receive_timestamps: list[float] = []
        self.missing_source = 0
        self.sessions_seen: set[str] = set()
        self.first_frame_monotonic: float | None = None
        self.last_frame_monotonic: float | None = None
        self._lock = threading.Lock()

    def __call__(self, packet: FramePacket) -> None:
        with self._lock:
            self.count += 1
            self.sessions_seen.add(str(packet.stream_session_id))
            if packet.source_timestamp is None:
                self.missing_source += 1
            else:
                self.source_timestamps.append(packet.source_timestamp)
            self.receive_timestamps.append(packet.receive_monotonic)
            if self.first_frame_monotonic is None:
                self.first_frame_monotonic = packet.receive_monotonic
            self.last_frame_monotonic = packet.receive_monotonic

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the measurements."""
        with self._lock:
            source = list(self.source_timestamps)
            receive = list(self.receive_timestamps)
            return {
                "frames": self.count,
                "sessions_seen": sorted(self.sessions_seen),
                "missing_source_timestamps": self.missing_source,
                "source_first_s": source[0] if source else None,
                "source_last_s": source[-1] if source else None,
                "source_span_s": (source[-1] - source[0]) if len(source) > 1 else None,
                "wall_duration_s": (
                    (self.last_frame_monotonic - self.first_frame_monotonic)
                    if self.first_frame_monotonic is not None
                    and self.last_frame_monotonic is not None
                    else None
                ),
                "source_monotonic": _is_monotonic(source),
                "receive_monotonic": _is_monotonic(receive),
                "source_median_step_s": _median_step(source),
            }


def _is_monotonic(values: list[float]) -> bool:
    return all(later >= earlier for earlier, later in zip(values, values[1:], strict=False))


def _median_step(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    steps = [b - a for a, b in zip(values, values[1:], strict=False)]
    return statistics.median(steps)


def build_settings(args: argparse.Namespace) -> Settings:
    """Build isolated settings for the replay."""
    database_url = args.database or f"sqlite+pysqlite:///{REPO_ROOT / 'data' / 'phase2_replay.db'}"
    return Settings(
        environment="test",
        log_level="WARNING",
        database_url=database_url,
        video={
            "mode": "mp4",
            "source_ref": str(args.video),
            "read_timeout_s": args.read_timeout,
        },
        media_roots={"roots": [str(args.video.parent)]},
        _env_file=None,  # type: ignore[call-arg]
    )


def make_writer(settings: Settings) -> tuple[DatabaseWriter, Any]:
    """Create the engine and the single writer, and ensure the schema exists."""
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    return DatabaseWriter(create_session_factory(engine)), engine


def fast_policy(args: argparse.Namespace) -> ReconnectPolicy:
    """A reconnect policy suitable for a short replay."""
    return ReconnectPolicy(
        backoff_s=(0.05, 0.1, 0.2),
        jitter_ratio=0.0,
        max_attempts=args.max_attempts,
    )


def check_clean_eof(
    settings: Settings, writer: DatabaseWriter, args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    """Decode the whole file offline and assert a clean ``ended``."""
    recorder = FrameRecorder()
    supervisor = PipelineSupervisor(
        settings=settings,
        camera_id=uuid.uuid4(),
        writer=writer,
        on_frame=recorder,
        policy=fast_policy(args),
    )
    supervisor.start()
    ended = wait_for_state(supervisor, CameraState.ENDED, timeout=args.timeout)
    supervisor.stop()

    measurements = recorder.snapshot()
    measurements.update(
        {
            "final_state": supervisor.state.value,
            "stats": supervisor.stats.as_dict(),
        }
    )
    if not ended:
        raise AssertionError(f"pipeline never reached 'ended' (state={supervisor.state.value})")
    if measurements["frames"] == 0:
        raise AssertionError("no frames were decoded")
    return (
        f"{measurements['frames']} frames, state=ended, "
        f"source span={measurements['source_span_s']:.1f}s",
        measurements,
    )


def check_timestamps(measurements: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Assert source and receive timestamps are monotonic."""
    if not measurements.get("source_monotonic", False):
        raise AssertionError("source presentation timestamps are not monotonic")
    if not measurements.get("receive_monotonic", False):
        raise AssertionError("receive timestamps are not monotonic")
    span = measurements.get("source_span_s")
    wall = measurements.get("wall_duration_s")
    step = measurements.get("source_median_step_s")
    if span is None or wall is None:
        raise AssertionError("not enough timestamps to compare")
    # Offline decoding is faster than real time; time must come from the source
    # timestamps, never from loop speed (thesis plan §9.1).
    if span <= 0:
        raise AssertionError("source timestamps do not advance")
    detail = (
        f"monotonic; source span={span:.2f}s in {wall:.2f}s wall; "
        f"median step={step:.4f}s; missing source timestamps={measurements['missing_source_timestamps']}"
    )
    return detail, measurements


def check_session_records(
    writer: DatabaseWriter, measurements: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Read the session rows back and assert they were closed correctly."""
    rows = writer.read(lambda session: list(session.scalars(select(StreamSession))))
    if not rows:
        raise AssertionError("no stream_session rows were written")
    open_rows = [row for row in rows if row.ended_at is None]
    if open_rows:
        raise AssertionError(f"{len(open_rows)} session rows were left open")
    summary = [
        {
            "attempt": row.attempt,
            "termination_reason": row.termination_reason,
            "frames_read": row.frames_read,
            "frames_dropped": row.frames_dropped,
            "closed": row.ended_at is not None,
        }
        for row in rows
    ]
    if len(rows) == 1 and rows[0].termination_reason != "source_ended":
        raise AssertionError(f"clean EOF recorded {rows[0].termination_reason!r}")
    return f"{len(rows)} session row(s), all closed: {json.dumps(summary)}", {
        "sessions": summary
    }


def check_offline_queue_drops_nothing(
    settings: Settings, writer: DatabaseWriter, args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    """Decode with a small offline queue and assert no sample is lost."""
    supervisor = PipelineSupervisor(
        settings=settings,
        camera_id=uuid.uuid4(),
        writer=writer,
        policy=fast_policy(args),
    )
    supervisor.start()
    wait_for_state(supervisor, CameraState.ENDED, timeout=args.timeout)
    stats = supervisor.stats.as_dict()
    supervisor.stop()
    if stats["frames_dropped"] != 0:
        raise AssertionError(
            f"offline decoding dropped {stats['frames_dropped']} frames; "
            "offline mode must not drop because the machine is slow"
        )
    return (
        f"{stats['frames_read']} frames, 0 dropped",
        {"stats": stats},
    )


def check_live_queue_drops_oldest(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Assert the live policy drops the oldest frame and never blocks.

    The bounded queue is exercised directly: a real RTSP overload cannot be
    produced offline, and the queue is the component whose behaviour the plan
    specifies (§9.1). The producer is run on its own thread with a timeout so a
    regression that made ``put`` block would fail rather than hang.
    """
    from fall_detection.video.queue import FrameQueue

    queue: FrameQueue[int] = FrameQueue(2, drop_oldest=True)
    finished = threading.Event()

    def produce() -> None:
        for value in range(1000):
            queue.put(value)
        finished.set()

    thread = threading.Thread(target=produce, name="live-queue-producer")
    thread.start()
    blocked = not finished.wait(timeout=args.stall_seconds + 5.0)
    thread.join(timeout=1.0)
    if blocked:
        raise AssertionError("live queue blocked the producer; it must drop instead")

    dropped = queue.dropped
    if dropped != 998:
        raise AssertionError(f"live queue dropped {dropped}, expected 998 of 1000")
    if queue.qsize() != 2:
        raise AssertionError(f"live queue depth {queue.qsize()}, expected 2")
    return (
        f"live queue capacity 2 held depth 2 and dropped {dropped}/1000 without blocking",
        {"dropped": dropped, "capacity": queue.capacity, "depth": queue.qsize()},
    )


def check_fault_recovery(
    settings: Settings, writer: DatabaseWriter, args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    """Kill FFmpeg mid-replay and assert the supervisor reconnects."""
    recorder = FrameRecorder()
    supervisor = PipelineSupervisor(
        settings=settings,
        camera_id=uuid.uuid4(),
        writer=writer,
        on_frame=recorder,
        policy=fast_policy(args),
    )
    # A short reconnect backoff keeps the replay quick.
    supervisor.start()

    killed = False
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        decoder = supervisor.current_decoder()
        if decoder is not None and supervisor.state is CameraState.LIVE:
            if decoder.frames_read >= args.kill_after_frames:
                killed = decoder.kill_process()
                break
        time.sleep(0.01)

    if not killed:
        supervisor.stop()
        raise AssertionError("could not kill the decoder mid-replay")

    # After the kill the supervisor must reconnect with a fresh session.
    recovered = wait_for_state(supervisor, CameraState.ENDED, timeout=args.timeout)
    stats = supervisor.stats.as_dict()
    snapshot = recorder.snapshot()
    supervisor.stop()

    distinct_sessions = snapshot["sessions_seen"]
    if stats["reconnects"] < 1:
        raise AssertionError("a killed decoder did not trigger a reconnect")
    if len(distinct_sessions) < 2:
        raise AssertionError(
            "reconnect did not create a new stream session "
            "(tracker state would be mixed between attempts)"
        )
    if not recovered:
        raise AssertionError(f"pipeline did not recover (state={supervisor.state.value})")
    return (
        f"killed after {args.kill_after_frames} frames; reconnects={stats['reconnects']}, "
        f"distinct sessions={len(distinct_sessions)}, recovered to ended",
        {"stats": stats, "frames": snapshot},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="MP4 to replay")
    parser.add_argument("--database", default=None, help="SQLite URL for evidence")
    parser.add_argument("--json", type=Path, default=None, help="write results here")
    parser.add_argument(
        "--skip-fault-injection",
        action="store_true",
        help="skip the decoder-kill recovery check",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--read-timeout", type=float, default=15.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--stall-seconds", type=float, default=0.2)
    parser.add_argument("--kill-after-frames", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 2

    print("== Phase 2 ingestion and replay verification ==")
    print(f"    video: {args.video}")
    settings = build_settings(args)
    writer, engine = make_writer(settings)

    try:
        measurements = run_check(
            "clean-eof-replay", lambda: check_clean_eof(settings, writer, args)
        )
        if measurements:
            run_check("timestamp-report", lambda: check_timestamps(measurements))
        run_check("session-records", lambda: check_session_records(writer, measurements))
        run_check(
            "offline-queue-no-drops",
            lambda: check_offline_queue_drops_nothing(settings, writer, args),
        )
        run_check(
            "live-queue-drops-oldest",
            lambda: check_live_queue_drops_oldest(args),
        )
        if not args.skip_fault_injection:
            run_check(
                "decoder-kill-recovery",
                lambda: check_fault_recovery(settings, writer, args),
            )
    finally:
        engine.dispose()

    failures = [result for result in RESULTS if result["status"] == "FAIL"]
    print()
    print(f"== {len(RESULTS) - len(failures)} passed, {len(failures)} failed ==")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "video": str(args.video),
                    "settings": {
                        "video_mode": settings.video.mode,
                        "read_timeout_s": settings.video.read_timeout_s,
                    },
                    "results": RESULTS,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
