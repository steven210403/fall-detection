"""Preview delivery and client backpressure (Phase 6).

Thesis plan §17.1 gives this package two jobs — "latest JPEG and client backpressure" —
and §18.2 chooses the transport: MJPEG for the MVP, with WebSocket binary JPEG as the
documented fallback if frame synchronisation fails.

What this package owns:

* ``frames``: :class:`~fall_detection.streaming.frames.PreviewStream`, which holds the
  newest annotated preview frame, encodes it at the §8 rate cap of 5 FPS and the
  960x540 size cap, preserves the aspect ratio so a normalized zone coordinate maps
  onto the displayed image by one scalar, and reports the measured JPEG size and the
  §18.2 bandwidth arithmetic.
* ``snapshots``: :class:`~fall_detection.streaming.snapshots.SnapshotBroadcaster`, which
  fans live envelopes out to connected clients through **bounded** queues — §17.2's
  "slow clients receive the latest snapshot, not a backlog" — throttles track
  snapshots to 2 Hz while letting events and status through immediately, and numbers
  every message so a client can notice a gap and recover it with the event cursor.

Neither module knows about WebSockets or MJPEG HTTP framing: the API layer owns those,
which keeps the backpressure rules testable without a socket or a browser. Neither
saves frames to disk either — §24.2's "do not save preview frames by default" is a
property of this package having no write path at all.
"""

from fall_detection.streaming.frames import (
    FrameEnvelope,
    PreviewStats,
    PreviewStream,
    fit_within,
)
from fall_detection.streaming.snapshots import (
    SNAPSHOT_SCHEMA_VERSION,
    BroadcasterStats,
    ClientHandle,
    MessageKind,
    SnapshotBroadcaster,
)

__all__ = [
    "SNAPSHOT_SCHEMA_VERSION",
    "BroadcasterStats",
    "ClientHandle",
    "FrameEnvelope",
    "MessageKind",
    "PreviewStats",
    "PreviewStream",
    "SnapshotBroadcaster",
    "fit_within",
]
