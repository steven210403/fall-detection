"""Fall events: the contract, and the writer that persists then publishes them."""

from __future__ import annotations

from fall_detection.events.contracts import FallEventPage, FallEventRecord, OpenFallEvent
from fall_detection.events.writer import FallEventStats, FallEventWriter

__all__ = [
    "FallEventPage",
    "FallEventRecord",
    "FallEventStats",
    "FallEventWriter",
    "OpenFallEvent",
]
