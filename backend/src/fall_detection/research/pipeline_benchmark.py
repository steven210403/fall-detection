"""§15.2 Experiment 5: throughput, stage latency, and unified-memory reporting.

The experiment is "n/s, 640/960, CPU/MPS; actual person counts available in each dataset"
with "Throughput, p50/p95 processing latency, stage times, CPU, RAM, accelerator memory",
and §23.1 adds the reporting rule that matters most on this hardware: *"On Apple Silicon,
RAM and accelerator memory share unified memory. Report process RSS, MPS allocated/driver
memory, and memory pressure; do not present them as discrete VRAM."*

This module owns three things and no measurement of its own:

* :class:`StageTimer` — per-stage wall time for one pipeline pass, collected in
  :class:`LatencySummary`. Stage times come from the services' own counters where they
  exist (detection reports its Ultralytics breakdown) and from the caller's stopwatch
  around each stage where they do not. A caller may also record the whole frame under
  :data:`TOTAL_STAGE`; that measurement *is* the total and is never added to the stages,
  because the span covers them and summing both would double every frame's latency.
* :func:`summarise_latency` — p50/p95/mean/max over samples, with the count included. A
  percentile of eleven samples is not a percentile of a thousand, and the count is what
  lets a reader see which it is.
* :func:`memory_snapshot` — process RSS, MPS allocated/reserved memory when the device is
  MPS, and the explicit statement that these are not VRAM.

**§9.1's latency rule.** A frame's *processing* latency is measurable (receive monotonic →
publish). Capture-to-display latency is not, because the camera clock is unknown; nothing
here computes one or claims to.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fall_detection.core.process import load_average, process_usage

__all__ = [
    "LATENCY_PERCENTILES",
    "TOTAL_STAGE",
    "LatencySummary",
    "StageTimer",
    "memory_snapshot",
    "percentile",
    "summarise_latency",
]

LATENCY_PERCENTILES: tuple[float, ...] = (50.0, 95.0)
"""§15.2 Experiment 5 names p50 and p95. Anything else is extra, never a substitute."""

TOTAL_STAGE = "frame_total"
"""The stage name a caller uses when it times the whole frame itself.

A recorded total is a *measurement*, not another stage: :meth:`StageTimer.total` reports
it as recorded instead of adding it to the per-stage samples, which would double every
frame's latency.
"""


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linear-interpolated percentile, or ``None`` for an empty sample.

    Uses the same definition as :func:`numpy.percentile` with ``method="linear"`` but
    without importing numpy, so a status payload can compute one without the scientific
    stack loaded.
    """
    if not values:
        return None
    if not 0.0 <= quantile <= 100.0:
        raise ValueError("quantile must be within [0, 100]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (quantile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Distribution summary for one measured quantity, in milliseconds.

    Attributes:
        name: What was measured.
        samples: Number of measurements the summary rests on.
        mean: Arithmetic mean.
        p50: Median.
        p95: 95th percentile.
        maximum: Largest sample.
        minimum: Smallest sample.
        note: What the figure does and does not claim.
    """

    name: str
    samples: int
    mean: float | None
    p50: float | None
    p95: float | None
    maximum: float | None
    minimum: float | None
    note: str = "processing latency on this machine; not a capture-to-display figure (§9.1)"

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "name": self.name,
            "samples": self.samples,
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "min": self.minimum,
            "max": self.maximum,
            "note": self.note,
        }


def summarise_latency(
    name: str, samples: Iterable[float], *, note: str | None = None
) -> LatencySummary:
    """Summarise ``samples`` (milliseconds) into mean/p50/p95/min/max."""
    values = [float(value) for value in samples]
    if not values:
        return LatencySummary(
            name=name,
            samples=0,
            mean=None,
            p50=None,
            p95=None,
            maximum=None,
            minimum=None,
            note=note or "no sample was collected; a percentile of nothing is not reported as 0.0",
        )
    return LatencySummary(
        name=name,
        samples=len(values),
        mean=sum(values) / len(values),
        p50=percentile(values, LATENCY_PERCENTILES[0]),
        p95=percentile(values, LATENCY_PERCENTILES[1]),
        maximum=max(values),
        minimum=min(values),
        note=note or "processing latency on this machine; not a capture-to-display figure (§9.1)",
    )


@dataclass
class StageTimer:
    """Collect per-stage wall times (milliseconds) for a run.

    The stages are the §9 chain: decode, detect, track, features, activity, publish. A
    caller wraps each stage and calls :meth:`record`; nothing here measures by itself,
    because only the caller knows where a stage begins.

    Attributes:
        samples: Stage name to collected milliseconds.
        counts: Stage name to the number of times it ran, including failures.
    """

    samples: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def record(self, stage: str, milliseconds: float) -> None:
        """Append one stage measurement."""
        self.samples.setdefault(stage, []).append(float(milliseconds))
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def extend(self, stage: str, milliseconds: Iterable[float]) -> None:
        """Append several measurements for one stage."""
        for value in milliseconds:
            self.record(stage, value)

    def summary(self, stage: str) -> LatencySummary:
        """Distribution summary for one stage."""
        return summarise_latency(stage, self.samples.get(stage, ()))

    def total(self) -> LatencySummary:
        """Distribution of the per-frame total.

        Two callers exist and they must not be confused. A caller that times the whole
        frame itself records it under :data:`TOTAL_STAGE`; that measurement *is* the
        total and is reported as recorded. A caller that only times individual stages
        has no such sample, and the total is then the per-frame sum across the stages
        that all ran, so a partially instrumented run cannot produce a total that omits
        a stage's cost.

        Adding a recorded total to the stage samples would count the same work twice:
        every caller records the total as the span covering its stages, so the sum
        would be exactly twice the measured frame time.
        """
        measured = self.samples.get(TOTAL_STAGE, ())
        if measured:
            return summarise_latency(
                TOTAL_STAGE,
                measured,
                note=(
                    "measured per-frame total, recorded by the caller around its stages; "
                    "processing time only (§9.1)"
                ),
            )
        stages = {stage: values for stage, values in self.samples.items() if stage != TOTAL_STAGE}
        if not stages:
            return summarise_latency(TOTAL_STAGE, ())
        lengths = {len(values) for values in stages.values()}
        if len(lengths) != 1:
            return summarise_latency(
                TOTAL_STAGE,
                (),
                note=(
                    "stages were measured a different number of times, so a per-frame total "
                    "would silently omit a stage's cost"
                ),
            )
        ordered = list(stages)
        totals = [sum(stages[stage][index] for stage in ordered) for index in range(lengths.pop())]
        return summarise_latency(
            TOTAL_STAGE,
            totals,
            note="sum of the measured stages per frame; processing time only (§9.1)",
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, one entry per stage plus the total."""
        payload = {
            stage: self.summary(stage).as_dict()
            for stage in sorted(self.samples)
            if stage != TOTAL_STAGE
        }
        payload[TOTAL_STAGE] = self.total().as_dict()
        payload["counts"] = dict(sorted(self.counts.items()))
        return payload


def memory_snapshot(*, device: str = "cpu") -> dict[str, Any]:
    """Process and accelerator memory, with §23.1's unified-memory statement.

    Args:
        device: The configured torch device. MPS figures are read only when the device is
            ``mps``, because asking torch for MPS memory on a CPU-only host would either
            raise or return a meaningless zero.

    Returns:
        Process RSS, peak RSS, thread count, and — on MPS — allocated and reserved
        accelerator memory, each labelled as unified rather than discrete.
    """
    usage = process_usage()
    payload: dict[str, Any] = {
        "process_rss_bytes": usage.rss_bytes,
        "process_peak_rss_bytes": usage.peak_rss_bytes,
        "threads": threading.active_count(),
        "cpu_user_s": round(usage.cpu_user_s, 3),
        "cpu_system_s": round(usage.cpu_system_s, 3),
        "device": device,
        "accelerator": _accelerator_memory(device),
        "note": (
            "On Apple Silicon the accelerator shares process memory, so the MPS figures "
            "below are unified-memory allocations, not discrete VRAM (§23.1)."
        ),
    }
    load = load_average()
    payload["load_average"] = None if load is None else [round(value, 3) for value in load]
    return payload


def _accelerator_memory(device: str) -> dict[str, Any]:
    """MPS allocated/reserved memory when the device is MPS, else a stated absence."""
    if device != "mps":
        return {
            "kind": "none",
            "reason": f"device is {device!r}; no accelerator memory is allocated",
        }
    try:
        import torch  # noqa: PLC0415 - imported only for a live accelerator figure

        if not torch.backends.mps.is_available():  # pragma: no cover - host dependent
            return {"kind": "mps", "available": False, "reason": "MPS not available on this host"}
        return {
            "kind": "mps",
            "available": True,
            "allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_bytes": int(torch.mps.driver_allocated_memory()),
            "unified_memory": True,
            "note": "unified memory shared with the process; not discrete VRAM (§23.1)",
        }
    except Exception as exc:  # noqa: BLE001 - a measurement that cannot be taken is reported
        return {"kind": "mps", "error": f"{type(exc).__name__}: {exc}"}


def summarise_throughput(
    frames: int, elapsed_s: float, *, people: int | None = None
) -> dict[str, Any]:
    """Frames per second over a measured wall time, with the sample size beside it.

    Args:
        frames: Frames processed.
        elapsed_s: Wall time over which they were processed.
        people: Total person-observations, so a per-person figure can be derived.

    Returns:
        Throughput figures, or a refusal when the elapsed time is not positive.
    """
    if elapsed_s <= 0:
        return {"status": "refused", "reason": "elapsed time must be positive"}
    payload: dict[str, Any] = {
        "frames": frames,
        "elapsed_s": round(elapsed_s, 3),
        "frames_per_second": round(frames / elapsed_s, 3),
        "note": (
            "wall-clock throughput on this machine over this clip; not a claim about a live "
            "camera, and not an accuracy figure"
        ),
    }
    if people is not None and frames > 0:
        payload["people"] = people
        payload["people_per_frame"] = round(people / frames, 3)
    return payload


def stage_map(timer: StageTimer) -> Mapping[str, LatencySummary]:
    """Read-only view of every stage summary, for a caller that wants objects."""
    return {stage: timer.summary(stage) for stage in sorted(timer.samples)}
