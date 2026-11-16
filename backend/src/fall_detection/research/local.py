"""Locally recorded test videos (THESIS_PLAN §6.4).

The draft also tests the system on the author's own recordings: falls and fall-like
activities (sitting down quickly, bending to pick something up, lying on a bed) at 2, 3 and
4 m from the camera and under different lighting. Each video is described by a small JSON
sidecar with the same name::

    {
      "is_fall": true,
      "activity": "fall_forward",
      "distance_m": 3,
      "lighting": "normal",
      "lying_from_s": 4.2,
      "notes": "optional"
    }

``lying_from_s`` (optional) is when the person is first on the floor; it gives the
detection and alarm delays. Frame times are the container's presentation timestamps, which
for a phone or webcam recording are the capture times.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fall_detection.research.sequences import LabelledSequence, SequenceFormatError

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    import numpy as np

__all__ = ["VIDEO_SUFFIXES", "load_local_sequences", "parse_sidecar"]

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")


def parse_sidecar(payload: dict[str, Any], *, name: str) -> dict[str, Any]:
    """Validate a sidecar.

    Raises:
        SequenceFormatError: ``is_fall`` is missing or not a boolean.
    """
    if not isinstance(payload.get("is_fall"), bool):
        raise SequenceFormatError(f"{name}: 'is_fall' must be true or false")
    lying = payload.get("lying_from_s")
    if lying is not None and (not isinstance(lying, int | float) or lying < 0):
        raise SequenceFormatError(f"{name}: 'lying_from_s' must be a non-negative number")
    return payload


def load_local_sequences(folder: Path) -> list[LabelledSequence]:
    """Every video in ``folder`` that has a JSON sidecar."""
    sequences = []
    for video in sorted(folder.iterdir()):
        if video.suffix.lower() not in VIDEO_SUFFIXES or video.name.startswith("._"):
            continue
        sidecar_path = video.with_suffix(".json")
        if not sidecar_path.is_file():
            continue
        sidecar = parse_sidecar(
            json.loads(sidecar_path.read_text(encoding="utf-8")), name=sidecar_path.name
        )
        timestamps = _timestamps(video)
        labels: dict[int, int] = {}
        lying = sidecar.get("lying_from_s")
        if lying is not None:
            labels = {index: (1 if t >= float(lying) else -1) for index, t in enumerate(timestamps)}
        sequences.append(
            LabelledSequence(
                dataset="local",
                sequence_id=video.stem,
                is_fall=bool(sidecar["is_fall"]),
                activity=str(sidecar.get("activity", "fall" if sidecar["is_fall"] else "adl")),
                group=video.stem,
                timestamps_s=timestamps,
                time_source="container presentation timestamps",
                reader=_reader(video),
                frame_labels=labels,
                metadata={
                    key: sidecar[key]
                    for key in ("distance_m", "lighting", "notes")
                    if key in sidecar
                },
            )
        )
    return sequences


def _timestamps(path: Path) -> tuple[float, ...]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        times: list[float] = []
        while capture.grab():
            times.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
    finally:
        capture.release()
    if not times:
        raise SequenceFormatError(f"{path.name} has no decodable frames")
    origin = times[0]
    relative = [value - origin for value in times]
    if any(later <= earlier for earlier, later in zip(relative, relative[1:], strict=False)):
        raise SequenceFormatError(f"{path.name}: presentation timestamps are not increasing")
    return tuple(relative)


def _reader(path: Path) -> Any:
    def read() -> Iterator[tuple[int, np.ndarray]]:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                yield index, frame
                index += 1
        finally:
            capture.release()

    return read
