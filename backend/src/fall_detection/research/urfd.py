"""UR Fall Detection Dataset adapter — the held-out test set (THESIS_PLAN §6).

Source: https://fenix.ur.edu.pl/~mkepski/ds/uf.html (Kwolek & Kępski, 2014), licensed
CC BY-NC-SA 4.0 for non-commercial academic use. Seventy sequences from the front camera
``cam0`` (parallel to the floor): 30 falls and 40 activities of daily living. Only cam0 is
used, because the ADLs were recorded with cam0 alone.

What this adapter relies on (checked against the published files):

* ``{seq}-data.csv`` has one row per frame: ``frame number, time (ms)[, SV_total]``.
  **That time column is the acquisition time**; the MP4's 30 fps container rate is never
  used as the time base.
* ``urfall-cam0-falls.csv`` / ``urfall-cam0-adls.csv`` give per-frame labels for a subset
  of frames: ``-1`` not lying, ``0`` the transitional "falling" pose, ``1`` lying on the
  ground. Frame numbers are 1-based.
* Frames come from either ``{seq}-cam0-rgb.zip`` (640×480 PNGs, the default) or the
  ``{seq}-cam0.mp4`` preview, whose right half is a 320×240 copy of the RGB stream (the
  left half is the depth map). The chosen source is recorded.

Two findings are encoded here because a report must state them:

* Every fall sequence **ends within ~2.1 s of the first lying frame**, so a 3 s long-lie
  alarm can never be observed on URFD. Fall detection is therefore scored at the impact
  stage on URFD, and :func:`post_impact_seconds` reports the available footage.
* 16 of the 40 ADL sequences contain intentional lying (label 1) — the bed/sofa
  scenarios the state machine must not raise an alarm for.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from fall_detection.research.sequences import LabelledSequence, SequenceFormatError

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    import numpy as np

__all__ = [
    "BASE_URL",
    "FALL_IDS",
    "ADL_IDS",
    "FrameSource",
    "UrfdAsset",
    "assets_for",
    "load_sequence",
    "load_sequences",
    "load_valid_sequences",
    "parse_label_csv",
    "parse_timestamps",
    "post_impact_seconds",
]

BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
FALL_IDS: tuple[str, ...] = tuple(f"fall-{index:02d}" for index in range(1, 31))
ADL_IDS: tuple[str, ...] = tuple(f"adl-{index:02d}" for index in range(1, 41))
LABEL_FILES: Mapping[str, str] = {"fall": "urfall-cam0-falls.csv", "adl": "urfall-cam0-adls.csv"}

FrameSource = Literal["rgb", "mp4"]


@dataclass(frozen=True, slots=True)
class UrfdAsset:
    """One file to download."""

    name: str
    url: str
    kind: str
    sequence_id: str


def assets_for(
    sequence_ids: Sequence[str], *, source: FrameSource = "rgb"
) -> tuple[UrfdAsset, ...]:
    """The files a set of sequences needs, label files included."""
    assets = [
        UrfdAsset(name=name, url=BASE_URL + name, kind="labels", sequence_id="all")
        for name in LABEL_FILES.values()
    ]
    for sequence_id in sequence_ids:
        video = f"{sequence_id}-cam0-rgb.zip" if source == "rgb" else f"{sequence_id}-cam0.mp4"
        assets.append(
            UrfdAsset(name=video, url=BASE_URL + video, kind=source, sequence_id=sequence_id)
        )
        data = f"{sequence_id}-data.csv"
        assets.append(
            UrfdAsset(name=data, url=BASE_URL + data, kind="timestamps", sequence_id=sequence_id)
        )
    return tuple(assets)


def parse_timestamps(text: str) -> tuple[float, ...]:
    """Parse ``{seq}-data.csv`` into seconds from the first frame.

    Raises:
        SequenceFormatError: Frame numbers are not 1..N in order, or time goes backwards.
    """
    times_ms: list[float] = []
    for line_number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if not row or not row[0].strip():
            continue
        try:
            frame = int(row[0])
            time_ms = float(row[1])
        except (ValueError, IndexError) as exc:
            raise SequenceFormatError(f"line {line_number}: not 'frame,time_ms': {row}") from exc
        if frame != len(times_ms) + 1:
            raise SequenceFormatError(
                f"line {line_number}: frame {frame} out of order (expected {len(times_ms) + 1})"
            )
        if times_ms and time_ms < times_ms[-1]:
            raise SequenceFormatError(f"line {line_number}: time goes backwards")
        times_ms.append(time_ms)
    if not times_ms:
        raise SequenceFormatError("no frames")
    origin = times_ms[0]
    return tuple((value - origin) / 1000.0 for value in times_ms)


def parse_label_csv(text: str) -> dict[str, dict[int, int]]:
    """Parse ``urfall-cam0-*.csv`` into ``{sequence: {zero-based frame: label}}``.

    Raises:
        SequenceFormatError: A label outside ``{-1, 0, 1}`` or a malformed row.
    """
    labels: OrderedDict[str, dict[int, int]] = OrderedDict()
    for line_number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if not row or not row[0].strip():
            continue
        try:
            sequence_id = row[0].strip()
            frame = int(row[1])
            label = int(row[2])
        except (ValueError, IndexError) as exc:
            raise SequenceFormatError(f"line {line_number}: malformed row {row}") from exc
        if label not in (-1, 0, 1):
            raise SequenceFormatError(f"line {line_number}: unknown label {label}")
        if frame < 1:
            raise SequenceFormatError(f"line {line_number}: frame numbers are 1-based")
        labels.setdefault(sequence_id, {})[frame - 1] = label
    return dict(labels)


def post_impact_seconds(sequence: LabelledSequence) -> float | None:
    """Footage after the first lying frame, in seconds; ``None`` without one."""
    first = sequence.first_label_time(1)
    if first is None or not sequence.timestamps_s:
        return None
    return sequence.timestamps_s[-1] - first


def load_sequence(
    root: Path,
    sequence_id: str,
    *,
    labels: Mapping[str, Mapping[int, int]],
    source: FrameSource = "rgb",
) -> LabelledSequence:
    """Build one sequence from downloaded files.

    Raises:
        FileNotFoundError: A required file has not been downloaded.
        SequenceFormatError: The frame count of the video and the timestamp file differ.
    """
    is_fall = sequence_id.startswith("fall-")
    timestamps = parse_timestamps((root / f"{sequence_id}-data.csv").read_text(encoding="utf-8"))
    if source == "rgb":
        archive = root / f"{sequence_id}-cam0-rgb.zip"
        names = _zip_frames(archive)
        count = len(names)
        resolution = "640x480"
    else:
        archive = root / f"{sequence_id}-cam0.mp4"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        count = _mp4_frame_count(archive)
        names = ()
        resolution = "320x240 (right half of the depth|RGB preview)"
    if count != len(timestamps):
        raise SequenceFormatError(
            f"{sequence_id}: {count} frames in {archive.name} but {len(timestamps)} rows in "
            f"{sequence_id}-data.csv; refusing to pair them"
        )

    def reader() -> Iterator[tuple[int, np.ndarray]]:
        if source == "rgb":
            yield from _read_zip(archive, names)
        else:
            yield from _read_mp4_rgb_half(archive)

    return LabelledSequence(
        dataset="urfd",
        sequence_id=sequence_id,
        is_fall=is_fall,
        activity="fall" if is_fall else "adl",
        group=sequence_id,
        timestamps_s=timestamps,
        time_source=f"{sequence_id}-data.csv time column (acquisition ms)",
        reader=reader,
        frame_labels=dict(labels.get(sequence_id, {})),
        metadata={"frame_source": source, "resolution": resolution, "camera": "cam0"},
    )


def load_sequences(
    root: Path,
    sequence_ids: Sequence[str] | None = None,
    *,
    source: FrameSource = "rgb",
) -> list[LabelledSequence]:
    """Load several sequences; label files must be present in ``root``."""
    labels: dict[str, dict[int, int]] = {}
    for name in LABEL_FILES.values():
        labels.update(parse_label_csv((root / name).read_text(encoding="utf-8")))
    ids = list(sequence_ids) if sequence_ids is not None else [*FALL_IDS, *ADL_IDS]
    return [load_sequence(root, sequence_id, labels=labels, source=source) for sequence_id in ids]


def load_valid_sequences(
    root: Path,
    sequence_ids: Sequence[str] | None = None,
    *,
    source: FrameSource = "rgb",
) -> tuple[list[LabelledSequence], dict[str, str]]:
    """Load every sequence that is well formed and list the ones that are not.

    A malformed sequence (for example adl-37, whose archive holds 350 frames but whose
    timestamp file has 330 rows) is excluded whole, with the reason, rather than repaired:
    pairing the first N frames with N timestamps would be a guess. A missing file still
    raises ``FileNotFoundError``, because that is an incomplete download, not bad data.
    """
    labels: dict[str, dict[int, int]] = {}
    for name in LABEL_FILES.values():
        labels.update(parse_label_csv((root / name).read_text(encoding="utf-8")))
    ids = list(sequence_ids) if sequence_ids is not None else [*FALL_IDS, *ADL_IDS]
    sequences: list[LabelledSequence] = []
    excluded: dict[str, str] = {}
    for sequence_id in ids:
        try:
            sequences.append(load_sequence(root, sequence_id, labels=labels, source=source))
        except SequenceFormatError as exc:
            excluded[sequence_id] = str(exc)
    return sequences, excluded


def _zip_frames(archive: Path) -> tuple[str, ...]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as handle:
        names = sorted(
            name
            for name in handle.namelist()
            if name.lower().endswith(".png") and not Path(name).name.startswith("._")
        )
    if not names:
        raise SequenceFormatError(f"{archive.name} contains no PNG frames")
    return tuple(names)


def _read_zip(archive: Path, names: Sequence[str]) -> Iterator[tuple[int, np.ndarray]]:
    import cv2
    import numpy as np

    with zipfile.ZipFile(archive) as handle:
        for index, name in enumerate(names):
            data = np.frombuffer(handle.read(name), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                raise SequenceFormatError(f"{archive.name}:{name} is not a decodable image")
            yield index, frame


def _mp4_frame_count(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        count = 0
        while capture.grab():
            count += 1
        return count
    finally:
        capture.release()


def _read_mp4_rgb_half(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            width = frame.shape[1]
            yield index, frame[:, width // 2 :].copy()
            index += 1
    finally:
        capture.release()
