"""CAUCAFall adapter — the calibration set (THESIS_PLAN §6).

Source: Eraso et al. (2022), *Dataset for human fall recognition in an uncontrolled
environment*, Data in Brief; Mendeley Data DOI 10.17632/7w7fccy7ky.4, CC BY 4.0.
Ten subjects in a real home, each recorded performing ten activities:

* five falls — ``Fall backwards``, ``Fall forward``, ``Fall left``, ``Fall right``,
  ``Fall sitting``;
* five activities of daily living — ``Hop``, ``Kneel``, ``Pick up object``, ``Sit down``,
  ``Walk``.

Each activity folder holds one AVI plus per-frame PNGs with YOLO-format labels
(``nofall``/``fall``). Calibration needs only the sequence-level truth, which the folder
name gives, so this adapter downloads the AVIs (~1 MB each) rather than ~5 GB of PNGs.

Two limits are recorded rather than hidden:

* **The frame rate is the AVI container's** (20 fps in the files checked). CAUCAFall
  publishes no per-frame acquisition timestamps, so ``rate_from_container`` is reported.
* **A video and its label folder can differ by a frame** (one checked folder has 125 AVI
  frames and 126 label files). Per-frame labels are therefore not paired with AVI frames;
  only the sequence-level truth is used.

The manifest of file URLs and checksums is built once from the Mendeley public API and
committed, so a later download needs no API listing and every file is verified.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fall_detection.research.sequences import LabelledSequence, SequenceFormatError

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    import numpy as np

__all__ = [
    "ADL_ACTIVITIES",
    "DATASET_ID",
    "FALL_ACTIVITIES",
    "CaucaVideo",
    "build_manifest",
    "load_manifest",
    "load_sequences",
]

DATASET_ID = "7w7fccy7ky"
VERSION = 4
API = "https://data.mendeley.com/public-api/datasets"
FALL_ACTIVITIES: tuple[str, ...] = (
    "Fall backwards",
    "Fall forward",
    "Fall left",
    "Fall right",
    "Fall sitting",
)
ADL_ACTIVITIES: tuple[str, ...] = ("Hop", "Kneel", "Pick up object", "Sit down", "Walk")

JsonFetcher = Callable[[str], Any]


@dataclass(frozen=True, slots=True)
class CaucaVideo:
    """One activity video in the manifest."""

    subject: str
    activity: str
    filename: str
    url: str
    sha256: str | None
    size: int | None

    @property
    def is_fall(self) -> bool:
        return self.activity in FALL_ACTIVITIES

    @property
    def sequence_id(self) -> str:
        return f"{self.subject}/{self.activity}"

    def local_path(self, root: Path) -> Path:
        return root / self.subject / f"{self.activity.replace(' ', '_')}.avi"

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "activity": self.activity,
            "filename": self.filename,
            "url": self.url,
            "sha256": self.sha256,
            "size": self.size,
        }


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "fall-detection-research/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https host
        return json.loads(response.read().decode("utf-8"))


def build_manifest(fetch: JsonFetcher = _fetch_json) -> list[CaucaVideo]:
    """List every activity AVI through the Mendeley public API.

    Raises:
        SequenceFormatError: A subject/activity folder has no AVI or an unknown activity.
    """
    folders = fetch(f"{API}/{DATASET_ID}/folders/{VERSION}")
    by_id = {folder["id"]: folder for folder in folders}
    videos: list[CaucaVideo] = []
    for folder in folders:
        parent = by_id.get(folder.get("parent_id"))
        if parent is None or not str(parent.get("name", "")).startswith("Subject"):
            continue
        activity = str(folder["name"])
        if activity not in FALL_ACTIVITIES and activity not in ADL_ACTIVITIES:
            raise SequenceFormatError(f"unknown CAUCAFall activity folder {activity!r}")
        files = fetch(f"{API}/{DATASET_ID}/files?folder_id={folder['id']}&version={VERSION}")
        avis = [item for item in files if str(item.get("filename", "")).lower().endswith(".avi")]
        if len(avis) != 1:
            raise SequenceFormatError(
                f"{parent['name']}/{activity}: expected one AVI, found {len(avis)}"
            )
        item = avis[0]
        details = item.get("content_details", {})
        videos.append(
            CaucaVideo(
                subject=str(parent["name"]),
                activity=activity,
                filename=str(item["filename"]),
                url=str(details["download_url"]),
                sha256=details.get("sha256_hash"),
                size=item.get("size"),
            )
        )
    videos.sort(key=lambda video: (_subject_number(video.subject), video.activity))
    return videos


def load_manifest(path: Path) -> list[CaucaVideo]:
    """Read a manifest written by ``scripts/caucafall_download.py``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [CaucaVideo(**entry) for entry in payload["videos"]]


def load_sequences(
    root: Path,
    videos: Sequence[CaucaVideo],
    *,
    fps_override: float | None = None,
) -> list[LabelledSequence]:
    """Build sequences from downloaded AVIs.

    Raises:
        FileNotFoundError: A video has not been downloaded.
        SequenceFormatError: A video reports no frame rate and none was given.
    """
    sequences = []
    for video in videos:
        path = video.local_path(root)
        if not path.is_file():
            raise FileNotFoundError(path)
        count, fps = _probe(path)
        rate = fps_override or fps
        if not rate or rate <= 0:
            raise SequenceFormatError(f"{path.name}: no frame rate; pass fps_override")
        timestamps = tuple(index / rate for index in range(count))
        sequences.append(
            LabelledSequence(
                dataset="caucafall",
                sequence_id=video.sequence_id,
                is_fall=video.is_fall,
                activity=video.activity,
                group=video.subject,
                timestamps_s=timestamps,
                time_source=f"AVI container rate ({rate:g} fps); no acquisition timestamps",
                reader=_reader(path),
                metadata={
                    "rate_from_container": fps_override is None,
                    "fps": rate,
                    "file": video.filename,
                },
            )
        )
    return sequences


def _reader(path: Path) -> Callable[[], Iterator[tuple[int, np.ndarray]]]:
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


def _probe(path: Path) -> tuple[int, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        count = 0
        while capture.grab():
            count += 1
        return count, fps
    finally:
        capture.release()


def _subject_number(name: str) -> int:
    digits = "".join(character for character in name if character.isdigit())
    return int(digits) if digits else 0


def manifest_payload(videos: Sequence[CaucaVideo]) -> Mapping[str, Any]:
    """The committed manifest's JSON content."""
    return {
        "dataset": "caucafall",
        "doi": "10.17632/7w7fccy7ky.4",
        "license": "CC BY 4.0",
        "citation": "Eraso, J. C., Munoz, E., Munoz, M., & Pinto, J. (2022). Dataset CAUCAFall: "
        "Recognition of human falls in an uncontrolled environment. Data in Brief, 45, 108610.",
        "fall_activities": list(FALL_ACTIVITIES),
        "adl_activities": list(ADL_ACTIVITIES),
        "videos": [video.as_dict() for video in videos],
    }
