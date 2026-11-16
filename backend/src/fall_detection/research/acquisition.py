"""Resumable acquisition of public dataset assets.

``docs/PUBLIC_DATASET_PLAN.md`` fixes the acquisition protocol: "``--continue-at -``
resumes an interrupted download. Keep an incomplete video as ``.part``, and rename it only
after curl succeeds." This module implements exactly that, plus the checks the plan
requires before a file may be used:

* a download that already exists and matches its recorded size **and** SHA-256 is never
  fetched again (the task's "do not download valid files again");
* an incomplete file stays ``.part`` and is resumed rather than restarted;
* completeness is verified against the publisher's declared content length *and*, when
  available, a recorded SHA-256;
* the verified file is renamed into place atomically.

FFmpeg is not involved. ``curl`` is invoked with an argument list and ``shell=False``,
following the same rule §24.2 applies to the decoder, and a URL is validated to be
``https`` before it is passed to a subprocess.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "AssetRequest",
    "AssetResult",
    "acquire",
    "head_content_length",
    "merge_asset_records",
    "sha256_file",
    "verify_asset",
]

_READ_CHUNK = 1024 * 1024
_TIMEOUT_S = 60.0


class AcquisitionError(RuntimeError):
    """A download could not be completed or failed verification."""


@dataclass(frozen=True, slots=True)
class AssetRequest:
    """One file to acquire.

    Attributes:
        url: Absolute ``https`` URL.
        destination: Final path. The partial file is this path plus ``.part``.
        dataset_id: Source dataset identifier, for the report.
        recording_id: Recording the asset belongs to.
        kind: Asset kind, e.g. ``"video"`` or ``"boxes_and_source_activity_labels"``.
        expected_bytes: Publisher-declared content length, when known.
        expected_sha256: Recorded checksum, when known. A verified file is reused only
            when this matches.
    """

    url: str
    destination: Path
    dataset_id: str
    recording_id: str
    kind: str
    expected_bytes: int | None = None
    expected_sha256: str | None = None

    @property
    def part_path(self) -> Path:
        """Path of the in-progress file."""
        return self.destination.with_name(self.destination.name + ".part")


@dataclass(frozen=True, slots=True)
class AssetResult:
    """Outcome of one acquisition.

    Attributes:
        request: The asset that was requested.
        status: ``reused``, ``downloaded``, ``resumed``, or ``failed``.
        size_bytes: Size of the file now in place, or of the partial file on failure.
        sha256: Digest of the completed file, when it completed.
        detail: Human-readable explanation, including the failure reason.
    """

    request: AssetRequest
    status: str
    size_bytes: int
    sha256: str | None
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Whether the asset is present and verified."""
        return self.status in {"reused", "downloaded", "resumed"}

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "dataset_id": self.request.dataset_id,
            "recording_id": self.request.recording_id,
            "kind": self.request.kind,
            "url": self.request.url,
            "local_path": str(self.request.destination),
            "status": self.status,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "detail": self.detail,
        }


def sha256_file(path: Path) -> str:
    """Return the file's SHA-256, streamed so a multi-gigabyte video fits in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_content_length(url: str) -> int | None:
    """Return the server's ``Content-Length``, or ``None`` when it is not offered.

    A missing length is not an error: the caller then verifies completeness by checksum
    or by a clean curl exit status alone, and says so in the report rather than assuming
    a size.
    """
    request = urllib.request.Request(url, method="HEAD")  # noqa: S310 - https validated by caller
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
            length = response.headers.get("Content-Length")
    except urllib.error.URLError, TimeoutError, OSError:
        return None
    if length is None:
        return None
    try:
        return int(length)
    except ValueError:
        return None


def verify_asset(
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[bool, str]:
    """Check a completed file against the recorded size and checksum.

    Returns:
        ``(True, "")`` when the file is usable, else ``(False, reason)``. When no
        expectation is recorded the file is accepted with a note, because refusing every
        unhashed file would block the first acquisition of that file.
    """
    if not path.is_file():
        return False, "file does not exist"
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        return False, f"size {actual_bytes} != recorded {expected_bytes}"
    if expected_sha256 is not None:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha256:
            return False, f"sha256 {actual_sha} != recorded {expected_sha256}"
    if expected_bytes is None and expected_sha256 is None:
        return True, "accepted without a recorded size or checksum"
    return True, ""


def _curl_available() -> str:
    """Return the curl path, or raise a clear error."""
    found = shutil.which("curl")
    if found is None:
        raise AcquisitionError("curl is not installed; resumable downloads require it")
    return found


def _run_curl(url: str, destination: Path, *, resume: bool) -> subprocess.CompletedProcess[str]:
    """Invoke curl with an argument list and ``shell=False``.

    ``--fail`` makes an HTTP error a non-zero exit rather than a saved error page, and
    ``--location`` follows the publisher's redirects. ``--continue-at -`` asks curl to
    resume from the partial file's own length.
    """
    command = [
        _curl_available(),
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "--retry-delay",
        "5",
    ]
    if resume:
        command += ["--continue-at", "-"]
    command += ["--output", str(destination), url]
    return subprocess.run(  # noqa: S603 - argument list, shell=False
        command, capture_output=True, text=True, check=False
    )


def acquire(
    request: AssetRequest,
    *,
    progress: Callable[[str], None] | None = None,
    force: bool = False,
) -> AssetResult:
    """Acquire one asset, reusing or resuming whenever possible.

    Args:
        request: What to fetch.
        progress: Optional callback for status lines.
        force: Re-download even when the existing file verifies.

    Returns:
        The outcome. A failure leaves the ``.part`` file in place so the next run resumes.

    Raises:
        AcquisitionError: The URL is not ``https``, which would make the download
            unverifiable over an unencrypted channel.
    """
    say = progress or (lambda _message: None)
    if not request.url.startswith("https://"):
        raise AcquisitionError(f"refusing a non-https URL: {request.url}")

    request.destination.parent.mkdir(parents=True, exist_ok=True)

    if request.destination.is_file() and not force:
        ok, reason = verify_asset(
            request.destination,
            expected_bytes=request.expected_bytes,
            expected_sha256=request.expected_sha256,
        )
        if ok:
            size = request.destination.stat().st_size
            digest = request.expected_sha256 or sha256_file(request.destination)
            say(f"reuse {request.destination.name} ({size} bytes)")
            return AssetResult(
                request=request,
                status="reused",
                size_bytes=size,
                sha256=digest,
                detail=reason or "verified against the recorded size and checksum",
            )
        say(f"existing {request.destination.name} is not valid ({reason}); re-fetching")
        request.destination.unlink(missing_ok=True)

    resume = request.part_path.is_file() and request.part_path.stat().st_size > 0
    if resume:
        say(f"resume {request.part_path.name} from {request.part_path.stat().st_size} bytes")
    else:
        say(f"download {request.url}")

    result = _run_curl(request.url, request.part_path, resume=resume)
    if result.returncode != 0:
        partial = request.part_path.stat().st_size if request.part_path.is_file() else 0
        return AssetResult(
            request=request,
            status="failed",
            size_bytes=partial,
            sha256=None,
            detail=f"curl exit {result.returncode}: {result.stderr.strip()[:300]}",
        )

    ok, reason = verify_asset(
        request.part_path,
        expected_bytes=request.expected_bytes,
        expected_sha256=request.expected_sha256,
    )
    if not ok:
        return AssetResult(
            request=request,
            status="failed",
            size_bytes=request.part_path.stat().st_size,
            sha256=None,
            detail=f"downloaded file failed verification: {reason}",
        )

    digest = request.expected_sha256 or sha256_file(request.part_path)
    request.part_path.replace(request.destination)
    size = request.destination.stat().st_size
    status = "resumed" if resume else "downloaded"
    say(f"{status} {request.destination.name} ({size} bytes)")
    return AssetResult(
        request=request,
        status=status,
        size_bytes=size,
        sha256=digest,
        detail=reason or "verified against the recorded size and checksum",
    )


def merge_asset_records(
    previous: Sequence[Mapping[str, Any]] | None,
    current: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge a previous acquisition record with this run's results.

    The evidence file accumulates across runs, and that is a correctness requirement rather
    than tidiness: a later run that only *verifies* an already-downloaded file produces
    results for the assets it was asked about, so replacing the file would erase the record
    of how every earlier asset was acquired. That record is what lets the next run reuse a
    valid file instead of re-downloading it, and it is the provenance §13.3 asks for.

    Args:
        previous: ``assets`` from the existing report, or ``None``.
        current: Results from this run.

    Returns:
        One entry per ``(recording_id, kind)``, sorted, with a later run's entry replacing an
        earlier one for the same asset and every other earlier entry kept. An entry missing
        either key is skipped, because it cannot be identified and merging it would create a
        duplicate.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in list(previous or ()) + list(current):
        recording_id = entry.get("recording_id")
        kind = entry.get("kind")
        if recording_id is None or kind is None:
            continue
        merged[(str(recording_id), str(kind))] = dict(entry)
    return [merged[key] for key in sorted(merged)]


def load_recorded_assets(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Read sizes and checksums from a previously written evidence report.

    The seed report (``experiments/reports/public_dataset_seed.json``) already records
    the size and SHA-256 of each asset it measured. Reading them back is what lets
    :func:`acquire` verify an existing file instead of downloading it again, which the
    task requires explicitly.

    Returns:
        ``{(url, local_path): {"size_bytes": int, "sha256": str}}``. A missing or
        malformed report yields an empty mapping rather than an error: the caller then
        simply has nothing to verify against.
    """
    recorded: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return recorded
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return recorded
    for asset in payload.get("assets", []):
        url = asset.get("url")
        local = asset.get("local_path")
        if not url or not local:
            continue
        entry: dict[str, Any] = {}
        if isinstance(asset.get("size_bytes"), int):
            entry["size_bytes"] = asset["size_bytes"]
        if isinstance(asset.get("sha256"), str):
            entry["sha256"] = asset["sha256"]
        if entry:
            recorded[(url, local)] = entry
    return recorded
