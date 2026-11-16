"""Tests for resumable, verified acquisition (Phase 7, ``docs/PUBLIC_DATASET_PLAN.md``).

The network is never used: an ``https`` URL is pointed at a local ``file://``-style server
only where the test needs a real transfer, and the verification logic — which is where the
risk lives — is tested directly against files on disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fall_detection.research.acquisition import (
    AcquisitionError,
    AssetRequest,
    acquire,
    head_content_length,
    load_recorded_assets,
    merge_asset_records,
    sha256_file,
    verify_asset,
)


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class TestVerifyAsset:
    def test_a_matching_size_and_checksum_verifies(self, tmp_path: Path) -> None:
        payload = b"dataset payload"
        path = _write(tmp_path / "day.mat", payload)
        digest = hashlib.sha256(payload).hexdigest()
        ok, reason = verify_asset(path, expected_bytes=len(payload), expected_sha256=digest)
        assert ok
        assert reason == ""

    def test_a_wrong_size_fails(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "day.mat", b"short")
        ok, reason = verify_asset(path, expected_bytes=999)
        assert not ok
        assert "size" in reason

    def test_a_wrong_checksum_fails(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "day.mat", b"payload")
        ok, reason = verify_asset(path, expected_sha256="0" * 64)
        assert not ok
        assert "sha256" in reason

    def test_a_missing_file_fails(self, tmp_path: Path) -> None:
        ok, reason = verify_asset(tmp_path / "absent.mat", expected_bytes=1)
        assert not ok
        assert "does not exist" in reason

    def test_an_unrecorded_file_is_accepted_with_a_note(self, tmp_path: Path) -> None:
        """The first acquisition of a file has nothing to verify against."""
        path = _write(tmp_path / "new.mat", b"payload")
        ok, reason = verify_asset(path)
        assert ok
        assert "without a recorded size or checksum" in reason

    def test_sha256_file_matches_hashlib(self, tmp_path: Path) -> None:
        payload = b"x" * 100_000
        path = _write(tmp_path / "big.bin", payload)
        assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


class TestAcquire:
    def test_a_valid_existing_file_is_reused_without_downloading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The task requires not re-downloading a valid file."""
        payload = b"already here"
        destination = _write(tmp_path / "day.mat", payload)
        digest = hashlib.sha256(payload).hexdigest()
        request = AssetRequest(
            url="https://example.invalid/day.mat",
            destination=destination,
            dataset_id="edinburgh_office",
            recording_id="day02",
            kind="boxes_and_source_activity_labels",
            expected_bytes=len(payload),
            expected_sha256=digest,
        )

        def explode(*_args, **_kwargs):  # pragma: no cover - only runs on a bug
            raise AssertionError("a valid file must not be re-downloaded")

        monkeypatch.setattr("fall_detection.research.acquisition._run_curl", explode, raising=True)
        result = acquire(request)
        assert result.status == "reused"
        assert result.ok
        assert result.size_bytes == len(payload)
        assert result.sha256 == digest

    def test_a_corrupt_existing_file_is_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        destination = _write(tmp_path / "day.mat", b"corrupt")
        payload = b"the real payload"
        digest = hashlib.sha256(payload).hexdigest()
        request = AssetRequest(
            url="https://example.invalid/day.mat",
            destination=destination,
            dataset_id="edinburgh_office",
            recording_id="day02",
            kind="boxes_and_source_activity_labels",
            expected_bytes=len(payload),
            expected_sha256=digest,
        )

        def fake_curl(url: str, target: Path, *, resume: bool):
            target.write_bytes(payload)
            return _Completed(0, "", "")

        monkeypatch.setattr(
            "fall_detection.research.acquisition._run_curl", fake_curl, raising=True
        )
        result = acquire(request)
        assert result.status == "downloaded"
        assert destination.read_bytes() == payload

    def test_a_resume_keeps_the_partial_file_and_reports_resumed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        destination = tmp_path / "day_14.avi"
        partial = destination.with_name(destination.name + ".part")
        _write(partial, b"half the video")
        payload = b"half the video and the rest"
        request = AssetRequest(
            url="https://example.invalid/day_14.avi",
            destination=destination,
            dataset_id="edinburgh_office",
            recording_id="day14",
            kind="video",
        )
        seen: dict[str, object] = {}

        def fake_curl(url: str, target: Path, *, resume: bool):
            seen["resume"] = resume
            seen["target"] = target
            target.write_bytes(payload)
            return _Completed(0, "", "")

        monkeypatch.setattr(
            "fall_detection.research.acquisition._run_curl", fake_curl, raising=True
        )
        result = acquire(request)
        assert seen["resume"] is True
        assert seen["target"] == partial
        assert result.status == "resumed"
        assert destination.read_bytes() == payload
        assert not partial.exists()

    def test_a_failure_leaves_the_part_file_for_the_next_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        destination = tmp_path / "day_14.avi"
        request = AssetRequest(
            url="https://example.invalid/day_14.avi",
            destination=destination,
            dataset_id="edinburgh_office",
            recording_id="day14",
            kind="video",
        )

        def fake_curl(url: str, target: Path, *, resume: bool):
            target.write_bytes(b"partial")
            return _Completed(22, "", "HTTP 404")

        monkeypatch.setattr(
            "fall_detection.research.acquisition._run_curl", fake_curl, raising=True
        )
        result = acquire(request)
        assert result.status == "failed"
        assert not result.ok
        assert "curl exit 22" in result.detail
        assert destination.with_name(destination.name + ".part").is_file()
        assert not destination.exists()

    def test_a_size_mismatch_after_download_is_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        destination = tmp_path / "day_14.avi"
        request = AssetRequest(
            url="https://example.invalid/day_14.avi",
            destination=destination,
            dataset_id="edinburgh_office",
            recording_id="day14",
            kind="video",
            expected_bytes=10_000,
        )

        def fake_curl(url: str, target: Path, *, resume: bool):
            target.write_bytes(b"too short")
            return _Completed(0, "", "")

        monkeypatch.setattr(
            "fall_detection.research.acquisition._run_curl", fake_curl, raising=True
        )
        result = acquire(request)
        assert result.status == "failed"
        assert "failed verification" in result.detail
        assert not destination.exists()

    def test_a_non_https_url_is_refused(self, tmp_path: Path) -> None:
        request = AssetRequest(
            url="http://example.invalid/day.mat",
            destination=tmp_path / "day.mat",
            dataset_id="edinburgh_office",
            recording_id="day02",
            kind="boxes_and_source_activity_labels",
        )
        with pytest.raises(AcquisitionError, match="non-https"):
            acquire(request)

    def test_force_re_fetches_a_valid_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = b"payload"
        destination = _write(tmp_path / "day.mat", payload)
        request = AssetRequest(
            url="https://example.invalid/day.mat",
            destination=destination,
            dataset_id="edinburgh_office",
            recording_id="day02",
            kind="boxes_and_source_activity_labels",
            expected_bytes=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )

        def fake_curl(url: str, target: Path, *, resume: bool):
            target.write_bytes(payload)
            return _Completed(0, "", "")

        monkeypatch.setattr(
            "fall_detection.research.acquisition._run_curl", fake_curl, raising=True
        )
        result = acquire(request, force=True)
        assert result.status == "downloaded"


class _Completed:
    """Minimal stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestRecordedAssets:
    def test_reads_sizes_and_checksums_from_an_evidence_report(self, tmp_path: Path) -> None:
        report = tmp_path / "seed.json"
        report.write_text(
            """
            {
              "assets": [
                {
                  "url": "https://example.invalid/day_2.avi",
                  "local_path": "datasets/raw/edinburgh_office/day02/day_2.avi",
                  "size_bytes": 695307302,
                  "sha256": "abc123"
                },
                {"url": "https://example.invalid/other", "local_path": "x", "size_bytes": 5}
              ]
            }
            """,
            encoding="utf-8",
        )
        recorded = load_recorded_assets(report)
        assert recorded[
            ("https://example.invalid/day_2.avi", "datasets/raw/edinburgh_office/day02/day_2.avi")
        ] == {"size_bytes": 695307302, "sha256": "abc123"}
        assert recorded[("https://example.invalid/other", "x")] == {"size_bytes": 5}

    def test_a_missing_report_yields_nothing(self, tmp_path: Path) -> None:
        assert load_recorded_assets(tmp_path / "absent.json") == {}

    def test_a_malformed_report_yields_nothing(self, tmp_path: Path) -> None:
        report = tmp_path / "broken.json"
        report.write_text("{not json", encoding="utf-8")
        assert load_recorded_assets(report) == {}

    def test_the_real_seed_report_is_readable(self) -> None:
        """The committed seed evidence must stay parseable by the acquisition tooling."""
        from pathlib import Path as _Path

        seed = (
            _Path(__file__).resolve().parents[2]
            / "experiments"
            / "reports"
            / "public_dataset_seed.json"
        )
        if not seed.is_file():
            pytest.skip("seed evidence report is not present")
        recorded = load_recorded_assets(seed)
        day2 = (
            "https://homepages.inf.ed.ac.uk/rbf/OFFICEDATA/AVI/day_2.avi",
            "datasets/raw/edinburgh_office/day02/day_2.avi",
        )
        assert day2 in recorded
        assert recorded[day2]["size_bytes"] == 695307302
        assert len(recorded[day2]["sha256"]) == 64


class TestHeadContentLength:
    def test_a_non_https_host_is_not_contacted(self) -> None:
        """An unreachable host returns ``None`` rather than raising."""
        assert head_content_length("https://127.0.0.1:1/nothing.avi") is None


class TestEvidenceAccumulation:
    """The CLI's evidence file must not lose earlier provenance on a later run.

    A verification-only run reuses files and produces results for the assets it was asked
    about. If the report were overwritten, the record of how every earlier asset was
    acquired would disappear — and that record is exactly what lets a later run reuse a
    valid file instead of re-downloading it.
    """

    def test_a_later_run_merges_rather_than_replaces(self) -> None:
        previous = [
            {"recording_id": "day02", "kind": "video", "sha256": "a" * 64},
            {"recording_id": "day02", "kind": "boxes_and_source_activity_labels"},
        ]
        current = [{"recording_id": "day16", "kind": "video", "sha256": "b" * 64}]
        merged = merge_asset_records(previous, current)
        assert {(item["recording_id"], item["kind"]) for item in merged} == {
            ("day02", "video"),
            ("day02", "boxes_and_source_activity_labels"),
            ("day16", "video"),
        }

    def test_a_re_verified_asset_replaces_its_own_earlier_entry(self) -> None:
        previous = [{"recording_id": "day02", "kind": "video", "status": "downloaded"}]
        current = [{"recording_id": "day02", "kind": "video", "status": "reused"}]
        merged = merge_asset_records(previous, current)
        assert len(merged) == 1
        assert merged[0]["status"] == "reused"

    def test_the_result_is_sorted_and_stable(self) -> None:
        previous = [{"recording_id": "day19", "kind": "video"}]
        current = [{"recording_id": "day02", "kind": "video"}]
        merged = merge_asset_records(previous, current)
        assert [item["recording_id"] for item in merged] == ["day02", "day19"]

    def test_an_unidentifiable_entry_is_skipped(self) -> None:
        """Merging an entry with no identity would create a duplicate, not a record."""
        previous = [{"kind": "video"}, {"recording_id": "day02"}]
        current = [{"recording_id": "day16", "kind": "video"}]
        merged = merge_asset_records(previous, current)
        assert len(merged) == 1
        assert merged[0]["recording_id"] == "day16"

    def test_no_previous_record_is_accepted(self) -> None:
        merged = merge_asset_records(None, [{"recording_id": "day02", "kind": "video"}])
        assert len(merged) == 1

    def test_an_empty_current_run_keeps_the_previous_record(self) -> None:
        previous = [{"recording_id": "day02", "kind": "video", "sha256": "a" * 64}]
        assert merge_asset_records(previous, []) == previous
