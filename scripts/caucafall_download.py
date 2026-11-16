#!/usr/bin/env python
"""Download the CAUCAFall activity videos into ``datasets/raw/caucafall``.

CAUCAFall (Mendeley Data DOI 10.17632/7w7fccy7ky.4, CC BY 4.0) is the calibration set.
The first run lists the 100 activity videos through the Mendeley public API and writes
``datasets/manifests/caucafall_videos.json`` (URLs and SHA-256 checksums); later runs read
that manifest, so every file is verified against a recorded checksum.

Examples::

    uv run --project backend python scripts/caucafall_download.py
    uv run --project backend python scripts/caucafall_download.py --refresh-manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fall_detection.research.acquisition import AssetRequest, acquire
from fall_detection.research.caucafall import build_manifest, load_manifest, manifest_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "datasets" / "manifests" / "caucafall_videos.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "datasets" / "raw" / "caucafall")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--refresh-manifest", action="store_true", help="re-list the files from the API"
    )
    parser.add_argument("--plan", action="store_true", help="list the videos and exit")
    args = parser.parse_args()

    if args.refresh_manifest or not args.manifest.is_file():
        videos = build_manifest()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest_payload(videos), indent=2) + "\n")
        print(f"wrote {args.manifest} ({len(videos)} videos)")
    else:
        videos = load_manifest(args.manifest)

    if args.plan:
        for video in videos:
            print(f"{video.sequence_id:32s} {video.size} {video.url}")
        return 0

    failed = 0
    for video in videos:
        result = acquire(
            AssetRequest(
                url=video.url,
                destination=video.local_path(args.root),
                dataset_id="caucafall",
                recording_id=video.sequence_id,
                kind="video",
                expected_bytes=video.size,
                expected_sha256=video.sha256,
            ),
            progress=lambda message: print(message, file=sys.stderr),
        )
        print(f"[{result.status}] {video.sequence_id}")
        failed += 0 if result.ok else 1
    print(f"{len(videos) - failed} ok, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
