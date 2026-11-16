#!/usr/bin/env python
"""Download the UR Fall Detection Dataset (cam0) into ``datasets/raw/urfd``.

Fetches the two per-frame label files and, for every requested sequence, its frames and
its ``-data.csv`` acquisition timestamps. Downloads resume after an interruption and a
verified file is never fetched again. ``--plan`` prints the files and fetches nothing.

The dataset is CC BY-NC-SA 4.0 (non-commercial academic use); cite Kwolek & Kępski (2014).

Examples::

    uv run --project backend python scripts/urfd_download.py --plan
    uv run --project backend python scripts/urfd_download.py --source rgb \\
        --json experiments/reports/urfd_acquisition.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fall_detection.research.acquisition import AssetRequest, acquire
from fall_detection.research.urfd import ADL_IDS, FALL_IDS, assets_for

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "datasets" / "raw" / "urfd")
    parser.add_argument(
        "--source",
        choices=("rgb", "mp4"),
        default="rgb",
        help="rgb: 640x480 PNG zips (default, ~4 GB); mp4: 320x240 previews (~90 MB)",
    )
    parser.add_argument(
        "--sequences", nargs="*", default=None, help="subset, e.g. fall-01 adl-01 (default: all)"
    )
    parser.add_argument("--plan", action="store_true", help="list the files and exit")
    parser.add_argument("--json", type=Path, default=None, help="write the acquisition report")
    args = parser.parse_args()

    sequences = args.sequences or [*FALL_IDS, *ADL_IDS]
    assets = assets_for(sequences, source=args.source)
    if args.plan:
        for asset in assets:
            print(f"{asset.kind:10s} {asset.url}")
        print(f"{len(assets)} files")
        return 0

    results = []
    for asset in assets:
        result = acquire(
            AssetRequest(
                url=asset.url,
                destination=args.root / asset.name,
                dataset_id="urfd",
                recording_id=asset.sequence_id,
                kind=asset.kind,
            ),
            progress=lambda message: print(message, file=sys.stderr),
        )
        print(f"[{result.status}] {asset.name} {result.size_bytes} bytes")
        results.append(result)

    failed = [result for result in results if not result.ok]
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "dataset": "urfd",
                    "source": args.source,
                    "license": "CC BY-NC-SA 4.0",
                    "citation": "Kwolek, B., & Kepski, M. (2014). Human fall detection on "
                    "embedded platform using depth maps and wireless accelerometer. Computer "
                    "Methods and Programs in Biomedicine, 117(3), 489-501.",
                    "assets": [result.as_dict() for result in results],
                    "failed": len(failed),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"{len(results) - len(failed)} ok, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
