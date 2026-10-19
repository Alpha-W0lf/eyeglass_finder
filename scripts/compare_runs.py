"""Compare two runs: same input should give the same finds.

Exits with status 1 and lists the differences when the two runs' final
targets (matched on image_url and face_bbox) are not the same set.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def target_keys(run_dir: Path) -> set:
    path = Path(run_dir) / "filtered_dataset.parquet"
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["image_url", "face_bbox"])
    return {(url, tuple(int(v) for v in bbox)) for url, bbox in zip(df["image_url"], df["face_bbox"])}


def compare(baseline: Path, current: Path) -> tuple[set, set]:
    """Return (only in baseline, only in current)."""
    base, curr = target_keys(baseline), target_keys(current)
    return base - curr, curr - base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--current-run", required=True)
    args = parser.parse_args()
    lost, gained = compare(Path(args.baseline_run), Path(args.current_run))
    print(f"Finds only in baseline: {len(lost)}; only in current: {len(gained)}")
    for url, bbox in sorted(lost):
        print(f"  - {url} {list(bbox)}")
    for url, bbox in sorted(gained):
        print(f"  + {url} {list(bbox)}")
    return 1 if lost or gained else 0


if __name__ == "__main__":
    sys.exit(main())
