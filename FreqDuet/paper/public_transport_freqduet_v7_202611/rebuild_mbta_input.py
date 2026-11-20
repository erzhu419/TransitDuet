"""Recreate the MBTA-111 simulator inputs from explicit public input paths."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apc-csv", type=Path, required=True)
    parser.add_argument("--gtfs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    if not args.apc_csv.is_file() or not (args.gtfs_dir / "stops.txt").is_file():
        parser.error("provide the Fall 2025 APC CSV and GTFS stops.txt")
    experiments = Path(__file__).resolve().parents[2] / "experiments"
    sys.path.insert(0, str(experiments / "v500_pathb_headroom_gate"))
    import build_corridors

    build_corridors.MBTA_APC_CSV = args.apc_csv
    build_corridors.MBTA_GTFS_DIR = args.gtfs_dir
    build_corridors.build_mbta_route111_wkdy(args.output)


if __name__ == "__main__":
    main()
