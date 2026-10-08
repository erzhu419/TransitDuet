#!/usr/bin/env python3
"""Run whole-scene cached-credit cross-validation on a scheduler node."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments.pointmaze_option_crossfit import run
from scripts import pointmaze_option_crossfit_stage144_spec as spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, choices=spec.ROOTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.optimizer_seed, args.output)


if __name__ == "__main__":
    main()
