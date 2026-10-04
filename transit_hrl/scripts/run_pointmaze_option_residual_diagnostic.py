#!/usr/bin/env python3
import argparse
from pathlib import Path

from freq_hrl.experiments import pointmaze_option_residual_diagnostic as experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment.run(args.optimizer_seed, output=args.output)


if __name__ == "__main__":
    main()
