#!/usr/bin/env python3
"""Run frozen response deployment on a scheduler compute node."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from freq_hrl.experiments.pointmaze_response_deployment import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.optimizer_seed, preflight=args.preflight, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
