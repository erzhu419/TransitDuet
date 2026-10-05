#!/usr/bin/env python3
"""Measure bounded plan authority with all controller weights frozen."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments import pointmaze_plan_authority as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    experiment.run(args.optimizer_seed, preflight=args.preflight, output=args.output)
    print("Eval complete: Stage120 result.json written", flush=True)


if __name__ == "__main__":
    main()
