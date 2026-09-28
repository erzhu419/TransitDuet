#!/usr/bin/env python3
"""Run one joint-training treatment on a scheduler compute node."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from freq_hrl.experiments.pointmaze_joint_renewal import train
from scripts import pointmaze_joint_renewal_stage35_spec as spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--method", choices=spec.METHODS, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    train(args.optimizer_seed, args.method, preflight=args.preflight, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
