#!/usr/bin/env python3
"""Run one preregistered critic-only clock information cell."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from freq_hrl.experiments.pointmaze_critic_calibration import train
from freq_hrl.experiments.pointmaze_critic_clock import make_model, worker_rollout
from scripts import pointmaze_critic_clock_stage42_spec as spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--method", choices=spec.METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    train(args.optimizer_seed, args.method, preflight=args.preflight, output=args.output,
          specification=spec, rollout_worker=worker_rollout, model_factory=make_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
