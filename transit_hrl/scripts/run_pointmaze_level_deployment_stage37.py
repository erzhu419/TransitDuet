#!/usr/bin/env python3
"""Run one upper/lower training or frozen gate-evaluation cell."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from freq_hrl.experiments.pointmaze_update_isolation import train
from freq_hrl.experiments.pointmaze_gate_deployment import evaluate
from scripts import pointmaze_level_deployment_stage37_spec as spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--method", choices=(*spec.METHODS, spec.GATE_TASK), required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.method == spec.GATE_TASK:
        evaluate(args.optimizer_seed, preflight=args.preflight, output=args.output)
    else:
        train(args.optimizer_seed, args.method, preflight=args.preflight, output=args.output, specification=spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
