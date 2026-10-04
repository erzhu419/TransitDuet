#!/usr/bin/env python3
"""Run Stage114 decision-aligned local upper credit."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments import pointmaze_upper_local_credit_train as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    experiment.base.run(args.optimizer_seed, preflight=args.preflight, output=args.output,
        protocol=experiment.spec, score_fn=experiment.score_upper_local, qualify_fn=experiment.qualify)
    print("Eval complete: Stage114 result.json written", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
