#!/usr/bin/env python3
"""Measure one frozen root on independent native diagnostic trajectories."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from freq_hrl.experiments.pointmaze_independent_credit import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    replay(args.optimizer_seed, preflight=args.preflight, output=args.output)
    print("Training complete: result.json written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
