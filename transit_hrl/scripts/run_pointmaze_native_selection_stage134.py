"""Run third-update candidate selection with separate native validation."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments.pointmaze_native_selection import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.optimizer_seed, args.output)


if __name__ == "__main__":
    main()
