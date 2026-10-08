#!/usr/bin/env python3
"""Run the native upper score-geometry and deployment-objective comparison."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments.pointmaze_score_geometry import run


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    run(args.optimizer_seed,args.output)
