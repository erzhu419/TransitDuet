#!/usr/bin/env python3
"""Run Stage115 with an explicit wider Bernstein plan-coordinate head."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments import pointmaze_upper_wide_plan_train as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    experiment.base.run(
        args.optimizer_seed,
        preflight=args.preflight,
        output=args.output,
        protocol=experiment.spec,
        score_fn=experiment.local_credit.score_upper_local,
        qualify_fn=experiment.qualify,
        upper_factory=experiment.upper_branch,
        training_pair_fn=experiment.training_pair,
        evaluation_group_fn=experiment.evaluation_group,
    )
    print("Eval complete: Stage115 result.json written", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

