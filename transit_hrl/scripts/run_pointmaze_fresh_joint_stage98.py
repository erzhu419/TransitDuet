#!/usr/bin/env python3
"""Run the fixed mean learner on the full new teacher sources."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from freq_hrl.experiments import pointmaze_fresh_joint as experiment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer-seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    experiment.run(a.optimizer_seed, preflight=a.preflight, output=a.output)
    print("Training complete: result.json written; fresh joint/lower final learners", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
