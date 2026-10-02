#!/usr/bin/env python3
"""Evaluate registered final-checkpoint native actor swaps."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from freq_hrl.experiments import pointmaze_actor_swap as experiment
from scripts import pointmaze_actor_swap_stage84_spec as spec


def main(*, protocol_spec=spec):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer-seed",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--preflight",action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    experiment.run(a.optimizer_seed,preflight=a.preflight,output=a.output,protocol=protocol_spec)
    print("Training complete: result.json written (evaluation only)",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
