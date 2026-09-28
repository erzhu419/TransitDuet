#!/usr/bin/env python3
"""Run the frozen Stage-33 controller/response phase on a compute node."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scripts import pointmaze_root_response_stage33_spec as spec
from freq_hrl.experiments.pointmaze_root_response import qualify, train_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--phase", choices=("train", "response", "pipeline"), required=True)
    parser.add_argument("--controller-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if cli.phase == "pipeline" and not cli.preflight:
        parser.error("full controller and response phases use separate resource allocations")
    if cli.phase == "response" and cli.controller_result is None:
        parser.error("response requires its completed fresh-root controller result")
    args = spec.arguments(cli.optimizer_seed, preflight=cli.preflight)
    torch.set_num_threads(1)
    if cli.phase == "train":
        train_controller(args, cli.output)
    else:
        source = cli.controller_result
        if cli.phase == "pipeline":
            source = cli.output.parent / "controller.json"
            train_controller(args, source)
        qualify(args, source, cli.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
