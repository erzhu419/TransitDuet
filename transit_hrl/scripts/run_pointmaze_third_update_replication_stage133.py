"""Run the frozen third update with the unchanged native continuation kernel."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments.pointmaze_compact_continuation import run
from scripts import pointmaze_third_update_replication_stage133_spec as spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.optimizer_seed, args.output, protocol_spec=spec)


if __name__ == "__main__":
    main()
