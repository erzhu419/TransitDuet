#!/usr/bin/env python3
"""Run historical nonlinear constraint calibration and paired native probes."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from freq_hrl.experiments import pointmaze_bounded_residual as bounded
from freq_hrl.experiments import pointmaze_calibrated_residual as shared
from scripts import pointmaze_bounded_residual_stage78_spec as spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer-seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    shared.run(a.optimizer_seed, preflight=a.preflight, output=a.output, protocol=spec,
        calibrator=bounded.calibrate, qualify_source=shared.qualify, calibration_check=bounded.check_calibration)
    print("Training complete: result.json written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
