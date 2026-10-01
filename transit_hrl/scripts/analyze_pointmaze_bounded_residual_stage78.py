#!/usr/bin/env python3
"""Summarize every registered four-curve native contrast and actual solver cost."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments import pointmaze_bounded_residual as bounded
from freq_hrl.experiments import pointmaze_calibrated_residual as shared
from freq_hrl.experiments.pointmaze_root_response import write_json
from scripts import pointmaze_bounded_residual_stage78_spec as spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    directory = spec.ROOT / "results" / a.run_name
    cells = [json.loads((directory / "cells" / f"replicate_{r}" / "result.json").read_text()) for r in spec.roots(preflight=a.preflight)]
    summary = shared.aggregate(cells, preflight=a.preflight, protocol=spec, calibration_check=bounded.check_calibration)
    summary["run_name"] = a.run_name
    write_json(directory / "qualification_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "root_rows"}, sort_keys=True), flush=True)
    print("Training complete: qualification_summary.json written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
