#!/usr/bin/env python3
"""Qualify the complete frozen local-credit cohort."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments import pointmaze_option_residual as experiment
from freq_hrl.experiments.pointmaze_root_response import write_json
from scripts import pointmaze_option_residual_stage111_spec as spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    base = spec.ROOT/"results"/args.run_name
    cells = [json.loads((base/"cells"/f"replicate_{r}"/"result.json").read_text()) for r in spec.roots(preflight=args.preflight)]
    summary = experiment.aggregate(cells, preflight=args.preflight)
    summary["run_name"] = args.run_name
    write_json(base/"qualification_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "root_rows"}, sort_keys=True), flush=True)
    print("Eval complete: qualification_summary.json written", flush=True)


if __name__ == "__main__":
    main()
