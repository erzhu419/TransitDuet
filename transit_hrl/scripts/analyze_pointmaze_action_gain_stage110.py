#!/usr/bin/env python3
"""Aggregate all36 directional endpoints without choosing an upper policy."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments import pointmaze_action_gain as experiment
from freq_hrl.experiments.pointmaze_root_response import write_json
from scripts import pointmaze_action_gain_stage110_spec as spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    base = spec.ROOT/"results"/a.run_name
    cells = [json.loads((base/"cells"/f"replicate_{r}"/"result.json").read_text()) for r in spec.roots(preflight=a.preflight)]
    summary = experiment.aggregate(cells, preflight=a.preflight)
    summary["run_name"] = a.run_name
    write_json(base/"qualification_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "root_rows"}, sort_keys=True), flush=True)
    print("Eval complete: qualification_summary.json written", flush=True)


if __name__ == "__main__":
    main()
