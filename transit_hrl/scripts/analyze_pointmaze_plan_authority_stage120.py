#!/usr/bin/env python3
"""Aggregate frozen plan-authority effects without fetching native arrays."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments import pointmaze_plan_authority as experiment
from freq_hrl.experiments.pointmaze_root_response import write_json
from scripts import pointmaze_plan_authority_stage120_spec as spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    base = spec.ROOT / "results" / args.run_name
    cells = [json.loads((base / "cells" / f"replicate_{root}" / "result.json").read_text())
        for root in spec.roots(preflight=args.preflight)]
    summary = experiment.aggregate(cells, preflight=args.preflight)
    summary["run_name"] = args.run_name
    write_json(base / "qualification_summary.json", summary)
    compact = {key: value for key, value in summary.items() if key != "root_rows"}
    write_json(base / "compact_summary.json", compact)
    print(json.dumps(compact, sort_keys=True), flush=True)
    print("Eval complete: Stage120 qualification_summary.json written", flush=True)


if __name__ == "__main__":
    main()
