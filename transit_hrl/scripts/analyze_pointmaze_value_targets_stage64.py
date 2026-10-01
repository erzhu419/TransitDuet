#!/usr/bin/env python3
"""Qualify the complete frozen critic-fit factorial without reward claims."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments.pointmaze_value_targets import aggregate
from freq_hrl.experiments.pointmaze_root_response import write_json
from scripts import pointmaze_value_targets_stage64_spec as spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    directory = spec.ROOT / "results" / args.run_name
    cells = [json.loads((directory / "cells" / f"replicate_{root}" / "result.json").read_text()) for root in spec.roots(preflight=args.preflight)]
    result = aggregate(cells, preflight=args.preflight)
    result["run_name"] = args.run_name
    write_json(directory / "qualification_summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "root_rows"}, sort_keys=True))
    print("Training complete: qualification_summary.json written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
