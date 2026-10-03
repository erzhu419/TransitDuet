#!/usr/bin/env python3
"""Qualify the matched fresh lower cohort and the 26-contrast family."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments import pointmaze_fresh_lower as experiment
from freq_hrl.experiments.pointmaze_root_response import write_json
from scripts import pointmaze_fresh_lower_stage99_spec as spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    base = spec.ROOT / "results" / a.run_name
    cells = [json.loads((base / "cells" / f"replicate_{r}" / "result.json").read_text())
        for r in spec.roots(preflight=a.preflight)]
    summary = experiment.aggregate(cells, preflight=a.preflight)
    summary["run_name"] = a.run_name
    write_json(base / "qualification_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "root_rows"}, sort_keys=True), flush=True)
    print("Training complete: qualification_summary.json written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
