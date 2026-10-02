#!/usr/bin/env python3
"""Qualify Stage88 independently and apply the frozen confirmation decision."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments import pointmaze_call_weighted as experiment
from freq_hrl.experiments.pointmaze_root_response import write_json
from scripts import pointmaze_call_weighted_replication_stage88_spec as spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name",required=True)
    p.add_argument("--preflight",action="store_true")
    a = p.parse_args()
    directory = spec.ROOT/"results"/a.run_name
    cells = [json.loads((directory/"cells"/f"replicate_{r}"/"result.json").read_text()) for r in spec.roots(preflight=a.preflight)]
    summary = experiment.aggregate(cells,preflight=a.preflight,protocol=spec)
    summary.update(run_name=a.run_name,independent_confirmation=spec.confirmation(summary))
    write_json(directory/"qualification_summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "root_rows"},sort_keys=True),flush=True)
    print("Training complete: qualification_summary.json written",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
