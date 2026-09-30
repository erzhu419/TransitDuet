#!/usr/bin/env python3
"""Diagnose Stage52 recorded forecast/control errors on the server."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import pointmaze_plan_error_stage53_spec as spec
from freq_hrl.experiments.pointmaze_plan_error import diagnose
from freq_hrl.experiments.pointmaze_root_response import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    summary = diagnose(spec.ROOT / "results" / spec.SOURCE_RUN)
    summary["run_name"] = args.run_name
    write_json(spec.ROOT / "results" / args.run_name / "qualification_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "primary_endpoints", "cost")}, sort_keys=True))
    print("Training complete: Stage53 offline qualification_summary.json written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
