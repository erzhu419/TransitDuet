#!/usr/bin/env python3
"""Qualify compact development labels without pulling server-side rollouts."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_expansion_stage187 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def result_path(directory, root, scenario, replica, probe):
    return directory/"cells"/scenario/f"seed_{root}"/f"replica_{replica}"/f"probe_{probe}"/"result.json"


def summarize(cells, *, preflight=False):
    required = spec.cells(preflight=preflight)
    if set(cells) != set(required):
        raise ValueError("Incomplete or mixed fresh development cohort")
    rows, controls = [], {}
    for key in required:
        cell = cells[key]
        if key != (cell["seed"], cell["scenario"], cell["replica"], cell["probe_index"]):
            raise ValueError("Result path does not match its development cell")
        rows.append(spec.qualify_cell(cell))
        group = key[:3]
        control = (cell["observer"], cell["policy_trace"])
        if group in controls and controls[group] != control:
            raise ValueError("Correlated probes do not reproduce the same original scene control")
        controls[group] = control
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "phase": "preflight" if preflight else "complete", "training_updates": 0,
        "new_native_episodes": 7*len(rows), "new_native_steps": 7*61380*len(rows),
        "training_states": len(rows), "paired_labels": 6*len(rows), "independent_scene_groups": len(controls),
        "exact_identity_reproductions": len(rows), "window_ledger_qualified_episodes": 7*len(rows),
        "training_data": rows, "third_scene_screen_eligible": False, "native_deployment_eligible": False,
        "stage": spec.contract()["boundary"]}


def load_summary(directory, *, preflight=False):
    return summarize({k: json.loads(result_path(directory, *k).read_text())
        for k in spec.cells(preflight=preflight)}, preflight=preflight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    summary = load_summary(directory, preflight=args.preflight)
    write_json(directory/("preflight_summary.json" if args.preflight else "summary.json"), summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("training_data", "contract")}, indent=2))


if __name__ == "__main__":
    main()
