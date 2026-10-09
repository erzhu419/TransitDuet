#!/usr/bin/env python3
"""Summarize signed execution qualification, not learned-channel performance."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_dispatch_stage150 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells):
    panels = []
    if set(cells) != set(spec.ROOTS):
        raise ValueError("Incomplete signed dispatch qualification roots")
    for root, cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["seed"] == root and cell["software_qualified"] and cell["baseline_reproduced"]
                and cell["zero_dispatch_reproduced"] and cell["training_updates"] == 0
                and cell["native_steps"] == len(spec.CONDITIONS) * 61380):
            raise ValueError("Signed dispatch execution did not qualify")
        rows = {row["condition"]: row for row in cell["evaluation"]}
        if set(rows) != set(spec.CONDITIONS) or len(rows) != len(cell["evaluation"]):
            raise ValueError("Missing or duplicate signed dispatch condition")
        zero = rows["signed_zero"]
        for condition, row in rows.items():
            for key in ("scenario", "scene_seed", "passengers_generated", "N_fleet", "simulation_end_time_s"):
                if row[key] != zero[key]:
                    raise ValueError("Unpaired signed dispatch physical scene")
            panels.append({"root": root, "condition": condition, "dispatch": row["dispatch"],
                "minus_zero": {key: row[key] - zero[key] for key in spec.frontier.source_spec.routing.METRICS}})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "software_qualified": True, "training_updates": 0,
        "native_steps": sum(cell["native_steps"] for cell in cells.values()),
        "stage": "signed_execution_qualification_with_frozen_HIRO_trained_lower",
        "panels": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {root: json.loads((directory / f"seed_{root}/result.json").read_text()) for root in spec.ROOTS}
    result = summarize(cells)
    write_json(directory / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
