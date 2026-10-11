#!/usr/bin/env python3
"""Qualify and merge opposite-direction labels without declaring a policy win."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_direction_stage175 as spec
from scripts.analyze_native_transit_value_data_stage174 import summarize as qualify_new
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells, sources, models, source_models, previous):
    current = qualify_new(cells, sources, models, source_models, protocol_spec=spec)
    for (root, scenario, index), cell in cells.items():
        paired = spec.paired_reference(root, scenario, index, previous)
        registered = sources[root, scenario]
        expected = registered["supply"]["decisions"][index-1]
        if not (paired["scene_seed"] == registered["scene_seed"]
                and all(paired[k] == expected[k] for k in ("time_s", "direction"))):
            raise ValueError("Changed registered first-direction training source")
        spec.validate_pair(cell, paired)
    data = sorted([*previous["training_data"], *current["training_data"]],
        key=lambda r: (r["root"], r["scenario"], r["probe_index"]))
    diagnostics = []
    for root in spec.ROOTS:
        group = [r for r in data if r["root"] == root]
        diagnostics.append({"root": root, "states": len(group), "labels": len(group)*6,
            "direction_states": {str(d): sum(r["direction"] == d for r in group) for d in (False, True)},
            "states_with_better_realized_action": sum(min(l["actual_cost_delta"] for l in r["labels"]) < 0 for r in group),
            "states_with_fleet_discrimination": sum(len({l["fleet_cost_delta"] for l in r["labels"]}) > 1 for r in group)})
    return {**current, "new_native_steps": current["native_steps"], "new_training_states": 30,
        "native_steps": previous["native_steps"]+current["native_steps"], "training_states": 60,
        "source_control_reproductions": 120, "paired_interventions": 360, "window_ledger_qualified_episodes": 420,
        "root_diagnostics": diagnostics, "training_data": data,
        "fitting_groups": [{"root": r, "scenario": s, "scene_seed": sources[r, s]["scene_seed"],
            "probes": sorted([*spec.source.PROBES, *spec.PROBES])} for r in spec.ROOTS for s in spec.SCENARIOS]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    cells = {(r, s, i): json.loads((directory/"cells"/s/f"seed_{r}/probe_{i}/result.json").read_text())
        for r in spec.ROOTS for s in spec.SCENARIOS for i in spec.PROBES}
    previous = json.loads((directory/"inputs/paired_source.json").read_text())
    source_dir = ROOT/"results"/spec.SOURCE_RUN
    if previous != json.loads((source_dir/"summary.json").read_text()):
        raise ValueError("Paired training input differs from the qualified source summary")
    result = summarize(cells, {(r, s): spec.source_record(r, s) for r in spec.ROOTS for s in spec.SCENARIOS},
        json.loads((directory/"inputs/models.json").read_text()),
        json.loads((source_dir/"inputs/models.json").read_text()), previous)
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "training_data", "fitting_groups"}}, indent=2))


if __name__ == "__main__":
    main()
