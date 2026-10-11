#!/usr/bin/env python3
"""Qualify passive native occupancy context and join unchanged value labels."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_context_stage177 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells, summary, sources, models, source_models):
    rows = spec.prior.training_rows(summary)
    if set(cells) != {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS} or models != source_models:
        raise ValueError("Missing passive occupancy cells or changed duration inputs")
    data = []
    for (root, scenario), cell in sorted(cells.items()):
        registered = sources[root, scenario]
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
                and cell["scene_seed"] == registered["scene_seed"] and cell["training_updates"] == 0
                and cell["native_steps"] == 61380):
            raise ValueError("Changed passive native scene, protocol or simulation budget")
        observer = cell["observer"]
        spec.shared.qualify(observer, {**registered, "probe": observer["probe"]})
        spec.shared.validate_scoring({**observer, "condition": "prefix_wait_full_fleet"}, models[str(root)],
            strengths=spec.shared.source.incumbent.CANDIDATES["rolling_plus_incumbent"])
        group = [r for r in rows if (r["root"], r["scenario"]) == (root, scenario)]
        spec.qualify_contexts(cell["contexts"], group, registered)
        if observer["policy_trace"] != spec.shared.policy_trace(observer["supply"]):
            raise ValueError("Changed passive native selector trace")
        by_index = {c["index"]: c for c in cell["contexts"]}
        for row in group:
            c = by_index[row["probe_index"]]
            data.append({**row, "occupancy_context": {k: c[k] for k in
                ("historical_peak", "active_buses", "queue_by_station", "route_speed_mps")}})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "new_native_steps": 613800, "new_native_episodes": 10,
        "source_control_reproductions": 10, "training_states": 60, "paired_labels_reused": 360,
        "training_data": data, "fitting_groups": summary["fitting_groups"],
        "next_step": "occupancy_conditioned_value_representation_fit_with_scene_grouped_validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    source = ROOT/"results"/spec.SOURCE_RUN
    summary = json.loads((directory/"inputs/training.json").read_text())
    if summary != json.loads((source/"summary.json").read_text()):
        raise ValueError("Passive value context changed the qualified training labels")
    result = summarize({(r, s): json.loads((directory/"cells"/s/f"seed_{r}/result.json").read_text())
        for r in spec.ROOTS for s in spec.SCENARIOS}, summary,
        {(r, s): spec.prior.source.source_record(r, s) for r in spec.ROOTS for s in spec.SCENARIOS},
        json.loads((directory/"inputs/models.json").read_text()), json.loads((source/"inputs/models.json").read_text()))
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "training_data", "fitting_groups"}}, indent=2))


if __name__ == "__main__":
    main()
