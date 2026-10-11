#!/usr/bin/env python3
"""Qualify matched native training labels; keep diagnosed checks out of fitting."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_data_stage174 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells, sources, models, source_models, *, protocol_spec=spec):
    spec = protocol_spec
    required = {(r, s, i) for r in spec.ROOTS for s in spec.SCENARIOS for i in spec.PROBES}
    if set(cells) != required or set(sources) != {(r, s) for r, s, _ in required} or models != source_models:
        raise ValueError("Incomplete value training data or changed fitted duration inputs")
    data = []
    for (root, scenario, index), cell in sorted(cells.items()):
        baseline, registered = cell["observer"], sources[root, scenario]
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
                and cell["probe_index"] == index and cell["training_updates"] == 0 and cell["native_steps"] == 7*61380
                and baseline["probe"]["index"] == index and baseline["scene_seed"] == registered["scene_seed"]
                and baseline["scene_seed"] != spec.shared.source.source.check_scene(root, scenario)):
            raise ValueError("Changed training-only value data scenes, indices or budget")
        spec.shared.qualify(baseline, {**registered, "probe": baseline["probe"]})
        spec.shared.validate_scoring({**baseline, "condition": "prefix_wait_full_fleet"}, models[str(root)],
            strengths=spec.shared.source.incumbent.CANDIDATES["rolling_plus_incumbent"])
        d = registered["supply"]["decisions"][index]
        p, context = baseline["probe"], cell["training_context"]
        values = [*p["native_state_v2"], *[v for name in ("queues", "active", "onboard") for v in context[name].values()],
            context["route_speed_mean_mps"], context["cost_before"]]
        if not (p["candidates"] == d["candidates"] and p["time_s"] == d["time_s"]
                and p["direction"] == d["direction"] and p["selected"] == p["greedy"] == d["selected"]
                and p["prefix_decisions_qualified"] == index+1 and len(p["native_state_v2"]) == 16
                and context["cost_before"] == p["cost_before"] and np.all(np.isfinite(values))
                and baseline["policy_trace"] == spec.shared.policy_trace(baseline["supply"])):
            raise ValueError("Changed causal native state, training context or source selector")
        spec.validate_continuations(baseline, cell["branches"])
        data.append(spec.records(cell))
    diagnostics = []
    for root in spec.ROOTS:
        group = [d for d in data if d["root"] == root]
        diagnostics.append({"root": root, "states": len(group), "labels": len(group)*6,
            "states_with_better_realized_action": sum(min(l["actual_cost_delta"] for l in d["labels"]) < 0 for d in group),
            "mean_best_realized_cost_delta": float(np.mean([min(l["actual_cost_delta"] for l in d["labels"]) for d in group])),
            "states_with_fleet_discrimination": sum(len({l["fleet_cost_delta"] for l in d["labels"]}) > 1 for d in group)})
    states = len(required)
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": 7*states*61380, "source_control_reproductions": 2*states,
        "window_ledger_qualified_episodes": 7*states, "paired_interventions": 6*states,
        "training_states": states, "root_diagnostics": diagnostics, "training_data": data,
        "stage": spec.contract()["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory, source = ROOT/"results"/args.run_name, ROOT/"results"/spec.SOURCE_RUN
    cells = {(r, s, i): json.loads((directory/"cells"/s/f"seed_{r}/probe_{i}/result.json").read_text())
        for r in spec.ROOTS for s in spec.SCENARIOS for i in spec.PROBES}
    result = summarize(cells, {(r, s): spec.source_record(r, s) for r in spec.ROOTS for s in spec.SCENARIOS},
        json.loads((directory/"inputs/models.json").read_text()), json.loads((source/"inputs/duration_models.json").read_text()))
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "training_data"}}, indent=2))


if __name__ == "__main__":
    main()
