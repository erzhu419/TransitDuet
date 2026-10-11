#!/usr/bin/env python3
"""Matched feature/objective effects without selecting a new primary post hoc."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_occupancy_value_stage178 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells):
    if set(cells) != set(spec.value.METHODS):
        raise ValueError("Missing matched occupancy/objective cells")
    for method, cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["method"] == method and cell["software_qualified"]
                and cell["critic_updates"] == 18*spec.value.STEPS and cell["actor_updates"] == 0
                and cell["new_native_steps"] == 0 and len(cell["fit_ledger"]) == 18
                and all(f["steps"] == spec.value.STEPS and f["parameter_count"] == 1586
                    and np.isfinite(f["parameter_update_l2"]) for f in cell["fit_ledger"])):
            raise ValueError("Changed paired critic architecture, updates or qualification")
        passed = all(v["mean_actual_cost_delta"] < 0 for split in cell["validation"].values()
            for v in split["by_root"].values())
        if cell["value_screen_passed"] != passed or cell["native_screen_eligible"] != (method == spec.PRIMARY and passed):
            raise ValueError("Changed pre-registered primary deployment gate")
    contrasts = {}
    key = lambda p: (p["root"], p["scene_seed"], p["probe_index"])
    for split in ("scene", "regime", "lower_root"):
        reference = {key(p) for p in cells["base_mse"]["validation"][split]["predictions"]}
        for cell in cells.values():
            rows = cell["validation"][split]["predictions"]
            if len(rows) != 60 or len(reference) != 60 or {key(p) for p in rows} != reference:
                raise ValueError("Unmatched native value validation states")
        means = {m: c["validation"][split]["overall"]["mean_actual_cost_delta"] for m, c in cells.items()}
        contrasts[split] = {"occupancy_under_mse": means["occupancy_mse"]-means["base_mse"],
            "occupancy_under_regret": means["occupancy_regret"]-means["base_regret"],
            "regret_under_base": means["base_regret"]-means["base_mse"],
            "regret_under_occupancy": means["occupancy_regret"]-means["occupancy_mse"]}
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "methods": {m: {"apparent_training": c["apparent_training"],
            "validation": {s: {k: v[k] for k in ("overall", "by_root")} for s, v in c["validation"].items()},
            "value_screen_passed": c["value_screen_passed"]} for m, c in cells.items()},
        "matched_cost_contrasts": contrasts, "critic_updates": 72*spec.value.STEPS, "new_native_steps": 0,
        "native_screen_eligible": cells[spec.PRIMARY]["native_screen_eligible"],
        "next_step": "separately_registered_native_primary_policy_screen" if cells[spec.PRIMARY]["native_screen_eligible"]
            else "use_fit_vs_transfer_and_cost_head_diagnostics_to_choose_next_data_or_training_fix"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    result = summarize({m: json.loads((directory/"cells"/m/"result.json").read_text()) for m in spec.value.METHODS})
    write_json(directory/"summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
