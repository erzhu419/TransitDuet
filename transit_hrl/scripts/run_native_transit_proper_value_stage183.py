#!/usr/bin/env python3
"""Remove decision-regret gradients with the qualified stopping schedule frozen."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_scene_stop_stage182 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

value, paired = source.value, source.paired
EXPERIMENT_PROTOCOL = "native_transit_proper_value_stage183_v1"
SOURCE_RUN = "native_transit_scene_stop_stage182_fit_20261012_r1"
PRIMARY = "proper_mse:wait_feasible"


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL, "primary": PRIMARY,
        "data": source.contract()["data"], "critic": "unchanged_floor_absolute_96_16_3_tanh_1603_parameters",
        "loss": "absolute_fleet_MSE_plus_three_head_paired_MSE_only_no_expected_regret_gradients",
        "control": "reuse_qualified_Stage182_scene_stop_values_and_all_28_inner_selected_step_counts",
        "matching": "same_outer_training_groups_states_seed178_scales_optimizer_parameters_and_optimizer_steps",
        "optimizer": source.contract()["optimizer"], "policies": list(value.POLICIES),
        "decision": source.contract()["decision"], "validation": source.contract()["validation"],
        "budget": "28_fits_sum_of_frozen_control_refit_steps_no_new_inner_control_actor_or_native_updates",
        "screen": source.contract()["screen"], "next": source.contract()["next"],
        "boundary": "objective_ablation_at_frozen_development_stopping_schedule_not_independent_confirmation_or_HRL"}


def fit_and_validate(first, labels, second, baselines, control):
    if not (control["protocol"] == source.EXPERIMENT_PROTOCOL and control["contract"] == source.contract()
            and control["software_qualified"] and len(control["selection_ledger"]) == 28
            and control["training_mode"] == "scene_stop"):
        raise ValueError("Proper value fitting requires the qualified frozen Stage182 stopping schedule")
    fits = []
    def fitter(rows, method):
        record = control["selection_ledger"][len(fits)]
        if sorted(map(list, {paired.scene_group(r) for r in rows})) != record["outer_training_groups"]:
            raise ValueError("Frozen stopping schedule changed its corresponding training scenes")
        model = value.fit_value(rows, method, steps=record["selected_steps"], objective="proper_mse")
        fits.append(model["steps"])
        return model
    result = source.source.fit_and_validate(first, labels, second, baselines, "floor_absolute", fitter=fitter)
    return {**result, "protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "training_mode": "proper_mse",
        "frozen_selected_steps": fits, "new_inner_updates": 0, "new_control_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "labels", "second", "baselines", "control", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = fit_and_validate(*(json.loads(getattr(args, n).read_text()) for n in ("first", "labels", "second", "baselines", "control")))
    write_json(args.output, result)
    print(json.dumps({"critic_updates": result["critic_updates"], "third_scene_screen_eligible": result["third_scene_screen_eligible"],
        "gates": result["gates"]}, indent=2), flush=True)
    print("NATIVE_PROPER_VALUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
