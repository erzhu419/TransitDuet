#!/usr/bin/env python3
"""Choose critic training duration inside each outer training scene bank."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_floor_value_stage181 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

value, paired, occupancy = source.value, source.paired, source.occupancy
EXPERIMENT_PROTOCOL = "native_transit_scene_stop_stage182_v1"
SOURCE_RUN = "native_transit_floor_value_stage181_fit_20261011_r1"
PRIMARY = "scene_stop:wait_feasible"


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL, "primary": PRIMARY,
        "data": "unchanged_120_states_720_labels_20_first_second_scene_groups_and_absolute_fleet_supervision",
        "critic": "unchanged_floor_absolute_96_16_3_tanh_1603_parameters_and_training_loss",
        "control": "reuse_qualified_Stage181_floor_absolute_fixed1500_predictions_no_retraining",
        "inner_split": "sort_outer_training_groups_by_scenario_root_scene_seed_hold_every_fifth_starting_at_zero",
        "checkpoint_steps": list(value.CHECKPOINT_STEPS),
        "selection": "minimum_inner_scene_absolute_fleet_MSE_plus_three_head_paired_MSE_earlier_on_exact_ties",
        "normalization": "inner_training_only_for_selection_full_outer_training_only_for_refit_no_outer_test_labels",
        "refit": "fresh_same_seed178_on_all_outer_training_scenes_for_selected_step_count",
        "policies": list(value.POLICIES), "decision": source.contract()["decision"],
        "optimizer": {"learning_rate": occupancy.LEARNING_RATE, "weight_decay": occupancy.WEIGHT_DECAY},
        "validation": source.contract()["validation"], "screen": source.contract()["screen"],
        "budget": {"inner_fits": 28, "refits": 28, "inner_updates": 28*occupancy.STEPS,
            "max_refit_updates": 28*occupancy.STEPS, "new_native_episodes": 0, "actor_updates": 0},
        "next": source.contract()["next"],
        "boundary": "nested_development_checkpoint_screen_extra_inner_fitting_compute_not_matched_compute_or_HRL_confirmation"}


def inner_split(rows):
    scenario = {paired.scene_group(r): r["scenario"] for r in rows}
    groups = sorted(scenario, key=lambda g: (scenario[g], *g))
    held = set(groups[::5])
    return ([r for r in rows if paired.scene_group(r) not in held],
        [r for r in rows if paired.scene_group(r) in held])


def selected_checkpoint(ledger):
    return min(ledger, key=lambda c: (c["validation_loss"], c["step"]))["step"]


def fit_and_validate(first, labels, second, baselines):
    selection_ledger = []
    def fitter(rows, method):
        train, validation = inner_split(rows)
        inner = value.fit_value(train, method, validation_rows=validation)
        step = selected_checkpoint(inner["checkpoint_ledger"])
        model = value.fit_value(rows, method, steps=step)
        record = {"outer_training_groups": model["training_groups"], "outer_training_states": len(rows),
            "inner_training_groups": inner["training_groups"], "inner_training_states": len(train),
            "inner_validation_groups": inner["validation_groups"], "inner_validation_states": len(validation),
            "inner_steps": inner["steps"], "selected_steps": step, "checkpoint_ledger": inner["checkpoint_ledger"],
            "inner_target_scale": inner["target_scale"], "inner_regret_cost_scale": inner["regret_cost_scale"]}
        selection_ledger.append(record)
        model["checkpoint_selection"] = record
        print(f"SCENE_STOP_SELECTED fit={len(selection_ledger)} inner={len(train)} "
            f"validation={len(validation)} refit={len(rows)} steps={step}", flush=True)
        return model
    result = source.fit_and_validate(first, labels, second, baselines, "floor_absolute", fitter=fitter)
    refit_updates = result["critic_updates"]
    inner_updates = sum(r["inner_steps"] for r in selection_ledger)
    return {**result, "protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "training_mode": "scene_stop",
        "selection_ledger": selection_ledger, "inner_critic_updates": inner_updates,
        "refit_critic_updates": refit_updates, "critic_updates": inner_updates+refit_updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "labels", "second", "baselines", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = fit_and_validate(*(json.loads(getattr(args, n).read_text()) for n in ("first", "labels", "second", "baselines")))
    write_json(args.output, result)
    print(json.dumps({"critic_updates": result["critic_updates"], "third_scene_screen_eligible": result["third_scene_screen_eligible"],
        "selected_steps": [r["selected_steps"] for r in result["selection_ledger"]], "gates": result["gates"]}, indent=2), flush=True)
    print("NATIVE_SCENE_STOP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
