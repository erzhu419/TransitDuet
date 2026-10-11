#!/usr/bin/env python3
"""Fit the complete timetable event encoder at the frozen value schedule."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_timetable_context_stage185 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

prior = source.prior
floor_source = prior.source.source.source
value, paired = prior.value, prior.paired
EXPERIMENT_PROTOCOL = "native_transit_sequence_value_stage186_v1"
SOURCE_RUN = "native_transit_timetable_context_stage185_frozen_20261012_r1"
CONTROL_RUN = source.SOURCE_RUN
PRIMARY = "timetable_sequence:wait_feasible"


def contract():
    return {**prior.contract(), "source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "primary": PRIMARY, "critic": "98_base_features_plus_16_learned_event_features_114_16_3_2147_parameters",
        "encoder": "all_real_events_shared_6_8_tanh_8_8_kernel3_tanh_masked_mean_max_no_truncation",
        "event_scaling": "all_real_training_events_only_per_channel_mean_std_no_padding_in_normalization",
        "initialization": "same_seed178_control_core_weights_biases_and_heads_zero_sequence_projection",
        "control": "reuse_qualified_Stage184_tail_summary_values_and_all_28_frozen_step_counts",
        "matching": "same_groups_states_labels_loss_base_target_scales_optimizer_steps_and_initial_predictor",
        "boundary": "learned_full_sequence_value_screen_not_repeated_upper_actions_or_end_to_end_HRL_confirmation"}


def training_rows(data, original):
    if not (data["protocol"] == source.EXPERIMENT_PROTOCOL and data["contract"] == source.contract()
            and data["software_qualified"] and data["source_control_reproductions"] == 20
            and data["training_states"] == 120 and data["paired_labels_reused"] == 720
            and data["new_native_episodes"] == 20 and data["new_native_steps"] == 20*61380
            and data["training_updates"] == 0):
        raise ValueError("Sequence fitting requires qualified complete timetable inputs")
    rows = data["training_data"]
    if [{k: v for k, v in r.items() if k != "timetable_context"} for r in rows] != original:
        raise ValueError("Sequence fitting changed original state order, context or cost/wait labels")
    return rows


def fit_and_validate(data, original, control):
    rows = training_rows(data, original)
    if not (control["protocol"] == prior.EXPERIMENT_PROTOCOL and control["contract"] == prior.contract()
            and control["software_qualified"] and control["training_mode"] == "tail_summary"
            and len(control["fit_ledger"]) == 28):
        raise ValueError("Sequence fitting requires the frozen Stage184 control")
    fits = []
    def fitter(train, method):
        old = control["fit_ledger"][len(fits)]
        if sorted(map(list, {paired.scene_group(r) for r in train})) != old["training_groups"]:
            raise ValueError("Sequence comparison changed training scenes")
        model = value.fit_value(train, method, steps=old["steps"], objective="proper_mse", representation="timetable_sequence")
        fits.append(model["steps"])
        return model
    result = floor_source.fit_rows_and_validate(rows, "floor_absolute", fitter=fitter)
    return {**result, "protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "training_mode": "timetable_sequence",
        "frozen_selected_steps": fits, "new_inner_updates": 0, "new_control_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "original", "control", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = fit_and_validate(*(json.loads(getattr(args, n).read_text()) for n in ("data", "original", "control")))
    write_json(args.output, result)
    print(json.dumps({"critic_updates": result["critic_updates"], "gates": result["gates"],
        "third_scene_screen_eligible": result["third_scene_screen_eligible"]}, indent=2), flush=True)
    print("NATIVE_SEQUENCE_VALUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
