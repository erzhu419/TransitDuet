#!/usr/bin/env python3
"""Restore causal full-tail summaries at the frozen proper-value schedule."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_proper_value_stage183 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

value, paired = source.value, source.paired
EXPERIMENT_PROTOCOL = "native_transit_tail_value_stage184_v1"
SOURCE_RUN = "native_transit_proper_value_stage183_fit_20261012_r1"
PRIMARY = "tail_summary:wait_feasible"


def contract():
    old = source.contract()
    return {**old, "source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "primary": PRIMARY, "critic": "98_16_3_tanh_1635_parameters_two_added_causal_full_tail_summaries",
        "added_features": list(value.TAIL_NAMES),
        "initialization": "same_seed178_control_weights_biases_and_heads_zero_added_columns",
        "control": "reuse_qualified_Stage183_proper_values_and_all_28_frozen_step_counts",
        "matching": "same_groups_states_loss_target_scales_optimizer_steps_and_initial_control_predictor",
        "boundary": "tail_summary_representation_ablation_not_complete_timetable_encoding_or_learned_HRL_confirmation"}


def alias_diagnostics(rows):
    old_aliases, remaining = [], []
    for row in rows:
        base, tail = value.features(row), value.features(row, "tail_summary")
        labels = {l["candidate"]: l for l in row["labels"]}
        for a in range(6):
            for b in range(a+1, 6):
                distinct = any(labels[a][h] != labels[b][h] for h in ("actual_cost_delta", "wait_delta_min"))
                if distinct and np.array_equal(base[a], base[b]):
                    record = {k: row[k] for k in ("root", "scene_seed", "probe_index")}
                    record.update(candidates=[a, b], tail_features=[tail[a, -2:].tolist(), tail[b, -2:].tolist()])
                    old_aliases.append(record)
                    if np.array_equal(tail[a], tail[b]): remaining.append(record)
    return {"old_distinct_outcome_aliases": old_aliases, "remaining_distinct_outcome_aliases": remaining}


def fit_and_validate(first, labels, second, baselines, control):
    if not (control["protocol"] == source.EXPERIMENT_PROTOCOL and control["contract"] == source.contract()
            and control["software_qualified"] and control["training_mode"] == "proper_mse"
            and len(control["fit_ledger"]) == 28):
        raise ValueError("Tail fitting requires the qualified frozen Stage183 proper-value control")
    fits = []
    def fitter(rows, method):
        old = control["fit_ledger"][len(fits)]
        if sorted(map(list, {paired.scene_group(r) for r in rows})) != old["training_groups"]:
            raise ValueError("Tail representation comparison changed training scenes")
        model = value.fit_value(rows, method, steps=old["steps"], objective="proper_mse", representation="tail_summary")
        fits.append(model["steps"])
        return model
    floor_source = source.source.source
    rows = floor_source.training_rows(first, labels, second, baselines)
    result = floor_source.fit_and_validate(first, labels, second, baselines, "floor_absolute", fitter=fitter)
    return {**result, "protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "training_mode": "tail_summary",
        "frozen_selected_steps": fits, "new_inner_updates": 0, "new_control_updates": 0,
        "representation_diagnostics": alias_diagnostics(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "labels", "second", "baselines", "control", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = fit_and_validate(*(json.loads(getattr(args, n).read_text()) for n in ("first", "labels", "second", "baselines", "control")))
    write_json(args.output, result)
    print(json.dumps({"critic_updates": result["critic_updates"], "gates": result["gates"],
        "representation_diagnostics": result["representation_diagnostics"]}, indent=2), flush=True)
    print("NATIVE_TAIL_VALUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
