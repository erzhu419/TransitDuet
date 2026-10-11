#!/usr/bin/env python3
"""Freeze the Stage178 primary and measure second-scene one-action transfer."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_data_stage174 as labels
from scripts import run_native_transit_value_context_stage177 as context
from scripts import run_native_transit_occupancy_value_stage178 as fitted
from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit import native_occupancy_value as value
from freq_hrl.domains.transit.native_value_context import NativeValueContextAudit, PROBES
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

shared, native = labels.shared, labels.native
ROOTS, SCENARIOS = labels.ROOTS, labels.SCENARIOS
EXPERIMENT_PROTOCOL = "native_transit_value_transfer_stage179_v1"
SOURCE_RUN = "native_transit_occupancy_value_stage178_fit_20261011_r1"
OCCUPANCY_KEYS = ("historical_peak", "active_buses", "queue_by_station", "route_speed_mps")
records, validate_continuations = labels.records, labels.validate_continuations


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": fitted.EXPERIMENT_PROTOCOL,
        "primary": fitted.PRIMARY, "roots": list(ROOTS), "scenarios": list(SCENARIOS), "probes": list(PROBES),
        "scene": "second_sorted_registered_Stage169_scene_per_regime_no_outcome_selection",
        "episodes_per_cell": 7, "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "intervention": shared.contract()["intervention"], "pairing": shared.contract()["pairing"],
        "model": "exact_Stage178_primary_full_fit_weights_and_training_only_normalization_no_refit",
        "prediction": "sealed_in_native_preaction_callback_before_observer_future_and_all_counterfactual_labels",
        "qualification": "60_exact_observers_60_exact_identity_controls_360_paired_labels_420_window_ledgers",
        "analysis": "frozen_selected_actual_cost_fleet_nonfleet_wait_reward_regret_by_root_regime_direction",
        "boundary": "development_scene_transfer_of_one_action_values_not_closed_loop_HRL_or_final_confirmation"}


def source_record(root, scenario):
    return labels.source_record(root, scenario, scene_index=1)


def model_input(cell):
    model = cell["model"]
    if not (cell["protocol"] == fitted.EXPERIMENT_PROTOCOL and cell["contract"] == fitted.contract()
            and cell["software_qualified"] and cell["method"] == fitted.PRIMARY
            and cell["critic_updates"] == 18*value.STEPS and cell["actor_updates"] == cell["new_native_steps"] == 0
            and model["method"] == fitted.PRIMARY and model["steps"] == value.STEPS
            and model["training_states"] == 60 and len(model["training_groups"]) == 10
            and model["parameter_count"] == 1586):
        raise ValueError("Transfer requires the exact completed Stage178 primary full fit")
    return {"source_run": SOURCE_RUN, "source_protocol": fitted.EXPERIMENT_PROTOCOL,
        "primary": fitted.PRIMARY, "model": model}


def frozen_prediction(row, model):
    heads = value.predict_value(row, model)
    return {"time_s": row["time_s"], "predicted_heads": heads.tolist(),
        "selected": paired.choice_from_cost(heads.sum(axis=1), row["original_choice"])}


class FrozenValueTransferAudit(NativeValueContextAudit):
    def __init__(self, env, model, probe_index, *, value_model, forced_candidate=None, reference=None):
        super().__init__(env, model, probe_index, forced_candidate=forced_candidate, reference=reference)
        self.value_model, self.prediction = value_model, None

    def select_candidate(self, candidates):
        selected = super().select_candidate(candidates)
        if len(self.decisions) == self.probe_index:
            causal = {**self.probe, "original_choice": self.probe["greedy"],
                "context": labels.training_context(self.snapshot),
                "occupancy_context": {k: self.contexts[-1][k] for k in OCCUPANCY_KEYS}}
            self.prediction = frozen_prediction(causal, self.value_model)
            print("FROZEN_VALUE_PREDICTION_SEALED " + json.dumps(self.prediction), flush=True)
        return selected


def collect(root, scenario, index, duration_model, value_model, output):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = native.NativeRoutingTracker
    torch.set_num_threads(1)
    shared.source.incumbent.source.distributional.validate_model(duration_model)
    registered = source_record(root, scenario)
    scene = registered["scene_seed"]
    if [root, scene] in value_model["training_groups"]:
        raise ValueError("Transfer scene was fitted by the frozen critic")
    _, checkpoint = native.load_source(root)
    raw = raw_directory(output)
    baseline, plan = shared.episode(root, scenario, scene, raw/"observer", checkpoint, duration_model, index,
        plan_class=partial(FrozenValueTransferAudit, value_model=value_model))
    shared.qualify(baseline, {**registered, "probe": baseline["probe"]})
    baseline["policy_trace"] = shared.policy_trace(baseline["supply"])
    cell = {"seed": root, "scenario": scenario, "probe_index": index, "observer": baseline, "branches": [],
        "training_context": labels.training_context(plan.snapshot),
        "context": next(c for c in plan.contexts if c["index"] == index), "prediction": plan.prediction}
    context.qualify_contexts([cell["context"]], [records(cell)], registered)
    write_json(output.parent/"prediction.json", plan.prediction)
    print("VALUE_TRANSFER_OBSERVER_SOURCE_QUALIFIED", flush=True)
    greedy = baseline["probe"]["greedy"]
    for candidate in [greedy, *[i for i in shared.CANDIDATES if i != greedy]]:
        row, _ = shared.episode(root, scenario, scene, raw/f"candidate_{candidate}", checkpoint,
            duration_model, index, candidate=candidate, reference=plan)
        shared.qualify(row, baseline, candidate=candidate)
        row["policy_trace"] = shared.policy_trace(row.pop("supply"))
        cell["branches"].append(row)
        print(f"value_transfer root={root} {scenario}/{index}/{candidate} "
            f"cost={row['service_cost_restricted']} peak={row['peak_fleet']} identity={row['source_control_reproduced']}", flush=True)
        if candidate == greedy:
            print("VALUE_TRANSFER_IDENTITY_SOURCE_QUALIFIED", flush=True)
    cell["branches"].sort(key=lambda r: r["probe"]["selected"])
    validate_continuations(baseline, cell["branches"])
    return {**cell, "software_qualified": True, "training_updates": 0, "native_steps": 7*61380}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--probe", type=int, choices=PROBES, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--value-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack = json.loads(args.value_model.read_text())
    if (pack["source_run"], pack["source_protocol"], pack["primary"], pack["model"]["method"]) != (
            SOURCE_RUN, fitted.EXPERIMENT_PROTOCOL, fitted.PRIMARY, fitted.PRIMARY):
        raise ValueError("Changed frozen primary model input")
    result = collect(args.seed, args.scenario, args.probe,
        json.loads(args.models.read_text())[str(args.seed)], pack["model"], args.output)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), **result})
    print("NATIVE_VALUE_TRANSFER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
