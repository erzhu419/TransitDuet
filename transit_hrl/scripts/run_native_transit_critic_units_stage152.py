#!/usr/bin/env python3
"""Isolate upper critic action units in learned native signed dispatch."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_dispatch_train_stage151 as training
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_critic_units_stage152_v1"
ROOTS = (293, 307)
METHODS = {
    "dispatch": ("channels", "physical_lower"),
    "dispatch_unit": ("channels", "physical_lower"),
    "dispatch_service_credit": ("channels", "physical_lower_service_credit"),
    "dispatch_service_credit_unit": ("channels", "physical_lower_service_credit"),
}
CONDITIONS = training.CONDITIONS
authority = training.authority
expected_updates = training.expected_updates
scene_seeds = training.scene_seeds


def contract(preflight):
    spec = training.contract(preflight)
    spec.update(roots=list(ROOTS),
        methods={key: {"coupling": mode, "authority_factor": factor}
                 for key, (mode, factor) in METHODS.items()},
        critic_action_units={key: "unit" if key.endswith("_unit") else "seconds" for key in METHODS},
        decision_clock={"channels": "max_zero_nominal_minus_120s"},
        statistics="two_root_descriptive_action_coordinates_by_credit_factorial",
        primary="unit_minus_seconds_cost_and_same_checkpoint_learned_minus_neutral_upper",
        decision="no_seed_expansion_without_useful_learned_upper_authority")
    return spec


def configure(base, method, root, *, preflight):
    _, factor = METHODS[method]
    cfg = authority.configure(base, factor, root, preflight=preflight)
    cfg["coupling"]["coupling_mode"] = "channels"
    cfg["upper"]["critic_action_units"] = contract(preflight)["critic_action_units"][method]
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = sys.modules[__name__]
    preflight = training.run_cell(args.method, args.seed, args.output.with_name("preflight.json"),
                                 preflight=True, experiment=experiment)
    result = training.run_cell(args.method, args.seed, args.output, preflight=False, experiment=experiment)
    result.update(worker_preflight_passed=True, worker_preflight_native_steps=preflight["native_steps"])
    write_json(args.output, result)
    print("NATIVE_CRITIC_UNITS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
