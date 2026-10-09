#!/usr/bin/env python3
"""Separate critic conditioning from the physical meaning of its L1 prior."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_critic_units_stage152 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_critic_regularization_stage153_v1"
ROOTS = (313, 331)
COORDINATES = {"seconds_sum": ("seconds", "sum"), "unit_sum": ("unit", "sum"),
    "unit_physical": ("unit", "physical_sum"), "unit_mean": ("unit", "mean")}
METHODS = {key: ("channels", "physical_lower_service_credit") for key in COORDINATES}
CONDITIONS = source.CONDITIONS
authority = source.authority
expected_updates = source.expected_updates
scene_seeds = source.scene_seeds


def contract(preflight):
    spec = source.contract(preflight)
    spec.update(roots=list(ROOTS),
        methods={key: {"coupling": mode, "authority_factor": factor}
                 for key, (mode, factor) in METHODS.items()},
        critic_action_units={key: units for key, (units, _) in COORDINATES.items()},
        weight_reg_mode={key: mode for key, (_, mode) in COORDINATES.items()},
        statistics="two_root_descriptive_coordinate_regularization_mechanism",
        primary="physical_regularizer_minus_unit_sum_and_same_checkpoint_upper_authority",
        regularization="same_coefficient_physical_sum_preserves_seconds_first_affine_layer_L1_mean_is_global_weakening_control")
    return spec


def configure(base, method, root, *, preflight):
    cfg = authority.configure(base, "physical_lower_service_credit", root, preflight=preflight)
    cfg["coupling"]["coupling_mode"] = "channels"
    units, mode = COORDINATES[method]
    cfg["upper"].update(critic_action_units=units, weight_reg_mode=mode)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = sys.modules[__name__]
    preflight = source.training.run_cell(args.method, args.seed, args.output.with_name("preflight.json"),
                                        preflight=True, experiment=experiment)
    result = source.training.run_cell(args.method, args.seed, args.output, preflight=False, experiment=experiment)
    result.update(worker_preflight_passed=True, worker_preflight_native_steps=preflight["native_steps"])
    write_json(args.output, result)
    print("NATIVE_CRITIC_REGULARIZATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
