#!/usr/bin/env python3
"""Test delayed service credit with a corrected eight-decision upper backup."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_critic_regularization_stage153 as source
from scripts import run_native_transit_dispatch_train_stage151 as training
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_trace_credit_stage155_v1"
ROOTS = (347, 359)
HORIZONS = {"one_step": 1, "retrace8": 8}
METHODS = {key: ("channels", "physical_lower_service_credit") for key in HORIZONS}
CONDITIONS = ("baseline", "neutral_upper", "fixed7", "zero_holding")
authority, scene_seeds, expected_updates = source.authority, source.scene_seeds, source.expected_updates


def contract(preflight):
    spec = source.contract(preflight)
    spec.update(roots=list(ROOTS), methods={key: {"coupling": mode, "authority_factor": factor}
        for key, (mode, factor) in METHODS.items()}, conditions=list(CONDITIONS),
        critic_action_units={key: "unit" for key in METHODS}, weight_reg_mode={key: "physical_sum" for key in METHODS},
        backup_horizon=HORIZONS, trace_lambda=.9,
        primary="retrace_minus_one_step_and_own_learned_minus_fixed7_or_neutral",
        statistics="two_root_descriptive_delayed_credit_not_confirmation",
        regularization="fixed_physical_sum_no_further_tuning",
        trace="stored_collection_density_clipped_pi_over_mu_soft_Bellman_residuals_terminal_masks",
        decision="no_expansion_without_state_adaptation_beyond_constant_phase")
    return spec


def configure(base, method, root, *, preflight):
    cfg = source.configure(base, "unit_physical", root, preflight=preflight)
    cfg["upper"].update(backup_horizon=HORIZONS[method], trace_lambda=.9)
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
    print("NATIVE_TRACE_CREDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
