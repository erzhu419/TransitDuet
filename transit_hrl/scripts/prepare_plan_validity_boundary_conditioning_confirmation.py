"""Reserve fresh prefix/future roles and a fixed readout; submit no native work."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_development import load_plan
from freq_hrl.experiments.plan_validity_preregistration import seed_cells
from freq_hrl.experiments.pointmaze_goal_validation import _training_seed
from scripts.submit_plan_validity_independent_calibration_scheduleurm import literal_seeds


def historical_seeds(protocols, plan):
    forbidden = set().union(*(literal_seeds(json.loads(p.read_text())) for p in protocols.glob("plan_validity*.json")), set())
    for c in seed_cells(plan):
        forbidden.update(literal_seeds(c))
        forbidden.update(_training_seed(optimizer_seed=c["optimizer_seed"], rollout_root=s, iteration=i)
            for s in c["seed_roles"]["train_seeds"]
            for i in range(2 if c["phase"] == "preflight" else plan["contract"]["primary_iterations"]))
    return forbidden


def prepare(design, *, freeze_manifest, forbidden):
    occupied = set(forbidden) | literal_seeds(design)
    horizon, checks, windows = design["horizon"], design["checks_per_prefix"], design["windows_per_state"]
    suffixes = list(range(40, 48))
    pairs_per_root = 16 * checks * windows * len(suffixes)
    steps_per_root = horizon * (16 + 2 * pairs_per_root)
    cells = []
    for source in design["cells"]:
        root = source["optimizer_seed"]
        prefixes, bootstrap = list(range(root + 69000, root + 69016)), root + 69800
        roles = [*prefixes, bootstrap]
        if len(set(roles)) != 17 or occupied.intersection(roles):
            raise ValueError("fresh confirmation overlaps historical, reserved or effective training seeds")
        occupied.update(roles)
        cells.append({"task": source["task"], "optimizer_seed": root, "readout_prefix_seeds": prefixes,
            "bootstrap_seed": bootstrap, "proposed_environment_steps": steps_per_root})
    return {"protocol_version": "frozen_conditioning_fresh_prefix_confirmation_plan_v1",
        "run_name": "plan_validity_boundary_conditioning_confirmation_20261111", "freeze_manifest": str(freeze_manifest),
        "primary_root": {"task": "reacher_tracking", "optimizer_seed": 1972210000}, "cells": cells,
        "horizon": horizon, "checks_per_prefix": checks, "windows_per_state": windows, "suffix_indices": suffixes,
        "methods": ["zero", "constant", "causal_t", "base", "wide", "tracking_distance", "state_rms"],
        "primary_comparators": ["wide", "state_rms", "base", "zero"],
        "seed_exclusion": "all_protocol_literal_roles_plus_effective_controller_training_seeds",
        "exclusion_audit": {"registered_and_effective_count": len(forbidden)},
        "sampling": "unchanged_C0_observe617_839_three_checks_original11401_windows_11402_suffix_indices_no_outcome_selection",
        "proposed_shards": len(cells) * 4, "prefixes_per_shard": 4, "proposed_steps_per_shard": steps_per_root // 4,
        "proposed_environment_steps": len(cells) * steps_per_root, "new_paired_futures": len(cells) * pairs_per_root,
        "factual_episodes": len(cells) * 16,
        "bootstrap_replicates": 10000, "bootstrap_unit": "whole_prefix_mean_paired_loss_contrast_all_checks_windows_futures_stay_together",
        "bootstrap_pairing": "same_resampled_prefix_indices_for_all_method_contrasts_on_each_root",
        "primary_rule": "tracking_minus_each_of_four_controls_has_nominal_one_sided_prefix_bootstrap95_upper_below_zero_and_negative_means_in_both_suffix_window_and_prefix_halves",
        "bound_scope": "each_comparison_nominal_bounds_not_simultaneous_coverage_no_suffix_MC_variance_subtraction_or_double_addition",
        "halves": {"suffix": [[40, 41, 42, 43], [44, 45, 46, 47]], "window": [[0, 1], [2, 3]], "prefix": "first8_last8_in_declared_order"},
        "statistics": "separate_roots_prefix_uncertainty_and_conditional_suffix_MC_SE_no_pseudotraining_seeds",
        "freeze_gate": "all_eight_models_and_all_old_controls_complete_before_any_fresh_prefix_access",
        "stopping": "fixed_16_prefixes_per_root_no_outcome_based_early_stop_or_budget_extension",
        "scope": "frozen_model_boundary_tail_prediction_not_complete_episode_control_or_independent_training_seed_confirmation",
        "native_sampling_released": False, "evaluation_prefixes_accessed": False, "jobs": [],
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "protocols/plan_validity_preregistration_2026-10-22.json")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve reserved confirmation roles and statistics")
    plan = load_plan(args.preregistration)
    result = prepare(json.loads(args.design.read_text()), freeze_manifest=args.freeze_manifest.resolve().relative_to(ROOT),
        forbidden=historical_seeds(ROOT / "protocols", plan))
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cells": len(result["cells"]), "proposed_environment_steps": result["proposed_environment_steps"],
        "native_sampling_released": False, "exclusion_audit": result["exclusion_audit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
