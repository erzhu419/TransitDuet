"""Training-only velocity calibration and frozen input-support audit."""

import math
import numpy as np
from scripts import pointmaze_upper_paths_stage75_spec as source
from scripts import pointmaze_learned_plan_stage55_spec as labels
from scripts import pointmaze_matched_upper_stage57_spec as clones

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_velocity_support_stage76_v1"
POLICY = "velocity_support"
RUNNER_SCRIPT = "scripts/run_pointmaze_velocity_support_stage76.py"
PERIODS, roots, arguments = source.PERIODS, source.roots, source.arguments
VARIANTS = ("as_label", "zero_velocity", "sampled_velocity")
COVERAGE_METRICS = ("outside_label_axis_range_rate", "above_label_speed_q99_rate")
ENDPOINTS = tuple(f"{p}/{m}/residual_minus_base" for p in PERIODS for m in COVERAGE_METRICS)
CHUNK_SIZE, SUPPORT_QUANTILE = 1024, .99
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (76, 76076)


def source_result(root, *, preflight):
    run = "pointmaze_upper_paths_stage75_" + ("preflight" if preflight else "full") + "_20261001_r1"
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def seed_roles(root, *, preflight):
    return {"calibration_labels": labels.seed_roles(root, preflight=preflight)["labels"],
        "native_plan_replay": source.seed_roles(root, preflight=preflight)["native_evaluation"]}


def counterfactual_seed(root, seed, period):
    return int(np.random.SeedSequence([76, root, seed, period, 76017]).generate_state(1)[0])


def label_archive(root, period, seed, *, preflight):
    file = clones.source_result(root, preflight=preflight)
    return file.parent.with_name(file.parent.name + "_raw") / str(period) / "teacher" / "labels" / "0" / "deterministic" / f"episode_{seed}.npz"


def budget(*, preflight):
    r = roots(preflight=preflight)[0]
    roles, h = seed_roles(r, preflight=preflight), arguments(r, preflight=preflight).horizon
    n, e = len(roles["calibration_labels"]), len(roles["native_plan_replay"])
    episodes, calls = len(PERIODS) * (n + e), (n + e) * sum(h // p for p in PERIODS)
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "layout_loads": 1,
        "label_archive_loads": len(PERIODS) * n, "label_state_rows": len(PERIODS) * n * h,
        "label_plan_checks": len(PERIODS) * n, "bc_mse_reproductions": len(PERIODS),
        "offline_actor_rows": len(VARIANTS) * len(PERIODS) * n * h,
        "offline_actor_forward_batches": len(VARIANTS) * len(PERIODS) * n * math.ceil(h / CHUNK_SIZE),
        "counterfactual_upper_proposals": n * sum(h // p for p in PERIODS),
        "native_driver_paths": len(PERIODS) * e, "native_plan_frame_checks": len(PERIODS) * e,
        "replayed_velocity_rows": 2 * episodes * h, "plan_decodes": calls,
        "plan_ols_fits": calls - episodes, "plan_ridge_predictions": calls - episodes,
        "bernstein_basis_evaluations": (n + e) * sum(p + 1 for p in PERIODS),
        "frozen_model_checks": len(PERIODS), "native_steps": 0, "optimizer_steps": 0,
        "forecaster_fits": 0, "native_trace_writes": 0, "checkpoint_writes": 0}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "labels": labels.EXPERIMENT_PROTOCOL,
        "calibration": "all_actual_Stage55_BC_label_states_only_no_warmup_train_evaluation_or_reward_selected_scale",
        "support": "coordinate_min_max_and_speed_q99_of_label_velocity_position_error_q99_in_physical_units",
        "native_replay": "existing_Stage75_seeds_and_upper_proposals_causal_target_driver_prefix_original_world_clipping_exact_energy_frame",
        "counterfactual": "historical_label_states_hold_first390_columns_fixed_replace_only_velocity2_zero_or_sampled_residual",
        "proposal_noise": "new_Stage76_RNG_zero_mean_frozen_upper_Gaussian_marginals_not_native_episode_rng_reconstruction",
        "actor": "frozen_clone_batched_conditional_tanh_command_change_and_Gaussian_KL_no_optimization",
        "state_check": "zero_residual_plan_and_velocity_exact_BC_command_MSE_reproduces_saved_clone_MSE",
        "statistics": "four_residual_minus_base_coverage_contrasts_equal_root_bootstrap_Bonferroni4_all_reported",
        "decision": "input_support_diagnosis_and_training_only_calibration_no_budget_application_no_policy_adoption_Stage67_HOLD_unchanged",
        "artifacts": "compact_JSON_only_no_raw_trace_or_checkpoint_writes",
        "limits": "coordinate_and_speed_envelopes_not_joint_state_support_conditional_actor_response_not_native_reward_improvement"}
