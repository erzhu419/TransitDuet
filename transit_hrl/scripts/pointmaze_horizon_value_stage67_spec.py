"""Finite-horizon value ablation; actors and upper stay frozen."""

import math
from scripts import pointmaze_value_targets_stage64_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_horizon_value_stage67_v1"
POLICY = "horizon_value"
RUNNER_SCRIPT = "scripts/run_pointmaze_horizon_value_stage67.py"
PERIODS, TRAIN_POLICIES = source.PERIODS, source.TRAIN_POLICIES
roots, arguments, seed_roles, training_result = source.roots, source.arguments, source.seed_roles, source.training_result
TREATMENTS = ("mc_normalized", "mc_factored")
CANDIDATE, WORKERS = "mc_factored", 2


def source_result(root, *, preflight):
    run = "pointmaze_value_targets_stage64_" + ("preflight" if preflight else "full") + "_20261001_r1"
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def options(*, preflight):
    return {**source.options(preflight=preflight), "workers": WORKERS}


def budget(*, preflight):
    old, opt = source.budget(preflight=preflight), options(preflight=preflight)
    cases = len(PERIODS) * len(TRAIN_POLICIES)
    n = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon * opt["rollouts_per_iteration"]
    fits = cases * len(TREATMENTS)
    chunks = math.ceil(n / 1024)
    return {"native_steps": 0, "actor_optimizer_steps": 0, "new_forecaster_fits": 0,
        "source_clone_loads": len(PERIODS), "forecaster_loads": 1,
        "archive_episodes": old["archive_episodes"], "reconstructed_lower_calls": old["reconstructed_lower_calls"],
        "reconstructed_upper_calls": old["reconstructed_upper_calls"], "archive_network_checks": old["archive_episodes"],
        "mc_target_calls": cases * (opt["critic_warmup_iterations"] + 1),
        "calibration_updates": fits * opt["critic_warmup_iterations"],
        "normalization_frame_fits": fits, "initialization_value_rows": fits * n,
        "probe_value_rows": fits * n, "source_critic_loads": cases, "source_control_checks": cases,
        "frozen_state_checks": 6 * fits, "probe_gae_calls": fits,
        "actor_score_forward_batches": fits * chunks, "actor_score_backward_batches": 3 * fits * chunks,
        "frozen_model_checks": fits, "critic_checkpoint_writes": cases}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "treatments": list(TREATMENTS),
        "pairing": "same_Stage57_calibration_and_disjoint_first_training_probe_paths_no_new_sampling",
        "control": "Stage64_mc_normalized_training_public_weights_Adam_and_probe_exact",
        "candidate": "V=m_gamma(round(H*existing_remaining_fraction))*(location+scale*rate_network)",
        "mass": "sum_gamma_k_for_remaining_steps_terminal_zero_no_new_clock_or_future_input",
        "initialization": "same_clone_hidden_weights_empty_Adam_head_rebased_at_full_horizon_initial_V_tapered_by_m_remaining/m_H",
        "normalization": "fixed_mean_std_of_MC/m_remaining_first_calibration_only",
        "training": "same_MC_paths_gamma_LR_epochs_minibatches_shuffle_value_coef_clip_and_optimizer_step_count",
        "objective": "normalized_rate_MSE_changes_time_weighting_not_only_inference_multiplier",
        "frozen": "actor_upper_and_their_Adam_bitexact_no_actor_updates",
        "fit_gate": "every_case_EV_at_least_0.10_global_MSE_tail_MSE_and_absolute_tail_bias_below_control",
        "credit_gate": "positive_mean_gradient_cosine_every_case_and_each_equal_root_group_mean_sign_disagreement_not_higher_mean_and_log_std_cosines_not_lower",
        "checkpoint": "explicit_factored_rate_weights_and_training_unit_Adam_not_ordinary_public_ValueNet_weights",
        "decision": "both_full_gates_required_before_guarded_actor_native_trial_no_seed_LR_or_gate_sweep",
        "limits": "development_archive_value_and_credit_test_not_reward_frequency_or_generalization_evidence"}
