"""Fixed causal state baselines on the exact Stage69 independent archives."""

import math
from scripts import pointmaze_independent_credit_stage69_spec as source
from scripts import pointmaze_horizon_value_stage67_spec as values_source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_mc_control_variate_stage70_v1"
POLICY = "mc_control_variate"
RUNNER_SCRIPT = "scripts/run_pointmaze_mc_control_variate_stage70.py"
PERIODS, TRAIN_POLICIES, TREATMENTS = source.PERIODS, source.TRAIN_POLICIES, source.TREATMENTS
roots, arguments = source.roots, source.arguments
ESTIMATORS = ("mc_common", "mc_control", "mc_factored", "gae_control", "gae_factored")


def options(*, preflight):
    return {**source.options(preflight=preflight), "workers": 2 if preflight else 4}


def seed_roles(root, *, preflight):
    return {"archive_batches": source.seed_roles(root, preflight=preflight)["fresh_batches"]}


def source_result(root, *, preflight):
    run = "pointmaze_independent_credit_stage69_" + ("preflight_20261001_r3" if preflight else "full_20261001_r1")
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def budget(*, preflight):
    opt = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    episodes = opt["batches"] * opt["episodes_per_batch"]
    cases = len(PERIODS) * len(TRAIN_POLICIES)
    forwards = cases * episodes * math.ceil(h / source.previous.CHUNK_SIZE)
    return {"archive_episodes": cases * episodes, "reconstructed_lower_calls": cases * episodes * h,
        "reconstructed_upper_calls": len(TRAIN_POLICIES) * episodes * sum(h // p for p in PERIODS),
        "archive_network_checks": cases * episodes, "source_clone_loads": len(PERIODS), "forecaster_loads": 1,
        "critic_checkpoint_loads": 2 * cases, "probe_value_rows": 2 * cases * episodes * h,
        "mc_calls": cases * opt["batches"], "gae_calls": 2 * cases * opt["batches"],
        "source_value_checks": 2 * cases * opt["batches"], "source_gradient_checks": cases,
        "actor_score_forward_batches": forwards, "actor_score_backward_batches": 7 * forwards,
        "control_variate_identity_checks": 2 * cases * 3, "frozen_model_checks": len(PERIODS) + 2 * cases}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "estimators": list(ESTIMATORS),
        "data": "all_Stage69_fresh_batches_same_eight_roots_periods_execution_and_episode_order_no_resampling",
        "baseline": "Stage64_control_Stage67_factored_critics_frozen_first_calibration_before_independent_archives",
        "causality": "critic_value_state_physical_history_executed_upper_plan_velocity_and_clock_before_current_lower_action",
        "reference": "unchanged_Stage69_time_only_rate_location_baseline_double_MC_recursion",
        "primary": "raw_uncentered_unscaled_MC_minus_fixed_state_baseline_episode_loss_gradients",
        "secondary": "separately_PPO_centered_scaled_each_batch_GAE_MC_directions_not_unbiased_raw_noise_estimates",
        "variance": "sample_covariance_trace_of_independent_episode_gradients_n32_full_n4_preflight",
        "decomposition": "g_state=g_common-h_var_state=var_common+var_h-2_trace_cov_common_h",
        "statistics": "raw_mean_snr_unbiased_signal_power_negative_estimates_retained_dependent_batch_pairs_descriptive_only",
        "checks": "Stage69_values_common_MC_and_GAE_noise_and_repeatability_reproduced_models_Adam_frozen",
        "decision": "diagnosis_only_no_actor_adoption_Stage67_HOLD_unchanged_no_fitted_CV_coefficient_or_baseline_selection",
        "limits": "reused_teacher_initialized_development_roots_uniform_time_discounted_surrogate_not_native_reward_gradient_or_frequency_proof"}
