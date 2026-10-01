"""Historical actor-covariance coefficients, frozen before independent probes."""

import math
from scripts import pointmaze_mc_control_variate_stage70_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_calibrated_cv_stage71_v1"
POLICY = "calibrated_cv"
RUNNER_SCRIPT = "scripts/run_pointmaze_calibrated_cv_stage71.py"
PERIODS, TRAIN_POLICIES = source.PERIODS, source.TRAIN_POLICIES
roots, arguments = source.roots, source.arguments
ESTIMATORS = (*source.ESTIMATORS, "mc_calibrated_control", "mc_calibrated_factored")


def options(*, preflight):
    old = source.values_source.options(preflight=preflight)
    return {**source.options(preflight=preflight),
        "calibration_batches": old["critic_warmup_iterations"],
        "calibration_episodes_per_batch": old["rollouts_per_iteration"]}


def seed_roles(root, *, preflight):
    roles = source.values_source.seed_roles(root, preflight=preflight)
    probes = source.seed_roles(root, preflight=preflight)["archive_batches"]
    if set(roles["calibration"]).intersection(s for batch in probes for s in batch):
        raise ValueError("Stage71 historical calibration/probe seeds overlap")
    return {"calibration": roles["calibration"], "archive_batches": probes}


def source_result(root, *, preflight):
    run = "pointmaze_mc_control_variate_stage70_" + ("preflight" if preflight else "full") + "_20261001_r1"
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def budget(*, preflight):
    opt = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    nc = opt["calibration_batches"] * opt["calibration_episodes_per_batch"]
    np = opt["batches"] * opt["episodes_per_batch"]
    cases = len(PERIODS) * len(TRAIN_POLICIES)
    chunks = math.ceil(h / source.source.previous.CHUNK_SIZE)
    return {"calibration_archive_episodes": cases * nc, "probe_archive_episodes": cases * np,
        "reconstructed_lower_calls": cases * (nc + np) * h,
        "reconstructed_upper_calls": len(TRAIN_POLICIES) * (nc + np) * sum(h // p for p in PERIODS),
        "archive_network_checks": cases * (nc + np), "source_clone_loads": len(PERIODS), "forecaster_loads": 1,
        "critic_checkpoint_loads": 2 * cases, "value_prediction_rows": 2 * cases * (nc + np) * h,
        "mc_calls": cases * (opt["calibration_batches"] + opt["batches"]),
        "gae_calls": 2 * cases * opt["batches"], "control_variate_coefficient_fits": 2 * cases,
        "source_gradient_checks": cases, "control_variate_identity_checks": 4 * cases * 3,
        "actor_score_forward_batches": cases * (nc + np) * chunks,
        "actor_score_backward_batches": cases * (5 * nc + 9 * np) * chunks,
        "frozen_model_checks": len(PERIODS) + 2 * cases}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "estimators": list(ESTIMATORS),
        "calibration": "all_original_Stage57_warmup_episodes_only_same_frozen_actor_and_Stage64_Stage67_critics",
        "coefficient": "one_signed_scalar_per_critic_case_trace_Cov(g0,h)/trace_Var(h)_all_actor_parameters_no_clipping_or_sweep",
        "zero_variance": "exact_zero_baseline_sample_variance_uses_alpha_zero",
        "candidate": "MC-b0-alpha*(V-b0)_detached_causal_value_state_before_current_lower_action",
        "reference": "unchanged_Stage69_time_only_baseline_first_calibration_rate_location_double_MC",
        "probe": "all_Stage69_archives_same_order_no_probe_labels_in_alpha_fit_no_resampling",
        "primary": "raw_uncentered_unscaled_independent_episode_loss_gradient_covariance_SNR_cross_batch_cosines",
        "secondary": "separate_per_batch_PPO_centered_scaled_directions_not_unbiased_noise_estimates",
        "statistics": "equal_root_descriptive_development_diagnosis_negative_signal_power_retained",
        "checks": "all_Stage70_probe_estimators_reproduced_models_and_Adam_bitexact_exact_counts",
        "decision": "no_actor_adoption_optimizer_or_critic_updates_Stage67_HOLD_unchanged",
        "limits": "critic_and_alpha_share_historical_calibration_probes_reused_from_Stage69_70_not_new_confirmation_uniform_time_discounted_surrogate_not_native_reward_or_frequency_proof"}
